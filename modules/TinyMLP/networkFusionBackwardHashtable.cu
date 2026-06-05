#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "TinyMLPHashGrid.h"

__device__ __forceinline__ void cp_async_128(void* smem, const void* global, bool valid) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    
    int src_size = valid ? 16 : 0; 
    #ifndef __INTELLISENSE__
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_int), "l"(global), "r"(src_size)
        : "memory"
    );
    #endif
}

__device__ __forceinline__ void cp_async_commit() {
    #ifndef __INTELLISENSE__
    asm volatile("cp.async.commit_group;\n" ::);
    #endif
}

__device__ __forceinline__ void cp_async_wait() {
    #ifndef __INTELLISENSE__
    asm volatile("cp.async.wait_group 0;\n" ::);
    #endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_n() {
    #ifndef __INTELLISENSE__
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    #endif
}

__device__ __forceinline__ void redAddF32(float* __restrict__ addr, float val) {
    #ifndef __INTELLISENSE__
    asm volatile(
        "red.global.add.f32 [%0], %1;"
        :: "l"(addr), "f"(val)
        : "memory"
    );
    #endif
}

__device__ __forceinline__ void loadVec4(const float* __restrict__ ptr, float dst[4]) {
    #ifndef __INTELLISENSE__
    asm volatile(
        "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "l"(ptr)
    );
    #endif
}

__device__ __forceinline__ uint32_t hash_coords(int cx, int cy, int cz, int T) {
    return ((static_cast<uint32_t>(cx) * 1U) ^ 
            (static_cast<uint32_t>(cy) * 2654435761U) ^ 
            (static_cast<uint32_t>(cz) * 805459861U)) & static_cast<uint32_t>(T - 1);
}

// Step 3 (Case A): Flat Index for dense 1:1 mapping at coarse levels
__device__ __forceinline__ uint32_t dense_index(int cx, int cy, int cz, int N_l) {
    return cx + cy * (N_l + 1) + cz * (N_l + 1) * (N_l + 1);
}


template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int TILE_FACTOR, int FEATURES_LEVEL>
__global__ void networkFusionMMA_Backward(
    half* d_dx_out,
    const half* __restrict__ d_loss_output,
    half** d_weights_array,
    half** d_activations,
    float* d_inputs,
    float** d_grad_weights,
    float** d_grad_biases,
    float* d_hashtable_grads,
    int batchSize,
    int inputDim,
    int hiddenDim,
    int outputDim,
    int numLayers,
    int tableSize,
    int numLevels,
    float b,
    int lowestSize,
    int denseLevelStart
) {
    int idx = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x + 1 - 1;

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    const int PAD = 8;

    __shared__ half shmem_A[WMMA_M * TILE_COUNT_Y * TILE_FACTOR][TILE_COUNT_X * WMMA_K + PAD];
    __shared__ half shmem_B[2][WMMA_K][TILE_COUNT_X * WMMA_N + PAD];

    #pragma unroll
    for (int i = 0; i < TILE_FACTOR; i++) {
        int chunk_row = (idx + i*256) / (TILE_COUNT_X * WMMA_K / 8);
        int chunk_col = (idx + i*256) % (TILE_COUNT_X * WMMA_K / 8) * 8;

        int global_row = (blockIdx.x * TILE_COUNT_Y * TILE_FACTOR) * WMMA_M + chunk_row;
        int global_col = chunk_col;

        bool a_valid = (global_row < batchSize) && (global_col < outputDim);
        const half* a_src = a_valid ? &d_loss_output[global_row * outputDim + global_col] : d_loss_output;
        cp_async_128(&shmem_A[chunk_row][chunk_col], a_src, a_valid);
        cp_async_commit();
    }

    cp_async_wait();
    __syncthreads();

    int layer = numLayers - 1;
    
    {
        int threads_per_row_b = WMMA_K / 8;
        int total_threads_needed_b = (TILE_COUNT_X * WMMA_N) * threads_per_row_b;
    
        int chunk_row_b = 0, chunk_col_b = 0, global_row_b = 0;
    
        if (idx < total_threads_needed_b) {
            chunk_row_b = idx / (TILE_COUNT_X * WMMA_K / 8);
            chunk_col_b = (idx % (TILE_COUNT_X * WMMA_K / 8)) * 8;
            global_row_b = blockIdx.x * TILE_COUNT_Y * TILE_FACTOR * WMMA_M + chunk_row_b;
            bool b_valid = (global_row_b < batchSize) && (chunk_col_b < hiddenDim);
            const half* b_src = b_valid ? &d_activations[layer][global_row_b * hiddenDim + chunk_col_b] : d_activations[layer];
            cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b], b_src, b_valid);
        }
        
        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        int buf = 0;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_FACTOR];
        #pragma unroll
        for (int i = 0; i < WARP_FACTOR; i++) {
            nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
        }

        #pragma unroll
        for (int k = 0; k < TILE_COUNT_Y * TILE_FACTOR * WMMA_M; k += WMMA_K) {
            int next_buf = 1 - buf;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_b;

            if (k + WMMA_K < TILE_COUNT_Y * TILE_FACTOR * WMMA_M && idx < total_threads_needed_b) {
                bool b_valid = (global_row_b + k + WMMA_K < batchSize) && (chunk_col_b < hiddenDim);
                const half* b_src = b_valid ? &d_activations[layer][(global_row_b + k + WMMA_K) * hiddenDim + chunk_col_b] : d_activations[layer];
                cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b], b_src, b_valid);
                cp_async_commit();
            }

            #pragma unroll
            for (int i = 0; i < WARP_FACTOR; i++) {
                nvcuda::wmma::load_matrix_sync(frag_a, &shmem_B[buf][0][(threadIdx.z + i*blockDim.z)*WMMA_M], TILE_COUNT_X * WMMA_N + PAD);
                nvcuda::wmma::load_matrix_sync(frag_b, &shmem_A[k][threadIdx.y*WMMA_K], TILE_COUNT_X*WMMA_K + PAD);
                nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
            }

            if (k + WMMA_K < TILE_COUNT_Y * TILE_FACTOR * WMMA_M) {
                cp_async_wait();
                __syncthreads();
            }
            buf = next_buf;
        }

        #pragma unroll
        for(int j = 0; j < WARP_FACTOR; j++) {
            #pragma unroll
            for (int i = 0; i < frag_acc[j].num_elements; i++) {
                int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                
                int global_row = (threadIdx.z + j*blockDim.z)*WMMA_M + local_row;
                int global_col = threadIdx.y * WMMA_N + local_col;

                float val = frag_acc[j].x[i];
                if (global_row < hiddenDim && global_col < outputDim) {
                    redAddF32(&d_grad_weights[layer][global_col * hiddenDim + global_row], val);
                }
            }
        }
    }

    float local_db = 0.0f;
    int global_col = threadIdx.y * 16 + (laneId % 16);
    
    #pragma unroll
    for (int i = 0; i < TILE_FACTOR; i++) {
        int row_start = (threadIdx.z + i*blockDim.z) * WMMA_M + (laneId / 16) * 8;
        #pragma unroll
        for(int j = 0; j < 8; j++) {
            local_db += __half2float(shmem_A[row_start + j][global_col]);
        }
    }
    local_db += __shfl_down_sync(0xFFFFFFFF, local_db, 16);

    if (laneId < 16 && global_col < outputDim) {
        redAddF32(&d_grad_biases[layer][global_col], local_db);
    }
    __syncthreads();

    {
        int threads_per_row_w = (TILE_COUNT_X * WMMA_N) / 8; 
        int total_threads_w = WMMA_K * threads_per_row_w; 

        int chunk_row_b = idx / threads_per_row_w;
        int chunk_col_b = (idx % threads_per_row_w) * 8;
        
        if (idx < total_threads_w) {
            bool b_valid = (chunk_row_b < outputDim) && (chunk_col_b < hiddenDim);
            const half* w_src = b_valid ? &d_weights_array[layer][chunk_row_b * hiddenDim + chunk_col_b] : d_weights_array[layer];
            cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b], w_src, b_valid);
        }
        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        int buf = 0;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[TILE_FACTOR];
        #pragma unroll
        for (int i = 0; i < TILE_FACTOR; i++) {
            nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
        }

        for (int k = 0; k < outputDim; k += WMMA_K) {
            int next_buf = 1 - buf;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_b;

            if (k + WMMA_K < outputDim) {
                if (idx < total_threads_w) {
                    bool isValid = (chunk_row_b + k + WMMA_K < outputDim) && (chunk_col_b < hiddenDim);
                    const half* w_src = isValid ? &d_weights_array[layer][(chunk_row_b + k + WMMA_K) * hiddenDim + chunk_col_b] : d_weights_array[layer];
                    cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b], w_src, isValid);
                }
                cp_async_commit();
            }

            nvcuda::wmma::load_matrix_sync(frag_b, &shmem_B[buf][0][threadIdx.y * WMMA_K], TILE_COUNT_X * WMMA_N + PAD);
            
            #pragma unroll
            for (int i = 0; i < TILE_FACTOR; i++) {
                nvcuda::wmma::load_matrix_sync(frag_a, &shmem_A[(threadIdx.z + i*blockDim.z)*WMMA_M][k], TILE_COUNT_X*WMMA_K + PAD);
                nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
            }

            if (k + WMMA_K < outputDim) {
                cp_async_wait();
                __syncthreads();
            }
            buf = next_buf;
        }

        __syncthreads();

        int base_col = threadIdx.y * WMMA_N;
        #pragma unroll
        for(int j = 0; j < TILE_FACTOR; j++) {
            #pragma unroll
            for (int i = 0; i < frag_acc[j].num_elements; i++) {
                int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;

                float val = frag_acc[j].x[i];
                shmem_A[(threadIdx.z + j*blockDim.z)*WMMA_M + local_row][base_col + local_col] = __float2half(val);
            }
        }
        __syncthreads();

        int total_threads = blockDim.x * blockDim.y * blockDim.z;
        int total_elements = (TILE_COUNT_Y * TILE_FACTOR * WMMA_M) * hiddenDim;

        for (int i = idx; i < total_elements; i += total_threads) {
            int r = i / hiddenDim; 
            int c = i % hiddenDim; 
            int global_r = (blockIdx.x * TILE_COUNT_Y * TILE_FACTOR * WMMA_M) + r;
            
            if (global_r < batchSize && c < hiddenDim) {
                float orig_act = __half2float(d_activations[layer][global_r * hiddenDim + c]);
                if (orig_act <= 0.0f) {
                    shmem_A[r][c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();
    }
    
    layer--;

    while (layer > 0) {
        {
            int threads_per_row_b = WMMA_K / 8;
            int total_threads_needed_b = (TILE_COUNT_X * WMMA_N) * threads_per_row_b;
        
            int chunk_row_b = 0, chunk_col_b = 0, global_row_b = 0;
        
            if (idx < total_threads_needed_b) {
                chunk_row_b = idx / (TILE_COUNT_X * WMMA_K / 8);
                chunk_col_b = (idx % (TILE_COUNT_X * WMMA_K / 8)) * 8;
                global_row_b = blockIdx.x * TILE_COUNT_Y * TILE_FACTOR * WMMA_M + chunk_row_b;
                bool b_valid = (global_row_b < batchSize) && (chunk_col_b < hiddenDim);
                const half* b_src = b_valid ? &d_activations[layer][global_row_b * hiddenDim + chunk_col_b] : d_activations[layer];
                cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b], b_src, b_valid);
            }
            
            cp_async_commit();
            cp_async_wait();
            __syncthreads();

            int buf = 0;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_FACTOR];
            #pragma unroll
            for (int i = 0; i < WARP_FACTOR; i++) {
                nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
            }

            #pragma unroll
            for (int k = 0; k < TILE_COUNT_Y * TILE_FACTOR * WMMA_M; k += WMMA_K) {
                int next_buf = 1 - buf;

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_a;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_b;

                if (k + WMMA_K < TILE_COUNT_Y * TILE_FACTOR * WMMA_M && idx < total_threads_needed_b) {
                    bool b_valid = (global_row_b + k + WMMA_K < batchSize) && (chunk_col_b < hiddenDim);
                    const half* b_src = b_valid ? &d_activations[layer][(global_row_b + k + WMMA_K) * hiddenDim + chunk_col_b] : d_activations[layer];
                    cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b], b_src, b_valid);
                    cp_async_commit();
                }

                #pragma unroll
                for (int i = 0; i < WARP_FACTOR; i++) {
                    nvcuda::wmma::load_matrix_sync(frag_a, &shmem_B[buf][0][(threadIdx.z + i*blockDim.z)*WMMA_M], TILE_COUNT_X * WMMA_N + PAD);
                    nvcuda::wmma::load_matrix_sync(frag_b, &shmem_A[k][threadIdx.y*WMMA_K], TILE_COUNT_X*WMMA_K + PAD);
                    nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
                }

                if (k + WMMA_K < TILE_COUNT_Y * TILE_FACTOR * WMMA_M) {
                    cp_async_wait();
                    __syncthreads();
                }
                buf = next_buf;
            }

            #pragma unroll
            for(int j = 0; j < WARP_FACTOR; j++) {
                #pragma unroll
                for (int i = 0; i < frag_acc[j].num_elements; i++) {
                    int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                    int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                    
                    int global_row = (threadIdx.z + j*blockDim.z)*WMMA_M + local_row;
                    int global_col = threadIdx.y * WMMA_N + local_col;

                    float val = frag_acc[j].x[i];

                    if (global_row < hiddenDim && global_col < hiddenDim) {
                        redAddF32(&d_grad_weights[layer][global_col * hiddenDim + global_row], val);
                    }
                }
            }

            local_db = 0.0f;
            global_col = threadIdx.y * 16 + (laneId % 16);
            
            #pragma unroll
            for (int i = 0; i < TILE_FACTOR; i++) {
                int row_start = (threadIdx.z + i*blockDim.z) * WMMA_M + (laneId / 16) * 8;
        
                #pragma unroll
                for(int j = 0; j < 8; j++) {
                    local_db += __half2float(shmem_A[row_start + j][global_col]);
                }
            }

            local_db += __shfl_down_sync(0xFFFFFFFF, local_db, 16);

            if (laneId < 16 && global_col < hiddenDim) {
                redAddF32(&d_grad_biases[layer][global_col], local_db);
            }

            __syncthreads();
        }

        {
            int threads_per_row_w = (TILE_COUNT_X * WMMA_N) / 8; 
            int total_threads_w = WMMA_K * threads_per_row_w; 

            int chunk_row_b = idx / threads_per_row_w;
            int chunk_col_b = (idx % threads_per_row_w) * 8;
            
            if (idx < total_threads_w) {
                bool b_valid = (chunk_row_b < hiddenDim) && (chunk_col_b < hiddenDim);
                const half* w_src = b_valid ? &d_weights_array[layer][chunk_row_b * hiddenDim + chunk_col_b] : d_weights_array[layer];
                cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b], w_src, b_valid);
            }

            cp_async_commit();
            cp_async_wait();
            __syncthreads();

            int buf = 0;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[TILE_FACTOR];
            #pragma unroll
            for (int i = 0; i < TILE_FACTOR; i++) {
                nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
            }

            for (int k = 0; k < hiddenDim; k += WMMA_K) {
                int next_buf = 1 - buf;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_b;

                if (k + WMMA_K < hiddenDim) {
                    if (idx < total_threads_w) {
                        bool isValid = (chunk_row_b + k + WMMA_K < hiddenDim) && (chunk_col_b < hiddenDim);
                        const half* w_src = isValid ? &d_weights_array[layer][(chunk_row_b + k + WMMA_K) * hiddenDim + chunk_col_b] : d_weights_array[layer];
                        cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b], w_src, isValid);
                    }
                    cp_async_commit();
                }

                nvcuda::wmma::load_matrix_sync(frag_b, &shmem_B[buf][0][threadIdx.y * WMMA_K], TILE_COUNT_X * WMMA_N + PAD);
                
                #pragma unroll
                for (int i = 0; i < TILE_FACTOR; i++) {
                    nvcuda::wmma::load_matrix_sync(frag_a, &shmem_A[(threadIdx.z + i*blockDim.z)*WMMA_M][k], TILE_COUNT_X*WMMA_K + PAD);
                    nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
                }

                if (k + WMMA_K < hiddenDim) {
                    cp_async_wait();
                    __syncthreads();
                }
                buf = next_buf;
            }

            __syncthreads();
            int base_col = threadIdx.y * WMMA_N;

            #pragma unroll
            for(int j = 0; j < TILE_FACTOR; j++) {
                #pragma unroll
                for (int i = 0; i < frag_acc[j].num_elements; i++) {
                    int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                    int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                    
                    float val = frag_acc[j].x[i];
                    shmem_A[(threadIdx.z + j*blockDim.z)*WMMA_M + local_row][base_col + local_col] = __float2half(val);
                }
            }
            __syncthreads();

            // PHASE 3 (ReLU)
            int total_threads = blockDim.x * blockDim.y * blockDim.z;
            int total_elements = (TILE_COUNT_Y * TILE_FACTOR * WMMA_M) * hiddenDim;

            for (int i = idx; i < total_elements; i += total_threads) {
                int r = i / hiddenDim; 
                int c = i % hiddenDim; 
                int global_r = (blockIdx.x * TILE_COUNT_Y * TILE_FACTOR * WMMA_M) + r;
                
                if (global_r < batchSize && c < hiddenDim) {
                    float orig_act = __half2float(d_activations[layer][global_r * hiddenDim + c]);
                    if (orig_act <= 0.0f) {
                        shmem_A[r][c] = __float2half(0.0f);
                    }
                }
            }
            __syncthreads();
        }
        layer--;
    }

    {
        int threads_per_row_b = WMMA_K / 8; // 2
        int total_threads_needed_b = (TILE_COUNT_X * WMMA_N) * threads_per_row_b; // 128
    
        int chunk_row_b = 0, chunk_col_b = 0, global_row_b = 0;
    
        if (idx < total_threads_needed_b) {
            chunk_row_b = idx / (TILE_COUNT_X * WMMA_K / 8);
            chunk_col_b = (idx % (TILE_COUNT_X * WMMA_K / 8)) * 8;
            global_row_b = blockIdx.x * TILE_COUNT_Y * TILE_FACTOR * WMMA_M + chunk_row_b;
            bool b_valid = (global_row_b < batchSize) && (chunk_col_b < inputDim);
            const half* b_src = b_valid ? &d_activations[layer][global_row_b * inputDim + chunk_col_b] : d_activations[layer];
            cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b], b_src, b_valid);
        }
        
        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        int buf = 0;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_FACTOR];
        #pragma unroll
        for (int i = 0; i < WARP_FACTOR; i++) {
            nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
        }

        #pragma unroll
        for (int k = 0; k < TILE_COUNT_Y * TILE_FACTOR * WMMA_M; k += WMMA_K) {
            int next_buf = 1 - buf;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_b;

            if (k + WMMA_K < TILE_COUNT_Y * TILE_FACTOR * WMMA_M && idx < total_threads_needed_b) {
                bool b_valid = (global_row_b + k + WMMA_K < batchSize) && (chunk_col_b < inputDim);
                const half* b_src = b_valid ? &d_activations[layer][(global_row_b + k + WMMA_K) * inputDim + chunk_col_b] : d_activations[layer];
                cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b], b_src, b_valid);
                cp_async_commit();
            }

            #pragma unroll
            for (int i = 0; i < WARP_FACTOR; i++) {
                nvcuda::wmma::load_matrix_sync(frag_a, &shmem_B[buf][0][(threadIdx.z + i*blockDim.z)*WMMA_M], TILE_COUNT_X * WMMA_N + PAD);
                nvcuda::wmma::load_matrix_sync(frag_b, &shmem_A[k][threadIdx.y*WMMA_K], TILE_COUNT_X*WMMA_K + PAD);
                nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
            }

            if (k + WMMA_K < TILE_COUNT_Y * TILE_FACTOR * WMMA_M) {
                cp_async_wait();
                __syncthreads();
            }
            buf = next_buf;
        }

        #pragma unroll
        for(int j = 0; j < WARP_FACTOR; j++) {
            #pragma unroll
            for (int i = 0; i < frag_acc[j].num_elements; i++) {
                int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                
                int global_row = (threadIdx.z + j*blockDim.z)*WMMA_M + local_row;
                int global_col = threadIdx.y * WMMA_N + local_col;

                float val = frag_acc[j].x[i];

                if (global_row < inputDim && global_col < hiddenDim) {
                    redAddF32(&d_grad_weights[layer][global_col * inputDim + global_row], val);
                }
            }
        }

        local_db = 0.0f;
        global_col = threadIdx.y * 16 + (laneId % 16);
        
        #pragma unroll
        for (int i = 0; i < TILE_FACTOR; i++) {
            int row_start = (threadIdx.z + i*blockDim.z) * WMMA_M + (laneId / 16) * 8;
    
            #pragma unroll
            for(int j = 0; j < 8; j++) {
                local_db += __half2float(shmem_A[row_start + j][global_col]);
            }
        }

        local_db += __shfl_down_sync(0xFFFFFFFF, local_db, 16);

        if (laneId < 16 && global_col < hiddenDim) {
            redAddF32(&d_grad_biases[layer][global_col], local_db);
        }
    }

    {
        int threads_per_row_w = (TILE_COUNT_X * WMMA_N) / 8; 
        int total_threads_w = WMMA_K * threads_per_row_w; 

        int chunk_row_b = idx / threads_per_row_w;
        int chunk_col_b = (idx % threads_per_row_w) * 8;
        
        if (idx < total_threads_w) {
            bool b_valid = (chunk_row_b < hiddenDim) && (chunk_col_b < inputDim);
            const half* w_src = b_valid ? &d_weights_array[layer][chunk_row_b * inputDim + chunk_col_b] : d_weights_array[layer];
            cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b], w_src, b_valid);
        }
        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        int buf = 0;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[TILE_FACTOR];
        #pragma unroll
        for (int i = 0; i < TILE_FACTOR; i++) {
            nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
        }

        for (int k = 0; k < hiddenDim; k += WMMA_K) {
            int next_buf = 1 - buf;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_b;

            if (k + WMMA_K < hiddenDim) {
                if (idx < total_threads_w) {
                    bool isValid = (chunk_row_b + k + WMMA_K < hiddenDim) && (chunk_col_b < inputDim);
                    const half* w_src = isValid ? &d_weights_array[layer][(chunk_row_b + k + WMMA_K) * inputDim + chunk_col_b] : d_weights_array[layer];
                    cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b], w_src, isValid);
                }
                cp_async_commit();
            }

            nvcuda::wmma::load_matrix_sync(frag_b, &shmem_B[buf][0][threadIdx.y * WMMA_K], TILE_COUNT_X * WMMA_N + PAD);
            
            #pragma unroll
            for (int i = 0; i < TILE_FACTOR; i++) {
                nvcuda::wmma::load_matrix_sync(frag_a, &shmem_A[(threadIdx.z + i*blockDim.z)*WMMA_M][k], TILE_COUNT_X*WMMA_K + PAD);
                nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
            }

            if (k + WMMA_K < hiddenDim) {
                cp_async_wait();
                __syncthreads();
            }
            buf = next_buf;
        }

        __syncthreads();

        int base_col = threadIdx.y * WMMA_N;
        #pragma unroll
        for(int j = 0; j < TILE_FACTOR; j++) {
            #pragma unroll
            for (int i = 0; i < frag_acc[j].num_elements; i++) {
                int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;

                float val = frag_acc[j].x[i];
                shmem_A[(threadIdx.z + j*blockDim.z)*WMMA_M + local_row][base_col + local_col] = __float2half(val);
            }
        }
        __syncthreads();
    }

    // so now the shmem_A is the local d_dx_out and which is used for gradients of hashtable weights
    // now one warp handles tile factor number of 16x16 tiles.
    // we make threads in one warp handle similar levels of features at one time to maximize warp level reduction
    // we have total of 16xTILE_FACTORxTILE_COUNT_Y vertical batches and numLevels x NUMFEATURES(2) rows

    __syncthreads();

    // {
    // int rows = WMMA_M * TILE_FACTOR * TILE_COUNT_Y;
    // int levelsThread = numLevels / (256 / rows);

    // int localRow = idx % rows;
    // int globalRow = blockIdx.x * rows  + localRow;

    // int levelStart = (idx / rows) * levelsThread;

    // if (globalRow < batchSize) {
    //     float inputVec[4];
    //     loadVec4(&d_inputs[globalRow * 4], inputVec);

    //     float N_l_float = lowestSize * __powf(b, levelStart);

    //     for (int l = levelStart; l < levelsThread + levelStart; l++) {
    //         int N_l = __float2int_rd(N_l_float);

    //         float x_l = inputVec[0] * N_l_float;
    //         float y_l = inputVec[1] * N_l_float;
    //         float z_l = inputVec[2] * N_l_float;

    //         int x0 = __float2int_rd(x_l);
    //         int y0 = __float2int_rd(y_l);
    //         int z0 = __float2int_rd(z_l);

    //         int x1 = min(x0 + 1, N_l);
    //         int y1 = min(y0 + 1, N_l);
    //         int z1 = min(z0 + 1, N_l);
            
    //         x0 = max(0, min(x0, N_l));
    //         y0 = max(0, min(y0, N_l));
    //         z0 = max(0, min(z0, N_l));

    //         float wx1 = 1 - x_l + floorf(x_l);
    //         float wy1 = 1 - y_l + floorf(y_l);
    //         float wz1 = 1 - z_l + floorf(z_l);

    //         bool isDense = l < denseLevelStart;

    //         auto get_table_index = [=](int cx, int cy, int cz) -> uint32_t {
    //             uint32_t table_index = isDense ? dense_index(cx, cy, cz, N_l)
    //                                         : hash_coords(cx, cy, cz, tableSize);
    //             return l * tableSize * FEATURES_LEVEL + table_index * FEATURES_LEVEL;
    //         };

    //         const float w000 = wx1 * wy1 * wz1, w100 = (1.0f - wx1) * wy1 * wz1;
    //         const float w010 = wx1 * (1.0f - wy1) * wz1, w110 = (1.0f - wx1) * (1.0f - wy1) * wz1;
    //         const float w001 = wx1 * wy1 * (1.0f - wz1), w101 = (1.0f - wx1) * wy1 * (1.0f - wz1);
    //         const float w011 = wx1 * (1.0f - wy1) * (1.0f - wz1), w111 = (1.0f - wx1) * (1.0f - wy1) * (1.0f - wz1);

    //         if constexpr (FEATURES_LEVEL == 2) {
    //             // Safely read from the 2D Shared Memory array
    //             const __half2* grad_ptr = reinterpret_cast<const __half2*>(&shmem_A[localRow][l * FEATURES_LEVEL]);
    //             float2 grads = __half22float2(*grad_ptr);
                
    //             int crrIdx;
    //             float weight = 0.0f;
    //             float valX, valY, sumX, sumY;
    //             unsigned group;
    //             int leader;

    //             #pragma unroll
    //             for (int i = 0; i < 8; i++) {
    //                 switch(i) {
    //                     case 0: crrIdx = get_table_index(x0, y0, z0); weight = w000; break;
    //                     case 1: crrIdx = get_table_index(x1, y0, z0); weight = w100; break;
    //                     case 2: crrIdx = get_table_index(x0, y1, z0); weight = w010; break;
    //                     case 3: crrIdx = get_table_index(x1, y1, z0); weight = w110; break;
    //                     case 4: crrIdx = get_table_index(x0, y0, z1); weight = w001; break;
    //                     case 5: crrIdx = get_table_index(x1, y0, z1); weight = w101; break;
    //                     case 6: crrIdx = get_table_index(x0, y1, z1); weight = w011; break;
    //                     case 7: crrIdx = get_table_index(x1, y1, z1); weight = w111; break;
    //                 }

    //                 // 1. Calculate THIS thread's weighted value BEFORE the shuffle
    //                 valX = grads.x * weight;
    //                 valY = grads.y * weight;
                    
    //                 // 2. Find all threads writing to the same bucket
    //                 group = __match_any_sync(0xFFFFFFFF, crrIdx);
    //                 leader = __ffs(group) - 1; // 0-indexed lane ID

    //                 sumX = 0.0f;
    //                 sumY = 0.0f;
    //                 unsigned tmp = group;
                    
    //                 // 3. Shuffle and sum the PRE-WEIGHTED values
    //                 while (tmp) {
    //                     int srcLane = __ffs(tmp) - 1;
    //                     sumX += __shfl_sync(group, valX, srcLane);
    //                     sumY += __shfl_sync(group, valY, srcLane);
    //                     tmp &= tmp - 1; 
    //                 }

    //                 // 4. Only the leader writes to Global Memory!
    //                 if (laneId == leader) {
    //                     redAddF32(&d_hashtable_grads[crrIdx], sumX);
    //                     redAddF32(&d_hashtable_grads[crrIdx + 1], sumY);
    //                 }
    //             }
    //         }

    //         N_l_float *= b;
    //     }
    // }
    // }

    int rows = WMMA_M * TILE_FACTOR * TILE_COUNT_Y;
    int levelsThread = numLevels / (256 / rows);

    int localRow = idx % rows;
    int globalRow = blockIdx.x * rows  + localRow;

    int levelStart = (idx / rows) * levelsThread;

    if (globalRow < batchSize) {
        float inputVec[4];
        loadVec4(&d_inputs[globalRow * 4], inputVec);

        float N_l_float = lowestSize * __powf(b, levelStart);

        for (int l = levelStart; l < levelsThread + levelStart; l++) {
            int N_l = __float2int_rd(N_l_float);

            float x_l = inputVec[0] * N_l_float;
            float y_l = inputVec[1] * N_l_float;
            float z_l = inputVec[2] * N_l_float;

            int x0 = __float2int_rd(x_l);
            int y0 = __float2int_rd(y_l);
            int z0 = __float2int_rd(z_l);

            int x1 = min(x0 + 1, N_l);
            int y1 = min(y0 + 1, N_l);
            int z1 = min(z0 + 1, N_l);
            
            x0 = max(0, min(x0, N_l));
            y0 = max(0, min(y0, N_l));
            z0 = max(0, min(z0, N_l));

            float wx1 = 1.0f - x_l + floorf(x_l);
            float wy1 = 1.0f - y_l + floorf(y_l);
            float wz1 = 1.0f - z_l + floorf(z_l);

            bool isDense = l < denseLevelStart;

            auto get_table_index = [=](int cx, int cy, int cz) -> uint32_t {
                uint32_t table_index = isDense ? dense_index(cx, cy, cz, N_l)
                                            : hash_coords(cx, cy, cz, tableSize);
                return l * tableSize * FEATURES_LEVEL + table_index * FEATURES_LEVEL;
            };

            const float w000 = wx1 * wy1 * wz1, w100 = (1.0f - wx1) * wy1 * wz1;
            const float w010 = wx1 * (1.0f - wy1) * wz1, w110 = (1.0f - wx1) * (1.0f - wy1) * wz1;
            const float w001 = wx1 * wy1 * (1.0f - wz1), w101 = (1.0f - wx1) * wy1 * (1.0f - wz1);
            const float w011 = wx1 * (1.0f - wy1) * (1.0f - wz1), w111 = (1.0f - wx1) * (1.0f - wy1) * (1.0f - wz1);

            if constexpr (FEATURES_LEVEL == 2) {
                const __half2* grad_ptr = reinterpret_cast<const __half2*>(&shmem_A[localRow][l * FEATURES_LEVEL]);
                float2 grads = __half22float2(*grad_ptr);
                
                int crrIdx;
                float weight = 0.0f;
                float valX, valY, sumX, sumY;
                unsigned group;
                int leader;

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    switch(i) {
                        case 0: crrIdx = get_table_index(x0, y0, z0); weight = w000; break;
                        case 1: crrIdx = get_table_index(x1, y0, z0); weight = w100; break;
                        case 2: crrIdx = get_table_index(x0, y1, z0); weight = w010; break;
                        case 3: crrIdx = get_table_index(x1, y1, z0); weight = w110; break;
                        case 4: crrIdx = get_table_index(x0, y0, z1); weight = w001; break;
                        case 5: crrIdx = get_table_index(x1, y0, z1); weight = w101; break;
                        case 6: crrIdx = get_table_index(x0, y1, z1); weight = w011; break;
                        case 7: crrIdx = get_table_index(x1, y1, z1); weight = w111; break;
                    }

                    // 1. Calculate THIS thread's weighted value BEFORE the shuffle
                    valX = grads.x * weight;
                    valY = grads.y * weight;
                    
                    // 2. Find all threads writing to the same bucket
                    unsigned active = __activemask();
                    group = __match_any_sync(active, crrIdx);
                    leader = __ffs(group) - 1; // 0-indexed lane ID

                    sumX = 0.0f;
                    sumY = 0.0f;
                    unsigned tmp = group;
                    
                    // 3. Shuffle and sum the PRE-WEIGHTED values
                    while (tmp) {
                        int srcLane = __ffs(tmp) - 1;
                        sumX += __shfl_sync(group, valX, srcLane);
                        sumY += __shfl_sync(group, valY, srcLane);
                        tmp &= tmp - 1; 
                    }

                    // 4. Only the leader writes to Global Memory!
                    if (laneId == leader) {
                        redAddF32(&d_hashtable_grads[crrIdx], sumX);
                        redAddF32(&d_hashtable_grads[crrIdx + 1], sumY);
                    }
                }
            }

            N_l_float *= b;
        }
    }
}

template <int FEATURES_LEVEL>
__global__ void hashTableGrad_Backward(
    const half* __restrict__ d_dx_out,
    const float* __restrict__ d_inputs,
    float* __restrict__ d_hashtable_grads,
    int batchSize,
    int tableSize,
    int numLevels,
    float b,
    int lowestSize,
    int denseLevelStart

) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * numLevels) return;
    int global_row = idx / numLevels;
    int level = idx % numLevels;
    int feature_offset = level * FEATURES_LEVEL;

    float inputVec[4];
    loadVec4(&d_inputs[global_row * 4], inputVec);

    float N_l_float = lowestSize * __powf(b, level);
    int N_l = __float2int_rd(N_l_float);

    float x_l = inputVec[0] * N_l_float;
    float y_l = inputVec[1] * N_l_float;
    float z_l = inputVec[2] * N_l_float;

    int x0 = __float2int_rd(x_l);
    int y0 = __float2int_rd(y_l);
    int z0 = __float2int_rd(z_l);

    int x1 = min(x0 + 1, N_l);
    int y1 = min(y0 + 1, N_l);
    int z1 = min(z0 + 1, N_l);

    x0 = max(0, min(x0, N_l));
    y0 = max(0, min(y0, N_l));
    z0 = max(0, min(z0, N_l));

    float wx1 = 1.0f - x_l + floorf(x_l);
    float wy1 = 1.0f - y_l + floorf(y_l);
    float wz1 = 1.0f - z_l + floorf(z_l);

    bool isDense = level < denseLevelStart;

    auto get_table_index = [=](int cx, int cy, int cz) -> uint32_t {
        uint32_t table_index = isDense ? dense_index(cx, cy, cz, N_l)
                                       : hash_coords(cx, cy, cz, tableSize);
        return level * tableSize * FEATURES_LEVEL + table_index * FEATURES_LEVEL;
    };

    const float w000 = wx1 * wy1 * wz1, w100 = (1.0f - wx1) * wy1 * wz1;
    const float w010 = wx1 * (1.0f - wy1) * wz1, w110 = (1.0f - wx1) * (1.0f - wy1) * wz1;
    const float w001 = wx1 * wy1 * (1.0f - wz1), w101 = (1.0f - wx1) * wy1 * (1.0f - wz1);
    const float w011 = wx1 * (1.0f - wy1) * (1.0f - wz1), w111 = (1.0f - wx1) * (1.0f - wy1) * (1.0f - wz1);

    if constexpr (FEATURES_LEVEL == 2) {
        int grad_in_idx = global_row * numLevels * FEATURES_LEVEL + feature_offset;
        float2 grads = __half22float2(*reinterpret_cast<const __half2*>(&d_dx_out[grad_in_idx]));

        redAddF32(&d_hashtable_grads[get_table_index(x0, y0, z0)], grads.x * w000);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y0, z0)], grads.x * w100);
        redAddF32(&d_hashtable_grads[get_table_index(x0, y1, z0)], grads.x * w010);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y1, z0)], grads.x * w110);
        redAddF32(&d_hashtable_grads[get_table_index(x0, y0, z1)], grads.x * w001);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y0, z1)], grads.x * w101);
        redAddF32(&d_hashtable_grads[get_table_index(x0, y1, z1)], grads.x * w011);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y1, z1)], grads.x * w111);

        redAddF32(&d_hashtable_grads[get_table_index(x0, y0, z0) + 1], grads.y * w000);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y0, z0) + 1], grads.y * w100);
        redAddF32(&d_hashtable_grads[get_table_index(x0, y1, z0) + 1], grads.y * w010);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y1, z0) + 1], grads.y * w110);
        redAddF32(&d_hashtable_grads[get_table_index(x0, y0, z1) + 1], grads.y * w001);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y0, z1) + 1], grads.y * w101);
        redAddF32(&d_hashtable_grads[get_table_index(x0, y1, z1) + 1], grads.y * w011);
        redAddF32(&d_hashtable_grads[get_table_index(x1, y1, z1) + 1], grads.y * w111);
    }
}

void launchHashTableBackwardKernel(
    half* d_dx_out,
    float* d_inputs,
    float* d_hashtable_grads,
    int batchSize,
    int tableSize,
    int numLevels,
    float b,
    int lowestSize,
    int featuresLevel,
    cudaStream_t stream
) {
    if (featuresLevel != 2) return;

    int numThreads = batchSize * numLevels;

    int threadsPerBlock = 256;

    int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(numBlocks, 1, 1);
    dim3 block(threadsPerBlock);

    float inner_term = (std::cbrt(static_cast<float>(tableSize)) - 1.0f) / lowestSize;

    // 2. Calculate the base-b logarithm
    float continuous_level = std::log2(inner_term) / std::log2(b);

    // 3. The first non-dense (hashed) level index
    int denseLevelStart = static_cast<int>(std::floor(continuous_level)) + 1;

    // 4. Clamp the result to ensure it stays within valid bounds [0, numLevels]
    denseLevelStart = std::max(0, std::min(denseLevelStart, numLevels));


    hashTableGrad_Backward<2><<<grid, block, 0, stream>>>(d_dx_out, d_inputs, d_hashtable_grads, batchSize, tableSize, numLevels, b, lowestSize, denseLevelStart);
}


void launchNetworkFusionHashTableBackwardKernel(
    MLPGridOptions*  opt,
    half*        d_loss_output,
    half**       d_weights_array,
    float**      d_biases_array, // Unused internally, kept for signature compatibility
    half**       d_activations,
    float*       d_inputs,             // NEW: Hash grid inputs
    float**      d_grad_weights,
    float**      d_grad_biases,
    float*       d_hashtable_grads,    // NEW: Hash grid gradients output
    half*        d_dx_out,
    int          batchSize,
    cudaStream_t stream
) {
    // Sanity check for the template parameter
    if (opt->featuresLevel != 2) return;

    int inputDim = opt->numLevels * opt->featuresLevel;
    int maxDim = (opt->hiddenDim > inputDim) ? opt->hiddenDim : inputDim;

    if (maxDim > 128) return;

    // --- Calculate Hash Grid 'denseLevelStart' ---
    float inner_term = (std::cbrt(static_cast<float>(opt->tableSize)) - 1.0f) / opt->lowestSize;
    float continuous_level = std::log2(inner_term) / std::log2(opt->b);
    int denseLevelStart = static_cast<int>(std::floor(continuous_level)) + 1;
    denseLevelStart = std::max(0, std::min(denseLevelStart, opt->numLevels));

    const int TILE_FACTOR = 4;

    int tileCountX = maxDim / 16;
    if (tileCountX == 0) tileCountX = 1;
    int tileCountY = 8 / tileCountX;

    int totalRows = tileCountY * TILE_FACTOR * 16;

    int numBlocks = (batchSize + totalRows - 1) / totalRows;
    dim3 grid(numBlocks, 1, 1);
    dim3 block(32, tileCountX, tileCountY);

    // Launch with the 5th template parameter (FEATURES_LEVEL = 2) and new arguments
    switch (maxDim)
    {
    case 16:
        networkFusionMMA_Backward<8, 1, 1, TILE_FACTOR, 2><<<grid, block, 0, stream>>>(
            d_dx_out, d_loss_output, d_weights_array, d_activations, d_inputs, 
            d_grad_weights, d_grad_biases, d_hashtable_grads, 
            batchSize, inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers, 
            opt->tableSize, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
        break;
    case 32:
        networkFusionMMA_Backward<4, 2, 1, TILE_FACTOR, 2><<<grid, block, 0, stream>>>(
            d_dx_out, d_loss_output, d_weights_array, d_activations, d_inputs, 
            d_grad_weights, d_grad_biases, d_hashtable_grads, 
            batchSize, inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers, 
            opt->tableSize, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
        break;
    case 64:
        networkFusionMMA_Backward<2, 4, 2, TILE_FACTOR, 2><<<grid, block, 0, stream>>>(
            d_dx_out, d_loss_output, d_weights_array, d_activations, d_inputs, 
            d_grad_weights, d_grad_biases, d_hashtable_grads, 
            batchSize, inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers, 
            opt->tableSize, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
        break;
    case 128:
        networkFusionMMA_Backward<1, 8, 8, TILE_FACTOR, 2><<<grid, block, 0, stream>>>(
            d_dx_out, d_loss_output, d_weights_array, d_activations, d_inputs, 
            d_grad_weights, d_grad_biases, d_hashtable_grads, 
            batchSize, inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers, 
            opt->tableSize, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
        break;
    default:
        break;
    }
}
