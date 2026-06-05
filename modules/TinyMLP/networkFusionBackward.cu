#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "TinyMLP.h"

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

template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int TILE_FACTOR, int CALC_DX = 0>
__global__ void networkFusionMMA_Backward(
    half* d_dx_out,
    const half* __restrict__ d_loss_output,
    half** d_weights_array,
    half** d_activations,
    float** d_grad_weights,
    float** d_grad_biases,
    int batchSize,
    int inputDim,
    int hiddenDim,
    int outputDim,
    int numLayers
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

    if constexpr (CALC_DX) {
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

        int total_threads = blockDim.x * blockDim.y * blockDim.z;
        int total_elements = (TILE_COUNT_Y * TILE_FACTOR * WMMA_M) * inputDim;

        for (int i = idx; i < total_elements; i += total_threads) {
            int r = i / inputDim; 
            int c = i % inputDim; 
            int global_r = (blockIdx.x * TILE_COUNT_Y * TILE_FACTOR * WMMA_M) + r;
            
            if (global_r < batchSize && c < inputDim) {
                d_dx_out[global_r * inputDim + c] = shmem_A[r][c];
            }
        }
        __syncthreads();
    }

}

void launchNetworkFusionBackwardKernel(
    MLPOption*   opt,
    half*        d_loss_output,
    half**       d_weights_array,
    float**      d_biases_array,
    half**       d_activations,
    float**      d_grad_weights,
    float**      d_grad_biases,
    half*        d_dx_out,
    int          batchSize,
    cudaStream_t stream
) {
    int maxDim = (opt->hiddenDim > opt->inputDim) ? opt->hiddenDim : opt->inputDim;

    if (maxDim > 128) return;

    const int TILE_FACTOR = 4;

    int tileCountX = maxDim / 16;
    if (tileCountX == 0) tileCountX = 1;
    int tileCountY = 8 / tileCountX;

    int totalRows = tileCountY * TILE_FACTOR * 16;

    int numBlocks = (batchSize + totalRows - 1) / totalRows;
    dim3 grid(numBlocks, 1, 1);

    dim3 block(32, tileCountX, tileCountY);

    switch (maxDim)
    {
    case 16:
        if (d_dx_out != nullptr) networkFusionMMA_Backward<8, 1, 1, TILE_FACTOR, 1><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        else networkFusionMMA_Backward<8, 1, 1, TILE_FACTOR, 0><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        break;
    case 32:
        if (d_dx_out != nullptr) networkFusionMMA_Backward<4, 2, 1, TILE_FACTOR, 1><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        else networkFusionMMA_Backward<4, 2, 1, TILE_FACTOR, 0><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        break;
    case 64:
        if (d_dx_out != nullptr) networkFusionMMA_Backward<2, 4, 2, TILE_FACTOR, 1><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        else networkFusionMMA_Backward<2, 4, 2, TILE_FACTOR, 0><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        break;
    case 128:
        if (d_dx_out != nullptr) networkFusionMMA_Backward<1, 8, 8, TILE_FACTOR, 1><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        else networkFusionMMA_Backward<1, 8, 8, TILE_FACTOR, 0><<<grid, block, 0, stream>>>(d_dx_out, d_loss_output, d_weights_array, d_activations, d_grad_weights, d_grad_biases, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);
        break;
    default:
        break;
    }
}