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

__device__ __forceinline__ void store_cg_two_f16(__half* ptr, __half a, __half b) {
    // Pack two fp16 into one 32-bit word
    // memory layout: [b (high 16 bits) | a (low 16 bits)]
    uint32_t packed = (uint32_t)__half_as_ushort(b) << 16
                    | (uint32_t)__half_as_ushort(a);

    #ifndef __INTELLISENSE__
    asm volatile(
        "st.global.cs.b32 [%0], %1;"
        :: "l"(ptr), "r"(packed)
        : "memory"
    );
    #endif
}


// Goal is to launch with block(32, 4, 2) if maxDim is 64, 4 in X dir and 2 in Y dir as we gurantee 8 warps per block exactly
template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int NUM_LAYERS, int OUT_ACT = 0>
__global__ void __launch_bounds__(256, 3) networkFusionMMA_2d(
    half* d_inputs,
    half** d_weights_array,
    float** d_biases_array,
    float* d_outputs,
    int batchSize,
    int inputDim,
    int hiddenDim,
    int outputDim
) {
    // 8 Warps per block is guranteed
    int idx = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    
    int warpM = blockIdx.x * TILE_COUNT_Y + threadIdx.z;
    int warpN = threadIdx.y;
    int laneId = threadIdx.x + 1 - 1;
 
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    const int PAD = 8;

    __shared__ half shmem_A[WMMA_M * TILE_COUNT_Y][TILE_COUNT_X*WMMA_K + PAD];
    __shared__ half shmem_B[2][TILE_COUNT_X*WMMA_N][WMMA_K + PAD];

    #pragma unroll
    for (int i = 0; i < WARP_FACTOR; i++) {
        int chunk_row = (idx + i*256) / (TILE_COUNT_X * WMMA_K / 8); // 256 threads per block
        int chunk_col = (idx + i*256) % (TILE_COUNT_X * WMMA_K / 8) * 8;

        int global_row = (blockIdx.x * TILE_COUNT_Y) * WMMA_M + chunk_row;
        int global_col = chunk_col;

        bool a_valid = (global_row < batchSize) && (global_col < inputDim);

        cp_async_128(&shmem_A[chunk_row][chunk_col],
                &d_inputs[global_row * inputDim + global_col],
                a_valid);
    }

    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    #pragma unroll
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int currentK = (layer == 0) ? inputDim : hiddenDim;
        int currentN = (layer == NUM_LAYERS - 1) ? outputDim : hiddenDim;
        bool isLastLayer = layer == NUM_LAYERS - 1;

        int chunk_row_b = idx / (WMMA_K / 8);
        int chunk_col_b = idx % (WMMA_K / 8) * 8;

        if (chunk_row_b < TILE_COUNT_X * WMMA_N) {
            bool b_valid = (chunk_row_b < currentN) && (chunk_col_b < currentK);
            cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b],
                        &d_weights_array[layer][chunk_row_b * currentK + chunk_col_b],
                        b_valid);
        }

        cp_async_commit();
        cp_async_wait();
        __syncthreads();
        
        int buf = 0;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_FACTOR];
        #pragma unroll
        for(int i = 0; i < WARP_FACTOR; i++) {
            nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
        }

        for (int k = 0; k < currentK; k += WMMA_K) {
            int next_buf = 1 - buf;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b;
            
            if (k + WMMA_K < currentK) {
                if (chunk_row_b < TILE_COUNT_X * WMMA_N) {
                    bool isValid = (chunk_row_b < currentN) && (chunk_col_b + k + WMMA_K) < currentK;
                    cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b],
                                &d_weights_array[layer][chunk_row_b * currentK + (chunk_col_b + k + WMMA_K)],
                                isValid);
                }
                cp_async_commit();
            }

            nvcuda::wmma::load_matrix_sync(frag_b, &shmem_B[buf][warpN * WMMA_K][0], WMMA_K + PAD);

            #pragma unroll
            for (int i = 0; i < WARP_FACTOR; i++) {
                nvcuda::wmma::load_matrix_sync(frag_a, &shmem_A[(threadIdx.z + i*blockDim.z)*WMMA_M][k], TILE_COUNT_X*WMMA_K + PAD);
                nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
            }

            if (k + WMMA_K < currentK) {
                cp_async_wait();
                __syncthreads();
            }

            buf = next_buf;
        }

        int base_col = warpN * WMMA_N;

        #pragma unroll
        for(int j = 0; j < WARP_FACTOR; j++) {
            #pragma unroll
            for (int i = 0; i < frag_acc[j].num_elements; i++) {
                int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                
                int global_r = (warpM + j*blockDim.z) * WMMA_M + local_row;
                int global_c = base_col + local_col; 

                float val = frag_acc[j].x[i];

                if (global_c < currentN)
                    val += d_biases_array[layer][global_c];

                if(!isLastLayer) {
                    val = fmaxf(val, 0.0f); // ReLU
                } else if constexpr (OUT_ACT == 1) {
                    val = 1.0f / (1.0f + expf(-val)); // Sigmoid
                }

                if (global_r < batchSize && global_c < currentN) {
                    if (isLastLayer) {
                        d_outputs[global_r * outputDim + global_c] = val;
                    } else {
                        shmem_A[(threadIdx.z + j*blockDim.z)*WMMA_M + local_row][base_col + local_col] = __float2half(val);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int NUM_LAYERS, int OUT_ACT = 0>
__global__ void __launch_bounds__(256, 3) networkFusionMMAGrad_2d(
    half* d_inputs,
    half** d_activations,
    half** d_weights_array,
    float** d_biases_array,
    float* d_outputs,
    int batchSize,
    int inputDim,
    int hiddenDim,
    int outputDim
) {
    // 8 Warps per block is guranteed
    int idx = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    
    int warpM = blockIdx.x * TILE_COUNT_Y + threadIdx.z;
    int warpN = threadIdx.y;
    int laneId = threadIdx.x + 1 - 1;
 
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    const int PAD = 8;

    __shared__ half shmem_A[WMMA_M * TILE_COUNT_Y][TILE_COUNT_X*WMMA_K + PAD];
    __shared__ half shmem_B[2][TILE_COUNT_X*WMMA_N][WMMA_K + PAD];

    #pragma unroll
    for (int i = 0; i < WARP_FACTOR; i++) {
        int chunk_row = (idx + i*256) / (TILE_COUNT_X * WMMA_K / 8); // 256 threads per block
        int chunk_col = (idx + i*256) % (TILE_COUNT_X * WMMA_K / 8) * 8;

        int global_row = (blockIdx.x * TILE_COUNT_Y) * WMMA_M + chunk_row;
        int global_col = chunk_col;

        bool a_valid = (global_row < batchSize) && (global_col < inputDim);

        cp_async_128(&shmem_A[chunk_row][chunk_col],
                &d_inputs[global_row * inputDim + global_col],
                a_valid);
    }

    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    #pragma unroll
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int currentK = (layer == 0) ? inputDim : hiddenDim;
        int currentN = (layer == NUM_LAYERS - 1) ? outputDim : hiddenDim;
        bool isLastLayer = layer == NUM_LAYERS - 1;

        int chunk_row_b = idx / (WMMA_K / 8);
        int chunk_col_b = idx % (WMMA_K / 8) * 8;

        if (chunk_row_b < TILE_COUNT_X * WMMA_N) {
            bool b_valid = (chunk_row_b < currentN) && (chunk_col_b < currentK);
            cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b],
                        &d_weights_array[layer][chunk_row_b * currentK + chunk_col_b],
                        b_valid);
        }

        cp_async_commit();
        cp_async_wait();
        __syncthreads();
        
        int buf = 0;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_FACTOR];
        #pragma unroll
        for(int i = 0; i < WARP_FACTOR; i++) {
            nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);
        }

        for (int k = 0; k < currentK; k += WMMA_K) {
            int next_buf = 1 - buf;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b;
            
            if (k + WMMA_K < currentK) {
                if (chunk_row_b < TILE_COUNT_X * WMMA_N) {
                    bool isValid = (chunk_row_b < currentN) && (chunk_col_b + k + WMMA_K) < currentK;
                    cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b],
                                &d_weights_array[layer][chunk_row_b * currentK + (chunk_col_b + k + WMMA_K)],
                                isValid);
                }
                cp_async_commit();
            }

            nvcuda::wmma::load_matrix_sync(frag_b, &shmem_B[buf][warpN * WMMA_K][0], WMMA_K + PAD);

            #pragma unroll
            for (int i = 0; i < WARP_FACTOR; i++) {
                nvcuda::wmma::load_matrix_sync(frag_a, &shmem_A[(threadIdx.z + i*blockDim.z)*WMMA_M][k], TILE_COUNT_X*WMMA_K + PAD);
                nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
            }

            if (k + WMMA_K < currentK) {
                cp_async_wait();
                __syncthreads();
            }

            buf = next_buf;
        }

        int base_col = warpN * WMMA_N;

        #pragma unroll
        for(int j = 0; j < WARP_FACTOR; j++) {
            #pragma unroll
            for (int i = 0; i < frag_acc[j].num_elements; i++) {
                int local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                int local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                
                int global_r = (warpM + j*blockDim.z) * WMMA_M + local_row;
                int global_c = base_col + local_col; 

                float val = frag_acc[j].x[i];

                if (global_c < currentN)
                    val += d_biases_array[layer][global_c];

                if(!isLastLayer) {
                    val = fmaxf(val, 0.0f); // ReLU
                } else if constexpr (OUT_ACT == 1) {
                    val = 1.0f / (1.0f + expf(-val)); // Sigmoid
                }

                if (global_r < batchSize && global_c < currentN) {
                    if (isLastLayer) {
                        d_outputs[global_r * outputDim + global_c] = val;
                    } else {
                        shmem_A[(threadIdx.z + j*blockDim.z)*WMMA_M + local_row][base_col + local_col] = __float2half(val);
                    }
                }
            }
        }
        __syncthreads();

        if (!isLastLayer) {
            int total_threads = blockDim.x * blockDim.y * blockDim.z;
            int currentN_half2 = currentN / 2; 
            int total_elements_half2 = (TILE_COUNT_Y * WMMA_M) * currentN_half2; 

            for (int i = idx; i < total_elements_half2; i += total_threads) {
                int local_r = i / currentN_half2; 
                int local_c = (i % currentN_half2) * 2;
                
                int global_r = (blockIdx.x * TILE_COUNT_Y) * WMMA_M + local_r;
                
                if (global_r < batchSize && local_c < currentN) {
                    store_cg_two_f16(&d_activations[layer + 1][global_r * currentN + local_c], shmem_A[local_r][local_c], shmem_A[local_r][local_c + 1]);
                }
            }
        }
        __syncthreads();
    }
}

template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int OUT_ACT>
static void launchWithLayers_2d(
    dim3 grid, dim3 block, cudaStream_t stream,
    half* d_inputs, half** d_weights_array, float** d_biases_array,
    float* d_outputs, int batchSize, int inputDim, int hiddenDim, int outputDim, int numLayers
) {
    switch (numLayers) {
        case 1: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 1, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 2: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 2, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 3: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 3, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 4: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 4, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 5: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 5, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 6: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 6, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 7: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 7, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 8: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 8, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 9: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 9, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 10: networkFusionMMA_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 10, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        default: break;
    }
}

void launchNetworkFusionKernel(
    MLPOption *opt,
    half *d_inputs,
    half **d_weights_array,
    float **d_biases_array, 
    float *d_outputs, 
    int batchSize, 
    cudaStream_t stream
) {
    int maxDim = (opt->hiddenDim > opt->inputDim) ? opt->hiddenDim : opt->inputDim;

    if (maxDim > 128) return;

    // Hardcode the warp factor to 4 (Each warp computes 4 tiles vertically)
    const int WARP_FACTOR = 4;
    
    // Calculate the Dynamic Warp Folding geometry
    int tileCountX = maxDim / 16;
    if (tileCountX == 0) tileCountX = 1; // Safety fallback for sizes < 16
    
    int tileCountY = 8 / tileCountX; // 8 Warps guaranteed per block
    
    // Calculate total batch rows processed by a single thread block
    int totalRows = tileCountY * WARP_FACTOR * 16;

    // Grid size scales to cover the entire Batch Dimension
    int numBlocks = (batchSize + totalRows - 1) / totalRows;
    dim3 grid(numBlocks, 1, 1);
    
    // Block size maps threads directly to the warp topology
    // X = 32 threads per warp
    // Y = Warps in the X-direction (Hidden Dim)
    // Z = Warps in the Y-direction (Batch Dim)
    dim3 block(32, tileCountX, tileCountY);

    // Macro to dispatch on output activation type
    #define DISPATCH_INF(TCY, TCX, WF) \
        if (opt->outputActivation == OUT_ACT_SIGMOID) \
            launchWithLayers_2d<TCY, TCX, WF, 1>(grid, block, stream, d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers); \
        else \
            launchWithLayers_2d<TCY, TCX, WF, 0>(grid, block, stream, d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);

    // Switchboard to instantiate the core blocks geometries
    switch (maxDim) {
        case 16:  DISPATCH_INF(8*WARP_FACTOR, 1, WARP_FACTOR); break;
        case 32:  DISPATCH_INF(4*WARP_FACTOR, 2, WARP_FACTOR); break;
        case 64:  DISPATCH_INF(2*WARP_FACTOR, 4, WARP_FACTOR); break;
        case 128: DISPATCH_INF(1*WARP_FACTOR, 8, WARP_FACTOR); break;
        default: break;
    }
    #undef DISPATCH_INF
}

template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int OUT_ACT>
static void launchWithLayersGrad_2d(
    dim3 grid, dim3 block, cudaStream_t stream,
    half* d_inputs, half** d_activations, half** d_weights_array, float** d_biases_array,
    float* d_outputs, int batchSize, int inputDim, int hiddenDim, int outputDim, int numLayers
) {
    switch (numLayers) {
        case 1: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 1, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 2: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 2, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 3: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 3, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 4: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 4, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 5: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 5, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 6: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 6, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 7: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 7, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 8: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 8, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 9: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 9, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        case 10: networkFusionMMAGrad_2d<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 10, OUT_ACT><<<grid, block, 0, stream>>>(d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, inputDim, hiddenDim, outputDim); break;
        default: break;
    }
}

void launchNetworkFusionGradKernel(
    MLPOption *opt,
    half *d_inputs,
    half **d_activations,
    half **d_weights_array,
    float **d_biases_array, 
    float *d_outputs, 
    int batchSize, 
    cudaStream_t stream
) {
    int maxDim = (opt->hiddenDim > opt->inputDim) ? opt->hiddenDim : opt->inputDim;

    if (maxDim > 128) return;

    const int WARP_FACTOR = 4;
    
    int tileCountX = maxDim / 16;
    if (tileCountX == 0) tileCountX = 1;
    int tileCountY = 8 / tileCountX;
    
    int totalRows = tileCountY * WARP_FACTOR * 16;
    int numBlocks = (batchSize + totalRows - 1) / totalRows;
    dim3 grid(numBlocks, 1, 1);
    dim3 block(32, tileCountX, tileCountY);

    #define DISPATCH_GRAD(TCY, TCX, WF) \
        if (opt->outputActivation == OUT_ACT_SIGMOID) \
            launchWithLayersGrad_2d<TCY, TCX, WF, 1>(grid, block, stream, d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers); \
        else \
            launchWithLayersGrad_2d<TCY, TCX, WF, 0>(grid, block, stream, d_inputs, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers);

    switch (maxDim) {
        case 16:  DISPATCH_GRAD(8*WARP_FACTOR, 1, WARP_FACTOR); break;
        case 32:  DISPATCH_GRAD(4*WARP_FACTOR, 2, WARP_FACTOR); break;
        case 64:  DISPATCH_GRAD(2*WARP_FACTOR, 4, WARP_FACTOR); break;
        case 128: DISPATCH_GRAD(1*WARP_FACTOR, 8, WARP_FACTOR); break;
        default: break;
    }
    #undef DISPATCH_GRAD
}
