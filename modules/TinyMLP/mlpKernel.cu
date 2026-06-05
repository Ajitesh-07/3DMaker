#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include "TinyMLP.h"

#define NO_WARPS 8

__global__ void singleLayerMMA(
    half* d_in,
    half* d_weight,
    float* d_bias,
    half* d_out,
    int M,
    int N,
    int K,
    bool applyRelu
) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    const int WARPS_PER_BLOCK = NO_WARPS;
    const int PAD = 8;
    
    __shared__ half smem_A[2][WARPS_PER_BLOCK * WMMA_M][WMMA_K+PAD];
    __shared__ half smem_B[2][WMMA_N][WMMA_K+PAD];

    __shared__ float smem_C[WARPS_PER_BLOCK * WMMA_M][WMMA_N];
    
    const int numALoad = WARPS_PER_BLOCK * WMMA_M * WMMA_K / numThreads;
    const int numBLoad = WMMA_K * WMMA_N / numThreads;

    // Preload the first block once
    for (int i = 0; i < numALoad; i++) {
        int localIdx = i * numThreads + idx;
        int local_row = localIdx / WMMA_K;
        int local_col = localIdx % WMMA_K;

        int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
        int global_col = local_col;

        if (global_row < M && global_col < K) {
            smem_A[0][local_row][local_col] = d_in[(global_row * K) + global_col];
        } else {
            smem_A[0][local_row][local_col] = __float2half(0.0f);
        }
    }
    for (int i = 0; i < numBLoad; i++) {
        int local_idx = (i * numThreads) + idx;

        int local_row = local_idx % WMMA_N;
        int local_col = local_idx / WMMA_N;

        int global_row = local_row;
        int global_col = (blockIdx.x * WMMA_N) + local_col;

        if (global_row < K && global_col < N) {
            smem_B[0][local_col][local_row] = d_weight[(global_col * K) + global_row];
        } else {
            smem_B[0][local_col][local_row] = __float2half(0.0f);
        }
    }

    __syncthreads();

    int buf = 0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
    nvcuda::wmma::fill_fragment(frag_acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int next_buf = 1 - buf;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b;

        nvcuda::wmma::load_matrix_sync(frag_a, &smem_A[buf][WMMA_M*threadIdx.y][0], WMMA_K+PAD);
        nvcuda::wmma::load_matrix_sync(frag_b, &smem_B[buf][0][0], WMMA_K+PAD);

        if (k + WMMA_K < K) {
            for (int i = 0; i < numALoad; i++) {
                int localIdx = i * numThreads + idx;
                int local_row = localIdx / WMMA_K;
                int local_col = localIdx % WMMA_K;

                int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
                int global_col = (k + WMMA_K) + local_col;

                if (global_row < M && global_col < K) {
                    smem_A[next_buf][local_row][local_col] = d_in[(global_row * K) + global_col];
                } else {
                    smem_A[next_buf][local_row][local_col] = __float2half(0.0f);
                }
            }

            for (int i = 0; i < numBLoad; i++) {
                int local_idx = (i * numThreads) + idx;

                int local_row = local_idx % WMMA_N;
                int local_col = local_idx / WMMA_N;

                int global_row = (k + WMMA_K) + local_row;
                int global_col = (blockIdx.x * WMMA_N) + local_col;

                if (global_row < K && global_col < N) {
                    smem_B[next_buf][local_col][local_row] = d_weight[(global_col * K) + global_row];
                } else {
                    smem_B[next_buf][local_col][local_row] = __float2half(0.0f);
                }
            }
        }

        nvcuda::wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);

        __syncthreads();
        buf = next_buf;
    }   

    nvcuda::wmma::store_matrix_sync(&smem_C[threadIdx.y * WMMA_M][0], frag_acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    int numCLoad = (WARPS_PER_BLOCK * WMMA_M * WMMA_N) / numThreads;
    for (int i = 0; i < numCLoad; i++) {
        int local_idx = (i * numThreads) + idx;
        int local_row = local_idx / WMMA_N;
        int local_col = local_idx % WMMA_N;

        int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
        int global_col = (blockIdx.x * WMMA_N) + local_col;
        if (global_row < M && global_col < N) {
            float val = smem_C[local_row][local_col];

            val += d_bias[global_col]; 
            
            if (applyRelu && val < 0.0f) val = 0.0f;

            d_out[(global_row * N) + global_col] = __float2half(val);
        }
    }

}

__global__ void singleLayerMMA_fp32(
    half* d_in,
    half* d_weight,
    float* d_bias,
    float* d_out,
    int M,
    int N,
    int K,
    bool applyRelu
) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    const int WARPS_PER_BLOCK = NO_WARPS;
    const int PAD = 8;
    
    __shared__ half smem_A[2][WARPS_PER_BLOCK * WMMA_M][WMMA_K+PAD];
    __shared__ half smem_B[2][WMMA_N][WMMA_K+PAD];

    __shared__ float smem_C[WARPS_PER_BLOCK * WMMA_M][WMMA_N];
    
    const int numALoad = WARPS_PER_BLOCK * WMMA_M * WMMA_K / numThreads;
    const int numBLoad = WMMA_K * WMMA_N / numThreads;

    // Preload the first block once
    for (int i = 0; i < numALoad; i++) {
        int localIdx = i * numThreads + idx;
        int local_row = localIdx / WMMA_K;
        int local_col = localIdx % WMMA_K;

        int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
        int global_col = local_col;

        if (global_row < M && global_col < K) {
            smem_A[0][local_row][local_col] = d_in[(global_row * K) + global_col];
        } else {
            smem_A[0][local_row][local_col] = __float2half(0.0f);
        }
    }
    for (int i = 0; i < numBLoad; i++) {
        int local_idx = (i * numThreads) + idx;

        int local_row = local_idx % WMMA_N;
        int local_col = local_idx / WMMA_N;

        int global_row = local_row;
        int global_col = (blockIdx.x * WMMA_N) + local_col;

        if (global_row < K && global_col < N) {
            smem_B[0][local_col][local_row] = d_weight[(global_col * K) + global_row];
        } else {
            smem_B[0][local_col][local_row] = __float2half(0.0f);
        }
    }

    __syncthreads();

    int buf = 0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
    nvcuda::wmma::fill_fragment(frag_acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int next_buf = 1 - buf;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b;

        nvcuda::wmma::load_matrix_sync(frag_a, &smem_A[buf][WMMA_M*threadIdx.y][0], WMMA_K+PAD);
        nvcuda::wmma::load_matrix_sync(frag_b, &smem_B[buf][0][0], WMMA_K+PAD);

        if (k + WMMA_K < K) {
            for (int i = 0; i < numALoad; i++) {
                int localIdx = i * numThreads + idx;
                int local_row = localIdx / WMMA_K;
                int local_col = localIdx % WMMA_K;

                int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
                int global_col = (k + WMMA_K) + local_col;

                if (global_row < M && global_col < K) {
                    smem_A[next_buf][local_row][local_col] = d_in[(global_row * K) + global_col];
                } else {
                    smem_A[next_buf][local_row][local_col] = __float2half(0.0f);
                }
            }

            for (int i = 0; i < numBLoad; i++) {
                int local_idx = (i * numThreads) + idx;

                int local_row = local_idx % WMMA_N;
                int local_col = local_idx / WMMA_N;

                int global_row = (k + WMMA_K) + local_row;
                int global_col = (blockIdx.x * WMMA_N) + local_col;

                if (global_row < K && global_col < N) {
                    smem_B[next_buf][local_col][local_row] = d_weight[(global_col * K) + global_row];
                } else {
                    smem_B[next_buf][local_col][local_row] = __float2half(0.0f);
                }
            }
        }

        nvcuda::wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);

        __syncthreads();
        buf = next_buf;
    }   

    nvcuda::wmma::store_matrix_sync(&smem_C[threadIdx.y * WMMA_M][0], frag_acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    int numCLoad = (WARPS_PER_BLOCK * WMMA_M * WMMA_N) / numThreads;
    for (int i = 0; i < numCLoad; i++) {
        int local_idx = (i * numThreads) + idx;
        int local_row = local_idx / WMMA_N;
        int local_col = local_idx % WMMA_N;

        int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
        int global_col = (blockIdx.x * WMMA_N) + local_col;
        if (global_row < M && global_col < N) {
            float val = smem_C[local_row][local_col];

            val += d_bias[global_col]; 
            if (applyRelu && val < 0.0f) val = 0.0f;

            d_out[(global_row * N) + global_col] = val;
        }
    }
}

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

__global__ void singleLayerMMA_LDGSTS(
    half* d_in,
    half* d_weight,
    float* d_bias,
    half* d_out,
    int M,
    int N,
    int K,
    bool applyRelu
) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    const int WARPS_PER_BLOCK = NO_WARPS;
    const int PAD = 8;
    
    __shared__ half smem_A[2][WARPS_PER_BLOCK * WMMA_M][WMMA_K+PAD];
    __shared__ half smem_B[2][WMMA_N][WMMA_K+PAD];
    __shared__ float smem_C[WARPS_PER_BLOCK * WMMA_M][WMMA_N];

    int a_chunk_row = idx / 2;
    int a_chunk_col = (idx % 2) * 8;
    
    int a_global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + a_chunk_row;
    int a_global_col = a_chunk_col;
    
    bool a_valid = (a_global_row < M) && (a_global_col < K);
    
    cp_async_128(&smem_A[0][a_chunk_row][a_chunk_col], 
                 &d_in[a_global_row * K + a_global_col], 
                 a_valid);

    if (idx < 32) {
        int b_chunk_col = idx / 2;
        int b_chunk_row = (idx % 2) * 8;
        
        int b_global_col = (blockIdx.x * WMMA_N) + b_chunk_col;
        int b_global_row = b_chunk_row;
        
        bool b_valid = (b_global_row < K) && (b_global_col < N);
        
        cp_async_128(&smem_B[0][b_chunk_col][b_chunk_row], 
                     &d_weight[b_global_col * K + b_global_row], 
                     b_valid);
    }
    
    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    int buf = 0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
    nvcuda::wmma::fill_fragment(frag_acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int next_buf = 1 - buf;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b;

        if (k + WMMA_K < K) {
            int a_next_col = (k + WMMA_K) + a_chunk_col;
            bool a_next_valid = (a_global_row < M) && (a_next_col < K);
            
            cp_async_128(&smem_A[next_buf][a_chunk_row][a_chunk_col], 
                         &d_in[a_global_row * K + a_next_col], 
                         a_next_valid);

            if (idx < 32) {
                int b_chunk_col = idx / 2;
                int b_chunk_row = (idx % 2) * 8;
                
                int b_global_col = (blockIdx.x * WMMA_N) + b_chunk_col;
                int b_next_row = (k + WMMA_K) + b_chunk_row;
                
                bool b_next_valid = (b_next_row < K) && (b_global_col < N);
                
                cp_async_128(&smem_B[next_buf][b_chunk_col][b_chunk_row], 
                             &d_weight[b_global_col * K + b_next_row], 
                             b_next_valid);
            }
            cp_async_commit();
        }

        nvcuda::wmma::load_matrix_sync(frag_a, &smem_A[buf][WMMA_M*threadIdx.y][0], WMMA_K+PAD);
        nvcuda::wmma::load_matrix_sync(frag_b, &smem_B[buf][0][0], WMMA_K+PAD);
        
        nvcuda::wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);

        if (k + WMMA_K < K) {
            cp_async_wait();
            __syncthreads();
        }
        buf = next_buf;
    }   
    
    nvcuda::wmma::store_matrix_sync(&smem_C[threadIdx.y * WMMA_M][0], frag_acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    int numCLoad = (WARPS_PER_BLOCK * WMMA_M * WMMA_N) / numThreads;
    for (int i = 0; i < numCLoad; i++) {
        int local_idx = (i * numThreads) + idx;
        int local_row = local_idx / WMMA_N;
        int local_col = local_idx % WMMA_N;

        int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
        int global_col = (blockIdx.x * WMMA_N) + local_col;
        
        if (global_row < M && global_col < N) {
            float val = smem_C[local_row][local_col];
            val += d_bias[global_col]; 
            
            if (applyRelu && val < 0.0f) val = 0.0f;

            d_out[(global_row * N) + global_col] = __float2half(val);
        }
    }
}

__global__ void singleLayerMMA_LDGSTS_fp32(
    half* d_in,
    half* d_weight,
    float* d_bias,
    float* d_out,
    int M,
    int N,
    int K,
    bool applyRelu
) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    const int WARPS_PER_BLOCK = NO_WARPS;
    const int PAD = 8;
    
    __shared__ half smem_A[2][WARPS_PER_BLOCK * WMMA_M][WMMA_K+PAD];
    __shared__ half smem_B[2][WMMA_N][WMMA_K+PAD];
    __shared__ float smem_C[WARPS_PER_BLOCK * WMMA_M][WMMA_N];
    
    int a_chunk_row = idx / 2;
    int a_chunk_col = (idx % 2) * 8;
    
    int a_global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + a_chunk_row;
    int a_global_col = a_chunk_col;
    
    bool a_valid = (a_global_row < M) && (a_global_col < K);
    
    cp_async_128(&smem_A[0][a_chunk_row][a_chunk_col], 
                 &d_in[a_global_row * K + a_global_col], 
                 a_valid);

    if (idx < 32) {
        int b_chunk_col = idx / 2;
        int b_chunk_row = (idx % 2) * 8;
        
        int b_global_col = (blockIdx.x * WMMA_N) + b_chunk_col;
        int b_global_row = b_chunk_row;
        
        bool b_valid = (b_global_row < K) && (b_global_col < N);
        
        cp_async_128(&smem_B[0][b_chunk_col][b_chunk_row], 
                     &d_weight[b_global_col * K + b_global_row], 
                     b_valid);
    }
    
    cp_async_commit();
    cp_async_wait();
    __syncthreads();
    
    int buf = 0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
    nvcuda::wmma::fill_fragment(frag_acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int next_buf = 1 - buf;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b;

        if (k + WMMA_K < K) {
            int a_next_col = (k + WMMA_K) + a_chunk_col;
            bool a_next_valid = (a_global_row < M) && (a_next_col < K);
            
            cp_async_128(&smem_A[next_buf][a_chunk_row][a_chunk_col], 
                         &d_in[a_global_row * K + a_next_col], 
                         a_next_valid);

            if (idx < 32) {
                int b_chunk_col = idx / 2;
                int b_chunk_row = (idx % 2) * 8;
                
                int b_global_col = (blockIdx.x * WMMA_N) + b_chunk_col;
                int b_next_row = (k + WMMA_K) + b_chunk_row;
                
                bool b_next_valid = (b_next_row < K) && (b_global_col < N);
                
                cp_async_128(&smem_B[next_buf][b_chunk_col][b_chunk_row], 
                             &d_weight[b_global_col * K + b_next_row], 
                             b_next_valid);
            }
            cp_async_commit();
        }

        nvcuda::wmma::load_matrix_sync(frag_a, &smem_A[buf][WMMA_M*threadIdx.y][0], WMMA_K+PAD);
        nvcuda::wmma::load_matrix_sync(frag_b, &smem_B[buf][0][0], WMMA_K+PAD);
        
        nvcuda::wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);

        if (k + WMMA_K < K) {
            cp_async_wait();
            __syncthreads();
        }
        buf = next_buf;
    }   

    nvcuda::wmma::store_matrix_sync(&smem_C[threadIdx.y * WMMA_M][0], frag_acc, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    int numCLoad = (WARPS_PER_BLOCK * WMMA_M * WMMA_N) / numThreads;
    for (int i = 0; i < numCLoad; i++) {
        int local_idx = (i * numThreads) + idx;
        int local_row = local_idx / WMMA_N;
        int local_col = local_idx % WMMA_N;

        int global_row = (blockIdx.y * WARPS_PER_BLOCK * WMMA_M) + local_row;
        int global_col = (blockIdx.x * WMMA_N) + local_col;
        
        if (global_row < M && global_col < N) {
            float val = smem_C[local_row][local_col];
            val += d_bias[global_col]; 
            
            if (applyRelu && val < 0.0f) val = 0.0f;

            d_out[(global_row * N) + global_col] = val;
        }
    }
}

extern "C" void launchForwardKernel(
    MLPOption*   opt,
    half*        d_inputs,
    half**       d_weights_array,
    float**      d_biases_array,
    float*       d_outputs,
    int          batchSize,
    half*        d_ping,          // pre-allocated by caller (class owns these)
    half*        d_pong,
    cudaStream_t stream
) {
    half* current_in  = d_inputs;
    half* current_out = d_ping;

    const int noWarps = NO_WARPS;   // must match WARPS_PER_BLOCK inside the kernels
    dim3 block(32, noWarps);

    for (int i = 0; i < opt->numLayers; i++) {
        int inDim  = (i == 0)                     ? opt->inputDim  : opt->hiddenDim;
        int outDim = (i == opt->numLayers - 1)    ? opt->outputDim : opt->hiddenDim;
        bool isLastLayer = (i == opt->numLayers - 1);
        bool applyReLU   = !isLastLayer;

        int tilesM = (batchSize + 15) / 16;
        int tilesN = (outDim    + 15) / 16;
        dim3 grid(tilesN, (tilesM + noWarps - 1) / noWarps);

        if (isLastLayer) {
            singleLayerMMA_LDGSTS_fp32<<<grid, block, 0, stream>>>(
                current_in, d_weights_array[i], d_biases_array[i],
                d_outputs, batchSize, outDim, inDim, /*applyRelu=*/false
            );
        } else {
            singleLayerMMA_LDGSTS<<<grid, block, 0, stream>>>(
                current_in, d_weights_array[i], d_biases_array[i],
                current_out, batchSize, outDim, inDim, applyReLU
            );
            current_in  = current_out;
            current_out = (current_out == d_ping) ? d_pong : d_ping;
        }
    }
}