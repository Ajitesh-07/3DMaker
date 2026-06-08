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

__device__ __forceinline__ void loadVec4(const float* __restrict__ ptr, float dst[4]) {
    #ifndef __INTELLISENSE__
    asm volatile(
        "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "l"(ptr)
    );
    #endif
}

__device__ __forceinline__ void store_cg_two_f16(__half* ptr, __half a, __half b) {
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


__device__ __forceinline__ uint32_t hash_coords(int cx, int cy, int cz, int T) {
    return ((static_cast<uint32_t>(cx) * 1U) ^ 
            (static_cast<uint32_t>(cy) * 2654435761U) ^ 
            (static_cast<uint32_t>(cz) * 805459861U)) & static_cast<uint32_t>(T - 1);
}

__device__ __forceinline__ uint32_t dense_index(int cx, int cy, int cz, int N_l) {
    return cx + cy * (N_l + 1) + cz * (N_l + 1) * (N_l + 1);
}


// this kernel assumes the d_inputs is of size batch size x 3
template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int NUM_LAYERS, int FEATURES_LEVEL>
__global__ void __launch_bounds__(256, 3) networkFusionHashTableMMA_2d_vector3(
    float* d_inputs,
    half* d_hashtable,
    half** d_weights_array,
    float** d_biases_array,
    float* d_outputs,
    int batchSize,
    int hiddenDim,
    int outputDim,
    int tableSize,
    int numLevels,
    float b,
    int lowestSize,
    int denseLevelStart
) {
    // 8 Warps per block is guranteed
    uint32_t idx = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int inputDim = numLevels * FEATURES_LEVEL;
    
    uint32_t warpM = blockIdx.x * TILE_COUNT_Y + threadIdx.z;
    uint32_t warpN = threadIdx.y;
    uint32_t laneId = threadIdx.x + 1 - 1;
 
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    const int PAD = 8;

    __shared__ half shmem_A[WMMA_M * TILE_COUNT_Y][TILE_COUNT_X*WMMA_K + PAD];
    __shared__ half shmem_B[2][TILE_COUNT_X*WMMA_N][WMMA_K + PAD];

    // Hash table
    // 8 elements comes from this fact, one warp processes 256 elements 16x16 tile, so 8 elemeents per thread
    // One thread processes 8 elements, so it processes 8 / featuresPerLevel total levels for a particular input

    {
        float inputVec[4];
        #pragma unroll
        for (int i = 0; i < WARP_FACTOR; i++) {
            int task_idx = idx + i * 256;
            uint32_t chunk_row = task_idx / (TILE_COUNT_X * WMMA_K / 8); 
            uint32_t chunk_col = (task_idx % (TILE_COUNT_X * WMMA_K / 8)) * 8;

            int global_row = blockIdx.x * (TILE_COUNT_Y * WMMA_M) + chunk_row;

            if (chunk_col < inputDim && global_row < batchSize) {
                loadVec4(&d_inputs[global_row * 4], inputVec);

                int levelStart = chunk_col / FEATURES_LEVEL;
                float N_l_float = lowestSize * __powf(b, levelStart);

                for (int l = levelStart; l < levelStart + (8 / FEATURES_LEVEL); l++) {
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

                    float wx1 = 1 - x_l + floorf(x_l);
                    float wy1 = 1 - y_l + floorf(y_l);
                    float wz1 = 1 - z_l + floorf(z_l);

                    bool isDense = l < denseLevelStart;

                    auto get_corner_pointer = [&](int cx, int cy, int cz) -> const half* {
                        uint32_t table_index = isDense ? dense_index(cx, cy, cz, N_l) 
                                                : hash_coords(cx, cy, cz, tableSize);
                        return &d_hashtable[l * tableSize * FEATURES_LEVEL + table_index * FEATURES_LEVEL];
                    };

                    const float w000 = wx1 * wy1 * wz1, w100 = (1 - wx1)  * wy1 * wz1;
                    const float w010 = wx1 * (1 - wy1)  * wz1, w110 = (1 - wx1)  * (1 - wy1)  * wz1;
                    const float w001 = wx1 * wy1 * (1 - wz1),  w101 = (1 - wx1) * wy1 * (1 - wz1);
                    const float w011 = wx1 * (1 - wy1)  * (1 - wz1),  w111 = (1 - wx1) * (1 - wy1)  * (1 - wz1);

                    const half* v000 = get_corner_pointer(x0, y0, z0);
                    const half* v100 = get_corner_pointer(x1, y0, z0);
                    const half* v010 = get_corner_pointer(x0, y1, z0);
                    const half* v110 = get_corner_pointer(x1, y1, z0);
                    const half* v001 = get_corner_pointer(x0, y0, z1);
                    const half* v101 = get_corner_pointer(x1, y0, z1);
                    const half* v011 = get_corner_pointer(x0, y1, z1);
                    const half* v111 = get_corner_pointer(x1, y1, z1);

                    if constexpr (FEATURES_LEVEL == 2) {
                        // load both features for each vertex as __half2
                        float2 f000 = __half22float2(*reinterpret_cast<const __half2*>(v000));
                        float2 f100 = __half22float2(*reinterpret_cast<const __half2*>(v100));
                        float2 f010 = __half22float2(*reinterpret_cast<const __half2*>(v010));
                        float2 f110 = __half22float2(*reinterpret_cast<const __half2*>(v110));
                        float2 f001 = __half22float2(*reinterpret_cast<const __half2*>(v001));
                        float2 f101 = __half22float2(*reinterpret_cast<const __half2*>(v101));
                        float2 f011 = __half22float2(*reinterpret_cast<const __half2*>(v011));
                        float2 f111 = __half22float2(*reinterpret_cast<const __half2*>(v111));

                        // accumulate in float — fma for both f=0 and f=1 lanes
                        float2 result;
                        result.x = __fmaf_rn(f000.x, w000, 
                                    __fmaf_rn(f100.x, w100, 
                                    __fmaf_rn(f010.x, w010, 
                                    __fmaf_rn(f110.x, w110,
                                    __fmaf_rn(f001.x, w001, 
                                    __fmaf_rn(f101.x, w101, 
                                    __fmaf_rn(f011.x, w011, 
                                            f111.x * w111)))))));

                        result.y = __fmaf_rn(f000.y, w000, 
                                    __fmaf_rn(f100.y, w100, 
                                    __fmaf_rn(f010.y, w010, 
                                    __fmaf_rn(f110.y, w110,
                                    __fmaf_rn(f001.y, w001, 
                                    __fmaf_rn(f101.y, w101, 
                                    __fmaf_rn(f011.y, w011, 
                                            f111.y * w111)))))));

                        // cast down to half only at store
                        int out_col = l * FEATURES_LEVEL;
                        *reinterpret_cast<__half2*>(&shmem_A[chunk_row][out_col]) = __float22half2_rn(result);
                    } else {
                        for (int f = 0; f < FEATURES_LEVEL; f++) {
                            float val = __half2float(__ldg(&v000[f])) * w000 + __half2float(__ldg(&v100[f])) * w100 +
                                        __half2float(__ldg(&v010[f])) * w010 + __half2float(__ldg(&v110[f])) * w110 +
                                        __half2float(__ldg(&v001[f])) * w001 + __half2float(__ldg(&v101[f])) * w101 +
                                        __half2float(__ldg(&v011[f])) * w011 + __half2float(__ldg(&v111[f])) * w111;
                            
                            int out_col = l * FEATURES_LEVEL + f;
                            shmem_A[chunk_row][out_col] = __float2half(val);
                        }
                    }

                    N_l_float *= b;
                }
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int currentK = (layer == 0) ? inputDim : hiddenDim;
        int currentN = (layer == NUM_LAYERS - 1) ? outputDim : hiddenDim;
        bool isLastLayer = layer == NUM_LAYERS - 1;

        uint32_t chunk_row_b = idx / (WMMA_K / 8);
        uint32_t chunk_col_b = idx % (WMMA_K / 8) * 8;

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
                uint32_t local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                uint32_t local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                
                int global_r = (warpM + j*blockDim.z) * WMMA_M + local_row;
                int global_c = base_col + local_col; 

                float val = frag_acc[j].x[i];

                if (global_c < currentN)
                    val += d_biases_array[layer][global_c];

                if(!isLastLayer) {
                    val = fmaxf(val, 0.0f); // ReLU
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


template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int FEATURES_LEVEL>
static void launchWithLayersHashGrid_2d(
    dim3 grid, dim3 block, cudaStream_t stream,
    float* d_inputs, half* d_hashtable, half** d_weights_array, float** d_biases_array,
    float* d_outputs, int batchSize, int tableSize, int hiddenDim, int outputDim, int numLayers,
    int numLevels, float b, int lowestSize, int denseLevelStart
) {
    switch (numLayers) {
        case 1: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 1, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 2: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 2, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 3: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 3, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 4: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 4, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 5: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 5, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 6: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 6, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 7: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 7, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 8: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 8, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 9: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 9, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 10: networkFusionHashTableMMA_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 10, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        default: break;
    }
}

void launchNetworkFusionHashTableKernel(
    MLPGridOptions*   opt,
    half*             d_hashtable,
    float*             d_inputs,
    half**            d_weights_array,
    float**           d_biases_array,
    float*            d_outputs,
    int               batchSize,
    cudaStream_t      stream
) {
    int inputDim = opt->numLevels * opt->featuresLevel;
    int maxDim = (opt->hiddenDim > inputDim) ? opt->hiddenDim : inputDim;

    if (maxDim > 128) return;

    const int WARP_FACTOR = 4;
    
    int tileCountX = maxDim / 16;
    if (tileCountX == 0) tileCountX = 1;
    int tileCountY = 8 / tileCountX;
    
    int totalRows = tileCountY * WARP_FACTOR * 16;
    int numBlocks = (batchSize + totalRows - 1) / totalRows;
    dim3 grid(numBlocks, 1, 1);
    dim3 block(32, tileCountX, tileCountY);

    // 1. Calculate the continuous threshold
    float inner_term = (std::cbrt(static_cast<float>(opt->tableSize)) - 1.0f) / opt->lowestSize;

    // 2. Calculate the base-b logarithm
    float continuous_level = std::log2(inner_term) / std::log2(opt->b);

    // 3. The first non-dense (hashed) level index
    int denseLevelStart = static_cast<int>(std::floor(continuous_level)) + 1;

    // 4. Clamp the result to ensure it stays within valid bounds [0, numLevels]
    denseLevelStart = std::max(0, std::min(denseLevelStart, opt->numLevels));
    

    switch (maxDim) {
        case 16:
            launchWithLayersHashGrid_2d<8*WARP_FACTOR, 1, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        case 32:
            launchWithLayersHashGrid_2d<4*WARP_FACTOR, 2, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        case 64:
            launchWithLayersHashGrid_2d<2*WARP_FACTOR, 4, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        case 128:
            launchWithLayersHashGrid_2d<1*WARP_FACTOR, 8, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        default:
            break;
    }
}


// ============================================================================
// TRAINING FORWARD PASS (Activation-Saving Variant)
// Identical to networkFusionHashTableMMA_2d_vector3 but writes intermediate
// activations to d_activations[layer+1] after each hidden layer for backprop.
// ============================================================================
template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int NUM_LAYERS, int FEATURES_LEVEL>
__global__ void __launch_bounds__(256, 3) networkFusionHashTableMMAGrad_2d_vector3(
    float* d_inputs,
    half* d_hashtable,
    half** d_activations,
    half** d_weights_array,
    float** d_biases_array,
    float* d_outputs,
    int batchSize,
    int hiddenDim,
    int outputDim,
    int tableSize,
    int numLevels,
    float b,
    int lowestSize,
    int denseLevelStart
) {
    uint32_t idx = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int inputDim = numLevels * FEATURES_LEVEL;
    
    uint32_t warpM = blockIdx.x * TILE_COUNT_Y + threadIdx.z;
    uint32_t warpN = threadIdx.y;
    uint32_t laneId = threadIdx.x + 1 - 1;
 
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    const int PAD = 8;

    __shared__ half shmem_A[WMMA_M * TILE_COUNT_Y][TILE_COUNT_X*WMMA_K + PAD];
    __shared__ half shmem_B[2][TILE_COUNT_X*WMMA_N][WMMA_K + PAD];

    // Hash table encoding into shmem_A (identical to inference variant)
    {
        float inputVec[4];
        #pragma unroll
        for (int i = 0; i < WARP_FACTOR; i++) {
            int task_idx = idx + i * 256;
            uint32_t chunk_row = task_idx / (TILE_COUNT_X * WMMA_K / 8); 
            uint32_t chunk_col = (task_idx % (TILE_COUNT_X * WMMA_K / 8)) * 8;

            int global_row = blockIdx.x * (TILE_COUNT_Y * WMMA_M) + chunk_row;

            if (chunk_col < inputDim && global_row < batchSize) {
                loadVec4(&d_inputs[global_row * 4], inputVec);

                int levelStart = chunk_col / FEATURES_LEVEL;
                float N_l_float = lowestSize * __powf(b, levelStart);

                for (int l = levelStart; l < levelStart + (8 / FEATURES_LEVEL); l++) {
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

                    float wx1 = 1 - x_l + floorf(x_l);
                    float wy1 = 1 - y_l + floorf(y_l);
                    float wz1 = 1 - z_l + floorf(z_l);

                    bool isDense = l < denseLevelStart;

                    auto get_corner_pointer = [&](int cx, int cy, int cz) -> const half* {
                        uint32_t table_index = isDense ? dense_index(cx, cy, cz, N_l) 
                                                : hash_coords(cx, cy, cz, tableSize);
                        return &d_hashtable[l * tableSize * FEATURES_LEVEL + table_index * FEATURES_LEVEL];
                    };

                    const float w000 = wx1 * wy1 * wz1, w100 = (1 - wx1)  * wy1 * wz1;
                    const float w010 = wx1 * (1 - wy1)  * wz1, w110 = (1 - wx1)  * (1 - wy1)  * wz1;
                    const float w001 = wx1 * wy1 * (1 - wz1),  w101 = (1 - wx1) * wy1 * (1 - wz1);
                    const float w011 = wx1 * (1 - wy1)  * (1 - wz1),  w111 = (1 - wx1) * (1 - wy1)  * (1 - wz1);

                    const half* v000 = get_corner_pointer(x0, y0, z0);
                    const half* v100 = get_corner_pointer(x1, y0, z0);
                    const half* v010 = get_corner_pointer(x0, y1, z0);
                    const half* v110 = get_corner_pointer(x1, y1, z0);
                    const half* v001 = get_corner_pointer(x0, y0, z1);
                    const half* v101 = get_corner_pointer(x1, y0, z1);
                    const half* v011 = get_corner_pointer(x0, y1, z1);
                    const half* v111 = get_corner_pointer(x1, y1, z1);

                    if constexpr (FEATURES_LEVEL == 2) {
                        float2 f000 = __half22float2(*reinterpret_cast<const __half2*>(v000));
                        float2 f100 = __half22float2(*reinterpret_cast<const __half2*>(v100));
                        float2 f010 = __half22float2(*reinterpret_cast<const __half2*>(v010));
                        float2 f110 = __half22float2(*reinterpret_cast<const __half2*>(v110));
                        float2 f001 = __half22float2(*reinterpret_cast<const __half2*>(v001));
                        float2 f101 = __half22float2(*reinterpret_cast<const __half2*>(v101));
                        float2 f011 = __half22float2(*reinterpret_cast<const __half2*>(v011));
                        float2 f111 = __half22float2(*reinterpret_cast<const __half2*>(v111));

                        float2 result;
                        result.x = __fmaf_rn(f000.x, w000, 
                                    __fmaf_rn(f100.x, w100, 
                                    __fmaf_rn(f010.x, w010, 
                                    __fmaf_rn(f110.x, w110,
                                    __fmaf_rn(f001.x, w001, 
                                    __fmaf_rn(f101.x, w101, 
                                    __fmaf_rn(f011.x, w011, 
                                            f111.x * w111)))))));

                        result.y = __fmaf_rn(f000.y, w000, 
                                    __fmaf_rn(f100.y, w100, 
                                    __fmaf_rn(f010.y, w010, 
                                    __fmaf_rn(f110.y, w110,
                                    __fmaf_rn(f001.y, w001, 
                                    __fmaf_rn(f101.y, w101, 
                                    __fmaf_rn(f011.y, w011, 
                                            f111.y * w111)))))));

                        int out_col = l * FEATURES_LEVEL;
                        *reinterpret_cast<__half2*>(&shmem_A[chunk_row][out_col]) = __float22half2_rn(result);
                    } else {
                        for (int f = 0; f < FEATURES_LEVEL; f++) {
                            float val = __half2float(__ldg(&v000[f])) * w000 + __half2float(__ldg(&v100[f])) * w100 +
                                        __half2float(__ldg(&v010[f])) * w010 + __half2float(__ldg(&v110[f])) * w110 +
                                        __half2float(__ldg(&v001[f])) * w001 + __half2float(__ldg(&v101[f])) * w101 +
                                        __half2float(__ldg(&v011[f])) * w011 + __half2float(__ldg(&v111[f])) * w111;
                            
                            int out_col = l * FEATURES_LEVEL + f;
                            shmem_A[chunk_row][out_col] = __float2half(val);
                        }
                    }

                    N_l_float *= b;
                }
            }
        }
    }
    __syncthreads();

    // Save the hash-encoded input to d_activations[0] for backward pass
    {
        int total_threads = blockDim.x * blockDim.y * blockDim.z;
        int inputDim_half2 = inputDim / 2;
        int total_elements_half2 = (TILE_COUNT_Y * WMMA_M) * inputDim_half2;

        for (int i = idx; i < total_elements_half2; i += total_threads) {
            int local_r = i / inputDim_half2;
            int local_c = (i % inputDim_half2) * 2;

            int global_r = (blockIdx.x * TILE_COUNT_Y) * WMMA_M + local_r;

            if (global_r < batchSize && local_c < inputDim) {
                store_cg_two_f16(&d_activations[0][global_r * inputDim + local_c], shmem_A[local_r][local_c], shmem_A[local_r][local_c + 1]);
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int currentK = (layer == 0) ? inputDim : hiddenDim;
        int currentN = (layer == NUM_LAYERS - 1) ? outputDim : hiddenDim;
        bool isLastLayer = layer == NUM_LAYERS - 1;

        uint32_t chunk_row_b = idx / (WMMA_K / 8);
        uint32_t chunk_col_b = idx % (WMMA_K / 8) * 8;

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
                uint32_t local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                uint32_t local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                
                int global_r = (warpM + j*blockDim.z) * WMMA_M + local_row;
                int global_c = base_col + local_col; 

                float val = frag_acc[j].x[i];

                if (global_c < currentN)
                    val += d_biases_array[layer][global_c];

                if(!isLastLayer) {
                    val = fmaxf(val, 0.0f); // ReLU
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

        // === ACTIVATION SAVING (only difference from inference variant) ===
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


template <int TILE_COUNT_Y, int TILE_COUNT_X, int WARP_FACTOR, int FEATURES_LEVEL>
static void launchWithLayersHashGridGrad_2d(
    dim3 grid, dim3 block, cudaStream_t stream,
    float* d_inputs, half* d_hashtable, half** d_activations, half** d_weights_array, float** d_biases_array,
    float* d_outputs, int batchSize, int tableSize, int hiddenDim, int outputDim, int numLayers,
    int numLevels, float b, int lowestSize, int denseLevelStart
) {
    switch (numLayers) {
        case 1: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 1, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 2: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 2, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 3: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 3, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 4: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 4, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 5: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 5, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 6: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 6, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 7: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 7, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 8: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 8, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 9: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 9, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        case 10: networkFusionHashTableMMAGrad_2d_vector3<TILE_COUNT_Y, TILE_COUNT_X, WARP_FACTOR, 10, FEATURES_LEVEL><<<grid, block, 0, stream>>>(d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, hiddenDim, outputDim, tableSize, numLevels, b, lowestSize, denseLevelStart); break;
        default: break;
    }
}

void launchNetworkFusionHashTableGradKernel(
    MLPGridOptions*   opt,
    half*             d_hashtable,
    float*            d_inputs,
    half**            d_activations,
    half**            d_weights_array,
    float**           d_biases_array,
    float*            d_outputs,
    int               batchSize,
    cudaStream_t      stream
) {
    int inputDim = opt->numLevels * opt->featuresLevel;
    int maxDim = (opt->hiddenDim > inputDim) ? opt->hiddenDim : inputDim;

    if (maxDim > 128) return;

    const int WARP_FACTOR = 4;
    
    int tileCountX = maxDim / 16;
    if (tileCountX == 0) tileCountX = 1;
    int tileCountY = 8 / tileCountX;
    
    int totalRows = tileCountY * WARP_FACTOR * 16;
    int numBlocks = (batchSize + totalRows - 1) / totalRows;
    dim3 grid(numBlocks, 1, 1);
    dim3 block(32, tileCountX, tileCountY);

    float inner_term = (std::cbrt(static_cast<float>(opt->tableSize)) - 1.0f) / opt->lowestSize;
    float continuous_level = std::log2(inner_term) / std::log2(opt->b);
    int denseLevelStart = static_cast<int>(std::floor(continuous_level)) + 1;
    denseLevelStart = std::max(0, std::min(denseLevelStart, opt->numLevels));

    switch (maxDim) {
        case 16:
            launchWithLayersHashGridGrad_2d<8*WARP_FACTOR, 1, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        case 32:
            launchWithLayersHashGridGrad_2d<4*WARP_FACTOR, 2, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        case 64:
            launchWithLayersHashGridGrad_2d<2*WARP_FACTOR, 4, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        case 128:
            launchWithLayersHashGridGrad_2d<1*WARP_FACTOR, 8, WARP_FACTOR, 2>(grid, block, stream,
                d_inputs, d_hashtable, d_activations, d_weights_array, d_biases_array, d_outputs, batchSize, opt->tableSize, opt->hiddenDim, opt->outputDim, opt->numLayers, opt->numLevels, opt->b, opt->lowestSize, denseLevelStart);
            break;
        default:
            break;
    }
}
