#include <cstdio>
#include "EmbeddingTable.h"

template <int NUM_FEATURES>
__global__ void rowAdagradOptimizer(
    const int* __restrict__ dense_row_id,
    const float* __restrict__ dense_grads,
    float* __restrict__ dense_row_moments,
    float* __restrict__ master_weights,
    const float* __restrict__ lr,
    float eps,
    int numUniqueRows
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numUniqueRows) return;

    int rowId = dense_row_id[idx];
    if (rowId < 0) return;

    const float* grads = dense_grads + ((size_t)idx * NUM_FEATURES);

    float current_moment_mean = 0.0f;
    float inv_features = 1.0f / NUM_FEATURES;
    #pragma unroll
    for(int i = 0; i < NUM_FEATURES; i++) {
        float val = grads[i];
        current_moment_mean += val*val;
    }
    current_moment_mean *= inv_features;
    current_moment_mean = current_moment_mean + dense_row_moments[rowId];

    dense_row_moments[rowId] = current_moment_mean;
    float normalizer = 1.0f / (sqrtf(current_moment_mean) + eps);

    float* weights = master_weights + ((size_t)rowId * NUM_FEATURES);
    #pragma unroll
    for(int i = 0; i < NUM_FEATURES; i++) {
        weights[i] -= lr[i] * grads[i] * normalizer;
    }
}

template <int NUM_FEATURES>
__global__ void rowAdagradOptimizerCoalesced(
    const int* __restrict__ dense_row_id,
    const float* __restrict__ dense_grads,
    float* __restrict__ dense_row_moments,
    float* __restrict__ master_weights,
    const float* __restrict__ lr,
    float eps,
    int numUniqueRows
) {
    int rowIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int laneId = threadIdx.x;
    if (rowIdx >= numUniqueRows) return;

    int rowId = dense_row_id[rowIdx];
    if (rowId < 0) return;

    const float* my_grads = dense_grads + ((size_t)rowIdx * NUM_FEATURES);
    float* my_weights = master_weights + ((size_t)rowId * NUM_FEATURES);

    bool is_valid_feature = laneId < NUM_FEATURES;
    float gradVal = is_valid_feature ? my_grads[laneId] : 0.0f;

    float row_sum_sq = gradVal * gradVal;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        row_sum_sq += __shfl_xor_sync(0xFFFFFFFF, row_sum_sq, o);

    float current_moment_mean = row_sum_sq / NUM_FEATURES + dense_row_moments[rowId];
    if (laneId == 0) dense_row_moments[rowId] = current_moment_mean;
    float normalizer = 1.0f / (sqrtf(current_moment_mean) + eps);

    if (is_valid_feature) {
        my_weights[laneId] -= lr[laneId] * gradVal * normalizer;
    }
}

void launchRowAdagradOptimizer(
    int num_features,
    const int* dense_row_id,
    const float* dense_grads,
    float* dense_row_moments,
    float* master_weights,
    const float* lr,
    float eps,
    int numUniqueRows,
    bool use_coalesced,
    cudaStream_t stream
) {
    if (numUniqueRows == 0) return;

    int threads_uncoal = 256;
    int blocks_uncoal = (numUniqueRows + threads_uncoal - 1) / threads_uncoal;

    dim3 threads_coal(32, 8);
    dim3 blocks_coal((numUniqueRows + threads_coal.y - 1) / threads_coal.y);
    #define LAUNCH_OPT(N) \
        case N: \
            if (use_coalesced) { \
                rowAdagradOptimizerCoalesced<N><<<blocks_coal, threads_coal, 0, stream>>>( \
                    dense_row_id, dense_grads, dense_row_moments, master_weights, lr, eps, numUniqueRows \
                ); \
            } else { \
                rowAdagradOptimizer<N><<<blocks_uncoal, threads_uncoal, 0, stream>>>( \
                    dense_row_id, dense_grads, dense_row_moments, master_weights, lr, eps, numUniqueRows \
                ); \
            } \
            break;

    switch (num_features) {
        LAUNCH_OPT(1)  LAUNCH_OPT(2)  LAUNCH_OPT(3)  LAUNCH_OPT(4)  LAUNCH_OPT(5)
        LAUNCH_OPT(6)  LAUNCH_OPT(7)  LAUNCH_OPT(8)  LAUNCH_OPT(9)  LAUNCH_OPT(10)
        LAUNCH_OPT(11) LAUNCH_OPT(12) LAUNCH_OPT(13) LAUNCH_OPT(14) LAUNCH_OPT(15)
        LAUNCH_OPT(16) LAUNCH_OPT(17) LAUNCH_OPT(18) LAUNCH_OPT(19) LAUNCH_OPT(20)
        default:
            printf("Error: launchRowAdagradOptimizer unsupported num_features: %d\n", num_features);
            break;
    }

    #undef LAUNCH_OPT
}
