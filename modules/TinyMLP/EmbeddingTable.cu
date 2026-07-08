#include "EmbeddingTable.h"
#include <cub/cub.cuh>
#include <algorithm>

__global__ void fillSequential(int* d_buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_buffer[idx] = idx;
}

template <int NUM_FEATURES>
__global__ void reduceGradsByRun(
    const int*   __restrict__ d_runOffsets,
    const int*   __restrict__ d_runLengths,
    const int*   __restrict__ d_indices,
    const float* __restrict__ d_grads_in,
    float*       __restrict__ d_grads_out,
    int numUniqueRows)
{
    int row  = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= numUniqueRows) return;

    int begin = d_runOffsets[row];
    int len   = d_runLengths[row];

    float acc[NUM_FEATURES];
    #pragma unroll
    for (int f = 0; f < NUM_FEATURES; f++) acc[f] = 0.0f;

    for (int i = lane; i < len; i += 32) {
        const float* g = d_grads_in + (size_t)d_indices[begin + i] * NUM_FEATURES;
        #pragma unroll
        for (int f = 0; f < NUM_FEATURES; f++) acc[f] += g[f];
    }

    #pragma unroll
    for (int f = 0; f < NUM_FEATURES; f++) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc[f] += __shfl_xor_sync(0xFFFFFFFF, acc[f], o);
    }

    if (lane == 0) {
        #pragma unroll
        for (int f = 0; f < NUM_FEATURES; f++)
            d_grads_out[(size_t)row * NUM_FEATURES + f] = acc[f];
    }
}

static void launchReduceGradsByRun(
    int num_features,
    const int* d_runOffsets,
    const int* d_runLengths,
    const int* d_indices,
    const float* d_grads_in,
    float* d_grads_out,
    int numUniqueRows,
    cudaStream_t stream)
{
    if (numUniqueRows == 0) return;

    dim3 threads(32, 8);
    dim3 blocks((numUniqueRows + threads.y - 1) / threads.y);
    #define LAUNCH_REDUCE(N) \
        case N: \
            reduceGradsByRun<N><<<blocks, threads, 0, stream>>>( \
                d_runOffsets, d_runLengths, d_indices, d_grads_in, d_grads_out, numUniqueRows \
            ); \
            break;

    switch (num_features) {
        LAUNCH_REDUCE(1)  LAUNCH_REDUCE(2)  LAUNCH_REDUCE(3)  LAUNCH_REDUCE(4)  LAUNCH_REDUCE(5)
        LAUNCH_REDUCE(6)  LAUNCH_REDUCE(7)  LAUNCH_REDUCE(8)  LAUNCH_REDUCE(9)  LAUNCH_REDUCE(10)
        LAUNCH_REDUCE(11) LAUNCH_REDUCE(12) LAUNCH_REDUCE(13) LAUNCH_REDUCE(14) LAUNCH_REDUCE(15)
        LAUNCH_REDUCE(16) LAUNCH_REDUCE(17) LAUNCH_REDUCE(18) LAUNCH_REDUCE(19) LAUNCH_REDUCE(20)
        default:
            printf("Error: launchReduceGradsByRun unsupported num_features: %d\n", num_features);
            break;
    }

    #undef LAUNCH_REDUCE
}

EmbeddingTable::EmbeddingTable(const EmbeddingTableOption& opt, int batchSize) {
    m_opt = opt;
    m_batchSize = batchSize;

    size_t rows = opt.rows;
    size_t F    = opt.num_features;

    CUDA_CHECK(cudaMalloc(&d_masterWeights, sizeof(float) * rows * F));
    CUDA_CHECK(cudaMalloc(&d_rowMeanMoment, sizeof(float) * rows));
    CUDA_CHECK(cudaMalloc(&d_lr,            sizeof(float) * F));
    CUDA_CHECK(cudaMemset(d_masterWeights, 0, sizeof(float) * rows * F));
    CUDA_CHECK(cudaMemset(d_rowMeanMoment, m_opt.priorS, sizeof(float) * rows));
    CUDA_CHECK(cudaMemcpy(d_lr, opt.lr, sizeof(float) * F, cudaMemcpyHostToDevice));
    m_opt.lr = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pairIdx,       sizeof(int) * batchSize));
    CUDA_CHECK(cudaMalloc(&d_rowidSorted,   sizeof(int) * batchSize));
    CUDA_CHECK(cudaMalloc(&d_indicesSorted, sizeof(int) * batchSize));
    CUDA_CHECK(cudaMalloc(&d_uniqueRows,    sizeof(int) * batchSize));
    CUDA_CHECK(cudaMalloc(&d_runLengths,    sizeof(int) * batchSize));
    CUDA_CHECK(cudaMalloc(&d_runOffsets,    sizeof(int) * batchSize));
    CUDA_CHECK(cudaMalloc(&d_numUnique,     sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_denseGrads,    sizeof(float) * batchSize * F));
    CUDA_CHECK(cudaMallocHost(&h_numUnique, sizeof(int)));

    constexpr int BS = 256;
    fillSequential<<<(batchSize + BS - 1) / BS, BS>>>(d_pairIdx, batchSize);
    CUDA_CHECK(cudaGetLastError());

    size_t sortBytes = 0, rleBytes = 0, scanBytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes,
        (const int*)d_rowidSorted, d_rowidSorted, (const int*)d_pairIdx, d_indicesSorted, batchSize));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr, rleBytes,
        (const int*)d_rowidSorted, d_uniqueRows, d_runLengths, d_numUnique, batchSize));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes,
        (const int*)d_runLengths, d_runOffsets, batchSize));
    m_tempStorageBytes = std::max(sortBytes, std::max(rleBytes, scanBytes));
    CUDA_CHECK(cudaMalloc(&d_tempStorage, m_tempStorageBytes));

    CUDA_CHECK(cudaDeviceSynchronize());
}

EmbeddingTable::~EmbeddingTable() {
    CUDA_CHECK(cudaFree(d_masterWeights));
    CUDA_CHECK(cudaFree(d_rowMeanMoment));
    CUDA_CHECK(cudaFree(d_lr));
    CUDA_CHECK(cudaFree(d_pairIdx));
    CUDA_CHECK(cudaFree(d_rowidSorted));
    CUDA_CHECK(cudaFree(d_indicesSorted));
    CUDA_CHECK(cudaFree(d_uniqueRows));
    CUDA_CHECK(cudaFree(d_runLengths));
    CUDA_CHECK(cudaFree(d_runOffsets));
    CUDA_CHECK(cudaFree(d_numUnique));
    CUDA_CHECK(cudaFree(d_denseGrads));
    CUDA_CHECK(cudaFree(d_tempStorage));
    CUDA_CHECK(cudaFreeHost(h_numUnique));
}

void EmbeddingTable::tableStep(const int* d_rowid, const float* d_grads, int numPairs,
                               cudaStream_t stream) {
    int n = (numPairs < 0) ? m_batchSize : numPairs;
    if (n == 0) return;
    if (n > m_batchSize) {
        fprintf(stderr, "EmbeddingTable::tableStep: numPairs %d > batchSize %d\n", n, m_batchSize);
        std::abort();
    }

    size_t tempBytes = m_tempStorageBytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_tempStorage, tempBytes,
        d_rowid, d_rowidSorted, (const int*)d_pairIdx, d_indicesSorted, n, 0, 32, stream));

    tempBytes = m_tempStorageBytes;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_tempStorage, tempBytes,
        (const int*)d_rowidSorted, d_uniqueRows, d_runLengths, d_numUnique, n, stream));

    CUDA_CHECK(cudaMemcpyAsync(h_numUnique, d_numUnique, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int numUniqueRows = *h_numUnique;
    if (numUniqueRows == 0) return;

    tempBytes = m_tempStorageBytes;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_tempStorage, tempBytes,
        (const int*)d_runLengths, d_runOffsets, numUniqueRows, stream));

    launchReduceGradsByRun(m_opt.num_features, d_runOffsets, d_runLengths, d_indicesSorted,
                           d_grads, d_denseGrads, numUniqueRows, stream);

    if (m_opt.op_type == ROW_ADAGRAD) {
        launchRowAdagradOptimizer(
            m_opt.num_features,
            d_uniqueRows,
            d_denseGrads,
            d_rowMeanMoment,
            d_masterWeights,
            d_lr,
            m_opt.eps,
            numUniqueRows,
            true,
            stream
        );
    } else {
        // TODO: GN_NORM — S_r += sum(w*t^2) needs a per-pair w*t^2 input alongside
        fprintf(stderr, "EmbeddingTable::tableStep: GN_NORM not implemented yet\n");
    }
}
