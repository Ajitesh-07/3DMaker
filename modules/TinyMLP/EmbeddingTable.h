#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            std::abort();                                                       \
        }                                                                       \
    } while (0)
#endif

enum OP_TYPE {
    ROW_ADAGRAD,
    GN_NORM
};

struct EmbeddingTableOption {
    uint32_t rows;
    int num_features;
    const float* lr;      // host pointer, one lr per feature; copied to device in the ctor
    float eps = 1e-8f;
    float priorS = 2.0f;  // warm-start S for GN_NORM (unused by ROW_ADAGRAD)
    OP_TYPE op_type = ROW_ADAGRAD;
};

class EmbeddingTable {
public:
    // batchSize = max (row, grad) pairs per tableStep call; all scratch is allocated
    // once here (tableStep itself never touches the allocator).
    EmbeddingTable(const EmbeddingTableOption& opt, int batchSize);
    ~EmbeddingTable();

    EmbeddingTable(const EmbeddingTable&) = delete;
    EmbeddingTable& operator=(const EmbeddingTable&) = delete;

    // One optimizer step: sort pairs by row, reduce duplicate rows to dense sums,
    // apply the row-wise update. d_rowid[i] pairs with d_grads[i*num_features ..].
    // Rows with id < 0 are padding and are skipped. numPairs < 0 means full batchSize.
    void tableStep(const int* d_rowid, const float* d_grads, int numPairs = -1,
                   cudaStream_t stream = 0);

    float*   masterWeights() { return d_masterWeights; }  // rows x num_features, fp32
    float*   rowMoments()    { return d_rowMeanMoment; }  // rows, accumulated S per row
    uint32_t rows()     const { return m_opt.rows; }
    int      features() const { return m_opt.num_features; }

private:
    EmbeddingTableOption m_opt;
    int m_batchSize;

    // parameters + optimizer state (rows-sized)
    float* d_masterWeights = nullptr;  // rows x F
    float* d_rowMeanMoment = nullptr;  // rows (the 4 B/row optimizer state)
    float* d_lr            = nullptr;  // F

    // per-step scratch (batchSize-sized, allocated once in the ctor)
    int*   d_pairIdx       = nullptr;  // 0..batchSize-1, filled once
    int*   d_rowidSorted   = nullptr;
    int*   d_indicesSorted = nullptr;  // sorted-order -> original pair index
    int*   d_uniqueRows    = nullptr;  // dense -> global row id
    int*   d_runLengths    = nullptr;  // pairs per unique row
    int*   d_runOffsets    = nullptr;  // exclusive scan of run lengths
    int*   d_numUnique     = nullptr;  // 1 int
    float* d_denseGrads    = nullptr;  // batchSize x F capacity, numUnique x F used
    void*  d_tempStorage   = nullptr;  // shared CUB scratch (max of sort/RLE/scan)
    size_t m_tempStorageBytes = 0;
    int*   h_numUnique     = nullptr;  // pinned readback
};

extern "C" void launchRowAdagradOptimizer(
    int num_features,
    const int* dense_row_id,
    const float* dense_grads,
    float* dense_row_moments,
    float* master_weights,
    const float* lr,
    float eps,
    int numUniqueRows,
    bool use_coalesced,
    cudaStream_t stream = 0
);
