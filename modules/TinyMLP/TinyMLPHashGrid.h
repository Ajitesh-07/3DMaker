#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "TinyMLP.h"

struct MLPGridOptions {
    int vectorDim; // the dimension before passing to the input grid, usually 3
    int hiddenDim;
    int outputDim;
    int numLayers;
    int activationType;

    int tableSize;
    int numLevels;
    float b;
    int lowestSize;
    int featuresLevel; // features per grid level
};

inline int inputDim(MLPGridOptions& opt) {
    return opt.featuresLevel * opt.numLevels;
} 

// ============================================================================
// TinyMLPHashGrid CLASS
// ============================================================================

class TinyMLPHashGrid {
public:
    explicit TinyMLPHashGrid(const MLPGridOptions& options, int maxBatchSize, int inferBatchSize = 0, unsigned int seed = 42);
    ~TinyMLPHashGrid();

    // Delete copy constructors to prevent accidental double-freeing of VRAM
    TinyMLPHashGrid(const TinyMLPHashGrid&) = delete;
    TinyMLPHashGrid& operator=(const TinyMLPHashGrid&) = delete;

    // 1. Clears gradients (MLP weights/biases AND hash grid)
    void zero_grad(cudaStream_t stream = 0);

    // 2. Training Forward Pass: Saves activations for backprop. Writes UNPADDED FP32 outputs to d_outputs.
    //    d_unpadded_inputs is [batchSize x 3] floats (user's raw 3D coordinates)
    void forward(const float* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream = 0);

    // 3. Loss Calculation: Calculates MSE, scales it, and populates internal gradients
    float calculate_loss_and_grad(const half* d_unpadded_targets, int batchSize, float loss_scale = 65536.0f, bool fetch_loss = false, cudaStream_t stream = 0);

    // 4. Backward Pass: Uses internal loss gradients to calculate weight/bias/hash gradients
    void backward(int batchSize, cudaStream_t stream = 0);
    void backward(const half* custom_loss_grad, int batchSize, cudaStream_t stream = 0);

    // 5. Optimizer Step: Fused Adam step for MLP weights, biases, AND hash grid
    void step(float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
              float epsilon = 1e-8f, float loss_scale = 65536.0f, cudaStream_t stream = 0);

    // 6. Reset Optimizer Step: Resets Adam moving average tracking step
    void reset_step();

    // --- Deployment API ---

    // Evaluation Forward: Bypasses activation saving. Writes UNPADDED FP32 outputs to d_outputs.
    void inference(const float* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream = 0);

    // Load/Save MLP Weights (hash grid NOT included — retrain or save separately)
    void loadWeights(const std::string& filename);
    void saveWeights(const std::string& filename);

    // In-memory overloads for testing infrastructure
    void loadWeights(const float* host_weights, const float* host_biases);
    void saveWeights(float* host_weights, float* host_biases);
    void loadHashgrid(const float* host_hashgrid);
    void saveHashgrid(float* host_hashgrid);

private:
    MLPGridOptions user_opt;    // The raw user-supplied options
    MLPGridOptions hw_opt;      // Options with padded outputDim for CUDA execution
    MLPOption      mlp_opt;     // The MLP-only options (inputDim=numLevels*featuresLevel, padded output)
    int m_maxBatchSize;
    int m_inferBatchSize;
    int current_step;

    // ==========================================
    // DATA BUFFERS
    // ==========================================

    float* d_padded_inputs;             // [maxBatch x 4] float inputs padded from 3->4
    const float* d_current_inputs_backward; // Pointer to inputs used in the last forward pass
    half* d_padded_targets;             // Padded ground truth for MSE
    float* d_padded_outputs_float;      // Padded raw network output

    // ==========================================
    // LOSS STATE
    // ==========================================

    half* d_dLoss_internal;             // FP16 gradients w.r.t network outputs
    float* d_total_loss;                // Device scalar to safely accumulate MSE loss

    // ==========================================
    // HASH GRID STATE
    // ==========================================

    float* d_master_hashtable;          // FP32 master hash table (for Adam)
    half* d_fwd_hashtable;              // FP16 hash table (used during forward)
    float* d_hashtable_grads;           // FP32 hash table gradients
    float* d_hash_m;                    // Adam 1st moment (hash grid)
    float* d_hash_v;                    // Adam 2nd moment (hash grid)

    int m_totalHashElements;            // numLevels * tableSize * featuresLevel

    // ==========================================
    // NETWORK STATE (Device Pointers)
    // ==========================================

    float** d_master_weights;
    float** d_master_biases;
    half** d_fwd_weights;

    half** d_activations;               // Array of pointers for intermediate outputs
    half* d_dx_out;                     // Gradients w.r.t inputs (required by backward)

    // ==========================================
    // OPTIMIZER STATE (Device Pointers)
    // ==========================================

    float** d_w_grad;
    float** d_b_grad;
    float** d_w_m;
    float** d_w_v;
    float** d_b_m;
    float** d_b_v;

    // ==========================================
    // PRIVATE HELPERS
    // ==========================================

    void allocate_memory();
    void free_pointer_array(float** d_arr, int count);
    void free_pointer_array(half** d_arr, int count, bool skip_first = false);

    inline int nextPowerOf2(int n) const {
        if (n <= 8) return 8;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }
};


// ============================================================================
// KERNEL LAUNCH DECLARATIONS
// ============================================================================

extern "C" void launchNetworkFusionHashTableKernel(
    MLPGridOptions*   opt,
    half*             d_hashtable,
    float*            d_inputs,
    half**            d_weights_array,
    float**           d_biases_array,
    float*            d_outputs,
    int               batchSize,
    cudaStream_t      stream
);

// Training forward pass: identical to inference but saves intermediate activations for backprop
extern "C" void launchNetworkFusionHashTableGradKernel(
    MLPGridOptions*   opt,
    half*             d_hashtable,
    float*            d_inputs,
    half**            d_activations,
    half**            d_weights_array,
    float**           d_biases_array,
    float*            d_outputs,
    int               batchSize,
    cudaStream_t      stream
);

extern "C" void launchNetworkFusionHashTableBackwardKernel(
    MLPGridOptions*  opt,
    half*        d_loss_output,
    half**       d_weights_array,
    float**      d_biases_array,
    half**       d_activations,
    float*       d_inputs,             
    float**      d_grad_weights,
    float**      d_grad_biases,
    float*       d_hashtable_grads,    
    half*        d_dx_out,
    int          batchSize,
    cudaStream_t stream
);

extern "C" void launchHashTableBackwardKernel(
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
);

extern "C" void launchAdamHashGridOptim(
    MLPGridOptions* opt,
    float*          master_hash,
    half*           fwd_hash,
    float*          gradients,
    float*          m,
    float*          v,
    float           lr,
    float           beta1,
    float           beta2,
    float           epsilon,
    float           bias_correction1,
    float           bias_correction2,
    float           inv_loss_scale,
    cudaStream_t    stream
);