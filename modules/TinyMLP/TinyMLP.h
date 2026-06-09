#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstdint>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define ACT_RELU 0

#define OUT_ACT_NONE    0
#define OUT_ACT_SIGMOID 1

struct MLPOption {
    int inputDim;
    int hiddenDim;
    int outputDim;
    int numLayers;
    int activationType;
    int outputActivation = OUT_ACT_NONE;  // Final layer activation (backward compatible)
};

class TinyMLP {
public:
    // Pass maxBatchSize here since we cannot modify MLPOption
    explicit TinyMLP(const MLPOption& options, int maxBatchSize, int inferBatchSize = 0, unsigned int seed = 42, bool isTraining = true);
    ~TinyMLP();

    // Delete copy constructors to prevent accidental double-freeing of VRAM
    TinyMLP(const TinyMLP&) = delete;
    TinyMLP& operator=(const TinyMLP&) = delete;
    
    // Switch between training and inference modes
    void switchToInferenceMode();
    void switchToTrainingMode();
    
    // 1. Clears gradients (mimics PyTorch zero_grad)
    void zero_grad(cudaStream_t stream = 0);

    // 2. Training Forward Pass: Saves activations for backprop. Writes UNPADDED FP32 outputs to d_outputs.
    void forward(const half* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream = 0);

    // 3. Loss Calculation: Calculates MSE, scales it, and populates internal gradients
    float calculate_loss_and_grad(const half* d_unpadded_targets, int batchSize, float loss_scale = 65536.0f, bool fetch_loss = false, cudaStream_t stream = 0);

    // 4. Backward Pass: Uses internal loss gradients to calculate weight/bias gradients
    void backward(int batchSize, cudaStream_t stream = 0);
    void backward(const half* custom_loss_grad, int batchSize, cudaStream_t stream = 0);

    // 4.1. Backward Pass with DX: Calculates gradients w.r.t input and saves to d_dx_out
    void backward(half* out_d_dx_out, int batchSize, cudaStream_t stream = 0);
    void backward(const half* custom_loss_grad, half* out_d_dx_out, int batchSize, cudaStream_t stream = 0);

    // 5. Optimizer Step: Fused Adam step for weights and biases
    void step(float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f, 
              float epsilon = 1e-8f, float loss_scale = 65536.0f, cudaStream_t stream = 0);

    // 6. Reset Optimizer Step: Resets Adam moving average tracking step
    void reset_step();

    // --- Deployment API ---
    
    // Evaluation Forward: Bypasses activation saving to save VRAM. Writes UNPADDED FP32 outputs to d_outputs.
    void inference(const half* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream = 0);

    // Load/Save Weights (Useful for saving training checkpoints)
    void loadWeights(const std::string& filename);
    void saveWeights(const std::string& filename);
    
    // In-memory overloads for testing infrastructure
    void loadWeights(const float* host_weights, const float* host_biases);
    void saveWeights(float* host_weights, float* host_biases);

private:
    MLPOption user_opt;     // The raw dimensions requested by the user
    MLPOption hw_opt;       // The power-of-2 padded dimensions for CUDA execution
    int m_maxBatchSize;     // Maximum batch size for buffer pre-allocation
    int m_inferBatchSize;   // Batch size for inference
    int current_step;       // Tracks Adam optimizer steps for bias correction
    bool m_isTraining;      // Tracks if we are in training mode

    cudaStream_t m_stream = nullptr; // Optional default stream

    // ==========================================
    // DATA BUFFERS (Padding & Formatting)
    // ==========================================
    
    half* d_padded_inputs = nullptr;          // Padded network input
    half* d_padded_targets = nullptr;         // Padded ground truth for MSE
    float* d_padded_outputs_float = nullptr;  // Padded raw network output

    // ==========================================
    // LOSS STATE
    // ==========================================
    
    half* d_dLoss_internal = nullptr;         // FP16 gradients w.r.t network outputs
    float* d_total_loss = nullptr;            // Device scalar to safely accumulate MSE loss

    // ==========================================
    // NETWORK STATE (Device Pointers)
    // ==========================================
    
    float** d_master_weights = nullptr;       // FP32 Master Weights (for precision Adam updates)
    float** d_master_biases = nullptr;        // FP32 Master Biases
    half** d_fwd_weights = nullptr;           // FP16 Casted Weights (used during forward pass)
    
    half** d_activations = nullptr;           // Array of pointers pointing to intermediate outputs

    // Optimizer State (Adam)
    half* d_dx_out = nullptr;                 // Gradients w.r.t inputs (required by backward kernel)
    
    // ==========================================
    // OPTIMIZER STATE (Device Pointers)
    // ==========================================
    
    float** d_w_grad = nullptr;               // FP32 Weight Gradients
    float** d_b_grad = nullptr;               // FP32 Bias Gradients
    float** d_w_m = nullptr;                  // Adam 1st Moment (Weights)
    float** d_w_v = nullptr;                  // Adam 2nd Moment (Weights)
    float** d_b_m = nullptr;                  // Adam 1st Moment (Biases)
    float** d_b_v = nullptr;                  // Adam 2nd Moment (Biases)

    // ==========================================
    // PRIVATE HELPERS
    // ==========================================
    
    void allocate_memory();
    void free_pointer_array(float** d_arr, int count);
    void free_pointer_array(half** d_arr, int count, bool skip_first = false);
    
    // Helper to calculate the next power of 2 (clamped to 8 minimum)
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


// legacy kernel dont use it
extern "C" void launchForwardKernel(
    MLPOption*   opt,
    half*        d_inputs,
    half**       d_weights_array,
    float**      d_biases_array,
    float*       d_outputs,
    int          batchSize,
    half*        d_ping,
    half*        d_pong,
    cudaStream_t stream
);


// The inference kernel dosent save activations in between for pure inference
extern "C" void launchNetworkFusionKernel(
    MLPOption*   opt,
    half*        d_inputs,
    half**       d_weights_array,
    float**      d_biases_array,
    float*       d_outputs,
    int          batchSize,
    cudaStream_t stream
);

// the backward kernel for calculating all d_weight_grad and d_bias_grad in one single pass
extern "C" void launchNetworkFusionBackwardKernel(
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
);

// the forward grad kernel which does the full fused forward pass and saves the activation weights as well
extern "C" void launchNetworkFusionGradKernel(
    MLPOption*   opt,
    half*        d_inputs,
    half**       d_activations,
    half**       d_weights_array,
    float**      d_biases_array,
    float*       d_outputs,
    int          batchSize,
    cudaStream_t stream
);

// the adam optimizer kernel for updating the weights only
extern "C" void launchAdamWeightsOptim(
    MLPOption*   opt,
    float**      master_weights,
    half**       fwd_weights,
    float**      gradients,
    float**      m,
    float**      v,
    float        lr,
    float        beta1,
    float        beta2,
    float        epsilon,
    float        bias_correction1,
    float        bias_correction2,
    float        inv_loss_scale,
    cudaStream_t stream
);

// the adam optimizer kernel for updating the bias only
extern "C" void launchAdamBiasOptim(
    MLPOption*   opt,
    float**      biases,
    float**      gradients,
    float**      m,
    float**      v,
    float        lr,
    float        beta1,
    float        beta2,
    float        epsilon,
    float        bias_correction1,
    float        bias_correction2,
    float        inv_loss_scale,
    cudaStream_t stream
);

// the mse grad kernel which calculates the mse loss value as well as the dLoss_out
extern "C" void launchMSELossGrad(
    const float* predictions,
    const half*  targets,
    half*        dLoss_out,
    float*       total_loss,
    int          batchSize,
    int          paddedDim,
    int          validDim,
    float        loss_scale,
    cudaStream_t stream,
    int          outputActivation = OUT_ACT_NONE
);

// kernel for zeroing out the gradients
extern "C" void launchZeroGradients(
    MLPOption*   opt,
    float**      w_grads,
    float**      b_grads,
    cudaStream_t stream
);

// extern "C" void padInputKernel(
//     const half* __restrict__ unpadded,
//     half* __restrict__       padded,                
//     int                      batchSize, 
//     int                      unpadded_dim,
//     int                      padded_dim
// );

// extern "C" void unpadOutputKernel(
//     const float* __restrict__ padded, 
//     float* __restrict__       unpadded, 
//     int                       batchSize, 
//     int                       padded_dim, 
//     int                       unpadded_dim
// );

// extern "C" void floatToHalfKernel(
//     const float* __restrict__ f, 
//     half* __restrict__        h, 
//     int                       n
// );


template<typename InType, typename OutType> __device__ __forceinline__ OutType cast_val(InType val);