#include "TinyMLP.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>

template<> __device__ __forceinline__ half cast_val<float, half>(float val) { return __float2half(val); }
template<> __device__ __forceinline__ half cast_val<half, half>(half val)   { return val; }
template<> __device__ __forceinline__ float cast_val<float, float>(float val) { return val; }
template<> __device__ __forceinline__ float cast_val<half, float>(half val)   { return __half2float(val); }

// 2. Zero Initialization Helpers
template<typename T>
__device__ __forceinline__ T zero_val();

template<> __device__ __forceinline__ float zero_val<float>() { return 0.0f; }
template<> __device__ __forceinline__ half zero_val<half>() { return __float2half(0.0f); }

// 3. Fused Pad & Cast Kernel
template <typename T_IN, typename T_OUT>
__global__ void fusedPadAndCastKernel(
    const T_IN* __restrict__ unpadded, 
    T_OUT* __restrict__ padded, 
    int batchSize, 
    int unpadded_dim, 
    int padded_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batchSize && col < padded_dim) {
        int padded_idx = row * padded_dim + col;
        
        if (col < unpadded_dim) {
            // Fetch and cast in a single instruction pipeline
            padded[padded_idx] = cast_val<T_IN, T_OUT>(unpadded[row * unpadded_dim + col]);
        } else {
            // Write type-safe zero
            padded[padded_idx] = zero_val<T_OUT>();
        }
    }
}

// 4. Fused Unpad & Cast Kernel
template <typename T_IN, typename T_OUT>
__global__ void fusedUnpadAndCastKernel(
    const T_IN* __restrict__ padded, 
    T_OUT* __restrict__ unpadded, 
    int batchSize, 
    int padded_dim, 
    int unpadded_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batchSize && col < unpadded_dim) {
        unpadded[row * unpadded_dim + col] = cast_val<T_IN, T_OUT>(padded[row * padded_dim + col]);
    }
}

__global__ void floatToHalfKernel(const float* __restrict__ f, half* __restrict__ h, int n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        h[idx] = __float2half(f[idx]);
    }
}

// ============================================================================
// CLASS IMPLEMENTATION
// ============================================================================

TinyMLP::TinyMLP(const MLPOption& options, int maxBatchSize, int inferBatchSize, unsigned int seed, bool isTraining) 
    : user_opt(options), m_maxBatchSize(maxBatchSize), m_inferBatchSize(inferBatchSize <= 0 ? maxBatchSize : inferBatchSize), current_step(0), m_isTraining(isTraining) {
    
    // Calculate Hardware Configuration (Power of 2 padding)
    hw_opt = options;
    hw_opt.inputDim = nextPowerOf2(user_opt.inputDim);
    hw_opt.hiddenDim = nextPowerOf2(user_opt.hiddenDim);
    hw_opt.outputDim = nextPowerOf2(user_opt.outputDim);

    allocate_memory();

    // Initialize Network Parameters (Kaiming Uniform)
    std::mt19937 gen(seed);
    std::vector<float*> host_master_w(hw_opt.numLayers);
    std::vector<float*> host_master_b(hw_opt.numLayers);
    std::vector<half*>  host_fwd_w(hw_opt.numLayers);

    for (int l = 0; l < hw_opt.numLayers; l++) {
        int in_d = (l == 0) ? hw_opt.inputDim : hw_opt.hiddenDim;
        int out_d = (l == hw_opt.numLayers - 1) ? hw_opt.outputDim : hw_opt.hiddenDim;
        
        int w_elements = in_d * out_d;
        int b_elements = out_d;

        float bound = 1.0f / std::sqrt((float)in_d);
        std::uniform_real_distribution<float> dist(-bound, bound);

        // Host buffers
        std::vector<float> h_w(w_elements);
        std::vector<float> h_b(b_elements, 0.0f); // Biases start at 0
        for (int i = 0; i < w_elements; i++) h_w[i] = dist(gen);

        // Device allocation and copy
        float *d_w, *d_b;
        half  *d_w_half;
        CUDA_CHECK(cudaMalloc(&d_w, w_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, b_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w_half, w_elements * sizeof(half)));

        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), w_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_elements * sizeof(float), cudaMemcpyHostToDevice));

        // Initial float2half cast for forward pass
        int blocks = (w_elements + 255) / 256;
        floatToHalfKernel<<<blocks, 256>>>(d_w, d_w_half, w_elements);

        host_master_w[l] = d_w;
        host_master_b[l] = d_b;
        host_fwd_w[l] = d_w_half;
    }

    // Copy array of pointers to device
    CUDA_CHECK(cudaMemcpy(d_master_weights, host_master_w.data(), hw_opt.numLayers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_master_biases, host_master_b.data(), hw_opt.numLayers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fwd_weights, host_fwd_w.data(), hw_opt.numLayers * sizeof(half*), cudaMemcpyHostToDevice));
}

TinyMLP::~TinyMLP() {
    // 1. Free contiguous buffers
    cudaFree(d_padded_inputs);
    cudaFree(d_padded_targets);
    cudaFree(d_padded_outputs_float);
    cudaFree(d_dLoss_internal);
    cudaFree(d_total_loss);
    cudaFree(d_dx_out);

    // 2. Free pointer arrays
    free_pointer_array(d_master_weights, hw_opt.numLayers);
    free_pointer_array(d_master_biases, hw_opt.numLayers);
    free_pointer_array(d_fwd_weights, hw_opt.numLayers);
    
    free_pointer_array(d_w_grad, hw_opt.numLayers);
    free_pointer_array(d_b_grad, hw_opt.numLayers);
    free_pointer_array(d_w_m, hw_opt.numLayers);
    free_pointer_array(d_w_v, hw_opt.numLayers);
    free_pointer_array(d_b_m, hw_opt.numLayers);
    free_pointer_array(d_b_v, hw_opt.numLayers);

    // 3. Free activations (skip index 0, as it points to d_padded_inputs)
    free_pointer_array(d_activations, hw_opt.numLayers, true);
}

void TinyMLP::allocate_memory() {
    int max_pad = std::max(m_maxBatchSize, m_inferBatchSize);
    bool needs_in_pad = (user_opt.inputDim != hw_opt.inputDim);
    bool needs_out_pad = (user_opt.outputDim != hw_opt.outputDim);

    if (m_isTraining || needs_in_pad) {
        if (!d_padded_inputs) CUDA_CHECK(cudaMalloc(&d_padded_inputs, max_pad * hw_opt.inputDim * sizeof(half)));
    }
    if (m_isTraining) {
        if (!d_padded_targets) CUDA_CHECK(cudaMalloc(&d_padded_targets, m_maxBatchSize * hw_opt.outputDim * sizeof(half)));
    }
    if (m_isTraining || needs_out_pad) {
        if (!d_padded_outputs_float) CUDA_CHECK(cudaMalloc(&d_padded_outputs_float, max_pad * hw_opt.outputDim * sizeof(float)));
    }
    
    if (m_isTraining) {
        if (!d_dLoss_internal) CUDA_CHECK(cudaMalloc(&d_dLoss_internal, m_maxBatchSize * hw_opt.outputDim * sizeof(half)));
        if (!d_total_loss) CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(float)));
        if (!d_dx_out) CUDA_CHECK(cudaMalloc(&d_dx_out, m_maxBatchSize * hw_opt.inputDim * sizeof(half)));
    }

    // Helper lambda to create zeroed out state arrays
    auto createZeroStateArray = [&](bool isWeight) {
        std::vector<float*> host_arr(hw_opt.numLayers);
        for (int l = 0; l < hw_opt.numLayers; l++) {
            int in_d = (l == 0) ? hw_opt.inputDim : hw_opt.hiddenDim;
            int out_d = (l == hw_opt.numLayers - 1) ? hw_opt.outputDim : hw_opt.hiddenDim;
            int elems = isWeight ? (in_d * out_d) : out_d;
            
            float* d_arr;
            CUDA_CHECK(cudaMalloc(&d_arr, elems * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_arr, 0, elems * sizeof(float)));
            host_arr[l] = d_arr;
        }
        float** d_arr_dev;
        CUDA_CHECK(cudaMalloc(&d_arr_dev, hw_opt.numLayers * sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(d_arr_dev, host_arr.data(), hw_opt.numLayers * sizeof(float*), cudaMemcpyHostToDevice));
        return d_arr_dev;
    };

    // Allocate Array Pointers
    if (!d_master_weights) CUDA_CHECK(cudaMalloc(&d_master_weights, hw_opt.numLayers * sizeof(float*)));
    if (!d_master_biases) CUDA_CHECK(cudaMalloc(&d_master_biases, hw_opt.numLayers * sizeof(float*)));
    if (!d_fwd_weights) CUDA_CHECK(cudaMalloc(&d_fwd_weights, hw_opt.numLayers * sizeof(half*)));

    if (m_isTraining) {
        if (!d_w_grad) d_w_grad = createZeroStateArray(true);
        if (!d_b_grad) d_b_grad = createZeroStateArray(false);
        if (!d_w_m)    d_w_m    = createZeroStateArray(true);
        if (!d_b_m)    d_b_m    = createZeroStateArray(false);
        if (!d_w_v)    d_w_v    = createZeroStateArray(true);
        if (!d_b_v)    d_b_v    = createZeroStateArray(false);

        if (!d_activations) {
            std::vector<half*> host_acts(hw_opt.numLayers);
            host_acts[0] = d_padded_inputs; // Index 0 strictly maps to padded inputs
            for (int l = 1; l < hw_opt.numLayers; l++) {
                half* d_arr;
                CUDA_CHECK(cudaMalloc(&d_arr, m_maxBatchSize * hw_opt.hiddenDim * sizeof(half)));
                host_acts[l] = d_arr;
            }
            CUDA_CHECK(cudaMalloc(&d_activations, hw_opt.numLayers * sizeof(half*)));
            CUDA_CHECK(cudaMemcpy(d_activations, host_acts.data(), hw_opt.numLayers * sizeof(half*), cudaMemcpyHostToDevice));
        }
    }
}

void TinyMLP::switchToInferenceMode() {
    if (!m_isTraining) return;

    if (user_opt.inputDim == hw_opt.inputDim && d_padded_inputs) {
        cudaFree(d_padded_inputs); d_padded_inputs = nullptr;
    }
    if (d_padded_targets) { cudaFree(d_padded_targets); d_padded_targets = nullptr; }
    if (user_opt.outputDim == hw_opt.outputDim && d_padded_outputs_float) {
        cudaFree(d_padded_outputs_float); d_padded_outputs_float = nullptr;
    }
    if (d_dLoss_internal) { cudaFree(d_dLoss_internal); d_dLoss_internal = nullptr; }
    if (d_total_loss) { cudaFree(d_total_loss); d_total_loss = nullptr; }
    if (d_dx_out) { cudaFree(d_dx_out); d_dx_out = nullptr; }

    free_pointer_array(d_w_grad, hw_opt.numLayers); d_w_grad = nullptr;
    free_pointer_array(d_b_grad, hw_opt.numLayers); d_b_grad = nullptr;
    free_pointer_array(d_w_m, hw_opt.numLayers); d_w_m = nullptr;
    free_pointer_array(d_w_v, hw_opt.numLayers); d_w_v = nullptr;
    free_pointer_array(d_b_m, hw_opt.numLayers); d_b_m = nullptr;
    free_pointer_array(d_b_v, hw_opt.numLayers); d_b_v = nullptr;

    free_pointer_array(d_activations, hw_opt.numLayers, true); d_activations = nullptr;

    m_isTraining = false;
}

void TinyMLP::switchToTrainingMode() {
    if (m_isTraining) return;
    m_isTraining = true;
    allocate_memory();
}

void TinyMLP::zero_grad(cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("zero_grad called while in inference mode!");
    launchZeroGradients(&hw_opt, d_w_grad, d_b_grad, stream);
}

void TinyMLP::forward(const half* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("forward called while in inference mode!");
    if (batchSize > m_maxBatchSize) throw std::runtime_error("Batch size exceeds maxBatchSize");

    if (user_opt.inputDim != hw_opt.inputDim) {
        dim3 block(16, 16);
        dim3 grid((batchSize + 15) / 16, (hw_opt.inputDim + 15) / 16);
        fusedPadAndCastKernel<half, half><<<grid, block, 0, stream>>>(d_unpadded_inputs, d_padded_inputs, batchSize, user_opt.inputDim, hw_opt.inputDim);
    } else {
        // We must snapshot the input for the backward pass since we own d_activations[0]
        CUDA_CHECK(cudaMemcpyAsync(d_padded_inputs, d_unpadded_inputs, batchSize * hw_opt.inputDim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    }

    launchNetworkFusionGradKernel(&hw_opt, d_padded_inputs, d_activations, d_fwd_weights, d_master_biases, d_padded_outputs_float, batchSize, stream);

    if (d_outputs != nullptr) {
        if (user_opt.outputDim == hw_opt.outputDim) {
            CUDA_CHECK(cudaMemcpyAsync(d_outputs, d_padded_outputs_float, batchSize * hw_opt.outputDim * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        } else {
            dim3 block(16, 16);
            dim3 out_grid((batchSize + 15) / 16, (user_opt.outputDim + 15) / 16);
            fusedUnpadAndCastKernel<float, float><<<out_grid, block, 0, stream>>>(d_padded_outputs_float, d_outputs, batchSize, hw_opt.outputDim, user_opt.outputDim);
        }
    }
}

void TinyMLP::inference(const half* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream) {
    if (batchSize > m_inferBatchSize) {
        for (int offset = 0; offset < batchSize; offset += m_inferBatchSize) {
            int current_batch = std::min(m_inferBatchSize, batchSize - offset);
            const half* current_in = d_unpadded_inputs + offset * user_opt.inputDim;
            float* current_out = (d_outputs != nullptr) ? (d_outputs + offset * user_opt.outputDim) : nullptr;
            inference(current_in, current_out, current_batch, stream);
        }
        return;
    }

    const half* final_inputs = d_unpadded_inputs;
    if (user_opt.inputDim != hw_opt.inputDim) {
        dim3 block(16, 16);
        dim3 grid((batchSize + 15) / 16, (hw_opt.inputDim + 15) / 16);
        fusedPadAndCastKernel<half, half><<<grid, block, 0, stream>>>(d_unpadded_inputs, d_padded_inputs, batchSize, user_opt.inputDim, hw_opt.inputDim);
        final_inputs = d_padded_inputs;
    }

    if (user_opt.outputDim == hw_opt.outputDim) {
        launchNetworkFusionKernel(&hw_opt, (half*)final_inputs, d_fwd_weights, d_master_biases, d_outputs, batchSize, stream);
    } else {
        launchNetworkFusionKernel(&hw_opt, (half*)final_inputs, d_fwd_weights, d_master_biases, d_padded_outputs_float, batchSize, stream);
        
        if (d_outputs != nullptr) {
            dim3 block(16, 16);
            dim3 out_grid((batchSize + 15) / 16, (user_opt.outputDim + 15) / 16);
            fusedUnpadAndCastKernel<float, float><<<out_grid, block, 0, stream>>>(d_padded_outputs_float, d_outputs, batchSize, hw_opt.outputDim, user_opt.outputDim);
        }
    }
}

// Update your TinyMLP.h definition to include: bool fetch_loss = false
float TinyMLP::calculate_loss_and_grad(const half* d_unpadded_targets, int batchSize, float loss_scale, bool fetch_loss, cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("calculate_loss_and_grad called while in inference mode!");
    // 1. Zero Loss Scalar
    CUDA_CHECK(cudaMemsetAsync(d_total_loss, 0, sizeof(float), stream));

    dim3 block(16, 16);
    dim3 grid((batchSize + 15) / 16, (hw_opt.outputDim + 15) / 16);

    // 2. Pad Targets (User Dim -> HW Dim) if necessary
    const half* final_targets = d_unpadded_targets;
    if (user_opt.outputDim != hw_opt.outputDim) {
        fusedPadAndCastKernel<half, half><<<grid, block, 0, stream>>>(
            d_unpadded_targets, d_padded_targets, batchSize, user_opt.outputDim, hw_opt.outputDim
        );
        final_targets = d_padded_targets;
    }

    // 3. Launch MSE using proper hw_opt layout directly on FP32 outputs
    launchMSELossGrad(d_padded_outputs_float, final_targets, d_dLoss_internal, d_total_loss, batchSize, hw_opt.outputDim, user_opt.outputDim, loss_scale, stream, hw_opt.outputActivation);

    // 4. Fetch scalar loss ONLY if requested
    if (fetch_loss) {
        float h_loss;
        // This blocks the CPU until the stream finishes!
        CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        return h_loss / (batchSize * user_opt.outputDim); 
    }

    // Return 0 if we didn't sync, keeping the GPU screaming fast
    return 0.0f; 
}

void TinyMLP::backward(int batchSize, cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("backward called while in inference mode!");
    launchNetworkFusionBackwardKernel(&hw_opt, d_dLoss_internal, d_fwd_weights, d_master_biases, d_activations, d_w_grad, d_b_grad, nullptr, batchSize, stream);
}

void TinyMLP::backward(const half* custom_loss_grad, int batchSize, cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("backward called while in inference mode!");
    if (user_opt.outputDim != hw_opt.outputDim) {
        dim3 block(16, 16);
        dim3 grid((batchSize + 15) / 16, (hw_opt.outputDim + 15) / 16);
        fusedPadAndCastKernel<half, half><<<grid, block, 0, stream>>>(custom_loss_grad, d_dLoss_internal, batchSize, user_opt.outputDim, hw_opt.outputDim);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_dLoss_internal, custom_loss_grad, batchSize * hw_opt.outputDim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    }
    launchNetworkFusionBackwardKernel(&hw_opt, d_dLoss_internal, d_fwd_weights, d_master_biases, d_activations, d_w_grad, d_b_grad, nullptr, batchSize, stream);
}

void TinyMLP::backward(half* user_d_dx_out, int batchSize, cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("backward called while in inference mode!");
    launchNetworkFusionBackwardKernel(&hw_opt, d_dLoss_internal, d_fwd_weights, d_master_biases, d_activations, d_w_grad, d_b_grad, user_d_dx_out, batchSize, stream);
}

void TinyMLP::backward(const half* custom_loss_grad, half* user_d_dx_out, int batchSize, cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("backward called while in inference mode!");
    if (user_opt.outputDim != hw_opt.outputDim) {
        dim3 block(16, 16);
        dim3 grid((batchSize + 15) / 16, (hw_opt.outputDim + 15) / 16);
        fusedPadAndCastKernel<half, half><<<grid, block, 0, stream>>>(custom_loss_grad, d_dLoss_internal, batchSize, user_opt.outputDim, hw_opt.outputDim);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_dLoss_internal, custom_loss_grad, batchSize * hw_opt.outputDim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    }
    launchNetworkFusionBackwardKernel(&hw_opt, d_dLoss_internal, d_fwd_weights, d_master_biases, d_activations, d_w_grad, d_b_grad, user_d_dx_out, batchSize, stream);
}

void TinyMLP::reset_step() {
    current_step = 0;
}

void TinyMLP::step(float lr, float beta1, float beta2, float epsilon, float loss_scale, cudaStream_t stream) {
    if (!m_isTraining) throw std::runtime_error("step called while in inference mode!");
    if (current_step < 50000) {
        current_step++;
    }
    
    float bias_correction1 = 1.0f - std::pow(beta1, (float)current_step);
    float bias_correction2 = 1.0f - std::pow(beta2, (float)current_step);
    float inv_loss_scale = 1.0f / loss_scale;

    launchAdamWeightsOptim(&hw_opt, d_master_weights, d_fwd_weights, d_w_grad, d_w_m, d_w_v, lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale, stream);
    launchAdamBiasOptim(&hw_opt, d_master_biases, d_b_grad, d_b_m, d_b_v, lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale, stream);
}

void TinyMLP::loadWeights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file for loading weights: " + filename);

    MLPOption file_opt;
    in.read(reinterpret_cast<char*>(&file_opt), sizeof(MLPOption));

    // Verify dimensions
    if (file_opt.inputDim != hw_opt.inputDim ||
        file_opt.hiddenDim != hw_opt.hiddenDim ||
        file_opt.outputDim != hw_opt.outputDim ||
        file_opt.numLayers != hw_opt.numLayers) {
        throw std::runtime_error("Weight file dimension mismatch!");
    }

    std::vector<float*> h_master_w(hw_opt.numLayers);
    std::vector<float*> h_master_b(hw_opt.numLayers);
    std::vector<half*>  h_fwd_w(hw_opt.numLayers);
    
    cudaMemcpy(h_master_w.data(), d_master_weights, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_master_b.data(), d_master_biases, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fwd_w.data(), d_fwd_weights, hw_opt.numLayers * sizeof(half*), cudaMemcpyDeviceToHost);

    for (int l = 0; l < hw_opt.numLayers; l++) {
        int in_d = (l == 0) ? hw_opt.inputDim : hw_opt.hiddenDim;
        int out_d = (l == hw_opt.numLayers - 1) ? hw_opt.outputDim : hw_opt.hiddenDim;
        
        int w_elements = in_d * out_d;
        int b_elements = out_d;

        std::vector<float> w_buf(w_elements);
        std::vector<float> b_buf(b_elements);

        in.read(reinterpret_cast<char*>(w_buf.data()), w_elements * sizeof(float));
        in.read(reinterpret_cast<char*>(b_buf.data()), b_elements * sizeof(float));

        CUDA_CHECK(cudaMemcpy(h_master_w[l], w_buf.data(), w_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_master_b[l], b_buf.data(), b_elements * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (w_elements + 255) / 256;
        floatToHalfKernel<<<blocks, 256>>>(h_master_w[l], h_fwd_w[l], w_elements);
    }
    cudaDeviceSynchronize();
    in.close();
}

void TinyMLP::saveWeights(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open file for saving weights: " + filename);

    out.write(reinterpret_cast<const char*>(&hw_opt), sizeof(MLPOption));

    std::vector<float*> h_master_w(hw_opt.numLayers);
    std::vector<float*> h_master_b(hw_opt.numLayers);
    
    cudaMemcpy(h_master_w.data(), d_master_weights, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_master_b.data(), d_master_biases, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);

    for (int l = 0; l < hw_opt.numLayers; l++) {
        int in_d = (l == 0) ? hw_opt.inputDim : hw_opt.hiddenDim;
        int out_d = (l == hw_opt.numLayers - 1) ? hw_opt.outputDim : hw_opt.hiddenDim;
        
        int w_elements = in_d * out_d;
        int b_elements = out_d;

        std::vector<float> w_buf(w_elements);
        std::vector<float> b_buf(b_elements);

        cudaMemcpy(w_buf.data(), h_master_w[l], w_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(b_buf.data(), h_master_b[l], b_elements * sizeof(float), cudaMemcpyDeviceToHost);

        out.write(reinterpret_cast<const char*>(w_buf.data()), w_elements * sizeof(float));
        out.write(reinterpret_cast<const char*>(b_buf.data()), b_elements * sizeof(float));
    }
    out.close();
}

void TinyMLP::loadWeights(const float* host_weights, const float* host_biases) {
    int w_offset = 0;
    int b_offset = 0;

    std::vector<float*> h_master_w(hw_opt.numLayers);
    std::vector<float*> h_master_b(hw_opt.numLayers);
    std::vector<half*>  h_fwd_w(hw_opt.numLayers);
    
    cudaMemcpy(h_master_w.data(), d_master_weights, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_master_b.data(), d_master_biases, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fwd_w.data(), d_fwd_weights, hw_opt.numLayers * sizeof(half*), cudaMemcpyDeviceToHost);

    for (int l = 0; l < hw_opt.numLayers; l++) {
        int in_d = (l == 0) ? hw_opt.inputDim : hw_opt.hiddenDim;
        int out_d = (l == hw_opt.numLayers - 1) ? hw_opt.outputDim : hw_opt.hiddenDim;
        
        int w_elements = in_d * out_d;
        int b_elements = out_d;

        CUDA_CHECK(cudaMemcpy(h_master_w[l], host_weights + w_offset, w_elements * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_master_b[l], host_biases + b_offset, b_elements * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (w_elements + 255) / 256;
        floatToHalfKernel<<<blocks, 256>>>(h_master_w[l], h_fwd_w[l], w_elements);

        w_offset += w_elements;
        b_offset += b_elements;
    }
    cudaDeviceSynchronize();
}

void TinyMLP::saveWeights(float* host_weights, float* host_biases) {
    int w_offset = 0;
    int b_offset = 0;

    std::vector<float*> h_master_w(hw_opt.numLayers);
    std::vector<float*> h_master_b(hw_opt.numLayers);
    
    cudaMemcpy(h_master_w.data(), d_master_weights, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_master_b.data(), d_master_biases, hw_opt.numLayers * sizeof(float*), cudaMemcpyDeviceToHost);

    for (int l = 0; l < hw_opt.numLayers; l++) {
        int in_d = (l == 0) ? hw_opt.inputDim : hw_opt.hiddenDim;
        int out_d = (l == hw_opt.numLayers - 1) ? hw_opt.outputDim : hw_opt.hiddenDim;
        
        int w_elements = in_d * out_d;
        int b_elements = out_d;

        CUDA_CHECK(cudaMemcpy(host_weights + w_offset, h_master_w[l], w_elements * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_biases + b_offset, h_master_b[l], b_elements * sizeof(float), cudaMemcpyDeviceToHost));

        w_offset += w_elements;
        b_offset += b_elements;
    }
}

// ============================================================================
// CLEANUP HELPERS
// ============================================================================

void TinyMLP::free_pointer_array(float** d_arr, int count) {
    if (!d_arr) return;
    std::vector<float*> h_arr(count);
    cudaMemcpy(h_arr.data(), d_arr, count * sizeof(float*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < count; i++) cudaFree(h_arr[i]);
    cudaFree(d_arr);
}

void TinyMLP::free_pointer_array(half** d_arr, int count, bool skip_first) {
    if (!d_arr) return;
    std::vector<half*> h_arr(count);
    cudaMemcpy(h_arr.data(), d_arr, count * sizeof(half*), cudaMemcpyDeviceToHost);
    for (int i = (skip_first ? 1 : 0); i < count; i++) cudaFree(h_arr[i]);
    cudaFree(d_arr);
}