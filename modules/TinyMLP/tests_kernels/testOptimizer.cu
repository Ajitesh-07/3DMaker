#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../TinyMLP.h"

using namespace std;

#define LOSS_SCALE 256.0f

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            cerr << "CUDA Error: " << cudaGetErrorString(err)                 \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------
// CPU Reference Implementations
// ---------------------------------------------------------

void cpuMSELossGrad(
    const vector<float>& predictions, 
    const vector<float>& targets,
    vector<float>& dLoss_out,
    float& total_loss,
    int batchSize, 
    int outputDim,
    float loss_scale
) {
    int num_elements = batchSize * outputDim;
    float grad_scale = (2.0f / batchSize) * loss_scale;
    
    double temp_total_loss = 0.0;

    for (int i = 0; i < num_elements; i++) {
        float diff = predictions[i] - targets[i];
        
        // Cast the math to double before adding to the accumulator
        temp_total_loss += (double)(diff * diff); 
        
        dLoss_out[i] = diff * grad_scale;
    }
    
    // Cast back to float for the final comparison
    total_loss = (float)temp_total_loss; 
}

void cpuAdamWeights(
    vector<vector<float>>& master_weights,
    vector<vector<float>>& fwd_weights,
    const vector<vector<float>>& gradients,
    vector<vector<float>>& m,
    vector<vector<float>>& v,
    const MLPOption& opt,
    float lr, float beta1, float beta2, float epsilon,
    float bias_correction1, float bias_correction2, float inv_loss_scale
) {
    for (int l = 0; l < opt.numLayers; l++) {
        int in_d = (l == 0) ? opt.inputDim : opt.hiddenDim;
        int out_d = (l == opt.numLayers - 1) ? opt.outputDim : opt.hiddenDim;
        int elements = in_d * out_d;

        for (int i = 0; i < elements; i++) {
            float g = gradients[l][i] * inv_loss_scale;
            
            float m_val = m[l][i];
            float v_val = v[l][i];
            float w_val = master_weights[l][i];

            m_val = beta1 * m_val + (1.0f - beta1) * g;
            v_val = beta2 * v_val + (1.0f - beta2) * g * g;

            float m_hat = m_val / bias_correction1;
            float v_hat = v_val / bias_correction2;

            w_val = w_val - (lr * m_hat / (sqrtf(v_hat) + epsilon));

            m[l][i] = m_val;
            v[l][i] = v_val;
            master_weights[l][i] = w_val;
            
            // fwd_weights is technically half, but we'll mock its float value for comparison
            // To be precise we could simulate FP16 cast
            fwd_weights[l][i] = __half2float(__float2half(w_val)); 
        }
    }
}

void cpuAdamBias(
    vector<vector<float>>& biases,
    const vector<vector<float>>& gradients,
    vector<vector<float>>& m,
    vector<vector<float>>& v,
    const MLPOption& opt,
    float lr, float beta1, float beta2, float epsilon,
    float bias_correction1, float bias_correction2, float inv_loss_scale
) {
    for (int l = 0; l < opt.numLayers; l++) {
        int out_d = (l == opt.numLayers - 1) ? opt.outputDim : opt.hiddenDim;

        for (int i = 0; i < out_d; i++) {
            float g = gradients[l][i] * inv_loss_scale;
            
            float m_val = m[l][i];
            float v_val = v[l][i];
            float b_val = biases[l][i];

            m_val = beta1 * m_val + (1.0f - beta1) * g;
            v_val = beta2 * v_val + (1.0f - beta2) * g * g;

            float m_hat = m_val / bias_correction1;
            float v_hat = v_val / bias_correction2;

            b_val = b_val - (lr * m_hat / (sqrtf(v_hat) + epsilon));

            m[l][i] = m_val;
            v[l][i] = v_val;
            biases[l][i] = b_val;
        }
    }
}

// ---------------------------------------------------------
// Helper: Random Data Generation
// ---------------------------------------------------------

void fillRandomFP16(vector<float>& host_float, half* d_ptr, int size, mt19937& gen) {
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    vector<half> host_half(size);
    for (int i = 0; i < size; i++) {
        float val = dist(gen);
        host_float[i] = val;
        host_half[i] = __float2half(val);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, host_half.data(), size * sizeof(half), cudaMemcpyHostToDevice));
}

void fillRandomFP32(vector<vector<float>>& host_data, float** d_ptr_array, const MLPOption& opt, bool isWeight, mt19937& gen) {
    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    vector<float*> host_ptr_array(opt.numLayers);

    for (int l = 0; l < opt.numLayers; l++) {
        int in_d = (l == 0) ? opt.inputDim : opt.hiddenDim;
        int out_d = (l == opt.numLayers - 1) ? opt.outputDim : opt.hiddenDim;
        int elements = isWeight ? (in_d * out_d) : out_d;

        host_data[l].resize(elements);
        for (int i = 0; i < elements; i++) {
            host_data[l][i] = dist(gen);
        }

        float* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_arr, host_data[l].data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        host_ptr_array[l] = d_arr;
    }
    
    float** d_ptr_array_dev;
    CUDA_CHECK(cudaMalloc(&d_ptr_array_dev, opt.numLayers * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_ptr_array_dev, host_ptr_array.data(), opt.numLayers * sizeof(float*), cudaMemcpyHostToDevice));
    
    // Copy the device array of pointers to the provided handle
    CUDA_CHECK(cudaMemcpy(d_ptr_array, &d_ptr_array_dev, sizeof(float*), cudaMemcpyHostToHost)); // Wait, we need to return d_ptr_array_dev.
    // Better signature: return float**
}

float** createDevicePointerArray(const vector<vector<float>>& host_data) {
    int numLayers = host_data.size();
    vector<float*> host_ptr_array(numLayers);
    
    for (int l = 0; l < numLayers; l++) {
        int elements = host_data[l].size();
        float* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_arr, host_data[l].data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        host_ptr_array[l] = d_arr;
    }

    float** d_ptr_array_dev;
    CUDA_CHECK(cudaMalloc(&d_ptr_array_dev, numLayers * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_ptr_array_dev, host_ptr_array.data(), numLayers * sizeof(float*), cudaMemcpyHostToDevice));
    
    return d_ptr_array_dev;
}

half** createDeviceHalfPointerArray(int numLayers, const vector<vector<float>>& sizes_ref, bool allocateOnly = false) {
    vector<half*> host_ptr_array(numLayers);
    for (int l = 0; l < numLayers; l++) {
        int elements = sizes_ref[l].size();
        half* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, elements * sizeof(half)));
        if (!allocateOnly) {
            vector<half> init_half(elements, __float2half(0.0f));
            CUDA_CHECK(cudaMemcpy(d_arr, init_half.data(), elements * sizeof(half), cudaMemcpyHostToDevice));
        }
        host_ptr_array[l] = d_arr;
    }
    half** d_ptr_array_dev;
    CUDA_CHECK(cudaMalloc(&d_ptr_array_dev, numLayers * sizeof(half*)));
    CUDA_CHECK(cudaMemcpy(d_ptr_array_dev, host_ptr_array.data(), numLayers * sizeof(half*), cudaMemcpyHostToDevice));
    return d_ptr_array_dev;
}

void verifyFP32(const vector<vector<float>>& cpu_data, float** d_ptr_array, const string& name) {
    int numLayers = cpu_data.size();
    vector<float*> host_ptr_array(numLayers);
    CUDA_CHECK(cudaMemcpy(host_ptr_array.data(), d_ptr_array, numLayers * sizeof(float*), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int l = 0; l < numLayers; l++) {
        int elements = cpu_data[l].size();
        vector<float> gpu_data(elements);
        CUDA_CHECK(cudaMemcpy(gpu_data.data(), host_ptr_array[l], elements * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < elements; i++) {
            float err = std::abs(cpu_data[l][i] - gpu_data[i]);
            if (err > max_err) max_err = err;
        }
    }
    
    if (max_err < 1e-4) {
        cout << "[SUCCESS] " << name << " verified. Max error: " << max_err << endl;
    } else {
        cout << "[FAILED] " << name << " verification failed. Max error: " << max_err << endl;
    }
}

void verifyFP16(const vector<vector<float>>& cpu_data, half** d_ptr_array, const string& name) {
    int numLayers = cpu_data.size();
    vector<half*> host_ptr_array(numLayers);
    CUDA_CHECK(cudaMemcpy(host_ptr_array.data(), d_ptr_array, numLayers * sizeof(half*), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int l = 0; l < numLayers; l++) {
        int elements = cpu_data[l].size();
        vector<half> gpu_data(elements);
        CUDA_CHECK(cudaMemcpy(gpu_data.data(), host_ptr_array[l], elements * sizeof(half), cudaMemcpyDeviceToHost));

        for (int i = 0; i < elements; i++) {
            float err = std::abs(cpu_data[l][i] - __half2float(gpu_data[i]));
            if (err > max_err) max_err = err;
        }
    }
    
    if (max_err < 1e-3) { // Slightly looser for half
        cout << "[SUCCESS] " << name << " verified. Max error: " << max_err << endl;
    } else {
        cout << "[FAILED] " << name << " verification failed. Max error: " << max_err << endl;
    }
}

// ---------------------------------------------------------
// Main Test Routines
// ---------------------------------------------------------

void testMSELossKernel() {
    cout << "\n--- Testing MSE Loss Kernel ---" << endl;
    int batchSize = 1 << 21;
    int outputDim = 64;
    float loss_scale = LOSS_SCALE;
    int numElements = batchSize * outputDim;

    mt19937 gen(42);

    vector<float> cpu_preds(numElements);
    vector<float> cpu_targets(numElements);
    vector<float> cpu_dLoss(numElements);
    float cpu_total_loss = 0.0f;

    float *d_preds;
    half *d_targets, *d_dLoss;
    float *d_total_loss;
    
    CUDA_CHECK(cudaMalloc(&d_preds, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, numElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dLoss, numElements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_total_loss, 0, sizeof(float)));

    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < numElements; i++) {
        cpu_preds[i] = dist(gen);
    }
    CUDA_CHECK(cudaMemcpy(d_preds, cpu_preds.data(), numElements * sizeof(float), cudaMemcpyHostToDevice));

    fillRandomFP16(cpu_targets, d_targets, numElements, gen);

    // CPU Pass
    cpuMSELossGrad(cpu_preds, cpu_targets, cpu_dLoss, cpu_total_loss, batchSize, outputDim, loss_scale);

    // GPU Pass
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    launchMSELossGrad(d_preds, d_targets, d_dLoss, d_total_loss, batchSize, outputDim, outputDim, loss_scale, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Verify Loss Scalar
    float gpu_total_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    float diff_loss = std::abs(cpu_total_loss - gpu_total_loss);
    // Relative error for scalar sum
    float rel_err = diff_loss / (cpu_total_loss + 1e-8f);
    cout << "Total Loss: CPU = " << cpu_total_loss << " | GPU = " << gpu_total_loss << " | Rel Err = " << rel_err << endl;
    if (rel_err < 1e-3) cout << "[SUCCESS] MSE Loss Scalar Match." << endl;
    else cout << "[FAILED] MSE Loss Scalar Mismatch." << endl;

    // Verify dLoss
    vector<half> gpu_dLoss_half(numElements);
    CUDA_CHECK(cudaMemcpy(gpu_dLoss_half.data(), d_dLoss, numElements * sizeof(half), cudaMemcpyDeviceToHost));
    
    float max_grad_err = 0.0f;
    for (int i = 0; i < numElements; i++) {
        float err = std::abs(cpu_dLoss[i] - __half2float(gpu_dLoss_half[i]));
        if (err > max_grad_err) max_grad_err = err;
    }
    
    if (max_grad_err < 1e-3) cout << "[SUCCESS] MSE Gradients Match. Max Err: " << max_grad_err << endl;
    else cout << "[FAILED] MSE Gradients Mismatch. Max Err: " << max_grad_err << endl;
    
    cout << "Performance: " << ms << " ms (" << ((float)numElements * sizeof(half) * 3) / (ms * 1e6) << " GB/s)" << endl;
    cudaStreamDestroy(stream);
}

void testAdamOptimKernels() {
    cout << "\n--- Testing Adam Optimizer Kernels ---" << endl;
    
    MLPOption opt;
    opt.inputDim = 32;
    opt.hiddenDim = 64;
    opt.outputDim = 32;
    opt.numLayers = 4;
    
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int step = 10;
    float inv_loss_scale = 1.0f / LOSS_SCALE;

    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    mt19937 gen(42);
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    vector<vector<float>> cpu_w_master(opt.numLayers), cpu_w_fwd(opt.numLayers);
    vector<vector<float>> cpu_w_m(opt.numLayers), cpu_w_v(opt.numLayers), cpu_w_grad(opt.numLayers);
    
    vector<vector<float>> cpu_b_master(opt.numLayers);
    vector<vector<float>> cpu_b_m(opt.numLayers), cpu_b_v(opt.numLayers), cpu_b_grad(opt.numLayers);

    for (int l = 0; l < opt.numLayers; l++) {
        int in_d = (l == 0) ? opt.inputDim : opt.hiddenDim;
        int out_d = (l == opt.numLayers - 1) ? opt.outputDim : opt.hiddenDim;
        int w_elems = in_d * out_d;
        int b_elems = out_d;

        cpu_w_master[l].resize(w_elems); cpu_w_fwd[l].resize(w_elems);
        cpu_w_m[l].resize(w_elems); cpu_w_v[l].resize(w_elems); cpu_w_grad[l].resize(w_elems);

        cpu_b_master[l].resize(b_elems);
        cpu_b_m[l].resize(b_elems); cpu_b_v[l].resize(b_elems); cpu_b_grad[l].resize(b_elems);

        for (int i = 0; i < w_elems; i++) {
            cpu_w_master[l][i] = dist(gen);
            cpu_w_m[l][i] = dist(gen) * 0.01f;
            cpu_w_v[l][i] = std::abs(dist(gen)) * 0.001f;
            cpu_w_grad[l][i] = dist(gen) * 100.0f; // Mock large scaled gradients
        }

        for (int i = 0; i < b_elems; i++) {
            cpu_b_master[l][i] = dist(gen);
            cpu_b_m[l][i] = dist(gen) * 0.01f;
            cpu_b_v[l][i] = std::abs(dist(gen)) * 0.001f;
            cpu_b_grad[l][i] = dist(gen) * 100.0f;
        }
    }

    float** d_w_master = createDevicePointerArray(cpu_w_master);
    float** d_w_m = createDevicePointerArray(cpu_w_m);
    float** d_w_v = createDevicePointerArray(cpu_w_v);
    float** d_w_grad = createDevicePointerArray(cpu_w_grad);
    half**  d_w_fwd = createDeviceHalfPointerArray(opt.numLayers, cpu_w_master, true);

    float** d_b_master = createDevicePointerArray(cpu_b_master);
    float** d_b_m = createDevicePointerArray(cpu_b_m);
    float** d_b_v = createDevicePointerArray(cpu_b_v);
    float** d_b_grad = createDevicePointerArray(cpu_b_grad);

    // CPU Pass
    cpuAdamWeights(cpu_w_master, cpu_w_fwd, cpu_w_grad, cpu_w_m, cpu_w_v, opt, lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale);
    cpuAdamBias(cpu_b_master, cpu_b_grad, cpu_b_m, cpu_b_v, opt, lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale);

    // GPU Pass
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    launchAdamWeightsOptim(&opt, d_w_master, d_w_fwd, d_w_grad, d_w_m, d_w_v, lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale, stream);
    launchAdamBiasOptim(&opt, d_b_master, d_b_grad, d_b_m, d_b_v, lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Verify
    verifyFP32(cpu_w_master, d_w_master, "Weights Master");
    verifyFP16(cpu_w_fwd, d_w_fwd, "Weights FWD (FP16)");
    verifyFP32(cpu_w_m, d_w_m, "Weights Momentum");
    verifyFP32(cpu_w_v, d_w_v, "Weights Variance");

    verifyFP32(cpu_b_master, d_b_master, "Bias Master");
    verifyFP32(cpu_b_m, d_b_m, "Bias Momentum");
    verifyFP32(cpu_b_v, d_b_v, "Bias Variance");

    cout << "Performance (Weights + Bias): " << ms << " ms" << endl;

    cudaStreamDestroy(stream);
}

int main() {
    testMSELossKernel();
    testAdamOptimKernels();
    return 0;
}
