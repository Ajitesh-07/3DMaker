#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cuda_fp16.h>
#include "../TinyMLP.h"

// ==========================================
// CPU REFERENCE IMPLEMENTATION
// ==========================================
void cpuReferenceMLP_MultiLayer(
    const std::vector<half>& inputs, 
    const std::vector<std::vector<half>>& weights, 
    const std::vector<std::vector<float>>& biases, 
    std::vector<float>& outputs, 
    int batchSize, int inputDim, int hiddenDim, int outputDim, int numLayers,
    std::vector<std::vector<half>>* out_activations = nullptr
) {
    int maxDim = std::max({inputDim, hiddenDim, outputDim});
    std::vector<float> current_act(batchSize * maxDim, 0.0f);
    std::vector<float> next_act(batchSize * maxDim, 0.0f);

    for (int i = 0; i < batchSize * inputDim; i++) {
        current_act[i] = __half2float(inputs[i]);
    }

    // --- NEW LOGIC: Initialize activations array ---
    if (out_activations) {
        out_activations->resize(numLayers);
        // Index 0 explicitly holds the inputs
        (*out_activations)[0] = inputs; 
    }

    for (int layer = 0; layer < numLayers; ++layer) {
        int current_K = (layer == 0) ? inputDim : hiddenDim;
        int current_N = (layer == numLayers - 1) ? outputDim : hiddenDim;
        
        const auto& W = weights[layer];
        const auto& B = biases[layer];

        for (int m = 0; m < batchSize; ++m) {
            for (int n = 0; n < current_N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < current_K; ++k) {
                    float a = current_act[m * current_K + k];
                    float b = __half2float(W[n * current_K + k]); 
                    sum += a * b;
                }
                
                sum += B[n];
                
                if (layer < numLayers - 1) {
                    sum = std::fmax(sum, 0.0f); 
                }
                
                next_act[m * current_N + n] = sum;
            }
        }

        // --- NEW LOGIC: Save into layer + 1 ---
        // (We skip the last layer as it writes to outputs, not intermediate activations)
        if (out_activations && layer < numLayers - 1) {
            (*out_activations)[layer + 1].resize(batchSize * current_N);
            for (int i = 0; i < batchSize * current_N; i++) {
                (*out_activations)[layer + 1][i] = __float2half(next_act[i]);
            }
        }

        current_act = next_act;
    }

    for (int i = 0; i < batchSize * outputDim; i++) {
        outputs[i] = current_act[i];
    }
}

// ==========================================
// VERIFICATION
// ==========================================
bool verifyResults(const std::vector<float>& cpu_outputs, const std::vector<float>& gpu_outputs, float tolerance = 2e-2f) {
    float max_error = 0.0f;
    int mismatch_count = 0;
    
    for (size_t i = 0; i < cpu_outputs.size(); ++i) {
        float diff = std::abs(cpu_outputs[i] - gpu_outputs[i]);
        if (diff > max_error) max_error = diff;

        if (diff > tolerance && mismatch_count < 5) {
            std::cout << "Mismatch at index " << i << "! CPU: " << cpu_outputs[i] << " | GPU: " << gpu_outputs[i] << std::endl;
            mismatch_count++;
        }
    }
    
    if (mismatch_count > 0) {
        std::cout << "FAILED! Max error was: " << max_error << std::endl;
        return false;
    }
    
    std::cout << "SUCCESS! Max error: " << max_error << std::endl;
    return true;
}

// ==========================================
// INFERENCE TEST
// ==========================================
void testMultiLayerMLP() {
    int batchSize = 16384; 
    int inputDim = 32;
    int hiddenDim = 64;
    int outputDim = 3;
    int numLayers = 4; 

    size_t size_inputs = batchSize * inputDim;
    size_t size_outputs = batchSize * outputDim;

    std::vector<half> h_inputs(size_inputs);
    std::vector<std::vector<half>> h_weights(numLayers);
    std::vector<std::vector<float>> h_biases(numLayers);
    std::vector<float> h_outputs_cpu(size_outputs, 0.0f);
    std::vector<float> h_outputs_gpu(size_outputs, 0.0f);

    for (auto& val : h_inputs) val = __float2half((rand() % 100) / 100.0f - 0.5f);

    for (int i = 0; i < numLayers; i++) {
        int current_K = (i == 0) ? inputDim : hiddenDim;
        int current_N = (i == numLayers - 1) ? outputDim : hiddenDim;
        h_weights[i].resize(current_N * current_K);
        h_biases[i].resize(current_N);
        for (auto& val : h_weights[i]) val = __float2half((rand() % 100) / 100.0f - 0.5f);
        for (auto& val : h_biases[i]) val = (rand() % 100) / 100.0f - 0.5f;
    }

    half* d_inputs;
    float* d_outputs;
    cudaMalloc(&d_inputs, size_inputs * sizeof(half));
    cudaMalloc(&d_outputs, size_outputs * sizeof(float));
    cudaMemcpy(d_inputs, h_inputs.data(), size_inputs * sizeof(half), cudaMemcpyHostToDevice);

    std::vector<half*> h_d_weights_ptrs(numLayers);
    std::vector<float*> h_d_biases_ptrs(numLayers);

    for (int i = 0; i < numLayers; i++) {
        int current_K = (i == 0) ? inputDim : hiddenDim;
        int current_N = (i == numLayers - 1) ? outputDim : hiddenDim;
        cudaMalloc(&h_d_weights_ptrs[i], current_N * current_K * sizeof(half));
        cudaMalloc(&h_d_biases_ptrs[i], current_N * sizeof(float));
        cudaMemcpy(h_d_weights_ptrs[i], h_weights[i].data(), current_N * current_K * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(h_d_biases_ptrs[i], h_biases[i].data(), current_N * sizeof(float), cudaMemcpyHostToDevice);
    }

    half** d_weights_array;
    float** d_biases_array;
    cudaMalloc(&d_weights_array, numLayers * sizeof(half*));
    cudaMalloc(&d_biases_array, numLayers * sizeof(float*));
    cudaMemcpy(d_weights_array, h_d_weights_ptrs.data(), numLayers * sizeof(half*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases_array, h_d_biases_ptrs.data(), numLayers * sizeof(float*), cudaMemcpyHostToDevice);

    MLPOption opt = {inputDim, hiddenDim, outputDim, numLayers, ACT_RELU}; 

    // CPU — also compute per-layer activations for forwardGrad verification
    std::vector<std::vector<half>> cpu_activations;
    std::cout << "\n--- Inference Verification ---\n";
    std::cout << "Running CPU Reference (" << numLayers << " Layers)..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuReferenceMLP_MultiLayer(h_inputs, h_weights, h_biases, h_outputs_cpu, batchSize, inputDim, hiddenDim, outputDim, numLayers, &cpu_activations);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();

    // GPU inference
    launchNetworkFusionKernel(&opt, d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu); cudaEventCreate(&stop_gpu);
    int num_runs = 100;
    cudaEventRecord(start_gpu, 0);
    for(int i = 0; i < num_runs; i++)
        launchNetworkFusionKernel(&opt, d_inputs, d_weights_array, d_biases_array, d_outputs, batchSize, 0);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    float gpu_total; cudaEventElapsedTime(&gpu_total, start_gpu, stop_gpu);
    float gpu_avg = gpu_total / num_runs;

    std::cout << "CPU: " << cpu_ms << " ms | GPU: " << gpu_avg << " ms | Speedup: " << cpu_ms / gpu_avg << "x\n";
    cudaMemcpy(h_outputs_gpu.data(), d_outputs, size_outputs * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Inference: "; verifyResults(h_outputs_cpu, h_outputs_gpu);

    // GPU forwardGrad — saves activations per layer
    // GPU forwardGrad — saves activations per layer
    std::vector<half*> h_d_act_ptrs(numLayers);
    
    // Index 0 points directly to the inputs (kernel doesn't need to write this)
    h_d_act_ptrs[0] = d_inputs; 

    // Indices 1 through numLayers-1 are all hidden layers
    for (int i = 1; i < numLayers; i++) {
        cudaMalloc(&h_d_act_ptrs[i], batchSize * hiddenDim * sizeof(half));
    }
    
    half** d_activations_array;
    cudaMalloc(&d_activations_array, numLayers * sizeof(half*));
    cudaMemcpy(d_activations_array, h_d_act_ptrs.data(), numLayers * sizeof(half*), cudaMemcpyHostToDevice);


    std::cout << "\n--- ForwardGrad Verification ---\n";
    std::vector<float> h_outputs_grad(size_outputs, 0.0f);
    launchNetworkFusionGradKernel(&opt, d_inputs, d_activations_array, d_weights_array, d_biases_array, d_outputs, batchSize, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ForwardGrad kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(h_outputs_grad.data(), d_outputs, size_outputs * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "ForwardGrad output: "; verifyResults(h_outputs_cpu, h_outputs_grad);

    // Verify per-layer activations against CPU (skip last layer — it writes to d_outputs directly)
    // Verify per-layer activations against CPU (Now including layer 0)
    for (int l = 0; l < numLayers; l++) {
        int current_N = (l == 0) ? inputDim : hiddenDim;
        int act_size = batchSize * current_N;

        std::vector<half> gpu_act(act_size);
        cudaMemcpy(gpu_act.data(), h_d_act_ptrs[l], act_size * sizeof(half), cudaMemcpyDeviceToHost);

        float max_diff = 0.f;
        int errors = 0;
        for (int i = 0; i < act_size; i++) {
            float gpu_val = __half2float(gpu_act[i]);
            float cpu_val = __half2float(cpu_activations[l][i]);
            float diff = std::abs(gpu_val - cpu_val);
            if (diff > max_diff) max_diff = diff;
            if (diff > 2e-2f && errors < 5) {
                std::cout << "  Layer " << l << " act mismatch [" << i << "]: GPU=" << gpu_val << " CPU=" << cpu_val << "\n";
                errors++;
            }
        }
        std::cout << "Layer " << l << " activations: MaxDiff=" << max_diff << " -> "
                  << ((errors == 0 && max_diff < 2e-2f) ? "[SUCCESS]" : "[FAILED]") << "\n";
    }

    // Cleanup
    cudaEventDestroy(start_gpu); cudaEventDestroy(stop_gpu);
    cudaFree(d_inputs); cudaFree(d_outputs);
    for (int i = 0; i < numLayers; i++) {
        cudaFree(h_d_weights_ptrs[i]); cudaFree(h_d_biases_ptrs[i]); cudaFree(h_d_act_ptrs[i]);
    }
    cudaFree(d_weights_array); cudaFree(d_biases_array); cudaFree(d_activations_array);
}

// ==========================================
// PERFORMANCE SWEEP (Side-by-Side)
// ==========================================
struct SweepResult { float infer_ms; float grad_ms; bool oom; };

SweepResult profileConfiguration(int batchSize, int hiddenDim, int numLayers) {
    SweepResult res = {0.f, 0.f, false};

    half* d_inputs = nullptr;
    float* d_outputs = nullptr;
    if (cudaMalloc(&d_inputs, batchSize * hiddenDim * sizeof(half)) != cudaSuccess) { res.oom = true; cudaGetLastError(); return res; }
    if (cudaMalloc(&d_outputs, batchSize * hiddenDim * sizeof(float)) != cudaSuccess) { res.oom = true; cudaGetLastError(); cudaFree(d_inputs); return res; }
    cudaMemset(d_inputs, 0, batchSize * hiddenDim * sizeof(half));

    std::vector<half*> h_wt(numLayers, nullptr);
    std::vector<float*> h_bi(numLayers, nullptr);
    std::vector<half*> h_act(numLayers, nullptr);

    for (int i = 0; i < numLayers && !res.oom; i++) {
        if (cudaMalloc(&h_wt[i], hiddenDim * hiddenDim * sizeof(half)) != cudaSuccess) res.oom = true;
        if (cudaMalloc(&h_bi[i], hiddenDim * sizeof(float)) != cudaSuccess) res.oom = true;
        if (cudaMalloc(&h_act[i], batchSize * hiddenDim * sizeof(half)) != cudaSuccess) res.oom = true;
        if (!res.oom) {
            cudaMemset(h_wt[i], 0, hiddenDim * hiddenDim * sizeof(half));
            cudaMemset(h_bi[i], 0, hiddenDim * sizeof(float));
        }
    }

    half** d_wt_arr = nullptr; float** d_bi_arr = nullptr; half** d_act_arr = nullptr;
    if (!res.oom) {
        cudaMalloc(&d_wt_arr, numLayers * sizeof(half*));
        cudaMalloc(&d_bi_arr, numLayers * sizeof(float*));
        cudaMalloc(&d_act_arr, numLayers * sizeof(half*));
        cudaMemcpy(d_wt_arr, h_wt.data(), numLayers * sizeof(half*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bi_arr, h_bi.data(), numLayers * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_act_arr, h_act.data(), numLayers * sizeof(half*), cudaMemcpyHostToDevice);
    }

    if (res.oom) { cudaGetLastError(); goto cleanup; }

    {
        MLPOption opt = {hiddenDim, hiddenDim, hiddenDim, numLayers, ACT_RELU}; 
        int num_runs = (batchSize < 500000) ? 50 : 10; 

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // Warmup + time inference
        launchNetworkFusionKernel(&opt, d_inputs, d_wt_arr, d_bi_arr, d_outputs, batchSize, 0);
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        for(int i = 0; i < num_runs; i++)
            launchNetworkFusionKernel(&opt, d_inputs, d_wt_arr, d_bi_arr, d_outputs, batchSize, 0);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float total; cudaEventElapsedTime(&total, start, stop);
        res.infer_ms = total / num_runs;

        // Warmup + time forwardGrad
        launchNetworkFusionGradKernel(&opt, d_inputs, d_act_arr, d_wt_arr, d_bi_arr, d_outputs, batchSize, 0);
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        for(int i = 0; i < num_runs; i++)
            launchNetworkFusionGradKernel(&opt, d_inputs, d_act_arr, d_wt_arr, d_bi_arr, d_outputs, batchSize, 0);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&total, start, stop);
        res.grad_ms = total / num_runs;

        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

cleanup:
    if (d_inputs) cudaFree(d_inputs);
    if (d_outputs) cudaFree(d_outputs);
    for (int i = 0; i < numLayers; i++) {
        if (h_wt[i]) cudaFree(h_wt[i]);
        if (h_bi[i]) cudaFree(h_bi[i]);
        if (h_act[i]) cudaFree(h_act[i]);
    }
    if (d_wt_arr) cudaFree(d_wt_arr);
    if (d_bi_arr) cudaFree(d_bi_arr);
    if (d_act_arr) cudaFree(d_act_arr);
    return res;
}

void runParameterSweep() {
    std::cout << "============================================================================================\n";
    std::cout << "              TINYMLP FUSED KERNEL PROFILER — Inference vs ForwardGrad                      \n";
    std::cout << "============================================================================================\n";
    std::cout << std::left 
              << std::setw(12) << "Batch"
              << std::setw(8)  << "Dim"
              << std::setw(8)  << "Layers"
              << std::setw(14) << "Infer (ms)"
              << std::setw(14) << "Grad (ms)"
              << std::setw(12) << "Overhead"
              << std::setw(14) << "Infer TFLOP"
              << std::setw(14) << "Grad TFLOP"
              << "Status\n";
    std::cout << std::string(96, '-') << "\n";

    std::vector<int> dimensions = {32, 64, 128};

    for (int dim : dimensions) {
        for (int layers = 1; layers <= 10; ++layers) {
            for (int batch = (1 << 16); batch <= (1 << 21); batch <<= 1) {
                
                auto res = profileConfiguration(batch, dim, layers);

                double total_flops = 2.0 * batch * dim * dim * layers;
                double infer_tflops = (!res.oom && res.infer_ms > 0.f) ? (total_flops / (res.infer_ms * 1e-3)) / 1e12 : 0.0;
                double grad_tflops  = (!res.oom && res.grad_ms > 0.f)  ? (total_flops / (res.grad_ms * 1e-3)) / 1e12 : 0.0;
                float overhead_pct  = (!res.oom && res.infer_ms > 0.f) ? ((res.grad_ms - res.infer_ms) / res.infer_ms) * 100.f : 0.f;

                std::cout << std::left 
                          << std::setw(12) << batch 
                          << std::setw(8)  << dim 
                          << std::setw(8)  << layers;

                if (res.oom) {
                    std::cout << std::setw(14) << "N/A" << std::setw(14) << "N/A"
                              << std::setw(12) << "N/A" << std::setw(14) << "N/A"
                              << std::setw(14) << "N/A" << "OOM\n";
                } else {
                    std::cout << std::setw(14) << std::fixed << std::setprecision(4) << res.infer_ms
                              << std::setw(14) << std::fixed << std::setprecision(4) << res.grad_ms
                              << std::setw(12) << std::fixed << std::setprecision(1) << overhead_pct << "%"
                              << std::setw(13) << std::fixed << std::setprecision(2) << infer_tflops
                              << std::setw(14) << std::fixed << std::setprecision(2) << grad_tflops
                              << "OK\n";
                }
            }
            std::cout << std::string(96, '-') << "\n";
        }
    }
}

int main() {
    std::cout << "Starting TinyMLP Fused Tensor Core Test..." << std::endl;
    testMultiLayerMLP();
    runParameterSweep();
    return 0;
}