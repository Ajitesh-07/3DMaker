#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../TinyMLP.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// ---------------------------------------------------------
// Device Info
// ---------------------------------------------------------
void printDeviceInfo() {
    int devId;
    CUDA_CHECK(cudaGetDevice(&devId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));
    std::cout << "========== Device Info ==========\n";
    std::cout << "Device                       : " << prop.name << "\n";
    std::cout << "Default shared mem per block : " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "=================================\n\n";
}

// ---------------------------------------------------------
// CPU Reference (Full Backpropagation Chain)
// ---------------------------------------------------------
void computeFullBackpropCPU(
    int batchSize, int inputDim, int hiddenDim, int outputDim, int numLayers,
    const std::vector<std::vector<half>>& activations,
    const std::vector<half>& loss_output,
    const std::vector<std::vector<half>>& weights,
    std::vector<std::vector<float>>& expected_dw,
    std::vector<std::vector<float>>& expected_db)
{
    std::vector<float> current_dz(batchSize * outputDim);
    for (size_t i = 0; i < loss_output.size(); ++i)
        current_dz[i] = __half2float(loss_output[i]);

    for (int layer = numLayers - 1; layer >= 0; --layer) {
        int in_d  = (layer == 0) ? inputDim : hiddenDim;
        int out_d = (layer == numLayers - 1) ? outputDim : hiddenDim;

        for (int o = 0; o < out_d; ++o) {
            float db_sum = 0.0f;
            for (int b = 0; b < batchSize; ++b) {
                float dz_val = current_dz[b * out_d + o];
                db_sum += dz_val;
                for (int h = 0; h < in_d; ++h) {
                    float a_val = __half2float(activations[layer][b * in_d + h]);
                    expected_dw[layer][o * in_d + h] += dz_val * a_val;
                }
            }
            expected_db[layer][o] = db_sum;
        }

        if (layer > 0) {
            std::vector<float> next_dz(batchSize * in_d, 0.0f);
            for (int b = 0; b < batchSize; ++b) {
                for (int h = 0; h < in_d; ++h) {
                    float sum = 0.0f;
                    for (int o = 0; o < out_d; ++o) {
                        sum += current_dz[b * out_d + o] * __half2float(weights[layer][o * in_d + h]);
                    }
                    float orig_act = __half2float(activations[layer][b * in_d + h]);
                    next_dz[b * in_d + h] = (orig_act > 0.0f) ? __half2float(__float2half(sum)) : 0.0f;
                }
            }
            current_dz = next_dz;
        }
    }
}

// ---------------------------------------------------------
// GPU Timer
// ---------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
    float Stop(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(stop, s));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ---------------------------------------------------------
// Compute total backward FLOPs
// ---------------------------------------------------------
double computeBackwardFlops(int batchSize, int inputDim, int hiddenDim, int outputDim, int numLayers) {
    double flops = 0.0;
    for (int l = 0; l < numLayers; ++l) {
        int in_d  = (l == 0) ? inputDim : hiddenDim;
        int out_d = (l == numLayers - 1) ? outputDim : hiddenDim;
        flops += 2.0 * batchSize * in_d * out_d;
        flops += (double)batchSize * out_d;
        if (l > 0) flops += 2.0 * batchSize * out_d * in_d;
    }
    return flops;
}

// ---------------------------------------------------------
// Verification Test
// ---------------------------------------------------------
void runVerification() {
    int batchSize = 1 << 13;
    int inputDim  = 64;
    int hiddenDim = 64;
    int outputDim = 3;
    int numLayers = 4;

    MLPOption opt = {inputDim, hiddenDim, outputDim, numLayers, ACT_RELU};
    printDeviceInfo();

    // --- Host data ---
    std::vector<half> loss_output(batchSize * outputDim);
    std::vector<std::vector<half>>  h_activations(numLayers);
    std::vector<std::vector<half>>  h_weights(numLayers);
    std::vector<std::vector<float>> h_grad_weights(numLayers);
    std::vector<std::vector<float>> h_expected_dw(numLayers);
    std::vector<std::vector<float>> h_grad_biases(numLayers);
    std::vector<std::vector<float>> h_expected_db(numLayers);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : loss_output) v = __float2half(dist(gen));

    for (int l = 0; l < numLayers; ++l) {
        int in_d  = (l == 0) ? inputDim : hiddenDim;
        int out_d = (l == numLayers - 1) ? outputDim : hiddenDim;
        h_activations[l].resize(batchSize * in_d);
        h_weights[l].resize(out_d * in_d);
        h_grad_weights[l].resize(out_d * in_d, 0.f);
        h_expected_dw[l].resize(out_d * in_d, 0.f);
        h_grad_biases[l].resize(out_d, 0.f);
        h_expected_db[l].resize(out_d, 0.f);
        for (auto& v : h_activations[l]) v = __float2half(dist(gen));
        for (auto& v : h_weights[l])     v = __float2half(dist(gen));
    }

    // --- Device allocations ---
    half* d_loss_output;
    CUDA_CHECK(cudaMalloc(&d_loss_output, batchSize * outputDim * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_loss_output, loss_output.data(), batchSize * outputDim * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<half*>  h_d_act_ptrs(numLayers);
    std::vector<half*>  h_d_wt_ptrs(numLayers);
    std::vector<float*> h_d_gw_ptrs(numLayers);
    std::vector<float*> h_d_gb_ptrs(numLayers);

    for (int l = 0; l < numLayers; ++l) {
        int in_d  = (l == 0) ? inputDim : hiddenDim;
        int out_d = (l == numLayers - 1) ? outputDim : hiddenDim;

        CUDA_CHECK(cudaMalloc(&h_d_act_ptrs[l], batchSize * in_d * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(h_d_act_ptrs[l], h_activations[l].data(), batchSize * in_d * sizeof(half), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&h_d_wt_ptrs[l], out_d * in_d * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(h_d_wt_ptrs[l], h_weights[l].data(), out_d * in_d * sizeof(half), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&h_d_gw_ptrs[l], out_d * in_d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_d_gb_ptrs[l], out_d * sizeof(float)));
    }

    half**  d_act_arr;
    half**  d_wt_arr;
    float** d_gw_arr;
    float** d_gb_arr;

    CUDA_CHECK(cudaMalloc(&d_act_arr, numLayers * sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&d_wt_arr,  numLayers * sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&d_gw_arr,  numLayers * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_gb_arr,  numLayers * sizeof(float*)));

    CUDA_CHECK(cudaMemcpy(d_act_arr, h_d_act_ptrs.data(), numLayers * sizeof(half*),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wt_arr,  h_d_wt_ptrs.data(),  numLayers * sizeof(half*),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gw_arr,  h_d_gw_ptrs.data(),  numLayers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gb_arr,  h_d_gb_ptrs.data(),  numLayers * sizeof(float*), cudaMemcpyHostToDevice));

    half* d_dx_out_dummy;
    CUDA_CHECK(cudaMalloc(&d_dx_out_dummy, sizeof(half)));
    cudaStream_t stream = 0;

    // --- GPU run ---
    const int NUM_GPU_RUNS = 10;
    GpuTimer gpuTimer;
    float totalGpuMs = 0.f;

    std::cout << "Timing GPU Fused Backward (" << NUM_GPU_RUNS << " runs)...\n";
    for (int r = 0; r < NUM_GPU_RUNS; ++r) {
        for (int l = 0; l < numLayers; ++l) {
            int in_d  = (l == 0) ? inputDim : hiddenDim;
            int out_d = (l == numLayers - 1) ? outputDim : hiddenDim;
            CUDA_CHECK(cudaMemset(h_d_gw_ptrs[l], 0, out_d * in_d * sizeof(float)));
            CUDA_CHECK(cudaMemset(h_d_gb_ptrs[l], 0, out_d * sizeof(float)));
        }

        gpuTimer.Start(stream);
        launchNetworkFusionBackwardKernel(
            &opt, d_loss_output, d_wt_arr, nullptr,
            d_act_arr, d_gw_arr, d_gb_arr,
            d_dx_out_dummy, batchSize, stream);
        totalGpuMs += gpuTimer.Stop(stream);
    }
    float avgGpuMs = totalGpuMs / NUM_GPU_RUNS;

    // --- CPU reference ---
    std::cout << "Computing CPU reference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    computeFullBackpropCPU(batchSize, inputDim, hiddenDim, outputDim, numLayers,
                           h_activations, loss_output, h_weights, h_expected_dw, h_expected_db);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double totalFlops = computeBackwardFlops(batchSize, inputDim, hiddenDim, outputDim, numLayers);
    double gpuTflops = (totalFlops / (avgGpuMs * 1e-3)) / 1e12;

    std::cout << "\n========== Timing Results ==========\n";
    std::cout << "GPU avg time  : " << avgGpuMs << " ms\n";
    std::cout << "CPU time      : " << cpuMs << " ms\n";
    std::cout << "Speedup       : " << cpuMs / avgGpuMs << "x\n";
    std::cout << "GPU TFLOPS    : " << gpuTflops << "\n";

    // --- Correctness check ---
    for (int l = numLayers - 1; l >= 0; --l) {
        std::cout << "\n========== Verification: LAYER " << l << " ==========\n";
        int in_d  = (l == 0) ? inputDim : hiddenDim;
        int out_d = (l == numLayers - 1) ? outputDim : hiddenDim;

        CUDA_CHECK(cudaMemcpy(h_grad_weights[l].data(), h_d_gw_ptrs[l], out_d * in_d * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_grad_biases[l].data(),  h_d_gb_ptrs[l], out_d * sizeof(float), cudaMemcpyDeviceToHost));

        float mse_w = 0.f, max_diff_w = 0.f; int errors_w = 0;
        for (size_t i = 0; i < h_grad_weights[l].size(); ++i) {
            float diff = std::abs(h_grad_weights[l][i] - h_expected_dw[l][i]);
            mse_w += diff * diff;
            if (diff > max_diff_w) max_diff_w = diff;
            if (diff > 0.5f && errors_w < 10) {
                std::cout << "  dW mismatch [" << i << "]: GPU=" << h_grad_weights[l][i] << " CPU=" << h_expected_dw[l][i] << "\n";
                ++errors_w;
            }
        }
        mse_w /= (float)h_grad_weights[l].size();

        float mse_b = 0.f, max_diff_b = 0.f; int errors_b = 0;
        for (size_t i = 0; i < h_grad_biases[l].size(); ++i) {
            float diff = std::abs(h_grad_biases[l][i] - h_expected_db[l][i]);
            mse_b += diff * diff;
            if (diff > max_diff_b) max_diff_b = diff;
            if (diff > 0.5f && errors_b < 10) {
                std::cout << "  db mismatch [" << i << "]: GPU=" << h_grad_biases[l][i] << " CPU=" << h_expected_db[l][i] << "\n";
                ++errors_b;
            }
        }
        mse_b /= (float)h_grad_biases[l].size();

        std::cout << "dW MaxDiff: " << max_diff_w << " | MSE: " << mse_w << " -> "
                  << ((errors_w == 0 && max_diff_w < 0.5f) ? "[SUCCESS]" : "[FAILED]") << "\n";
        std::cout << "db MaxDiff: " << max_diff_b << " | MSE: " << mse_b << " -> "
                  << ((errors_b == 0 && max_diff_b < 0.5f) ? "[SUCCESS]" : "[FAILED]") << "\n";
    }
    std::cout << "====================================\n";

    // --- Cleanup ---
    cudaFree(d_loss_output); cudaFree(d_dx_out_dummy);
    cudaFree(d_act_arr); cudaFree(d_wt_arr);
    cudaFree(d_gw_arr); cudaFree(d_gb_arr);
    for (int l = 0; l < numLayers; ++l) {
        cudaFree(h_d_act_ptrs[l]); cudaFree(h_d_wt_ptrs[l]);
        cudaFree(h_d_gw_ptrs[l]); cudaFree(h_d_gb_ptrs[l]);
    }
}

// ---------------------------------------------------------
// Performance Sweep
// ---------------------------------------------------------
void runPerformanceSweep() {
    std::vector<int> batchSizes = {1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21};
    std::vector<int> dims       = {16, 32, 64, 128};
    std::vector<int> layerCounts= {2, 4, 6, 8, 10};

    std::cout << "\nStarting Performance Sweep...\n\n";
    std::cout << std::left
              << std::setw(12) << "BatchSize"
              << std::setw(8)  << "Dim"
              << std::setw(10) << "Layers"
              << std::setw(15) << "Time (ms)"
              << std::setw(12) << "TFLOPS"
              << "Status" << "\n";
    std::cout << std::string(67, '-') << "\n";

    GpuTimer gpuTimer;
    cudaStream_t stream = 0;

    for (int batchSize : batchSizes) {
        for (int dim : dims) {
            for (int numLayers : layerCounts) {
                bool oom = false;
                MLPOption opt = {dim, dim, dim, numLayers, ACT_RELU};

                half* d_loss = nullptr;
                half* d_dx   = nullptr;
                if (cudaMalloc(&d_loss, batchSize * dim * sizeof(half)) != cudaSuccess) oom = true;
                if (cudaMalloc(&d_dx,   sizeof(half)) != cudaSuccess) oom = true;

                std::vector<half*>  h_act(numLayers, nullptr);
                std::vector<half*>  h_wt(numLayers, nullptr);
                std::vector<float*> h_gw(numLayers, nullptr);
                std::vector<float*> h_gb(numLayers, nullptr);

                for (int l = 0; l < numLayers && !oom; ++l) {
                    if (cudaMalloc(&h_act[l], batchSize * dim * sizeof(half))  != cudaSuccess) oom = true;
                    if (cudaMalloc(&h_wt[l],  dim * dim * sizeof(half))        != cudaSuccess) oom = true;
                    if (cudaMalloc(&h_gw[l],  dim * dim * sizeof(float))       != cudaSuccess) oom = true;
                    if (cudaMalloc(&h_gb[l],  dim * sizeof(float))             != cudaSuccess) oom = true;
                }

                half**  d_act_arr = nullptr;
                half**  d_wt_arr  = nullptr;
                float** d_gw_arr  = nullptr;
                float** d_gb_arr  = nullptr;

                if (!oom) {
                    cudaMalloc(&d_act_arr, numLayers * sizeof(half*));
                    cudaMalloc(&d_wt_arr,  numLayers * sizeof(half*));
                    cudaMalloc(&d_gw_arr,  numLayers * sizeof(float*));
                    cudaMalloc(&d_gb_arr,  numLayers * sizeof(float*));

                    cudaMemcpy(d_act_arr, h_act.data(), numLayers * sizeof(half*),  cudaMemcpyHostToDevice);
                    cudaMemcpy(d_wt_arr,  h_wt.data(),  numLayers * sizeof(half*),  cudaMemcpyHostToDevice);
                    cudaMemcpy(d_gw_arr,  h_gw.data(),  numLayers * sizeof(float*), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_gb_arr,  h_gb.data(),  numLayers * sizeof(float*), cudaMemcpyHostToDevice);
                }

                float avgGpuMs = 0.0f;
                if (!oom) {
                    // Warmup
                    launchNetworkFusionBackwardKernel(&opt, d_loss, d_wt_arr, nullptr,
                        d_act_arr, d_gw_arr, d_gb_arr,
                        d_dx, batchSize, stream);
                    cudaStreamSynchronize(stream);

                    const int RUNS = 10;
                    gpuTimer.Start(stream);
                    for (int r = 0; r < RUNS; ++r) {
                        launchNetworkFusionBackwardKernel(&opt, d_loss, d_wt_arr, nullptr,
                            d_act_arr, d_gw_arr, d_gb_arr,
                            d_dx, batchSize, stream);
                    }
                    avgGpuMs = gpuTimer.Stop(stream) / RUNS;
                } else {
                    cudaGetLastError();
                }

                double flops = computeBackwardFlops(batchSize, dim, dim, dim, numLayers);
                double tflops = (!oom && avgGpuMs > 0.f) ? (flops / (avgGpuMs * 1e-3)) / 1e12 : 0.0;

                std::cout << std::left
                          << std::setw(12) << batchSize
                          << std::setw(8)  << dim
                          << std::setw(10) << numLayers;
                if (oom) {
                    std::cout << std::setw(15) << "N/A" << std::setw(12) << "N/A" << "OOM\n";
                } else {
                    std::cout << std::setw(15) << std::fixed << std::setprecision(3) << avgGpuMs
                              << std::setw(12) << std::fixed << std::setprecision(2) << tflops << "OK\n";
                }

                // Cleanup
                if (d_loss) cudaFree(d_loss);
                if (d_dx)   cudaFree(d_dx);
                for (int l = 0; l < numLayers; ++l) {
                    if (h_act[l]) cudaFree(h_act[l]);
                    if (h_wt[l])  cudaFree(h_wt[l]);
                    if (h_gw[l])  cudaFree(h_gw[l]);
                    if (h_gb[l])  cudaFree(h_gb[l]);
                }
                if (d_act_arr) cudaFree(d_act_arr);
                if (d_wt_arr)  cudaFree(d_wt_arr);
                if (d_gw_arr)  cudaFree(d_gw_arr);
                if (d_gb_arr)  cudaFree(d_gb_arr);
            }
        }
    }
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main() {
    runVerification();
    runPerformanceSweep();
    return 0;
}