#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../TinyMLPHashGrid.h"

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
// CPU Reference (Hash Grid Helpers)
// ---------------------------------------------------------
uint32_t hash_coords_cpu(int cx, int cy, int cz, int T) {
    return ((static_cast<uint32_t>(cx) * 1U) ^ 
            (static_cast<uint32_t>(cy) * 2654435761U) ^ 
            (static_cast<uint32_t>(cz) * 805459861U)) & static_cast<uint32_t>(T - 1);
}

uint32_t dense_index_cpu(int cx, int cy, int cz, int N_l) {
    return cx + cy * (N_l + 1) + cz * (N_l + 1) * (N_l + 1);
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
    std::vector<std::vector<float>>& expected_db,
    std::vector<half>& expected_dx_out) 
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
                    next_dz[b * in_d + h] = (orig_act > 0.0f) ? sum : 0.0f;
                }
            }
            current_dz = next_dz;
        } else {
            for (int b = 0; b < batchSize; ++b) {
                for (int h = 0; h < in_d; ++h) {
                    float sum = 0.0f;
                    for (int o = 0; o < out_d; ++o) {
                        sum += current_dz[b * out_d + o] * __half2float(weights[layer][o * in_d + h]);
                    }
                    expected_dx_out[b * in_d + h] = __float2half(sum);
                }
            }
        }
    }
}

void computeHashGridBackpropCPU(
    int batchSize, const MLPGridOptions& opt,
    const std::vector<float>& inputs,
    const std::vector<half>& d_dx_out,
    std::vector<float>& expected_hash_grads) 
{
    float inner_term = (std::cbrt(static_cast<float>(opt.tableSize)) - 1.0f) / opt.lowestSize;
    float continuous_level = std::log2(inner_term) / std::log2(opt.b);
    int denseLevelStart = static_cast<int>(std::floor(continuous_level)) + 1;
    denseLevelStart = std::max(0, std::min(denseLevelStart, opt.numLevels));

    for (int b_idx = 0; b_idx < batchSize; ++b_idx) {
        float x = inputs[b_idx * 4 + 0];
        float y = inputs[b_idx * 4 + 1];
        float z = inputs[b_idx * 4 + 2];

        for (int l = 0; l < opt.numLevels; ++l) {
            float N_l_float = opt.lowestSize * std::pow(opt.b, (float)l);
            int N_l = static_cast<int>(N_l_float);

            float x_l = x * N_l_float;
            float y_l = y * N_l_float;
            float z_l = z * N_l_float;

            int x0 = static_cast<int>(x_l);
            int y0 = static_cast<int>(y_l);
            int z0 = static_cast<int>(z_l);

            int x1 = std::min(x0 + 1, N_l);
            int y1 = std::min(y0 + 1, N_l);
            int z1 = std::min(z0 + 1, N_l);

            x0 = std::max(0, std::min(x0, N_l));
            y0 = std::max(0, std::min(y0, N_l));
            z0 = std::max(0, std::min(z0, N_l));

            float fx = x_l - std::floor(x_l);
            float fy = y_l - std::floor(y_l);
            float fz = z_l - std::floor(z_l);

            bool isDense = l < denseLevelStart;

            auto get_table_idx = [&](int cx, int cy, int cz) -> uint32_t {
                uint32_t table_index = isDense ? dense_index_cpu(cx, cy, cz, N_l)
                                               : hash_coords_cpu(cx, cy, cz, opt.tableSize);
                return (l * opt.tableSize + table_index) * opt.featuresLevel;
            };

            float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
            float w100 = fx          * (1.0f - fy) * (1.0f - fz);
            float w010 = (1.0f - fx) * fy          * (1.0f - fz);
            float w110 = fx          * fy          * (1.0f - fz);
            float w001 = (1.0f - fx) * (1.0f - fy) * fz;
            float w101 = fx          * (1.0f - fy) * fz;
            float w011 = (1.0f - fx) * fy          * fz;
            float w111 = fx          * fy          * fz;

            int grad_offset = b_idx * opt.numLevels * opt.featuresLevel + (l * opt.featuresLevel);
            float grad_x = __half2float(d_dx_out[grad_offset]);
            float grad_y = __half2float(d_dx_out[grad_offset + 1]);

            auto add_grad = [&](int cx, int cy, int cz, float weight) {
                uint32_t base_idx = get_table_idx(cx, cy, cz);
                expected_hash_grads[base_idx + 0] += grad_x * weight;
                expected_hash_grads[base_idx + 1] += grad_y * weight;
            };

            add_grad(x0, y0, z0, w000);
            add_grad(x1, y0, z0, w100);
            add_grad(x0, y1, z0, w010);
            add_grad(x1, y1, z0, w110);
            add_grad(x0, y0, z1, w001);
            add_grad(x1, y0, z1, w101);
            add_grad(x0, y1, z1, w011);
            add_grad(x1, y1, z1, w111);
        }
    }
}

// ---------------------------------------------------------
// GPU Timer & FLOPs
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
    int hiddenDim = 64;
    int outputDim = 8;
    int numLayers = 4;

    MLPGridOptions opt;
    opt.vectorDim = 3;
    opt.hiddenDim = hiddenDim;
    opt.outputDim = outputDim;
    opt.numLayers = numLayers;
    opt.activationType = 1; 
    opt.tableSize = 1 << 19;
    opt.numLevels = 16;
    opt.b = 1.38f;
    opt.lowestSize = 16;
    opt.featuresLevel = 2;

    int inputDim = opt.numLevels * opt.featuresLevel; 

    printDeviceInfo();

    std::vector<half> loss_output(batchSize * outputDim);
    std::vector<std::vector<half>>  h_activations(numLayers);
    std::vector<std::vector<half>>  h_weights(numLayers);
    std::vector<std::vector<float>> h_grad_weights(numLayers);
    std::vector<std::vector<float>> h_expected_dw(numLayers);
    std::vector<std::vector<float>> h_grad_biases(numLayers);
    std::vector<std::vector<float>> h_expected_db(numLayers);

    std::vector<float> h_inputs(batchSize * 4); 
    std::vector<half>  h_expected_dx_out(batchSize * inputDim);
    std::vector<float> h_gpu_hash_grads(opt.numLevels * opt.tableSize * opt.featuresLevel, 0.0f);
    std::vector<float> h_expected_hash_grads(opt.numLevels * opt.tableSize * opt.featuresLevel, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::uniform_real_distribution<float> pos_dist(0.f, 1.f); 

    for (auto& v : loss_output) v = __float2half(dist(gen));
    for (int i = 0; i < batchSize; ++i) {
        h_inputs[i * 4 + 0] = pos_dist(gen);
        h_inputs[i * 4 + 1] = pos_dist(gen);
        h_inputs[i * 4 + 2] = pos_dist(gen);
        h_inputs[i * 4 + 3] = 0.0f; 
    }

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

    half* d_loss_output;
    float* d_inputs;
    float* d_hashtable_grads;
    half* d_dx_out;

    CUDA_CHECK(cudaMalloc(&d_loss_output, batchSize * outputDim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_inputs, batchSize * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hashtable_grads, h_gpu_hash_grads.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_out, batchSize * inputDim * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_loss_output, loss_output.data(), batchSize * outputDim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(), batchSize * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hashtable_grads, 0, h_gpu_hash_grads.size() * sizeof(float)));

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

    cudaStream_t stream = 0;

    const int NUM_GPU_RUNS = 10;
    GpuTimer gpuTimer;
    float totalGpuMs = 0.f;

    std::cout << "Timing FUSED GPU Backward Kernel (" << NUM_GPU_RUNS << " runs)...\n";
    for (int r = 0; r < NUM_GPU_RUNS; ++r) {
        for (int l = 0; l < numLayers; ++l) {
            int in_d  = (l == 0) ? inputDim : hiddenDim;
            int out_d = (l == numLayers - 1) ? outputDim : hiddenDim;
            CUDA_CHECK(cudaMemset(h_d_gw_ptrs[l], 0, out_d * in_d * sizeof(float)));
            CUDA_CHECK(cudaMemset(h_d_gb_ptrs[l], 0, out_d * sizeof(float)));
        }
        CUDA_CHECK(cudaMemset(d_hashtable_grads, 0, h_gpu_hash_grads.size() * sizeof(float)));

        gpuTimer.Start(stream);
        
        launchNetworkFusionHashTableBackwardKernel(
            &opt, d_loss_output, d_wt_arr, nullptr,
            d_act_arr, d_inputs, d_gw_arr, d_gb_arr, d_hashtable_grads,
            d_dx_out, batchSize, stream);
            
        totalGpuMs += gpuTimer.Stop(stream);
    }
    float avgGpuMs = totalGpuMs / NUM_GPU_RUNS;

    std::cout << "Computing CPU reference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    
    computeFullBackpropCPU(batchSize, inputDim, hiddenDim, outputDim, numLayers,
                           h_activations, loss_output, h_weights, h_expected_dw, h_expected_db, h_expected_dx_out);
                           
    computeHashGridBackpropCPU(batchSize, opt, h_inputs, h_expected_dx_out, h_expected_hash_grads);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double totalFlops = computeBackwardFlops(batchSize, inputDim, hiddenDim, outputDim, numLayers);
    double gpuTflops = (totalFlops / (avgGpuMs * 1e-3)) / 1e12;

    std::cout << "\n========== Timing Results ==========\n";
    std::cout << "GPU avg time  : " << avgGpuMs << " ms\n";
    std::cout << "CPU time      : " << cpuMs << " ms\n";
    std::cout << "Speedup       : " << cpuMs / avgGpuMs << "x\n";
    std::cout << "GPU TFLOPS    : " << gpuTflops << " (MLP Math only)\n";

    std::cout << "\n========== Verification: HASH GRID ==========\n";
    CUDA_CHECK(cudaMemcpy(h_gpu_hash_grads.data(), d_hashtable_grads, h_gpu_hash_grads.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    float mse_hash = 0.f, max_diff_hash = 0.f; int errors_hash = 0;
    for (size_t i = 0; i < h_gpu_hash_grads.size(); ++i) {
        float diff = std::abs(h_gpu_hash_grads[i] - h_expected_hash_grads[i]);
        mse_hash += diff * diff;
        if (diff > max_diff_hash) max_diff_hash = diff;
        if (diff > 0.5f && errors_hash < 5) {
            std::cout << "  HashGrad mismatch [" << i << "]: GPU=" << h_gpu_hash_grads[i] << " CPU=" << h_expected_hash_grads[i] << "\n";
            ++errors_hash;
        }
    }
    mse_hash /= (float)h_gpu_hash_grads.size();
    std::cout << "HashGrid MaxDiff: " << max_diff_hash << " | MSE: " << mse_hash << " -> "
              << ((errors_hash == 0 && max_diff_hash < 0.5f) ? "[SUCCESS]" : "[FAILED]") << "\n";

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
            if (diff > 0.5f && errors_w < 5) {
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
            if (diff > 0.5f && errors_b < 5) {
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

    cudaFree(d_loss_output); cudaFree(d_dx_out); cudaFree(d_inputs); cudaFree(d_hashtable_grads);
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
    std::vector<int> dims       = {32, 64}; 
    std::vector<int> layerCounts= {2, 4, 6, 8};

    std::cout << "\nStarting Performance Sweep (Fused Pipeline)...\n\n";
    std::cout << std::left
              << std::setw(12) << "BatchSize"
              << std::setw(8)  << "Dim"
              << std::setw(10) << "Layers"
              << std::setw(18) << "Fused Time (ms)"
              << std::setw(10) << "TFLOPS"
              << "Status" << "\n";
    std::cout << std::string(67, '-') << "\n";

    cudaStream_t stream = 0;

    for (int batchSize : batchSizes) {
        for (int dim : dims) {
            for (int numLayers : layerCounts) {
                bool oom = false;
                
                MLPGridOptions opt = {3, dim, dim, numLayers, 1, 1<<19, 16, 1.38f, 16, 2};

                half* d_loss = nullptr;
                half* d_dx   = nullptr;
                float* d_inputs = nullptr;
                float* d_hash_grads = nullptr;
                
                if (cudaMalloc(&d_loss, batchSize * dim * sizeof(half)) != cudaSuccess) oom = true;
                if (cudaMalloc(&d_dx,   batchSize * dim * sizeof(half)) != cudaSuccess) oom = true;
                if (cudaMalloc(&d_inputs, batchSize * 4 * sizeof(float)) != cudaSuccess) oom = true;
                if (cudaMalloc(&d_hash_grads, opt.numLevels * opt.tableSize * opt.featuresLevel * sizeof(float)) != cudaSuccess) oom = true;

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

                float avgTotalMs = 0.0f;

                if (!oom) {
                    // Warmup
                    launchNetworkFusionHashTableBackwardKernel(&opt, d_loss, d_wt_arr, nullptr,
                        d_act_arr, d_inputs, d_gw_arr, d_gb_arr, d_hash_grads,
                        d_dx, batchSize, stream);
                    cudaStreamSynchronize(stream);

                    // Create fine-grained events
                    cudaEvent_t e_start, e_end;
                    cudaEventCreate(&e_start);
                    cudaEventCreate(&e_end);

                    const int RUNS = 10;
                    float totalFusedMs = 0.0f;
                    
                    for (int r = 0; r < RUNS; ++r) {
                        cudaEventRecord(e_start, stream);
                        
                        launchNetworkFusionHashTableBackwardKernel(&opt, d_loss, d_wt_arr, nullptr,
                            d_act_arr, d_inputs, d_gw_arr, d_gb_arr, d_hash_grads,
                            d_dx, batchSize, stream);
                            
                        cudaEventRecord(e_end, stream);
                        cudaEventSynchronize(e_end);

                        float msFused = 0;
                        cudaEventElapsedTime(&msFused, e_start, e_end);
                        totalFusedMs += msFused;
                    }

                    cudaEventDestroy(e_start);
                    cudaEventDestroy(e_end);

                    avgTotalMs = totalFusedMs / RUNS;
                } else {
                    cudaGetLastError(); // Clear OOM error context
                }

                double flops = computeBackwardFlops(batchSize, dim, dim, dim, numLayers);
                double tflops = (!oom && avgTotalMs > 0.f) ? (flops / (avgTotalMs * 1e-3)) / 1e12 : 0.0;

                std::cout << std::left
                          << std::setw(12) << batchSize
                          << std::setw(8)  << dim
                          << std::setw(10) << numLayers;
                          
                if (oom) {
                    std::cout << std::setw(18) << "N/A" 
                              << std::setw(10) << "N/A" << "OOM\n";
                } else {
                    std::cout << std::setw(18) << std::fixed << std::setprecision(3) << avgTotalMs
                              << std::setw(10) << std::fixed << std::setprecision(2) << tflops 
                              << "OK\n";
                }

                if (d_loss) cudaFree(d_loss);
                if (d_dx)   cudaFree(d_dx);
                if (d_inputs) cudaFree(d_inputs);
                if (d_hash_grads) cudaFree(d_hash_grads);
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