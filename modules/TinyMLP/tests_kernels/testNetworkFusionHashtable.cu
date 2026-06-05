#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../TinyMLPHashGrid.h"

// ==========================================
// PERFORMANCE SWEEP
// ==========================================
struct SweepResult { float infer_ms; bool oom; };

SweepResult profileConfiguration(int batchSize, int hiddenDim, int numLayers) {
    SweepResult res = {0.f, false};

    int tableSize     = 1 << 19;
    int numLevels     = 16;
    int featuresLevel = 2;
    int inDim         = numLevels * featuresLevel;

    // vectorDim=4: kernel reads float4 so input stride is 4 floats (x,y,z,w=0)
    const int INPUT_STRIDE = 4;

    MLPGridOptions opt = {
        4,              // vectorDim -- float4 load path
        hiddenDim, hiddenDim, numLayers, 1,
        tableSize, numLevels, 1.3819f, 16, featuresLevel
    };

    float* d_inputs    = nullptr;
    float* d_outputs   = nullptr;
    half*  d_hashtable = nullptr;

    if (cudaMalloc(&d_inputs,    batchSize * INPUT_STRIDE * sizeof(float)) != cudaSuccess)
        { res.oom = true; cudaGetLastError(); return res; }
    if (cudaMalloc(&d_outputs,   batchSize * hiddenDim    * sizeof(float)) != cudaSuccess)
        { res.oom = true; cudaGetLastError(); cudaFree(d_inputs); return res; }
    if (cudaMalloc(&d_hashtable, numLevels * tableSize * featuresLevel * sizeof(half)) != cudaSuccess)
        { res.oom = true; cudaGetLastError(); cudaFree(d_inputs); cudaFree(d_outputs); return res; }

    cudaMemset(d_inputs,    0, batchSize * INPUT_STRIDE * sizeof(float));
    cudaMemset(d_hashtable, 0, numLevels * tableSize * featuresLevel * sizeof(half));

    std::vector<half*>  h_wt(numLayers, nullptr);
    std::vector<float*> h_bi(numLayers, nullptr);

    for (int i = 0; i < numLayers && !res.oom; i++) {
        int K = (i == 0)             ? inDim     : hiddenDim;
        int N = (i == numLayers - 1) ? hiddenDim : hiddenDim;
        if (cudaMalloc(&h_wt[i], N * K * sizeof(half))  != cudaSuccess) res.oom = true;
        if (cudaMalloc(&h_bi[i], N     * sizeof(float)) != cudaSuccess) res.oom = true;
        if (!res.oom) {
            cudaMemset(h_wt[i], 0, N * K * sizeof(half));
            cudaMemset(h_bi[i], 0, N     * sizeof(float));
        }
    }

    half**  d_wt_arr = nullptr;
    float** d_bi_arr = nullptr;
    if (!res.oom) {
        cudaMalloc(&d_wt_arr, numLayers * sizeof(half*));
        cudaMalloc(&d_bi_arr, numLayers * sizeof(float*));
        cudaMemcpy(d_wt_arr, h_wt.data(), numLayers * sizeof(half*),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_bi_arr, h_bi.data(), numLayers * sizeof(float*), cudaMemcpyHostToDevice);
    }

    if (res.oom) { cudaGetLastError(); goto cleanup; }

    {
        int num_runs = (batchSize < 500000) ? 50 : 10;

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // Warmup
        launchNetworkFusionHashTableKernel(&opt, d_hashtable, d_inputs,
                                           d_wt_arr, d_bi_arr, d_outputs,
                                           batchSize, 0);
        cudaDeviceSynchronize();

        // Timing
        cudaEventRecord(start, 0);
        for (int i = 0; i < num_runs; i++)
            launchNetworkFusionHashTableKernel(&opt, d_hashtable, d_inputs,
                                               d_wt_arr, d_bi_arr, d_outputs,
                                               batchSize, 0);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float total; cudaEventElapsedTime(&total, start, stop);
        res.infer_ms = total / num_runs;

        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

cleanup:
    if (d_inputs)    cudaFree(d_inputs);
    if (d_outputs)   cudaFree(d_outputs);
    if (d_hashtable) cudaFree(d_hashtable);
    for (int i = 0; i < numLayers; i++) {
        if (h_wt[i]) cudaFree(h_wt[i]);
        if (h_bi[i]) cudaFree(h_bi[i]);
    }
    if (d_wt_arr) cudaFree(d_wt_arr);
    if (d_bi_arr) cudaFree(d_bi_arr);
    return res;
}

void runParameterSweep() {
    std::cout << "============================================================================================\n";
    std::cout << "              TINYMLP FUSED HASHGRID KERNEL PROFILER -- Inference Performance               \n";
    std::cout << "============================================================================================\n";
    std::cout << std::left
              << std::setw(12) << "Batch"
              << std::setw(8)  << "Dim"
              << std::setw(8)  << "Layers"
              << std::setw(14) << "Infer (ms)"
              << std::setw(14) << "Infer TFLOP"
              << std::setw(20) << "Items / sec"
              << "Status\n";
    std::cout << std::string(85, '-') << "\n";

    std::vector<int> dimensions = {32, 64, 128};

    for (int dim : dimensions) {
        for (int layers = 1; layers <= 10; ++layers) {
            for (int batch = (1 << 16); batch <= (1 << 21); batch <<= 1) {
                auto res = profileConfiguration(batch, dim, layers);

                double total_flops   = 2.0 * batch * dim * dim * layers;
                double infer_tflops  = (!res.oom && res.infer_ms > 0.f)
                                         ? (total_flops / (res.infer_ms * 1e-3)) / 1e12 : 0.0;
                double items_per_sec = (!res.oom && res.infer_ms > 0.f)
                                         ? (batch / (res.infer_ms * 1e-3)) : 0.0;

                std::cout << std::left
                          << std::setw(12) << batch
                          << std::setw(8)  << dim
                          << std::setw(8)  << layers;

                if (res.oom) {
                    std::cout << std::setw(14) << "N/A"
                              << std::setw(14) << "N/A"
                              << std::setw(20) << "N/A" << "OOM\n";
                } else {
                    std::cout << std::setw(14) << std::fixed << std::setprecision(4) << res.infer_ms
                              << std::setw(14) << std::fixed << std::setprecision(2) << infer_tflops
                              << std::setw(20) << std::scientific << std::setprecision(2) << items_per_sec
                              << "OK\n";
                }
            }
            std::cout << std::string(85, '-') << "\n";
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fflush(stdout);                                                     \
            printf("CUDA ERROR at %s:%d\n  call:   %s\n  reason: %s\n",        \
                   __FILE__, __LINE__, #call, cudaGetErrorString(_e));          \
            fflush(stdout);                                                     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static inline float h2f(half h) { return __half2float(h); }
static inline half  f2h(float f) { return __float2half(f); }

// ──────────────────────────────────────────────────────────────────────────────
// Hash function -- must match the GPU kernel exactly
// ──────────────────────────────────────────────────────────────────────────────

static constexpr uint64_t P1 = 1;
static constexpr uint64_t P2 = 2654435761ULL;
static constexpr uint64_t P3 = 805459861ULL;

static inline int hashCoord(int ix, int iy, int iz, int tableSize) {
    uint64_t h = ((uint64_t)(uint32_t)ix * P1)
               ^ ((uint64_t)(uint32_t)iy * P2)
               ^ ((uint64_t)(uint32_t)iz * P3);
    return (int)(h % (uint64_t)tableSize);
}

// ──────────────────────────────────────────────────────────────────────────────
// CPU reference: hash-grid encoding
//   coords is [batchSize x INPUT_STRIDE] -- only x,y,z (indices 0,1,2) used;
//   index 3 (w) is the dummy padding and is ignored.
// ──────────────────────────────────────────────────────────────────────────────

static void cpuHashGridEncoding(
        const MLPGridOptions& opt,
        const std::vector<half>& hashtable,   // [numLevels x tableSize x featuresLevel]
        const float* coords,                  // [batchSize x INPUT_STRIDE]
        int INPUT_STRIDE,
        int batchSize,
        std::vector<float>& encoded)          // out: [batchSize x inputDim]
{
    const int inDim = opt.featuresLevel * opt.numLevels;
    encoded.assign(batchSize * inDim, 0.0f);

    for (int b = 0; b < batchSize; ++b) {
        float x = coords[b * INPUT_STRIDE + 0];
        float y = coords[b * INPUT_STRIDE + 1];
        float z = coords[b * INPUT_STRIDE + 2];
        // coords[b * INPUT_STRIDE + 3] == 0.f, ignored by both CPU and GPU

        for (int lv = 0; lv < opt.numLevels; ++lv) {
            float scale = (float)opt.lowestSize * std::pow(opt.b, (float)lv);
            float sx = x * scale, sy = y * scale, sz = z * scale;

            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);
            int z0 = (int)std::floor(sz);
            float tx = sx - x0, ty = sy - y0, tz = sz - z0;

            int   cx[2] = {x0, x0 + 1};
            int   cy[2] = {y0, y0 + 1};
            int   cz[2] = {z0, z0 + 1};
            float wx[2] = {1.f - tx, tx};
            float wy[2] = {1.f - ty, ty};
            float wz[2] = {1.f - tz, tz};

            for (int f = 0; f < opt.featuresLevel; ++f) {
                float val = 0.f;
                for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    int entry   = hashCoord(cx[dx], cy[dy], cz[dz], opt.tableSize);
                    int flatIdx = lv * (opt.tableSize * opt.featuresLevel)
                                + entry * opt.featuresLevel
                                + f;
                    val += wx[dx] * wy[dy] * wz[dz] * h2f(hashtable[flatIdx]);
                }
                encoded[b * inDim + lv * opt.featuresLevel + f] = val;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CPU reference: MLP forward pass
// ──────────────────────────────────────────────────────────────────────────────

static float applyActivation(float x, int type) {
    switch (type) {
        case 0:  return x > 0.f ? x : 0.f;
        case 1:  return 1.f / (1.f + std::exp(-x));
        default: return x;
    }
}

static void cpuMLP(
        const MLPGridOptions& opt,
        const std::vector<std::vector<half>>&  weights,
        const std::vector<std::vector<float>>& biases,
        const std::vector<float>& encoded,
        int batchSize,
        std::vector<float>& output)
{
    const int inDim = opt.featuresLevel * opt.numLevels;
    output.resize(batchSize * opt.outputDim);

    for (int b = 0; b < batchSize; ++b) {
        std::vector<float> act(encoded.begin() + b * inDim,
                               encoded.begin() + b * inDim + inDim);

        for (int l = 0; l < opt.numLayers; ++l) {
            int inW  = (l == 0)                 ? inDim         : opt.hiddenDim;
            int outW = (l == opt.numLayers - 1) ? opt.outputDim : opt.hiddenDim;
            bool last = (l == opt.numLayers - 1);

            std::vector<float> next(outW, 0.f);
            for (int o = 0; o < outW; ++o) {
                float sum = biases[l][o];
                for (int i = 0; i < inW; ++i)
                    sum += h2f(weights[l][o * inW + i]) * act[i];
                next[o] = last ? sum : applyActivation(sum, opt.activationType);
            }
            act = next;
        }

        for (int o = 0; o < opt.outputDim; ++o)
            output[b * opt.outputDim + o] = act[o];
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main correctness test
// ──────────────────────────────────────────────────────────────────────────────

void testHashGridMLP() {
    MLPGridOptions opt;
    opt.vectorDim      = 4;   // float4 load path: kernel reads (x, y, z, w=0)
    opt.hiddenDim      = 64;
    opt.outputDim      = 16;
    opt.numLayers      = 3;
    opt.activationType = 0;   // ReLU

    opt.tableSize      = 1 << 14;
    opt.numLevels      = 8;
    opt.b              = 1.3819f;
    opt.lowestSize     = 16;
    opt.featuresLevel  = 2;

    // Each input sample is 4 floats: [x, y, z, 0.f]
    const int INPUT_STRIDE = 4;
    const int batchSize    = 256;
    const int inDim        = inputDim(opt);

    printf("=== testHashGridMLP ===\n");
    printf("  vectorDim=%d (float4, w=0 dummy)  hiddenDim=%d  outputDim=%d  numLayers=%d\n",
           opt.vectorDim, opt.hiddenDim, opt.outputDim, opt.numLayers);
    printf("  tableSize=%d  numLevels=%d  b=%.4f  lowestSize=%d  featuresLevel=%d\n",
           opt.tableSize, opt.numLevels, opt.b, opt.lowestSize, opt.featuresLevel);
    printf("  inputDim=%d   batchSize=%d   inputStride=%d\n\n",
           inDim, batchSize, INPUT_STRIDE);
    fflush(stdout);

    // ── Random data ───────────────────────────────────────────────────────────
    std::mt19937 rng(42);
    auto randF = [&](float lo, float hi) -> float {
        return lo + (hi - lo) * (float)rng() / (float)rng.max();
    };
    auto randH = [&](float lo, float hi) -> half { return f2h(randF(lo, hi)); };

    const int tableElems = opt.numLevels * opt.tableSize * opt.featuresLevel;
    std::vector<half> h_hashtable(tableElems);
    for (auto& v : h_hashtable) v = randH(-0.01f, 0.01f);

    // Build padded input [batchSize x 4]: x,y,z random in (0,1), w=0
    std::vector<float> h_inputs(batchSize * INPUT_STRIDE, 0.f);
    for (int b = 0; b < batchSize; ++b) {
        h_inputs[b * INPUT_STRIDE + 0] = randF(0.001f, 0.999f); // x
        h_inputs[b * INPUT_STRIDE + 1] = randF(0.001f, 0.999f); // y
        h_inputs[b * INPUT_STRIDE + 2] = randF(0.001f, 0.999f); // z
        h_inputs[b * INPUT_STRIDE + 3] = 0.f;                   // w -- dummy
    }

    std::vector<std::vector<half>>  h_weights(opt.numLayers);
    std::vector<std::vector<float>> h_biases (opt.numLayers);
    for (int l = 0; l < opt.numLayers; ++l) {
        int inW  = (l == 0)                 ? inDim         : opt.hiddenDim;
        int outW = (l == opt.numLayers - 1) ? opt.outputDim : opt.hiddenDim;
        h_weights[l].resize(outW * inW);
        h_biases [l].resize(outW);
        float scale = std::sqrt(2.f / inW);
        for (auto& v : h_weights[l]) v = randH(-scale, scale);
        for (auto& v : h_biases [l]) v = randF(-0.1f, 0.1f);
    }

    // ── GPU allocations ───────────────────────────────────────────────────────
    half*  d_hashtable;
    float* d_inputs;
    float* d_outputs;

    CUDA_CHECK(cudaMalloc(&d_hashtable, tableElems * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_hashtable, h_hashtable.data(),
                          tableElems * sizeof(half), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_inputs, batchSize * INPUT_STRIDE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(),
                          batchSize * INPUT_STRIDE * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_outputs, batchSize * opt.outputDim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_outputs, 0, batchSize * opt.outputDim * sizeof(float)));

    std::vector<half*>  d_weights_vec(opt.numLayers);
    std::vector<float*> d_biases_vec (opt.numLayers);
    for (int l = 0; l < opt.numLayers; ++l) {
        CUDA_CHECK(cudaMalloc(&d_weights_vec[l], h_weights[l].size() * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_weights_vec[l], h_weights[l].data(),
                              h_weights[l].size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_biases_vec[l], h_biases[l].size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_biases_vec[l], h_biases[l].data(),
                              h_biases[l].size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    half**  d_weights_array;
    float** d_biases_array;
    CUDA_CHECK(cudaMalloc(&d_weights_array, opt.numLayers * sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&d_biases_array,  opt.numLayers * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_weights_array, d_weights_vec.data(),
                          opt.numLayers * sizeof(half*),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases_array,  d_biases_vec.data(),
                          opt.numLayers * sizeof(float*), cudaMemcpyHostToDevice));

    // ── Launch ────────────────────────────────────────────────────────────────
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    launchNetworkFusionHashTableKernel(
        &opt, d_hashtable, d_inputs,
        d_weights_array, d_biases_array,
        d_outputs, batchSize, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> gpu_output(batchSize * opt.outputDim);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_outputs,
                          batchSize * opt.outputDim * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ── CPU reference ─────────────────────────────────────────────────────────
    std::vector<float> encoded;
    cpuHashGridEncoding(opt, h_hashtable, h_inputs.data(), INPUT_STRIDE, batchSize, encoded);

    std::vector<float> cpu_output;
    cpuMLP(opt, h_weights, h_biases, encoded, batchSize, cpu_output);

    // ── Compare ───────────────────────────────────────────────────────────────
    const int    totalElems = batchSize * opt.outputDim;
    const double TOLERANCE  = 1e-2;
    double maxAbsErr = 0.0, sumAbsErr = 0.0, sumSqErr = 0.0;
    int    numFail   = 0;

    for (int i = 0; i < totalElems; ++i) {
        double diff = std::abs((double)gpu_output[i] - (double)cpu_output[i]);
        maxAbsErr = std::max(maxAbsErr, diff);
        sumAbsErr += diff;
        sumSqErr  += diff * diff;
        if (diff > TOLERANCE) ++numFail;
    }

    printf("-- Comparison results (%d elements) --------------------------\n", totalElems);
    printf("  Max absolute error  : %.6e\n", maxAbsErr);
    printf("  Mean absolute error : %.6e\n", sumAbsErr / totalElems);
    printf("  RMSE                : %.6e\n", std::sqrt(sumSqErr / totalElems));
    printf("  Elements > tol(%.0e): %d / %d\n", TOLERANCE, numFail, totalElems);

    if (numFail == 0) {
        printf("\n  PASS -- GPU output matches CPU reference within tolerance.\n");
    } else {
        printf("\n  FAIL -- %d elements exceed tolerance %.1e\n", numFail, TOLERANCE);
        printf("\n  First mismatches (batch x outputDim):\n");
        int shown = 0;
        for (int i = 0; i < totalElems && shown < 10; ++i) {
            double diff = std::abs((double)gpu_output[i] - (double)cpu_output[i]);
            if (diff > TOLERANCE) {
                printf("    [batch=%3d out=%d]  GPU=%.6f  CPU=%.6f  diff=%.2e\n",
                       i / opt.outputDim, i % opt.outputDim,
                       gpu_output[i], cpu_output[i], diff);
                ++shown;
            }
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_hashtable);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_weights_array);
    cudaFree(d_biases_array);
    for (int l = 0; l < opt.numLayers; ++l) {
        cudaFree(d_weights_vec[l]);
        cudaFree(d_biases_vec[l]);
    }
}

// ──────────────────────────────────────────────────────────────────────────────

int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Running on GPU: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    testHashGridMLP();

    std::cout << "\nStarting TinyMLP Fused HashGrid Tensor Core Profiler...\n";
    runParameterSweep();
    return 0;
}
