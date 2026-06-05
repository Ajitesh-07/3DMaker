#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "TinyMLPHashGrid.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#define DEMO_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            cerr << "CUDA Error: " << cudaGetErrorString(err)                 \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

void loadDatasetFromImage(const string& filename, int& outWidth, int& outHeight,
                          vector<float>& inputs, vector<half>& targets) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    if (!data) {
        cerr << "Failed to load image: " << filename << endl;
        exit(1);
    }

    outWidth = width;
    outHeight = height;
    int batchSize = width * height;

    // Hash grid inputs: [x, y, z] floats in [0, 1]
    // We use z=0 for a 2D image fitting task
    inputs.resize(batchSize * 3);
    targets.resize(batchSize * 3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;

            float nx = x / (float)(width - 1);
            float ny = y / (float)(height - 1);

            inputs[i * 3 + 0] = nx;
            inputs[i * 3 + 1] = ny;
            inputs[i * 3 + 2] = 0.0f;

            int p_idx = (y * width + x) * 3;
            targets[i * 3 + 0] = __float2half(data[p_idx + 0] / 255.0f);
            targets[i * 3 + 1] = __float2half(data[p_idx + 1] / 255.0f);
            targets[i * 3 + 2] = __float2half(data[p_idx + 2] / 255.0f);
        }
    }
    stbi_image_free(data);
    cout << "Loaded Image: " << width << "x" << height << " (" << batchSize << " pixels)" << endl;
}

void savePredictedImage(const string& filename, int width, int height, float* d_outputs_float) {
    int batchSize = width * height;
    vector<float> h_outputs(batchSize * 3);
    DEMO_CHECK(cudaMemcpy(h_outputs.data(), d_outputs_float, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    vector<unsigned char> img_data(batchSize * 3);
    for (int i = 0; i < batchSize; i++) {
        float r = fmaxf(0.0f, fminf(1.0f, h_outputs[i * 3 + 0]));
        float g = fmaxf(0.0f, fminf(1.0f, h_outputs[i * 3 + 1]));
        float b = fmaxf(0.0f, fminf(1.0f, h_outputs[i * 3 + 2]));

        img_data[i * 3 + 0] = (unsigned char)(r * 255.0f);
        img_data[i * 3 + 1] = (unsigned char)(g * 255.0f);
        img_data[i * 3 + 2] = (unsigned char)(b * 255.0f);
    }
    stbi_write_jpg(filename.c_str(), width, height, 3, img_data.data(), 100);
}

int main() {
    cout << "--- TinyMLPHashGrid Demo: Image Fitting with Hash Grid Encoding ---" << endl;

    int width = 0, height = 0;
    vector<float> h_inputs;
    vector<half>  h_targets;

    // Hash Grid + MLP Configuration
    MLPGridOptions opt;
    opt.vectorDim     = 3;       // Raw 3D coordinate input
    opt.hiddenDim     = 64;      // Must be power of 2
    opt.outputDim     = 3;       // RGB output
    opt.numLayers     = 3;       // 3-layer MLP after encoding
    opt.activationType = ACT_RELU;

    // Hash grid parameters (instant-ngp style)
    opt.tableSize     = 1 << 19; // 16384 entries per level
    opt.numLevels     = 16;      // 16 resolution levels
    opt.b             = 1.5f;    // Geometric growth factor
    opt.lowestSize    = 16;      // Coarsest grid resolution
    opt.featuresLevel = 2;       // 2 features per level -> inputDim = 32

    // Load Image
    loadDatasetFromImage("../images/image1.jpg", width, height, h_inputs, h_targets);
    int batchSize = width * height;

    // Initialize Network
    TinyMLPHashGrid mlp(opt, batchSize, 42);

    float *d_inputs;
    half  *d_targets;
    float *d_outputs; // User-allocated output buffer (Design Option 1)
    DEMO_CHECK(cudaMalloc(&d_inputs,  batchSize * 3 * sizeof(float)));
    DEMO_CHECK(cudaMalloc(&d_targets, batchSize * 3 * sizeof(half)));
    DEMO_CHECK(cudaMalloc(&d_outputs, batchSize * opt.outputDim * sizeof(float)));
    DEMO_CHECK(cudaMemcpy(d_inputs,  h_inputs.data(),  batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice));
    DEMO_CHECK(cudaMemcpy(d_targets, h_targets.data(), batchSize * 3 * sizeof(half),  cudaMemcpyHostToDevice));

    cudaStream_t stream;
    DEMO_CHECK(cudaStreamCreate(&stream));

    // Training Hyperparams
    int numSteps = 10000;
    float lr = 1e-3f;
    float LOSS_SCALE = 65536.0f;

    cout << "Network inputDim (from hash grid): " << opt.numLevels * opt.featuresLevel << endl;
    cout << "Starting Training Loop... (" << numSteps << " steps)" << endl;

    // CUDA events for per-phase timing
    cudaEvent_t eStart, eAfterFwd, eAfterBwd, eAfterOpt;
    DEMO_CHECK(cudaEventCreate(&eStart));
    DEMO_CHECK(cudaEventCreate(&eAfterFwd));
    DEMO_CHECK(cudaEventCreate(&eAfterBwd));
    DEMO_CHECK(cudaEventCreate(&eAfterOpt));

    float totalFwdMs = 0.0f, totalBwdMs = 0.0f, totalOptMs = 0.0f;

    auto start_time = chrono::high_resolution_clock::now();

    for (int step = 1; step <= numSteps; step++) {
        bool fetch_loss = (step % 1000 == 0 || step == 1);

        // API Showcase: Pass nullptr if we don't need output this step, completely skipping unpad overhead!
        float* current_d_outputs = fetch_loss ? d_outputs : nullptr;

        mlp.zero_grad(stream);

        // --- Forward (includes hash encoding + MLP forward + loss grad) ---
        DEMO_CHECK(cudaEventRecord(eStart, stream));
        mlp.forward(d_inputs, current_d_outputs, batchSize, stream);
        float loss = mlp.calculate_loss_and_grad(d_targets, batchSize, LOSS_SCALE, fetch_loss, stream);
        DEMO_CHECK(cudaEventRecord(eAfterFwd, stream));

        // --- Backward ---
        mlp.backward(batchSize, stream);
        DEMO_CHECK(cudaEventRecord(eAfterBwd, stream));

        // --- Optimizer step ---
        mlp.step(lr, 0.9f, 0.999f, 1e-8f, LOSS_SCALE, stream);
        DEMO_CHECK(cudaEventRecord(eAfterOpt, stream));

        DEMO_CHECK(cudaEventSynchronize(eAfterOpt));

        float fwdMs = 0.0f, bwdMs = 0.0f, optMs = 0.0f;
        DEMO_CHECK(cudaEventElapsedTime(&fwdMs, eStart, eAfterFwd));
        DEMO_CHECK(cudaEventElapsedTime(&bwdMs, eAfterFwd, eAfterBwd));
        DEMO_CHECK(cudaEventElapsedTime(&optMs, eAfterBwd, eAfterOpt));
        totalFwdMs += fwdMs;
        totalBwdMs += bwdMs;
        totalOptMs += optMs;

        if (fetch_loss) {
            cout << "Step " << step << " | Loss: " << loss << endl;
            if (step % 2000 == 0) {
                string out_name = "../images/hashgrid_pred_step_" + to_string(step) + ".jpg";
                savePredictedImage(out_name, width, height, current_d_outputs);
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration = end_time - start_time;

    cout << "Training Completed. Saving final image..." << endl;
    
    // API Showcase: User passes their own output pointer for inference
    mlp.inference(d_inputs, d_outputs, batchSize, stream);
    savePredictedImage("../images/hashgrid_pred_final.jpg", width, height, d_outputs);

    float avgFwdMs = totalFwdMs / numSteps;
    float avgBwdMs = totalBwdMs / numSteps;
    float avgOptMs = totalOptMs / numSteps;
    float avgStepMs = avgFwdMs + avgBwdMs + avgOptMs;
    float totalGpuSec = (totalFwdMs + totalBwdMs + totalOptMs) / 1000.0f;
    float throughput = (batchSize * 1000.0f) / avgStepMs;

    cout << "\n==============================================" << endl;
    cout << "          PERFORMANCE METRICS" << endl;
    cout << "==============================================" << endl;
    cout << "Total Wall Time:   " << duration.count() / 1000.0f << " seconds" << endl;
    cout << "Total GPU Time:    " << totalGpuSec << " seconds" << endl;
    cout << "----------------------------------------------" << endl;
    cout << "Avg Forward/Step:  " << avgFwdMs << " ms" << endl;
    cout << "Avg Backward/Step: " << avgBwdMs << " ms" << endl;
    cout << "Avg Optim/Step:    " << avgOptMs << " ms" << endl;
    cout << "Avg Total/Step:    " << avgStepMs << " ms" << endl;
    cout << "----------------------------------------------" << endl;
    cout << "Throughput:        " << throughput << " pixels/sec" << endl;
    cout << "==============================================" << endl;

    DEMO_CHECK(cudaEventDestroy(eStart));
    DEMO_CHECK(cudaEventDestroy(eAfterFwd));
    DEMO_CHECK(cudaEventDestroy(eAfterBwd));
    DEMO_CHECK(cudaEventDestroy(eAfterOpt));

    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_outputs); // User is responsible for freeing output buffer
    cudaStreamDestroy(stream);
    return 0;
}
