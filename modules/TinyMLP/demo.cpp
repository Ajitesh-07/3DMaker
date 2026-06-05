#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "TinyMLP.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            cerr << "CUDA Error: " << cudaGetErrorString(err)                 \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

void loadDatasetFromImage(const string& filename, int& outWidth, int& outHeight, int inputDim, vector<half>& inputs, vector<half>& targets) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    if (!data) {
        cerr << "Failed to load image: " << filename << endl;
        exit(1);
    }
    
    outWidth = width;
    outHeight = height;
    int batchSize = width * height;
    
    inputs.resize(batchSize * inputDim);
    // User dimension doesn't need to be padded. So target size is just 3 for RGB!
    targets.resize(batchSize * 3); 
    
    int num_freqs = (inputDim - 2) / 4; 

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            
            float nx = (x / (float)width) * 2.0f - 1.0f;
            float ny = (y / (float)height) * 2.0f - 1.0f;
            
            int in_idx = i * inputDim;
            inputs[in_idx + 0] = __float2half(nx);
            inputs[in_idx + 1] = __float2half(ny);
            
            int offset = 2;
            for (int f = 0; f < num_freqs; f++) {
                float freq = powf(2.0f, (float)f) * 3.1415926535f;
                inputs[in_idx + offset++] = __float2half(sinf(nx * freq));
                inputs[in_idx + offset++] = __float2half(cosf(nx * freq));
                inputs[in_idx + offset++] = __float2half(sinf(ny * freq));
                inputs[in_idx + offset++] = __float2half(cosf(ny * freq));
            }
            while (offset < inputDim) {
                inputs[in_idx + offset++] = __float2half(0.0f);
            }

            int p_idx = (y * width + x) * 3;
            float r = data[p_idx + 0] / 255.0f;
            float g = data[p_idx + 1] / 255.0f;
            float b = data[p_idx + 2] / 255.0f;
            
            int out_idx = i * 3; // EXACT output dimension!
            targets[out_idx + 0] = __float2half(r);
            targets[out_idx + 1] = __float2half(g);
            targets[out_idx + 2] = __float2half(b);
        }
    }
    stbi_image_free(data);
    cout << "Loaded Image: " << width << "x" << height << " (" << batchSize << " pixels)" << endl;
}

void savePredictedImage(const string& filename, int width, int height, float* d_outputs_float) {
    int batchSize = width * height;
    vector<float> h_outputs(batchSize * 3); // EXACT output dimension!
    CUDA_CHECK(cudaMemcpy(h_outputs.data(), d_outputs_float, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
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
    cout << "--- End-to-End TinyMLP Demo (No Manual Padding) ---" << endl;
    
    int width = 0, height = 0;
    vector<half> h_inputs, h_targets;
    
    MLPOption opt;
    opt.inputDim = 32;   
    opt.hiddenDim = 64; 
    opt.outputDim = 3;   // EXACTLY 3! No manual padding to 8 required here!
    opt.numLayers = 3;
    opt.activationType = ACT_RELU;

    // Load Image
    loadDatasetFromImage("../images/image1.jpg", width, height, opt.inputDim, h_inputs, h_targets);
    int batchSize = width * height;
    
    // Initialize Network (API handles all padding & memory internally!)
    TinyMLP mlp(opt, batchSize, 42);

    half *d_inputs, *d_targets;
    float *d_outputs; // User-allocated output buffer (Design Option 1)
    CUDA_CHECK(cudaMalloc(&d_inputs, batchSize * opt.inputDim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_targets, batchSize * opt.outputDim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_outputs, batchSize * opt.outputDim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(), batchSize * opt.inputDim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), batchSize * opt.outputDim * sizeof(half), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Training Hyperparams
    int numSteps = 10000;
    float lr = 1e-3f;
    float LOSS_SCALE = 65536.0f;
    
    cout << "Starting Training Loop... (" << numSteps << " steps)" << endl;

    auto start_time = chrono::high_resolution_clock::now();

    for (int step = 1; step <= numSteps; step++) {
        bool fetch_loss = (step % 1000 == 0 || step == 1);

        // API Showcase: If we don't need to save the image this step, pass nullptr to skip VRAM copy overhead!
        float* current_d_outputs = fetch_loss ? d_outputs : nullptr;

        mlp.zero_grad(stream);
        mlp.forward(d_inputs, current_d_outputs, batchSize, stream);
        float loss = mlp.calculate_loss_and_grad(d_targets, batchSize, LOSS_SCALE, fetch_loss, stream);
        mlp.backward(batchSize, stream);
        mlp.step(lr, 0.9f, 0.999f, 1e-8f, LOSS_SCALE, stream);

        if (fetch_loss) {
            cout << "Step " << step << " | Loss: " << loss << endl;
            if (step % 1000 == 0) {
                string out_name = "../images/demo_pred_step_" + to_string(step) + ".jpg";
                savePredictedImage(out_name, width, height, current_d_outputs);
            }
        }
    }
    
    cudaStreamSynchronize(stream);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration = end_time - start_time;
    
    cout << "Training Completed. Saving final image..." << endl;
    
    // API Showcase: User passes their own allocated pointer for pure inference
    mlp.inference(d_inputs, d_outputs, batchSize, stream);
    savePredictedImage("../images/demo_pred_final.jpg", width, height, d_outputs);

    float avg_ms = duration.count() / numSteps;
    float throughput = (batchSize * 1000.0f) / avg_ms;

    cout << "\n==============================================" << endl;
    cout << "          PERFORMANCE METRICS" << endl;
    cout << "==============================================" << endl;
    cout << "Total Time:        " << duration.count() / 1000.0f << " seconds" << endl;
    cout << "Avg Time per Step: " << avg_ms << " ms" << endl;
    cout << "Throughput:        " << throughput << " pixels/sec" << endl;
    cout << "==============================================" << endl;
    
    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_outputs); // User is responsible for freeing memory
    cudaStreamDestroy(stream);
    return 0;
}
