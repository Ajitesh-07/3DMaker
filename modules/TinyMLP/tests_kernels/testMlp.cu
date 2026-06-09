#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../TinyMLP.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h" // Since you put it in the images folder
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

using namespace std;

#define LOSS_SCALE 65336.0f

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            cerr << "CUDA Error: " << cudaGetErrorString(err)                 \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void floatToHalf(const float* f, half* h, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) h[idx] = __float2half(f[idx]);
}

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
    targets.resize(batchSize * 8); // outputDim is padded to 8
    
    int num_freqs = (inputDim - 2) / 4; // Frequencies per coordinate

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            
            // Normalize to [-1, 1]
            float nx = (x / (float)width) * 2.0f - 1.0f;
            float ny = (y / (float)height) * 2.0f - 1.0f;
            
            int in_idx = i * inputDim;
            inputs[in_idx + 0] = __float2half(nx);
            inputs[in_idx + 1] = __float2half(ny);
            
            // Positional Encoding (Fourier Features)
            int offset = 2;
            for (int f = 0; f < num_freqs; f++) {
                float freq = powf(2.0f, (float)f) * 3.1415926535f;
                inputs[in_idx + offset++] = __float2half(sinf(nx * freq));
                inputs[in_idx + offset++] = __float2half(cosf(nx * freq));
                inputs[in_idx + offset++] = __float2half(sinf(ny * freq));
                inputs[in_idx + offset++] = __float2half(cosf(ny * freq));
            }
            
            // Pad remaining
            while (offset < inputDim) {
                inputs[in_idx + offset++] = __float2half(0.0f);
            }

            int p_idx = (y * width + x) * 3;
            float r = data[p_idx + 0] / 255.0f;
            float g = data[p_idx + 1] / 255.0f;
            float b = data[p_idx + 2] / 255.0f;
            
            int out_idx = i * 8;
            targets[out_idx + 0] = __float2half(r);
            targets[out_idx + 1] = __float2half(g);
            targets[out_idx + 2] = __float2half(b);
            for (int p = 3; p < 8; p++) {
                targets[out_idx + p] = __float2half(0.0f);
            }
        }
    }
    stbi_image_free(data);
    cout << "Loaded Image: " << width << "x" << height << " (" << batchSize << " pixels)" << endl;
}

void savePredictedImage(const string& filename, int width, int height, float* d_outputs_float) {
    int batchSize = width * height;
    vector<float> h_outputs(batchSize * 8); // outputDim is 8
    CUDA_CHECK(cudaMemcpy(h_outputs.data(), d_outputs_float, batchSize * 8 * sizeof(float), cudaMemcpyDeviceToHost));
    
    vector<unsigned char> img_data(batchSize * 3);
    for (int i = 0; i < batchSize; i++) {
        float r = fmaxf(0.0f, fminf(1.0f, h_outputs[i * 8 + 0]));
        float g = fmaxf(0.0f, fminf(1.0f, h_outputs[i * 8 + 1]));
        float b = fmaxf(0.0f, fminf(1.0f, h_outputs[i * 8 + 2]));
        
        img_data[i * 3 + 0] = (unsigned char)(r * 255.0f);
        img_data[i * 3 + 1] = (unsigned char)(g * 255.0f);
        img_data[i * 3 + 2] = (unsigned char)(b * 255.0f);
    }
    
    stbi_write_jpg(filename.c_str(), width, height, 3, img_data.data(), 100);
}

float** createDevicePointerArrayFloat(int numLayers, const MLPOption& opt, bool isWeight, mt19937& gen) {
    vector<float*> host_ptr_array(numLayers);

    for (int l = 0; l < numLayers; l++) {
        int in_d = (l == 0) ? opt.inputDim : opt.hiddenDim;
        int out_d = (l == numLayers - 1) ? opt.outputDim : opt.hiddenDim;
        int elements = isWeight ? (in_d * out_d) : out_d;

        // PyTorch Kaiming/LeCun Uniform Initialization
        float bound = 1.0f / sqrtf((float)in_d);
        uniform_real_distribution<float> dist(-bound, bound);

        vector<float> host_data(elements);
        for (int i = 0; i < elements; i++) {
            host_data[i] = dist(gen);
        }

        float* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_arr, host_data.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        host_ptr_array[l] = d_arr;
    }
    float** d_ptr_array_dev;
    CUDA_CHECK(cudaMalloc(&d_ptr_array_dev, numLayers * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_ptr_array_dev, host_ptr_array.data(), numLayers * sizeof(float*), cudaMemcpyHostToDevice));
    return d_ptr_array_dev;
}

float** createDevicePointerArrayFloatZero(int numLayers, const MLPOption& opt, bool isWeight) {
    vector<float*> host_ptr_array(numLayers);

    for (int l = 0; l < numLayers; l++) {
        int in_d = (l == 0) ? opt.inputDim : opt.hiddenDim;
        int out_d = (l == numLayers - 1) ? opt.outputDim : opt.hiddenDim;
        int elements = isWeight ? (in_d * out_d) : out_d;

        float* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_arr, 0, elements * sizeof(float)));
        host_ptr_array[l] = d_arr;
    }
    float** d_ptr_array_dev;
    CUDA_CHECK(cudaMalloc(&d_ptr_array_dev, numLayers * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_ptr_array_dev, host_ptr_array.data(), numLayers * sizeof(float*), cudaMemcpyHostToDevice));
    return d_ptr_array_dev;
}

half** createDeviceHalfPointerArrayForWeights(int numLayers, const MLPOption& opt, float** d_float_weights) {
    vector<half*> host_ptr_array(numLayers);
    vector<float*> host_float_ptrs(numLayers);
    CUDA_CHECK(cudaMemcpy(host_float_ptrs.data(), d_float_weights, numLayers * sizeof(float*), cudaMemcpyDeviceToHost));

    for (int l = 0; l < numLayers; l++) {
        int in_d = (l == 0) ? opt.inputDim : opt.hiddenDim;
        int out_d = (l == numLayers - 1) ? opt.outputDim : opt.hiddenDim;
        int elements = in_d * out_d;

        half* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, elements * sizeof(half)));
        
        // Convert initial weights to half
        int threadsPerBlock = 256;
        int blocks = (elements + threadsPerBlock - 1) / threadsPerBlock;
        floatToHalf<<<blocks, threadsPerBlock>>>(host_float_ptrs[l], d_arr, elements);
        
        host_ptr_array[l] = d_arr;
    }
    half** d_ptr_array_dev;
    CUDA_CHECK(cudaMalloc(&d_ptr_array_dev, numLayers * sizeof(half*)));
    CUDA_CHECK(cudaMemcpy(d_ptr_array_dev, host_ptr_array.data(), numLayers * sizeof(half*), cudaMemcpyHostToDevice));
    return d_ptr_array_dev;
}

half** createDeviceActivationsArray(int numLayers, const MLPOption& opt, int batchSize, half* d_inputs) {
    vector<half*> host_ptr_array(numLayers);

    // Index 0 is the input to the network
    host_ptr_array[0] = d_inputs;

    // Index 1 to numLayers-1 are the outputs of the hidden layers
    for (int l = 1; l < numLayers; l++) {
        int out_d = opt.hiddenDim; // Only hidden layers store activations
        int elements = batchSize * out_d;

        half* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, elements * sizeof(half)));
        CUDA_CHECK(cudaMemset(d_arr, 0, elements * sizeof(half)));
        host_ptr_array[l] = d_arr;
    }
    half** d_ptr_array_dev;
    CUDA_CHECK(cudaMalloc(&d_ptr_array_dev, numLayers * sizeof(half*)));
    CUDA_CHECK(cudaMemcpy(d_ptr_array_dev, host_ptr_array.data(), numLayers * sizeof(half*), cudaMemcpyHostToDevice));
    return d_ptr_array_dev;
}

void runPerformanceSweep() {
    cout << "\n=========================================================================================" << endl;
    cout << "                             TINYMLP PERFORMANCE SWEEP" << endl;
    cout << "=========================================================================================" << endl;
    cout << "Dim\tLayers\tBatch Size\tTime/Step (ms)\tThroughput (GB/s)\tThroughput (items/sec)" << endl;
    cout << "-----------------------------------------------------------------------------------------" << endl;

    int dims[] = {32, 64, 128};
    int batchSizes[] = {1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20};
    int numSteps = 50; // Warmup + Profile steps
    
    mt19937 gen(42);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    for (int dim : dims) {
        for (int layers = 2; layers <= 10; layers++) {
            for (int batchSize : batchSizes) {
                
                MLPOption opt;
                opt.inputDim = dim;
                opt.hiddenDim = dim;
                opt.outputDim = dim;
                opt.numLayers = layers;

                // 1. Allocate Dataset (random data for speed test)
                half *d_inputs, *d_targets;
                CUDA_CHECK(cudaMalloc(&d_inputs, batchSize * opt.inputDim * sizeof(half)));
                CUDA_CHECK(cudaMalloc(&d_targets, batchSize * opt.outputDim * sizeof(half)));

                // 2. Allocate Network Parameters
                float** d_master_weights = createDevicePointerArrayFloat(opt.numLayers, opt, true, gen);
                float** d_master_biases = createDevicePointerArrayFloat(opt.numLayers, opt, false, gen);
                half** d_fwd_weights = createDeviceHalfPointerArrayForWeights(opt.numLayers, opt, d_master_weights);
                
                // 3. Allocate Optimizer States
                float** d_w_m = createDevicePointerArrayFloatZero(opt.numLayers, opt, true);
                float** d_w_v = createDevicePointerArrayFloatZero(opt.numLayers, opt, true);
                float** d_w_grad = createDevicePointerArrayFloatZero(opt.numLayers, opt, true);
                
                float** d_b_m = createDevicePointerArrayFloatZero(opt.numLayers, opt, false);
                float** d_b_v = createDevicePointerArrayFloatZero(opt.numLayers, opt, false);
                float** d_b_grad = createDevicePointerArrayFloatZero(opt.numLayers, opt, false);

                // 4. Allocate Forward/Backward Buffers
                float* d_outputs_float;
                half* d_dLoss;
                float* d_total_loss;
                half** d_activations = createDeviceActivationsArray(opt.numLayers, opt, batchSize, d_inputs);
                
                CUDA_CHECK(cudaMalloc(&d_outputs_float, batchSize * opt.outputDim * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_dLoss, batchSize * opt.outputDim * sizeof(half)));
                CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(float)));
                
                half* d_dx_out;
                CUDA_CHECK(cudaMalloc(&d_dx_out, batchSize * opt.inputDim * sizeof(half)));

                // Profile loop
                cudaEventRecord(start_event, stream);

                for (int step = 1; step <= numSteps; step++) {
                    CUDA_CHECK(cudaMemsetAsync(d_total_loss, 0, sizeof(float), stream));

                    launchNetworkFusionGradKernel(&opt, d_inputs, d_activations, d_fwd_weights, d_master_biases, d_outputs_float, batchSize, stream);

                    launchMSELossGrad(d_outputs_float, d_targets, d_dLoss, d_total_loss, batchSize, opt.outputDim, opt.outputDim, LOSS_SCALE, stream, OUT_ACT_NONE);

                    launchNetworkFusionBackwardKernel(&opt, d_dLoss, d_fwd_weights, d_master_biases, d_activations, d_w_grad, d_b_grad, d_dx_out, batchSize, stream);

                    float bias_correction1 = 1.0f - powf(0.9f, (float)step);
                    float bias_correction2 = 1.0f - powf(0.999f, (float)step);
                    float inv_loss_scale = 1.0f / LOSS_SCALE;

                    launchAdamWeightsOptim(&opt, d_master_weights, d_fwd_weights, d_w_grad, d_w_m, d_w_v, 1e-4f, 0.9f, 0.999f, 1e-8f, bias_correction1, bias_correction2, inv_loss_scale, stream);
                    launchAdamBiasOptim(&opt, d_master_biases, d_b_grad, d_b_m, d_b_v, 1e-4f, 0.9f, 0.999f, 1e-8f, bias_correction1, bias_correction2, inv_loss_scale, stream);
                }

                cudaEventRecord(stop_event, stream);
                cudaEventSynchronize(stop_event);

                float ms = 0;
                cudaEventElapsedTime(&ms, start_event, stop_event);
                float ms_per_step = ms / numSteps;

                // Memory operations per step (approximate GB/s)
                // Just calculating raw items throughput for simplicity, but can add memory BW if needed
                float throughput_items = (batchSize * 1000.0f) / ms_per_step;
                
                // Calculate bytes read/written to estimate memory bandwidth (very rough)
                float bytes_per_step = (float)batchSize * (opt.inputDim * 2 + opt.outputDim * 2 * 3) + 
                                       (float)batchSize * opt.hiddenDim * 2 * opt.numLayers * 2; // VERY approx
                float gbps = (bytes_per_step * 1000.0f) / (ms_per_step * 1e9f);

                cout << dim << "\t" << layers << "\t" << batchSize << "\t\t" << ms_per_step << "\t\t" << gbps << "\t\t" << throughput_items << endl;

                // Free memory
                cudaFree(d_inputs); cudaFree(d_targets);
                cudaFree(d_outputs_float);
                cudaFree(d_dLoss); cudaFree(d_total_loss); cudaFree(d_dx_out);
                
                // Free pointer arrays (need host cleanup)
                auto freeArray = [](float** d_arr, int count) {
                    float* h_arr[20];
                    cudaMemcpy(h_arr, d_arr, count * sizeof(float*), cudaMemcpyDeviceToHost);
                    for(int i=0; i<count; i++) cudaFree(h_arr[i]);
                    cudaFree(d_arr);
                };
                auto freeHalfArray = [](half** d_arr, int count) {
                    half* h_arr[20];
                    cudaMemcpy(h_arr, d_arr, count * sizeof(half*), cudaMemcpyDeviceToHost);
                    for(int i=0; i<count; i++) {
                        // For activations, index 0 is d_inputs which is freed separately!
                        if (i > 0 || d_arr != d_arr /*hack to not free d_inputs from activation array since we freed it above*/) {
                             cudaFree(h_arr[i]);
                        }
                    }
                    cudaFree(d_arr);
                };

                freeArray(d_master_weights, layers); freeArray(d_master_biases, layers);
                freeArray(d_w_m, layers); freeArray(d_w_v, layers); freeArray(d_w_grad, layers);
                freeArray(d_b_m, layers); freeArray(d_b_v, layers); freeArray(d_b_grad, layers);
                
                // Fwd weights and activations
                half* h_fwd_arr[20]; cudaMemcpy(h_fwd_arr, d_fwd_weights, layers * sizeof(half*), cudaMemcpyDeviceToHost);
                for(int i=0; i<layers; i++) cudaFree(h_fwd_arr[i]); cudaFree(d_fwd_weights);

                half* h_act_arr[20]; cudaMemcpy(h_act_arr, d_activations, layers * sizeof(half*), cudaMemcpyDeviceToHost);
                for(int i=1; i<layers; i++) cudaFree(h_act_arr[i]); cudaFree(d_activations);
            }
        }
    }
    
    cudaStreamDestroy(stream);
}

int main() {
    cout << "--- End-to-End TinyMLP Training Test ---" << endl;
    
    int width = 0, height = 0;
    vector<half> h_inputs, h_targets;
    
    MLPOption opt;
    // You can adjust these for image learning
    opt.inputDim = 32;   // 32 handles (X, Y) + 14 Fourier frequencies (7 for X, 7 for Y)
    opt.hiddenDim = 32; 
    opt.outputDim = 8;   // Padded to 8 for alignment
    opt.numLayers = 3;
    
    mt19937 gen(42);
    
    // Update path since we're running from the build/ directory!
    loadDatasetFromImage("../images/image1.jpg", width, height, opt.inputDim, h_inputs, h_targets);
    int batchSize = width * height;
    
    half *d_inputs, *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs, batchSize * opt.inputDim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_targets, batchSize * opt.outputDim * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(), batchSize * opt.inputDim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), batchSize * opt.outputDim * sizeof(half), cudaMemcpyHostToDevice));

    // 2. Allocate Network Parameters
    float** d_master_weights = createDevicePointerArrayFloat(opt.numLayers, opt, true, gen);
    float** d_master_biases = createDevicePointerArrayFloat(opt.numLayers, opt, false, gen);
    
    half** d_fwd_weights = createDeviceHalfPointerArrayForWeights(opt.numLayers, opt, d_master_weights);
    
    // 3. Allocate Optimizer States
    float** d_w_m = createDevicePointerArrayFloatZero(opt.numLayers, opt, true);
    float** d_w_v = createDevicePointerArrayFloatZero(opt.numLayers, opt, true);
    float** d_w_grad = createDevicePointerArrayFloatZero(opt.numLayers, opt, true);
    
    float** d_b_m = createDevicePointerArrayFloatZero(opt.numLayers, opt, false);
    float** d_b_v = createDevicePointerArrayFloatZero(opt.numLayers, opt, false);
    float** d_b_grad = createDevicePointerArrayFloatZero(opt.numLayers, opt, false);

    // 4. Allocate Forward/Backward Buffers
    float* d_outputs_float;
    half* d_dLoss;
    float* d_total_loss;
    half** d_activations = createDeviceActivationsArray(opt.numLayers, opt, batchSize, d_inputs);
    
    CUDA_CHECK(cudaMalloc(&d_outputs_float, batchSize * opt.outputDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dLoss, batchSize * opt.outputDim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(float)));
    
    // Not explicitly checking inputs grads, but Backward kernel requires dx_out parameter
    half* d_dx_out;
    CUDA_CHECK(cudaMalloc(&d_dx_out, batchSize * opt.inputDim * sizeof(half)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start_loop, end_loop;
    cudaEventCreate(&start_loop); cudaEventCreate(&end_loop);
    cudaEvent_t start_fwd, end_fwd, start_mse, end_mse, start_bwd, end_bwd, start_opt, end_opt;
    cudaEventCreate(&start_fwd); cudaEventCreate(&end_fwd);
    cudaEventCreate(&start_mse); cudaEventCreate(&end_mse);
    cudaEventCreate(&start_bwd); cudaEventCreate(&end_bwd);
    cudaEventCreate(&start_opt); cudaEventCreate(&end_opt);

    // Training Hyperparams
    int numSteps = 10000;
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    
    cout << "Starting Training Loop... (" << numSteps << " steps)" << endl;

    cudaEventRecord(start_loop, stream);

    for (int step = 1; step <= numSteps; step++) {
        // Zero Total Loss
        CUDA_CHECK(cudaMemsetAsync(d_total_loss, 0, sizeof(float), stream));

        launchZeroGradients(&opt, d_w_grad, d_b_grad, stream);

        if (step == numSteps) cudaEventRecord(start_fwd, stream);
        // Forward Pass
        launchNetworkFusionGradKernel(
            &opt, d_inputs, d_activations, d_fwd_weights, d_master_biases, 
            d_outputs_float, batchSize, stream
        );

        if (step == numSteps) cudaEventRecord(end_fwd, stream);

        if (step == numSteps) cudaEventRecord(start_mse, stream);
        // MSE Loss & Grad
        launchMSELossGrad(
            d_outputs_float, d_targets, d_dLoss, d_total_loss, 
            batchSize, opt.outputDim, opt.outputDim, LOSS_SCALE, stream, OUT_ACT_NONE
        );
        if (step == numSteps) cudaEventRecord(end_mse, stream);

        // Print Loss and Save Progress
        if (step % 1000 == 0 || step == 1) {
            float h_total_loss;
            CUDA_CHECK(cudaMemcpyAsync(&h_total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
            cout << "Step " << step << " | Loss: " << (h_total_loss / batchSize) << endl;
            
            // Save predicted image every 1000 steps
            if (step % 1000 == 0) {
                string out_name = "../images/pred_step_" + to_string(step) + ".jpg";
                savePredictedImage(out_name, width, height, d_outputs_float);
            }
        }

        if (step == numSteps) cudaEventRecord(start_bwd, stream);
        // Backward Pass
        launchNetworkFusionBackwardKernel(
            &opt, d_dLoss, d_fwd_weights, d_master_biases, d_activations, 
            d_w_grad, d_b_grad, d_dx_out, batchSize, stream
        );
        if (step == numSteps) cudaEventRecord(end_bwd, stream);

        if (step == numSteps) cudaEventRecord(start_opt, stream);
        // Optimizer Step
        float bias_correction1 = 1.0f - powf(beta1, (float)step);
        float bias_correction2 = 1.0f - powf(beta2, (float)step);
        float inv_loss_scale = 1.0f / LOSS_SCALE;

        launchAdamWeightsOptim(
            &opt, d_master_weights, d_fwd_weights, d_w_grad, d_w_m, d_w_v, 
            lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale, stream
        );

        launchAdamBiasOptim(
            &opt, d_master_biases, d_b_grad, d_b_m, d_b_v, 
            lr, beta1, beta2, epsilon, bias_correction1, bias_correction2, inv_loss_scale, stream
        );
        if (step == numSteps) cudaEventRecord(end_opt, stream);
    }
    
    cudaEventRecord(end_loop, stream);
    cudaStreamSynchronize(stream);
    cout << "Training Completed. Saving final image..." << endl;
    savePredictedImage("../images/pred_final.jpg", width, height, d_outputs_float);

    float total_ms = 0, fwd_ms = 0, mse_ms = 0, bwd_ms = 0, opt_ms = 0;
    cudaEventElapsedTime(&total_ms, start_loop, end_loop);
    cudaEventElapsedTime(&fwd_ms, start_fwd, end_fwd);
    cudaEventElapsedTime(&mse_ms, start_mse, end_mse);
    cudaEventElapsedTime(&bwd_ms, start_bwd, end_bwd);
    cudaEventElapsedTime(&opt_ms, start_opt, end_opt);

    float avg_ms = total_ms / numSteps;
    float throughput = (batchSize * 1000.0f) / avg_ms;

    cout << "\n==============================================" << endl;
    cout << "          PERFORMANCE METRICS" << endl;
    cout << "==============================================" << endl;
    cout << "Avg Time per Step: " << avg_ms << " ms" << endl;
    cout << "Throughput:        " << throughput << " items/sec" << endl;
    cout << "\nTime Spent Breakdown (Measured on final step):" << endl;
    cout << "  [1] Forward Kernel:   " << fwd_ms << " ms" << endl;
    cout << "  [2] MSE Loss Kernel:  " << mse_ms << " ms" << endl;
    cout << "  [3] Backward Kernel:  " << bwd_ms << " ms" << endl;
    cout << "  [4] Optimizer Kernels:" << opt_ms << " ms" << endl;
    cout << "==============================================" << endl;
    
    // Cleanup (Not strictly necessary for single run, but good practice)
    cudaStreamDestroy(stream);

    runPerformanceSweep();
    return 0;
}
