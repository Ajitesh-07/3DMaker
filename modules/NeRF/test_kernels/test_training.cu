#include "../InstantNerf.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void initRaysKernel(float3* o, float3* d, float* true_rgb, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // All rays point straight down Z axis from outside AABB
        o[i] = make_float3(0.0f, 0.0f, -2.0f);
        d[i] = make_float3(0.0f, 0.0f, 1.0f);

        // Target color: Solid Red
        true_rgb[i * 3 + 0] = 1.0f;
        true_rgb[i * 3 + 1] = 0.0f;
        true_rgb[i * 3 + 2] = 0.0f;
    }
}

__global__ void computeMSEKernel(const float* pred_rgb, const float* true_rgb, float* loss_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff_r = pred_rgb[i * 3 + 0] - true_rgb[i * 3 + 0];
        float diff_g = pred_rgb[i * 3 + 1] - true_rgb[i * 3 + 1];
        float diff_b = pred_rgb[i * 3 + 2] - true_rgb[i * 3 + 2];
        
        float squared_err = diff_r*diff_r + diff_g*diff_g + diff_b*diff_b;
        atomicAdd(loss_out, squared_err);
    }
}

int main() {
    std::cout << "Starting NeRF Training Convergence Test..." << std::endl;

    int numRays = 1920 * 1080; // 1080p resolution

    float3* d_rays_o;
    float3* d_rays_d;
    float* d_rgb_true;
    float* d_rgb_out;
    float* d_loss;

    CUDA_CHECK(cudaMalloc(&d_rays_o, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rgb_true, numRays * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb_out, numRays * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    int blocks = (numRays + 255) / 256;
    initRaysKernel<<<blocks, 256>>>(d_rays_o, d_rays_d, d_rgb_true, numRays);
    CUDA_CHECK(cudaDeviceSynchronize());

    NerfOptions opts;
    opts.rayChunkSize = 32 * 1024;
    opts.batchSize = 256 * 1024;
    opts.learningRate = 1e-2f; // Higher LR for fast test convergence
    opts.isProfiling = false; // Disable profiling to run fast
    
    InstantNerf nerf;
    nerf.init(opts);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Training for 3 Epochs targeting a solid RED volume at 1080p..." << std::endl;

    for (int epoch = 1; epoch <= 3; ++epoch) {
        int trainSteps = 0;
        
        int chunks = (numRays + opts.rayChunkSize - 1) / opts.rayChunkSize;
        std::vector<uint32_t> dummyHitCounts(chunks, 0);
        
        CUDA_CHECK(cudaEventRecord(start));
        // Train and capture the network's volume rendering output
        nerf.trainWithRays(d_rays_o, d_rays_d, d_rgb_true, numRays, trainSteps, dummyHitCounts.data(), d_rgb_out, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        // Compute MSE between the network's prediction and the Target RED color
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        computeMSEKernel<<<blocks, 256>>>(d_rgb_out, d_rgb_true, d_loss, numRays);
        
        float h_loss = 0;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        
        float mse = h_loss / (numRays * 3.0f);
        float stepsPerSec = (trainSteps * 1000.0f) / ms;

        std::cout << "Epoch " << epoch << " | Steps: " << trainSteps << " | MSE Loss: " << mse 
                  << " | Speed: " << stepsPerSec << " steps/sec" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_rays_o));
    CUDA_CHECK(cudaFree(d_rays_d));
    CUDA_CHECK(cudaFree(d_rgb_true));
    CUDA_CHECK(cudaFree(d_rgb_out));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Test Complete." << std::endl;
    return 0;
}
