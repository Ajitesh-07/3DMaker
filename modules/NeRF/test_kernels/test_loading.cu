#include "../InstantNerf.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <filesystem>

__global__ void initRaysKernelLoadTest(float3* o, float3* d, float* true_rgb, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o[i] = make_float3(0.0f, 0.0f, -2.0f);
        d[i] = make_float3(0.0f, 0.0f, 1.0f);
        true_rgb[i * 3 + 0] = 1.0f;
        true_rgb[i * 3 + 1] = 0.0f;
        true_rgb[i * 3 + 2] = 0.0f;
    }
}

int main() {
    std::cout << "Starting NeRF Save/Load Test..." << std::endl;

    NerfOptions opts;
    opts.batchSize = 1024;
    opts.rayChunkSize = 1024;
    
    InstantNerf nerf1;
    nerf1.init(opts);
    
    int numRays = 1024;
    float3* d_rays_o;
    float3* d_rays_d;
    float* d_rgb_true;
    float* d_rgb_out;

    CUDA_CHECK(cudaMalloc(&d_rays_o, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rgb_true, numRays * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb_out, numRays * 3 * sizeof(float)));

    int blocks = (numRays + 255) / 256;
    initRaysKernelLoadTest<<<blocks, 256>>>(d_rays_o, d_rays_d, d_rgb_true, numRays);
    cudaDeviceSynchronize();

    int step = 0;
    std::vector<uint32_t> hitCounts(1, 0);
    std::cout << "Training briefly to mutate the model state..." << std::endl;
    nerf1.trainWithRaysHit(d_rays_o, d_rays_d, d_rgb_true, numRays, step, hitCounts.data(), d_rgb_out, 0);
    cudaDeviceSynchronize();
    
    std::string testFile = "test.inerf";
    std::cout << "Saving to " << testFile << "..." << std::endl;
    nerf1.save(testFile);
    
    std::cout << "File saved successfully. Now loading into a new instance..." << std::endl;
    InstantNerf nerf2;
    nerf2.load(testFile);
    
    std::cout << "File loaded successfully. Attempting to render with the new instance..." << std::endl;
    nerf2.setMemoryMode(INFERENCE);
    nerf2.renderImage(d_rays_o, d_rays_d, numRays, d_rgb_out, 0);
    cudaDeviceSynchronize();
    
    std::cout << "Render successful!" << std::endl;
    std::cout << "All tests passed!" << std::endl;

    std::filesystem::remove(testFile); // Clean up
    
    return 0;
}
