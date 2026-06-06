#include "../InstantNerf.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

struct Resolution {
    std::string name;
    int width;
    int height;
};

__global__ void initRaysKernel(float3* o, float3* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o[i] = make_float3(0.0f, 0.0f, -2.0f);
        d[i] = make_float3(0.0f, 0.0f, 1.0f);
    }
}

void printVRAMUsage() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    size_t used_byte = total_byte - free_byte;
    std::cout << "VRAM Used  : " << used_byte / (1024.0 * 1024.0) << " MB" << std::endl;
}

int main() {
    std::cout << "Comparing Ray-Centric vs Hit-Centric Training Pipeline...\n" << std::endl;

    int numRays = 1920 * 1080; // 1080p
    std::cout << "Target Resolution: 1920x1080 (1080p)" << std::endl;
    std::cout << "Total Rays: " << numRays << "\n" << std::endl;

    float3* d_rays_o;
    float3* d_rays_d;
    float* d_rgb_true;

    CUDA_CHECK(cudaMalloc(&d_rays_o, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rgb_true, numRays * 3 * sizeof(float)));

    int blocks = (numRays + 255) / 256;
    initRaysKernel<<<blocks, 256>>>(d_rays_o, d_rays_d, numRays);
    CUDA_CHECK(cudaMemset(d_rgb_true, 0, numRays * 3 * sizeof(float)));

    NerfOptions opts;
    opts.rayChunkSize = 256 * 1024;
    opts.batchSize = 256 * 1024;
    opts.isProfiling = true;
    opts.aabbMin = make_float3(-1.5f, -1.5f, -1.5f);
    opts.aabbMax = make_float3(1.5f, 1.5f, 1.5f);
    
    InstantNerf nerf;
    nerf.init(opts);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    int dummySteps = 0;
    int chunks = (numRays + opts.rayChunkSize - 1) / opts.rayChunkSize;
    std::vector<uint32_t> dummyHitCounts(chunks, 0);
    nerf.trainWithRays(d_rays_o, d_rays_d, d_rgb_true, 8192, dummySteps, dummyHitCounts.data());
    cudaDeviceSynchronize();
    nerf.resetStats();

    // ==========================================
    // 1. Ray-Centric (Original)
    // ==========================================
    std::cout << "======================================" << std::endl;
    std::cout << " 1. Original (trainWithRays)          " << std::endl;
    std::cout << "======================================" << std::endl;
    
    int trainSteps1 = 0;
    CUDA_CHECK(cudaEventRecord(start));
    nerf.trainWithRays(d_rays_o, d_rays_d, d_rgb_true, numRays, trainSteps1, dummyHitCounts.data());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms1 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));
    std::cout << "Time Taken : " << ms1 << " ms" << std::endl;
    printVRAMUsage();
    nerf.printStats();
    nerf.resetStats();

    // ==========================================
    // 2. Hit-Centric (New)
    // ==========================================
    std::cout << "\n======================================" << std::endl;
    std::cout << " 2. Hit-Centric (trainWithRaysHit)    " << std::endl;
    std::cout << "======================================" << std::endl;
    
    int trainSteps2 = 0;
    try {
        CUDA_CHECK(cudaEventRecord(start));
        nerf.trainWithRaysHit(d_rays_o, d_rays_d, d_rgb_true, numRays, trainSteps2, dummyHitCounts.data(), nullptr, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION CAUGHT: " << e.what() << "\n";
        return 1;
    }

    float ms2 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));
    std::cout << "Time Taken : " << ms2 << " ms" << std::endl;
    printVRAMUsage();
    nerf.printStats();
    nerf.resetStats();

    std::cout << "\n======================================" << std::endl;
    std::cout << " SPEEDUP: " << ms1 / ms2 << "x faster!" << std::endl;
    std::cout << "======================================" << std::endl;

    CUDA_CHECK(cudaFree(d_rays_o));
    CUDA_CHECK(cudaFree(d_rays_d));
    CUDA_CHECK(cudaFree(d_rgb_true));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
