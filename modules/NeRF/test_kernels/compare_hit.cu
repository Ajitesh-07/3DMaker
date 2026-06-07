#include "../InstantNerf.h"
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cmath>

struct Resolution {
    std::string name;
    int width;
    int height;
};

__global__ void initRaysKernel(float3* o, float3* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float u = (float)(i % 1920) / 1920.0f;
        float v = (float)(i / 1920) / 1080.0f;
        o[i] = make_float3((u - 0.5f) * 2.0f, (v - 0.5f) * 2.0f, -2.0f);
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
    std::cout << "Comparing Ray-Centric vs Hit-Centric Rendering Pipeline...\n" << std::endl;

    int numRays = 1920 * 1080; // 1080p
    std::cout << "Target Resolution: 1920x1080 (1080p)" << std::endl;
    std::cout << "Total Rays: " << numRays << "\n" << std::endl;

    float3* d_rays_o;
    float3* d_rays_d;
    float* d_rgb_out1;
    float* d_rgb_out2;

    CUDA_CHECK(cudaMalloc(&d_rays_o, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rgb_out1, numRays * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb_out2, numRays * 3 * sizeof(float)));

    int blocks = (numRays + 255) / 256;
    initRaysKernel<<<blocks, 256>>>(d_rays_o, d_rays_d, numRays);
    CUDA_CHECK(cudaMemset(d_rgb_out1, 0, numRays * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rgb_out2, 0, numRays * 3 * sizeof(float)));

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
    nerf.renderImage(d_rays_o, d_rays_d, numRays, d_rgb_out1);
    cudaDeviceSynchronize();
    nerf.resetStats();

    // ==========================================
    // 1. Ray-Centric (Original)
    // ==========================================
    std::cout << "======================================" << std::endl;
    std::cout << " 1. Original (renderImage)            " << std::endl;
    std::cout << "======================================" << std::endl;
    
    CUDA_CHECK(cudaEventRecord(start));
    nerf.renderImage(d_rays_o, d_rays_d, numRays, d_rgb_out1);
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
    std::cout << " 2. Hit-Centric (renderImageHit)      " << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        CUDA_CHECK(cudaEventRecord(start));
        nerf.renderImageHit(d_rays_o, d_rays_d, numRays, d_rgb_out2, 0);
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

    // Verify output
    std::vector<float> h_out1(numRays * 3);
    std::vector<float> h_out2(numRays * 3);
    CUDA_CHECK(cudaMemcpy(h_out1.data(), d_rgb_out1, numRays * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out2.data(), d_rgb_out2, numRays * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < numRays * 3; i++) {
        float diff = std::abs(h_out1[i] - h_out2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "\n======================================" << std::endl;
    std::cout << " Max Diff Between Methods: " << max_diff << std::endl;
    if (max_diff < 1e-4) {
        std::cout << " [SUCCESS] Both rendering methods output the same image." << std::endl;
    } else {
        std::cout << " [FAILURE] Outputs differ!" << std::endl;
    }
    std::cout << "======================================" << std::endl;

    CUDA_CHECK(cudaFree(d_rays_o));
    CUDA_CHECK(cudaFree(d_rays_d));
    CUDA_CHECK(cudaFree(d_rgb_out1));
    CUDA_CHECK(cudaFree(d_rgb_out2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
