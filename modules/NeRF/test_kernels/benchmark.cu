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
        // Pointing straight down the Z axis from outside the AABB
        o[i] = make_float3(0.0f, 0.0f, -2.0f);
        d[i] = make_float3(0.0f, 0.0f, 1.0f);
    }
}

void printVRAMUsage() {
    size_t free_byte;
    size_t total_byte;
    cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
    if (err == cudaSuccess) {
        size_t used_byte = total_byte - free_byte;
        std::cout << "VRAM Used  : " << used_byte / (1024.0 * 1024.0) << " MB / " 
                  << total_byte / (1024.0 * 1024.0) << " MB" << std::endl;
    }
}

int main() {
    std::cout << "Starting NeRF Training Benchmark..." << std::endl;

    std::vector<Resolution> resolutions = {
        {"720p", 1280, 720},
        {"1080p", 1920, 1080},
        {"1440p (2K)", 2560, 1440},
        {"2160p (4K)", 3840, 2160}
    };

    int maxRays = 3840 * 2160;

    float3* d_rays_o;
    float3* d_rays_d;
    float* d_rgb_true;

    CUDA_CHECK(cudaMalloc(&d_rays_o, maxRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, maxRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rgb_true, maxRays * 3 * sizeof(float)));

    // Initialize with dummy data that guarantees AABB intersection for realistic benchmarking
    int blocks = (maxRays + 255) / 256;
    initRaysKernel<<<blocks, 256>>>(d_rays_o, d_rays_d, maxRays);
    CUDA_CHECK(cudaMemset(d_rgb_true, 0, maxRays * 3 * sizeof(float)));

    NerfOptions opts;
    opts.rayChunkSize = 256 * 1024; // Configurable chunk size
    opts.isProfiling = true;
    
    InstantNerf nerf;
    std::cout << "Initializing MLPs and Grid..." << std::endl;
    nerf.init(opts);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup step to compile kernels, allocate context, etc.
    std::cout << "Running Warmup Pass..." << std::endl;
    std::cout.flush();
    std::cerr.flush();
    int dummySteps = 0;
    try {
        nerf.trainWithRays(d_rays_o, d_rays_d, d_rgb_true, 8192, dummySteps);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after warmup trainWithRays: %s\n", cudaGetErrorString(err));
            fflush(stderr);
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception during warmup: %s\n", e.what());
        fflush(stderr);
        return 1;
    }
    std::cout << "Warmup complete." << std::endl;
    nerf.resetStats(); // Clear warmup stats before actual benchmark
    std::cout << "---------------------------------" << std::endl;

    for (const auto& res : resolutions) {
        int numRays = res.width * res.height;
        int trainSteps = 0;

        try {
            CUDA_CHECK(cudaEventRecord(start));
            
            nerf.trainWithRays(d_rays_o, d_rays_d, d_rgb_true, numRays, trainSteps);
            
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

            std::cout << "Resolution : " << res.name << " (" << res.width << "x" << res.height << ")" << std::endl;
            std::cout << "Total Rays : " << numRays << std::endl;
            std::cout << "Train Steps: " << trainSteps << std::endl;
            std::cout << "Time Taken : " << ms << " ms (" << ms / 1000.0f << " seconds)" << std::endl;
            std::cout << "Throughput : " << (numRays / (ms / 1000.0f)) / 1e6f << " Million Rays / sec" << std::endl;
            printVRAMUsage();
            nerf.printStats();
            nerf.resetStats(); // Clear stats for the next resolution
            std::cout << "---------------------------------" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception during benchmark loop: " << e.what() << std::endl;
            std::cerr.flush();
            return 1;
        }

    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_rays_o));
    CUDA_CHECK(cudaFree(d_rays_d));
    CUDA_CHECK(cudaFree(d_rgb_true));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Benchmark Complete." << std::endl;
    return 0;
}
