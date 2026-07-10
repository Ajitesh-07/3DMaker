#include <iostream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include "../DataLoader.h"
#include "../InstantNerf.h"
#include "../BakedNerf.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> [checkpoint_path]" << std::endl;
        return 1;
    }
    
    std::string datasetPath = argv[1];
    
    // 1. Setup Data Loader
    std::cout << "Loading dataset: " << datasetPath << std::endl;
    uint32_t rayChunkSize = 32 * 1024;
    uint32_t batchSize = 256 * 1024;
    DataLoader dataloader(datasetPath, rayChunkSize, true, false, 0.0f);
    
    // 2. Setup Teacher
    std::cout << "Initializing Teacher..." << std::endl;
    InstantNerf teacher;
    if (argc >= 3) {
        std::cout << "Loading teacher checkpoint: " << argv[2] << std::endl;
        teacher.load(argv[2], INFERENCE, 256*1024);
    } else {
        std::cerr << "Teacher checkpoint required!" << std::endl;
        return 1;
    }
    
    // 3. Setup BakedNerf
    std::cout << "Initializing BakedNerf..." << std::endl;
    const NerfOptions& topts = teacher.options();
    BakeOptions opts;
    opts.isProfiling = true;
    opts.jointFitBatch = batchSize; // Match dataloader chunk
    opts.voxelGridResolution = make_uint3(topts.gridResolution.x * SPARSE_B, topts.gridResolution.y * SPARSE_B, topts.gridResolution.z * SPARSE_B);
    
    BakedNerf bakedNerf;
    bakedNerf.init(opts);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 4. Bake Geometry (Phase 1)
    std::cout << "Baking Geometry..." << std::endl;
    auto start_bake = std::chrono::high_resolution_clock::now();
    
    bakedNerf.bakeGeometry(teacher, stream);
    cudaStreamSynchronize(stream);
    
    auto end_bake = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_bake = end_bake - start_bake;
    std::cout << "Bake Geometry Time: " << diff_bake.count() << " seconds" << std::endl;
    std::cout << "Baked Blocks: " << bakedNerf.numBlocks() << std::endl;
    std::cout << "Baked Voxels: " << bakedNerf.numVoxels() << std::endl;
    
    // 5. Joint Fit (Distillation Phase 2)
    int testSteps = 100;
    std::cout << "Running Joint Fit for " << testSteps << " steps..." << std::endl;
    int trainSteps = 0;
    
    auto start_fit = std::chrono::high_resolution_clock::now();
    
    int i = 0;
    while(trainSteps < testSteps) {
        float3 bg = make_float3(0.0f, 0.0f, 0.0f);
        int cursor = (i * rayChunkSize) % std::max((uint32_t)1, dataloader.getTotalRays());
        dataloader.fetchRayChunk(cursor, rayChunkSize, 42 + i, bg, stream, false, 0);
        
        bakedNerf.jointFit(
            teacher,
            dataloader.getChunkRaysO(),
            dataloader.getChunkRaysD(),
            rayChunkSize,
            trainSteps,
            stream
        );
        i++;
    }
    cudaStreamSynchronize(stream);
    
    auto end_fit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_fit = end_fit - start_fit;
    std::cout << "Joint Fit Time (" << trainSteps << " steps): " << diff_fit.count() << " seconds" << std::endl;
    std::cout << "Average time per step: " << (diff_fit.count() / trainSteps) * 1000.0 << " ms" << std::endl;
    
    // 6. Benchmark renderImage
    std::cout << "\nBenchmarking renderImage (800x800 = 640000 rays)..." << std::endl;
    int render_rays = 800 * 800;
    
    float3* d_test_rays_o;
    float3* d_test_rays_d;
    float* d_rgb_out;
    cudaMalloc(&d_test_rays_o, render_rays * sizeof(float3));
    cudaMalloc(&d_test_rays_d, render_rays * sizeof(float3));
    cudaMalloc(&d_rgb_out, render_rays * 3 * sizeof(float));

    // Fill with valid dataloader rays to ensure they hit geometry
    for(int offset = 0; offset < render_rays; offset += rayChunkSize) {
        int copy_count = std::min((uint32_t)(render_rays - offset), rayChunkSize);
        cudaMemcpyAsync(d_test_rays_o + offset, dataloader.getChunkRaysO(), copy_count * sizeof(float3), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_test_rays_d + offset, dataloader.getChunkRaysD(), copy_count * sizeof(float3), cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    // Warmup
    bakedNerf.renderImage(d_test_rays_o, d_test_rays_d, render_rays, d_rgb_out, stream);
    cudaStreamSynchronize(stream);

    // Timing
    int render_iters = 1;
    auto start_render = std::chrono::high_resolution_clock::now();

    for(int k = 0; k < render_iters; k++) {
        bakedNerf.renderImage(d_test_rays_o, d_test_rays_d, render_rays, d_rgb_out, stream);
    }

    cudaStreamSynchronize(stream);
    auto end_render = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_render = end_render - start_render;

    std::cout << "Render Time for " << render_iters << " frames (800x800): " << diff_render.count() << " seconds" << std::endl;
    std::cout << "Average FPS: " << render_iters / diff_render.count() << " fps" << std::endl;
    
    bakedNerf.printStats();

    cudaFree(d_test_rays_o);
    cudaFree(d_test_rays_d);
    cudaFree(d_rgb_out);

    cudaStreamDestroy(stream);

    std::cout << "Done!" << std::endl;
    
    return 0;
}
