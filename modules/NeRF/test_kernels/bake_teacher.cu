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
    opts.jointFitBatch = rayChunkSize; // Match dataloader chunk
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
    
    for (int i = 0; i < testSteps; i++) {
        // Fetch a random batch of rays using DataLoader
        float3 bg = make_float3(0.0f, 0.0f, 0.0f);
        int cursor = (i * opts.jointFitBatch) % std::max((uint32_t)1, dataloader.getTotalRays());
        dataloader.fetchRayChunk(cursor, opts.jointFitBatch, 42 + i, bg, stream, false, 0);
        
        bakedNerf.jointFit(
            teacher,
            dataloader.getChunkRaysO(),
            dataloader.getChunkRaysD(),
            opts.jointFitBatch,
            trainSteps,
            stream
        );
    }
    cudaStreamSynchronize(stream);
    
    auto end_fit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_fit = end_fit - start_fit;
    std::cout << "Joint Fit Time (" << testSteps << " steps): " << diff_fit.count() << " seconds" << std::endl;
    std::cout << "Average time per step: " << (diff_fit.count() / testSteps) * 1000.0 << " ms" << std::endl;
    
    cudaStreamDestroy(stream);
    std::cout << "Done!" << std::endl;
    
    return 0;
}
