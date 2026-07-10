#include <iostream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include "../DataLoader.h"
#include "../InstantNerf.h"
#include "../BakedNerf.h"
#include <filesystem>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../third_party/stb_image_write.h"

static void saveImagePNG(const std::vector<float>& rgb, int width, int height, const std::string& path) {
    if (width <= 0 || height <= 0) return;
    std::vector<uint8_t> bytes(width * height * 3);
    for (size_t i = 0; i < bytes.size(); ++i) {
        float v = std::min(1.0f, std::max(0.0f, rgb[i]));
        bytes[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    stbi_write_png(path.c_str(), width, height, 3, bytes.data(), width * 3);
}

static std::string formatETA(float seconds) {
    if (!(seconds >= 0.0f)) return "--:--";
    int t = (int)(seconds + 0.5f);
    int h = t / 3600; t %= 3600;
    int m = t / 60;   t %= 60;
    char buf[32];
    if (h > 0) snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, t);
    else       snprintf(buf, sizeof(buf), "%02d:%02d", m, t);
    return buf;
}

static void renderBakingBar(int step, int targetSteps, double emaMsPerStep) {
    const int kWidth = 28;
    float frac = (targetSteps > 0) ? (float)step / (float)targetSteps : 0.0f;
    frac = std::min(1.0f, std::max(0.0f, frac));
    int filled = (int)(frac * kWidth + 0.5f);
    std::string bar(filled, '#');
    bar.append(kWidth - filled, ' ');
    
    float etaSeconds = (float)(emaMsPerStep * std::max(0, targetSteps - step) / 1000.0);
    printf("\r[%s] %d/%d steps | %.2f ms/step | ETA %s   ",
           bar.c_str(), step, targetSteps, emaMsPerStep,
           formatETA(etaSeconds).c_str());
    fflush(stdout);
}

void saveStudentImage(BakedNerf& student, DataLoader& dataloader, int img_idx, const std::string& prefix, cudaStream_t stream) {
    int width = dataloader.getWidth();
    int height = dataloader.getHeight();
    int pixels = width * height;
    
    std::vector<float> img_student(pixels * 3);
    float* d_student_rgb;
    cudaMalloc(&d_student_rgb, pixels * 3 * sizeof(float));
    
    int tile = dataloader.getRayChunkSize();
    float3 bg = make_float3(1.0f, 1.0f, 1.0f);
    
    // Allocate buffers for all rays of the image
    float3* d_all_rays_o;
    float3* d_all_rays_d;
    cudaMalloc(&d_all_rays_o, pixels * sizeof(float3));
    cudaMalloc(&d_all_rays_d, pixels * sizeof(float3));
    
    for (int off = 0; off < pixels; off += tile) {
        int count = std::min(tile, pixels - off);
        dataloader.fetchRayChunk(img_idx * pixels + off, count, 0, bg, stream, true, 0);
        
        // Copy fetched chunk into the full image buffers
        cudaMemcpyAsync(d_all_rays_o + off, dataloader.getChunkRaysO(), count * sizeof(float3), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_all_rays_d + off, dataloader.getChunkRaysD(), count * sizeof(float3), cudaMemcpyDeviceToDevice, stream);
    }
    
    // Call renderImage exactly once on the full image rays
    student.renderImage(d_all_rays_o, d_all_rays_d, pixels, d_student_rgb, stream);
    cudaStreamSynchronize(stream);
    
    cudaMemcpy(img_student.data(), d_student_rgb, pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "[" << prefix << "] Saved student image." << std::endl;
    saveImagePNG(img_student, width, height, "../benchmarks/frames_baked/" + prefix + "_student.png");
    
    cudaFree(d_student_rgb);
    cudaFree(d_all_rays_o);
    cudaFree(d_all_rays_d);
}

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
    opts.isProfiling = false;
    opts.mlpLearningRate = 1e-2;
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

    std::cout << "\nSaving Student Image (Post-Geometry Bake)..." << std::endl;
    saveStudentImage(bakedNerf, dataloader, 0, "post_geometry", stream);
    
    // 5. Joint Fit (Distillation Phase 2)
    int testSteps = 1000;
    std::cout << "Running Joint Fit for " << testSteps << " steps..." << std::endl;
    int trainSteps = 0;
    
    auto start_fit = std::chrono::high_resolution_clock::now();
    double emaMsPerStep = 0.0;
    
    int i = 0;
    while(trainSteps < testSteps) {
        auto t0 = std::chrono::high_resolution_clock::now();
        float3 bg = make_float3(0.0f, 0.0f, 0.0f);
        int cursor = (i * rayChunkSize) % std::max((uint32_t)1, dataloader.getTotalRays());
        dataloader.fetchRayChunk(cursor, rayChunkSize, 42 + i, bg, stream, false, 0);
        
        int prevSteps = trainSteps;
        bakedNerf.jointFit(
            teacher,
            dataloader.getChunkRaysO(),
            dataloader.getChunkRaysD(),
            rayChunkSize,
            trainSteps,
            stream
        );
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        int stepsDone = trainSteps - prevSteps;
        if (stepsDone > 0) {
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double msPerStep = ms / stepsDone;
            emaMsPerStep = (emaMsPerStep > 0.0) ? (0.70 * emaMsPerStep + 0.30 * msPerStep) : msPerStep;
        }
        
        renderBakingBar(trainSteps, testSteps, emaMsPerStep);
        i++;
    }
    std::cout << std::endl;
    cudaStreamSynchronize(stream);
    
    auto end_fit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_fit = end_fit - start_fit;
    std::cout << "Joint Fit Time (" << trainSteps << " steps): " << diff_fit.count() << " seconds" << std::endl;
    std::cout << "Average time per step: " << (diff_fit.count() / trainSteps) * 1000.0 << " ms" << std::endl;

    std::cout << "\nSaving Student Image (Post-Joint Fit)..." << std::endl;
    // saveStudentImage(bakedNerf, dataloader, 0, "post_jointfit", stream);
    
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

    bakedNerf.resetStats();

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
