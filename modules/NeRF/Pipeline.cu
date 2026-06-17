#include "InstantNerf.h"
#include "DataLoader.h"
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../third_party/stb_image_write.h"

// ------------------------------------------------------------------
// Utility Functions & Kernels
// ------------------------------------------------------------------

// Prints the current VRAM usage of the GPU. 
// Note: This includes the WDDM Windows OS reservation (usually ~1.1GB).
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

// Calculates the Mean Squared Error (MSE) between the rendered RGB and ground truth
__global__ void mse_kernel(const float* pred, const float* target, float* loss_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = pred[idx * 3 + 0] - target[idx * 3 + 0];
        float g = pred[idx * 3 + 1] - target[idx * 3 + 1];
        float b = pred[idx * 3 + 2] - target[idx * 3 + 2];
        float err = (r*r + g*g + b*b) / 3.0f;
        atomicAdd(loss_out, err);
    }
}

// Converts float[0, 1] RGB output to byte[0, 255] RGB for saving to PNG
__global__ void float_to_byte_kernel(const float* rgb_float, uint8_t* rgb_byte, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixels) {
        float r = fmaxf(0.0f, fminf(1.0f, rgb_float[idx * 3 + 0]));
        float g = fmaxf(0.0f, fminf(1.0f, rgb_float[idx * 3 + 1]));
        float b = fmaxf(0.0f, fminf(1.0f, rgb_float[idx * 3 + 2]));
        rgb_byte[idx * 3 + 0] = (uint8_t)(r * 255.0f);
        rgb_byte[idx * 3 + 1] = (uint8_t)(g * 255.0f);
        rgb_byte[idx * 3 + 2] = (uint8_t)(b * 255.0f);
    }
}

// Generates rays for an arbitrary virtual camera (used for the 360 video)
__global__ void generate_custom_rays_kernel(
    int width, int height, float focal_length, 
    float c00, float c01, float c02, float c03,
    float c10, float c11, float c12, float c13,
    float c20, float c21, float c22, float c23,
    float3* d_rays_o, float3* d_rays_d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int px = idx % width;
    int py = idx / width;

    // NeRF Camera space direction (X right, Y up, Z backwards)
    float dir_x = (px - width * 0.5f) / focal_length;
    float dir_y = -(py - height * 0.5f) / focal_length;
    float dir_z = -1.0f;

    float inv_norm = rsqrtf(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    dir_x *= inv_norm;
    dir_y *= inv_norm;
    dir_z *= inv_norm;

    // C2W * dir (Rotate camera space to world space)
    float world_dir_x = c00*dir_x + c01*dir_y + c02*dir_z;
    float world_dir_y = c10*dir_x + c11*dir_y + c12*dir_z;
    float world_dir_z = c20*dir_x + c21*dir_y + c22*dir_z;

    d_rays_o[idx] = make_float3(c03, c13, c23);
    d_rays_d[idx] = make_float3(world_dir_x, world_dir_y, world_dir_z);
}

// ------------------------------------------------------------------
// Main Execution
// ------------------------------------------------------------------

void run_training_pipeline(
    const std::string& dataset_path,
    int totalEpochsParam,
    int totalSteps,
    int numCascades,
    float lambdaDist,
    int samplesPerVoxel,
    bool runValidation,
    bool renderVideo,
    float maxLR,
    float minLR
) {
    std::cout << "Starting NeRF Hit-Centric Training with Streaming DataLoader..." << std::endl;

    // ==========================================
    // 1. Configuration & Setup  (kept in parity with train_hit)
    // ==========================================

    NerfOptions opts;
    opts.densityBias = 0.0f;                  // match train_hit
    opts.rayChunkSize = 64 * 1024;            // Limit VRAM per training step
    opts.colorHiddenDim = 32;
    opts.colorNumLayers = 3;
    opts.renderBatchSize = 256 * 1024;
    opts.batchSize = 256 * 1024;              // Neural Network internal batch size
    opts.samplesPerVoxel = samplesPerVoxel;   // K sub-voxel samples per occupied finest voxel
    opts.minDensityThreshold = 0.01f;
    opts.learningRate = maxLR;                // cosine-schedule start (max_lr)
    opts.lossScale = 128.0f;                  // FP16 Mixed Precision scaling
    opts.epsilon = 1e-8f;
    opts.isProfiling = false;
    opts.legacyRenderFlag = false;
    opts.lambdaDist = lambdaDist;
    opts.aabbMin = make_float3(-1.5f, -1.5f, -1.5f);
    opts.aabbMax = make_float3(1.5f, 1.5f, 1.5f);

    std::cout << "\nLoading training dataset..." << std::endl;
    DataLoader dataset(dataset_path, opts.rayChunkSize);

    // Reuse the train loader for between-epoch + validation rendering. A separate test loader
    // needs transforms_test.json (the COLMAP path does NOT produce one) and maps a second full-res
    // zero-copy image buffer that OOMs 8GB cards -- reuse keeps one frame + one mapping.
    DataLoader& test_dataset = dataset;
    std::cout << "\nReusing train dataset for rendering (no separate test load)." << std::endl;

    // Use the cascade count estimated by colmap2nerf for this scene (defaults to 1 for
    // bounded/legacy datasets). This drives the occupancy-grid extent and ray AABB scale.
    opts.numCascades = dataset.getNumCascades();
    if (numCascades != 0 ) opts.numCascades = numCascades;
    std::cout << "Using numCascades = " << opts.numCascades << " (from dataset)" << std::endl;

    const int HOLDOUT_EVERY = 8;   // mip-360 protocol: every 8th image held out for validation
    int holdoutEvery = runValidation ? HOLDOUT_EVERY : 0;
    std::cout << "lambdaDist=" << lambdaDist << " K=" << samplesPerVoxel << " steps=" << totalSteps
              << " validate=" << runValidation << " video=" << renderVideo
              << " maxLR=" << maxLR << " minLR=" << minLR << std::endl;

    std::cout << "\nInitializing NeRF Model..." << std::endl;
    InstantNerf nerf;
    nerf.init(opts);
    printVRAMUsage();

    // ==========================================
    // 2. Training Loop Variables
    // ==========================================
    int totalEpochs = totalEpochsParam; 
    int total_rays = dataset.getTotalRays();
    int trainSteps = 0;
    
    int chunks_per_epoch = (total_rays + opts.rayChunkSize - 1) / opts.rayChunkSize;
    int total_chunks = totalEpochs * chunks_per_epoch;
    int current_chunk = 0;
    
    // Cosine Annealing Learning Rate limits.
    // NOTE: min_lr must be well below max_lr or the cosine schedule below is a
    // no-op and the LR stays pinned at the (high) max for all of training. A
    // constant 1e-2 LR is a major instability source late in training â€” exactly
    // when the hash grid has learned large features and gradients spike â€” so we
    // decay down to a small floor to keep late-stage updates stable.
    float max_lr = opts.learningRate;   // = maxLR
    float min_lr = minLR;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> seed_dist(0, 0xFFFFFFFF);

    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    
    float* d_chunk_rgb_out;
    CUDA_CHECK(cudaMalloc(&d_chunk_rgb_out, opts.rayChunkSize * 3 * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "\nStarting Training Loop (" << totalEpochs << " Epochs, " << total_rays << " rays per epoch)..." << std::endl;

    // ==========================================
    // 3. Main Training Epochs
    // ==========================================
    for (int epoch = 1; epoch <= totalEpochs; ++epoch) {
        if (totalSteps > 0 && trainSteps >= totalSteps) break;
        
        // Generate a new random seed for the DataLoader to perfectly randomize rays this epoch!
        uint32_t epoch_seed = seed_dist(rng);

        CUDA_CHECK(cudaEventRecord(start));
        auto start_time = std::chrono::high_resolution_clock::now();
        int last_trainSteps = trainSteps;
        
        for (int offset = 0; offset < total_rays; offset += opts.rayChunkSize) {

            if (totalSteps > 0 && trainSteps >= totalSteps) break;
            
            // 3.1 Cosine Annealing Learning Rate Update
            // Drive the cosine by trainSteps/totalSteps so the LR truly sweeps max->min on a
            // step-capped run (current_chunk/total_chunks barely advances and pins LR near max,
            // the late-training divergence source).
            float progress = (totalSteps > 0)
                ? fminf((float)trainSteps / (float)totalSteps, 1.0f)
                : (float)current_chunk / (float)total_chunks;
            float current_lr = min_lr + 0.5f * (max_lr - min_lr) * (1.0f + cosf(3.14159265359f * progress));
            nerf.setLearningRate(current_lr);
            current_chunk++;
            
            // 3.2 Prepare the Batch
            nerf.setMemoryMode(TRAINING);
            int chunkSize = std::min((int)opts.rayChunkSize, total_rays - offset);
            
            // Random background augmentation for robustness against solid colors
            float3 random_bg = make_float3(dist(rng), dist(rng), dist(rng));
            nerf.setBgColor(random_bg);
            
            // Fetch streaming data from host RAM -> GPU over PCIe using the zero-copy dataloader.
            // holdoutEvery (>=2 when validating) keeps every Nth image out of the randomized draw.
            dataset.fetchRayChunk(offset, chunkSize, epoch_seed, random_bg, 0, false, holdoutEvery);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            bool should_print = (current_chunk % 10 == 0) || (offset + opts.rayChunkSize >= total_rays);
            if (should_print) {
                CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
            }

            // 3.3 Train the NeRF Model
            std::vector<uint32_t> h_hitCounts(chunkSize, 0); // CPU vector to track active hits per ray
            nerf.trainWithRaysHit(
                dataset.getChunkRaysO(), 
                dataset.getChunkRaysD(), 
                dataset.getChunkRgbTrue(), 
                chunkSize, 
                trainSteps, 
                h_hitCounts.data(),
                should_print ? d_chunk_rgb_out : nullptr,
                0
            );
            
            // 3.4 Print Metrics
            if (should_print) {
                int threads = 256;
                int blocks = (chunkSize + threads - 1) / threads;
                mse_kernel<<<blocks, threads>>>(d_chunk_rgb_out, dataset.getChunkRgbTrue(), d_loss, chunkSize);
                
                float h_loss;
                CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
                float mse = h_loss / chunkSize;
                float psnr = -10.0f * log10f(fmaxf(mse, 1e-8f)); // Peak Signal to Noise Ratio
                
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> elapsed = current_time - start_time;
                float elapsed_sec = elapsed.count() > 0.0f ? elapsed.count() : 1e-5f;
                int steps_passed = std::max(1, trainSteps - last_trainSteps);
                float steps_per_sec = steps_passed / elapsed_sec;
                float ms_per_step = (elapsed_sec * 1000.0f) / steps_passed;
                
                int barWidth = 20;
                float ep_progress = (float)(offset + chunkSize) / total_rays;
                int pos = barWidth * ep_progress;
                
                std::cout << "\rEpoch " << epoch << "/" << totalEpochs << " [";
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(ep_progress * 100.0f) << "% "
                          << "| Step " << trainSteps << " "
                          << "| Loss: " << std::fixed << std::setprecision(5) << mse << " "
                          << "(PSNR: " << std::setprecision(1) << psnr << ") "
                          << "| " << std::setprecision(0) << steps_per_sec << " steps/s "
                          << "| " << std::setprecision(1) << ms_per_step << " ms/step    " << std::flush;
                          
                start_time = std::chrono::high_resolution_clock::now();
                last_trainSteps = trainSteps;
            }
        }
        std::cout << std::endl;
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        std::cout << "Epoch " << epoch << " completed in " << ms / 1000.0f << " seconds." << std::endl;

        // ==========================================
        // 4. End-of-Epoch Render Testing
        // ==========================================
        std::cout << "Rendering Test Images for Epoch " << epoch << "..." << std::endl;
        int test_w = test_dataset.getWidth();
        int test_h = test_dataset.getHeight();
        int pixels = test_w * test_h;
        
        float* d_render_out;
        CUDA_CHECK(cudaMalloc(&d_render_out, pixels * 3 * sizeof(float)));
        uint8_t* d_render_byte;
        CUDA_CHECK(cudaMalloc(&d_render_byte, pixels * 3 * sizeof(uint8_t)));
        std::vector<uint8_t> h_render_byte(pixels * 3);

        // ALWAYS switch back to INFERENCE mode before rendering full images
        nerf.setMemoryMode(INFERENCE);
        nerf.setBgColor(make_float3(1.0f, 1.0f, 1.0f)); // White background for testing

        // Just test on the first 3 images in the dataset
        int num_test_frames = std::min(3, (int)(test_dataset.getTotalRays() / pixels));
        // The (reused) train loader sizes its ray buffers to rayChunkSize, so a full image does NOT
        // fit one fetch -- render each image tile-by-tile into its slice of d_render_out.
        int render_tile = (int)opts.rayChunkSize;
        for (int i = 0; i < num_test_frames; ++i) {
            for (int off = 0; off < pixels; off += render_tile) {
                int count = std::min(render_tile, pixels - off);
                test_dataset.fetchRayChunk(i * pixels + off, count, 0, make_float3(1.0f, 1.0f, 1.0f), 0, /*forceSequential=*/true);
                CUDA_CHECK(cudaDeviceSynchronize());
                nerf.renderImageHit(test_dataset.getChunkRaysO(), test_dataset.getChunkRaysD(), count, d_render_out + (size_t)off * 3, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Convert HDR float to LDR byte and save
            int blocks = (pixels + 255) / 256;
            float_to_byte_kernel<<<blocks, 256>>>(d_render_out, d_render_byte, pixels);
            CUDA_CHECK(cudaMemcpy(h_render_byte.data(), d_render_byte, pixels * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            
            std::filesystem::create_directories("../benchmarks/frames");
            std::string filename = "../benchmarks/frames/epoch_" + std::to_string(epoch) + "_view_" + std::to_string(i) + ".png";
            stbi_write_png(filename.c_str(), test_w, test_h, 3, h_render_byte.data(), test_w * 3);
            std::cout << "Saved " << filename << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_render_out));
        CUDA_CHECK(cudaFree(d_render_byte));

        // Save a checkpoint model halfway through
        if (epoch == 2) {
            std::cout << "Saving Epoch 2 checkpoint..." << std::endl;
            std::filesystem::create_directories("../benchmarks/saved");
            nerf.save("../benchmarks/saved/model_epoch_2.inerf");
        }
    }

    // ==========================================
    // 4.5 Held-out Validation PSNR (every Nth image is never trained on)
    // ==========================================
    if (runValidation) {
        std::cout << "\nComputing held-out validation PSNR (every " << HOLDOUT_EVERY
                  << "th image, excluded from training)..." << std::endl;
        nerf.setMemoryMode(INFERENCE);
        float3 valBg = make_float3(1.0f, 1.0f, 1.0f);   // fixed white bg: render + GT both over it
        nerf.setBgColor(valBg);

        int valW = test_dataset.getWidth();
        int valH = test_dataset.getHeight();
        int valPixels = valW * valH;
        int numImgs = (int)(test_dataset.getTotalRays() / (uint32_t)valPixels);
        int renderTile = (int)opts.rayChunkSize;

        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        long long totalValPixels = 0;
        int numValImgs = 0;
        for (int i = 0; i < numImgs; ++i) {
            if (i % HOLDOUT_EVERY != 0) continue;       // only the held-out images
            for (int off = 0; off < valPixels; off += renderTile) {
                int count = std::min(renderTile, valPixels - off);
                test_dataset.fetchRayChunk(i * valPixels + off, count, 0, valBg, 0, /*forceSequential=*/true);
                CUDA_CHECK(cudaDeviceSynchronize());
                nerf.renderImageHit(test_dataset.getChunkRaysO(), test_dataset.getChunkRaysD(),
                                    count, d_chunk_rgb_out, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
                int threads = 256, blocks = (count + threads - 1) / threads;
                mse_kernel<<<blocks, threads>>>(d_chunk_rgb_out, test_dataset.getChunkRgbTrue(), d_loss, count);
                totalValPixels += count;
            }
            numValImgs++;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        float valMse  = h_loss / (float)std::max(1LL, totalValPixels);
        float valPsnr = -10.0f * log10f(fmaxf(valMse, 1e-8f));
        std::cout << "\n========================================" << std::endl;
        std::cout << "  HELD-OUT VALIDATION PSNR: " << std::fixed << std::setprecision(2)
                  << valPsnr << " dB" << std::endl;
        std::cout << "  (" << numValImgs << " held-out images, " << totalValPixels << " pixels)" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // ==========================================
    // 5. Export 360-Degree Novel View Synthesis
    // ==========================================
    if (renderVideo) {
    std::cout << "\nGenerating 360 degree video frames..." << std::endl;
    int video_w = 800;
    int video_h = 800;
    int video_pixels = video_w * video_h;
    
    // We approximate standard camera extrinsics
    float camera_angle_x = dataset.getCameraAngleX();
    float focal_length = 0.5f * video_w / tanf(0.5f * camera_angle_x);

    float* d_video_rays_o;
    float* d_video_rays_d;
    float* d_video_out;
    uint8_t* d_video_byte;
    CUDA_CHECK(cudaMalloc(&d_video_rays_o, video_pixels * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_video_rays_d, video_pixels * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_video_out, video_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_video_byte, video_pixels * 3 * sizeof(uint8_t)));
    std::vector<uint8_t> h_video_byte(video_pixels * 3);

    nerf.setBgColor(make_float3(1.0f, 1.0f, 1.0f));
    
        float3 center, up;
    dataset.getSceneOrientation(center, up);

    float O_up[3] = {up.x, up.y, up.z};
    float O_forward[3] = {1.0f, 0.0f, 0.0f};
    if (abs(O_up[0]) > 0.9f) { O_forward[0] = 0.0f; O_forward[1] = 1.0f; O_forward[2] = 0.0f; }
    float dot_uf = O_up[0]*O_forward[0] + O_up[1]*O_forward[1] + O_up[2]*O_forward[2];
    O_forward[0] -= dot_uf * O_up[0]; O_forward[1] -= dot_uf * O_up[1]; O_forward[2] -= dot_uf * O_up[2];
    float len_f = sqrtf(O_forward[0]*O_forward[0] + O_forward[1]*O_forward[1] + O_forward[2]*O_forward[2]);
    if(len_f>0){O_forward[0]/=len_f; O_forward[1]/=len_f; O_forward[2]/=len_f;}
    
    float O_right[3] = {
        O_up[1]*O_forward[2] - O_up[2]*O_forward[1],
        O_up[2]*O_forward[0] - O_up[0]*O_forward[2],
        O_up[0]*O_forward[1] - O_up[1]*O_forward[0]
    };

    // Fly a virtual camera in a circle around the model
    for (int frame = 0; frame < 120; ++frame) {
        float angle = frame * (2.0f * 3.14159265f / 120.0f);
        float scale = 1.0f;
        float radius = 4.0311f * scale; 
        float elev = 30.0f * 3.14159265f / 180.0f; // 30 degrees elevation
        
        float r_z = radius * sinf(elev);
        float r_xy = radius * cosf(elev);
        float local_x = r_xy * cosf(angle);
        float local_y = r_xy * sinf(angle);
        float local_z = r_z;

        float cx = center.x + O_forward[0]*local_x + O_right[0]*local_y + O_up[0]*local_z;
        float cy = center.y + O_forward[1]*local_x + O_right[1]*local_y + O_up[1]*local_z;
        float cz = center.z + O_forward[2]*local_x + O_right[2]*local_y + O_up[2]*local_z;

        float zc_x = cx - center.x, zc_y = cy - center.y, zc_z = cz - center.z;
        float len_zc = sqrtf(zc_x*zc_x + zc_y*zc_y + zc_z*zc_z);
        if(len_zc > 0) { zc_x/=len_zc; zc_y/=len_zc; zc_z/=len_zc; }

        float xc_x = O_up[1]*zc_z - O_up[2]*zc_y;
        float xc_y = O_up[2]*zc_x - O_up[0]*zc_z;
        float xc_z = O_up[0]*zc_y - O_up[1]*zc_x;
        float len_xc = sqrtf(xc_x*xc_x + xc_y*xc_y + xc_z*xc_z);
        if (len_xc > 0.0f) { xc_x /= len_xc; xc_y /= len_xc; xc_z /= len_xc; }
        else { xc_x = O_right[0]; xc_y = O_right[1]; xc_z = O_right[2]; }

        float yc_x = zc_y * xc_z - zc_z * xc_y;
        float yc_y = zc_z * xc_x - zc_x * xc_z;
        float yc_z = zc_x * xc_y - zc_y * xc_x;

        int blocks = (video_pixels + 255) / 256;
        generate_custom_rays_kernel<<<blocks, 256>>>(
            video_w, video_h, focal_length,
            xc_x, yc_x, zc_x, cx,
            xc_y, yc_y, zc_y, cy,
            xc_z, yc_z, zc_z, cz,
            (float3*)d_video_rays_o, (float3*)d_video_rays_d
        );        CUDA_CHECK(cudaDeviceSynchronize());

        nerf.renderImageHit((float3*)d_video_rays_o, (float3*)d_video_rays_d, video_pixels, d_video_out, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float_to_byte_kernel<<<blocks, 256>>>(d_video_out, d_video_byte, video_pixels);
        CUDA_CHECK(cudaMemcpy(h_video_byte.data(), d_video_byte, video_pixels * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        
        char filename[256];
        sprintf(filename, "../benchmarks/frames/video_frame_%03d.png", frame);
        stbi_write_png(filename, video_w, video_h, 3, h_video_byte.data(), video_w * 3);
        if (frame % 10 == 0) std::cout << "Rendered " << frame << "/120 frames..." << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_video_rays_o));
    CUDA_CHECK(cudaFree(d_video_rays_d));
    CUDA_CHECK(cudaFree(d_video_out));
    CUDA_CHECK(cudaFree(d_video_byte));

    std::cout << "Creating MP4 with FFmpeg..." << std::endl;
    system("ffmpeg -y -framerate 30 -i ../benchmarks/frames/video_frame_%03d.png -c:v libx264 -pix_fmt yuv420p ../benchmarks/nerf_360.mp4");
    } // end if (renderVideo)

    std::cout << "\nSaving trained NeRF to ../benchmarks/saved/model.inerf..." << std::endl;
    std::filesystem::create_directories("../benchmarks/saved");
    nerf.save("../benchmarks/saved/model.inerf");
    std::cout << "Save complete!" << std::endl;

    // ==========================================
    // 6. Cleanup
    // ==========================================
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_chunk_rgb_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Training Complete." << std::endl;
}

