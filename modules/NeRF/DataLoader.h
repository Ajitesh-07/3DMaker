#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

class DataLoader {
public:
    DataLoader(const std::string& dataset_path, bool is_training = true);
    ~DataLoader();

    // Generates rays for all loaded images and pushes them to VRAM
    void loadDataToGPU();

    // Getters for chunk pointers
    const float3* getChunkRaysO() const { return d_chunk_rays_o; }
    const float3* getChunkRaysD() const { return d_chunk_rays_d; }
    const float* getChunkRgbTrue() const { return d_chunk_rgb_true; }

    // Shuffles the pixel indices on the GPU
    void shuffleRays(cudaStream_t stream = 0);

    // Dynamically generates a chunk of rays on the GPU
    void fetchRayChunk(int offset, int size, cudaStream_t stream = 0);

    // Frees the compact VRAM arrays
    void freeVRAM();

    uint32_t getTotalRays() const { return total_rays; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    std::string dataset_path;
    bool is_training;

    void* gen; // curandGenerator_t hidden as void* to avoid pulling curand.h in the header

    int width;
    int height;
    float camera_angle_x;
    float focal_length;
    uint32_t total_rays;

    struct Frame {
        std::string file_path;
        float transform_matrix[16]; // 4x4 matrix
    };

    std::vector<Frame> frames;
    // Replaced full vectors with a byte array for images
    std::vector<uint8_t> h_images_rgba; // CPU buffer for 8-bit images

    // GPU Buffers for compact representation
    uint8_t* d_images_rgba = nullptr;
    float* d_transforms = nullptr;

    // GPU chunk buffers (sized for rayChunkSize)
    float3* d_chunk_rays_o = nullptr;
    float3* d_chunk_rays_d = nullptr;
    float* d_chunk_rgb_true = nullptr;

    // GPU shuffle indices
    uint32_t* d_indices = nullptr;
    uint32_t* d_shuffled_indices = nullptr;
    float* d_random_keys = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    void parseTransformsJson();
    void loadImages();
    void setupCUBShuffle();
};
