#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

class DataLoader {
public:
    DataLoader(const std::string& dataset_path, uint32_t ray_chunk_size, bool is_training = true);
    ~DataLoader();

    const float3* getChunkRaysO() const { return d_chunk_rays_o; }
    const float3* getChunkRaysD() const { return d_chunk_rays_d; }
    const float* getChunkRgbTrue() const { return d_chunk_rgb_true; }

    void loadDataToGPU();
    void fetchRayChunk(int offset, int size, uint32_t seed, float3 bg_color = make_float3(1.0f, 1.0f, 1.0f), cudaStream_t stream = 0);
    void freeVRAM();

    uint32_t getTotalRays() const { return total_rays; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    float getCameraAngleX() const { return camera_angle_x; }

private:
    std::string dataset_path;
    bool m_is_training;

    int width;
    int height;
    
    float camera_angle_x;
    float focal_length;
    uint32_t  m_ray_chunk_size;

    struct Frame {
        std::string file_path;
        float transform_matrix[16];
    };

    std::vector<Frame> frames;
    
    // Pinned host memory and mapped device pointer
    uint8_t* h_images_rgba = nullptr;
    uint8_t* d_images_rgba = nullptr;
    
    // GPU Buffers
    float* d_transforms = nullptr;
    float3* d_chunk_rays_o = nullptr;
    float3* d_chunk_rays_d = nullptr;
    float* d_chunk_rgb_true = nullptr;

    uint32_t total_rays;


    void parseTransformsJson();
    void loadImages();
};