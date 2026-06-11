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

    void getSceneOrientation(float3& center, float3& up) const {
        if (frames.empty()) {
            center = make_float3(0, 0, 0);
            up = make_float3(0, 0, 1);
            return;
        }
        float sum_px = 0, sum_py = 0, sum_pz = 0;
        float sum_ux = 0, sum_uy = 0, sum_uz = 0;
        for (const auto& f : frames) {
            sum_px += f.transform_matrix[3];
            sum_py += f.transform_matrix[7];
            sum_pz += f.transform_matrix[11];
            sum_ux += f.transform_matrix[1];
            sum_uy += f.transform_matrix[5];
            sum_uz += f.transform_matrix[9];
        }
        center = make_float3(sum_px / frames.size(), sum_py / frames.size(), sum_pz / frames.size());
        float len = sqrtf(sum_ux*sum_ux + sum_uy*sum_uy + sum_uz*sum_uz);
        if (len > 0) {
            up = make_float3(sum_ux/len, sum_uy/len, sum_uz/len);
        } else {
            up = make_float3(0, 0, 1);
        }
    }
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
