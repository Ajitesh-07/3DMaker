#include "DataLoader.h"
#include <fstream>
#include <iostream>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb_image.h"
#include "../../third_party/json.hpp"

#include <cub/cub.cuh>
#include <curand.h>

using json = nlohmann::json;
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


DataLoader::DataLoader(const std::string& dataset_path, uint32_t ray_chunk_size, bool is_training)
        : dataset_path(dataset_path), m_is_training(is_training) {
    int width = 0;
    int height = 0;
    m_ray_chunk_size = ray_chunk_size;

    parseTransformsJson();
    loadImages();
    loadDataToGPU();
}

DataLoader::~DataLoader() {
    freeVRAM();
    if (h_images_rgba) cudaFreeHost(h_images_rgba);
}

void DataLoader::freeVRAM() {
    if (d_transforms) cudaFree(d_transforms);
    if (d_chunk_rays_o) cudaFree(d_chunk_rays_o);
    if (d_chunk_rays_d) cudaFree(d_chunk_rays_d);
    if (d_chunk_rgb_true) cudaFree(d_chunk_rgb_true);
}


void DataLoader::parseTransformsJson() {
    std::string json_file = dataset_path + (m_is_training ? "/transforms_train.json" : "/transforms_test.json");
    std::ifstream file(json_file);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << json_file << std::endl;
        exit(1);
    }

    json j;
    file >> j;

    camera_angle_x = j["camera_angle_x"];

    // Optional: number of spatial-occupancy cascades estimated by colmap2nerf for this scene.
    // Absent in legacy/synthetic datasets -> stays 1 (bounded, single occupancy grid).
    if (j.contains("num_cascades")) {
        num_cascades = j["num_cascades"];
        if (num_cascades < 1) num_cascades = 1;
    }

    for (auto& frame : j["frames"]) {
        Frame f;
        std::string path = frame["file_path"];
        if (path.find(".png") == std::string::npos && 
            path.find(".jpg") == std::string::npos && 
            path.find(".jpeg") == std::string::npos) {
            path += ".png";
        }

        f.file_path = dataset_path + "/" + path;

                int idx = 0;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                f.transform_matrix[idx++] = frame["transform_matrix"][row][col];
            }
        }
        frames.push_back(f);
    }
}

void DataLoader::loadImages() {
    if (frames.empty()) return;
    std::cout << "Loading " << frames.size() << " images..." << std::endl;

    int w, h, channels;
    unsigned char* img0 = stbi_load(frames[0].file_path.c_str(), &w, &h, &channels, 4);
    if (!img0) {
        std::cerr << "Failed to load image: " << frames[0].file_path << std::endl;
        exit(1);
    }

    width = w;
    height = h;
    focal_length = 0.5f * width / tanf(0.5f * camera_angle_x);

    size_t pixels_per_image = width * height;
    size_t bytes_per_image = pixels_per_image * 4;

    CUDA_CHECK(cudaHostAlloc((void**)&h_images_rgba, frames.size() * bytes_per_image, cudaHostAllocMapped));

    std::memcpy(h_images_rgba, img0, bytes_per_image);
    stbi_image_free(img0);

    for (int i = 1; i < (int)frames.size(); ++i) {
        int thread_w, thread_h, thread_channels;
        
        unsigned char* img = stbi_load(frames[i].file_path.c_str(), &thread_w, &thread_h, &thread_channels, 4);
        
        if (!img) {
            #pragma omp critical
            {
                std::cerr << "Failed to load image: " << frames[i].file_path << std::endl;
            }
            continue;
        }

        size_t offset = i * bytes_per_image;
        std::memcpy(h_images_rgba + offset, img, bytes_per_image);

        stbi_image_free(img);
    }

    total_rays = width * height * frames.size();
    std::cout << "Loaded " << total_rays << " total rays compactly via Pinned Memory. Focal length: " << focal_length << std::endl;
}

void DataLoader::loadDataToGPU() {
    std::cout << "Mapping Pinned Memory to GPU..." << std::endl;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_images_rgba, (void*)h_images_rgba, 0));

    std::vector<float> h_transforms(frames.size() * 16);
    for (size_t i = 0; i < frames.size(); ++i) {
        for (int j = 0; j < 16; ++j) {
            h_transforms[i * 16 + j] = frames[i].transform_matrix[j];
        }
    }
    CUDA_CHECK(cudaMalloc(&d_transforms, frames.size() * 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transforms, h_transforms.data(), frames.size() * 16 * sizeof(float), cudaMemcpyHostToDevice));

    if (!m_is_training) {
        m_ray_chunk_size = width * height; // Ensure buffer is large enough for a full image
    }

    CUDA_CHECK(cudaMalloc(&d_chunk_rays_o, m_ray_chunk_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_chunk_rays_d, m_ray_chunk_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_chunk_rgb_true, m_ray_chunk_size * 3 * sizeof(float)));
}

__device__ uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__global__ void fetchRayChunkStreamingKernel(
    int offset, int size, int total_rays, uint32_t seed, bool randomize,
    int width, int height, float focal_length,
    const float* d_transforms,
    const uint8_t* d_images_rgba,
    float3* d_chunk_rays_o, float3* d_chunk_rays_d, float* d_chunk_rgb,
    float3 bg_color
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    uint32_t global_idx;
    if (randomize) {
        uint32_t rand_val = pcg_hash(offset + idx + seed);
        global_idx = rand_val % total_rays;
    } else {
        global_idx = (offset + idx) % total_rays;
    }

    int img_idx = global_idx / (width * height);
    int pixel_idx = global_idx % (width * height);
    int px = pixel_idx % width;
    int py = pixel_idx / width;

    float dir_x = (px - width * 0.5f) / focal_length;
    float dir_y = -(py - height * 0.5f) / focal_length;
    float dir_z = -1.0f;

    float inv_norm = rsqrtf(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    dir_x *= inv_norm;
    dir_y *= inv_norm;
    dir_z *= inv_norm;

    const float* c2w = &d_transforms[img_idx * 16];

    float world_dir_x = c2w[0]*dir_x + c2w[1]*dir_y + c2w[2]*dir_z;
    float world_dir_y = c2w[4]*dir_x + c2w[5]*dir_y + c2w[6]*dir_z;
    float world_dir_z = c2w[8]*dir_x + c2w[9]*dir_y + c2w[10]*dir_z;

    float world_o_x = c2w[3];
    float world_o_y = c2w[7];
    float world_o_z = c2w[11];

    d_chunk_rays_o[idx] = make_float3(world_o_x, world_o_y, world_o_z);
    d_chunk_rays_d[idx] = make_float3(world_dir_x, world_dir_y, world_dir_z);

    uint8_t r = d_images_rgba[global_idx * 4 + 0];
    uint8_t g = d_images_rgba[global_idx * 4 + 1];
    uint8_t b = d_images_rgba[global_idx * 4 + 2];
    uint8_t a = d_images_rgba[global_idx * 4 + 3];

    float fr = r / 255.0f;
    float fg = g / 255.0f;
    float fb = b / 255.0f;
    float fa = a / 255.0f;

    fr = fr * fa + bg_color.x * (1.0f - fa);
    fg = fg * fa + bg_color.y * (1.0f - fa);
    fb = fb * fa + bg_color.z * (1.0f - fa);

    d_chunk_rgb[idx * 3 + 0] = fr;
    d_chunk_rgb[idx * 3 + 1] = fg;
    d_chunk_rgb[idx * 3 + 2] = fb;
}

void DataLoader::fetchRayChunk(int offset, int size, uint32_t seed, float3 bg_color, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    fetchRayChunkStreamingKernel<<<gridSize, blockSize, 0, stream>>>(
        offset, size, total_rays, seed, m_is_training,
        width, height, focal_length,
        d_transforms, d_images_rgba,
        d_chunk_rays_o, d_chunk_rays_d, d_chunk_rgb_true,
        bg_color
    );
}

