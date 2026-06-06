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


DataLoader::DataLoader(const std::string& dataset_path, bool is_training) 
    : dataset_path(dataset_path), is_training(is_training), gen(nullptr), width(0), height(0), total_rays(0) {
    parseTransformsJson();
    loadImages();
    loadDataToGPU();
}

DataLoader::~DataLoader() {
    freeVRAM();
    if (gen) curandDestroyGenerator((curandGenerator_t)gen);
}

void DataLoader::freeVRAM() {
    if (d_images_rgba) cudaFree(d_images_rgba);
    if (d_transforms) cudaFree(d_transforms);

    if (d_chunk_rays_o) cudaFree(d_chunk_rays_o);
    if (d_chunk_rays_d) cudaFree(d_chunk_rays_d);
    if (d_chunk_rgb_true) cudaFree(d_chunk_rgb_true);
    
    if (d_indices) cudaFree(d_indices);
    if (d_shuffled_indices) cudaFree(d_shuffled_indices);
    if (d_random_keys) cudaFree(d_random_keys);
    if (d_temp_storage) cudaFree(d_temp_storage);
}

void DataLoader::parseTransformsJson() {
    std::string json_file = dataset_path + (is_training ? "/transforms_train.json" : "/transforms_test.json");
    std::ifstream file(json_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << json_file << std::endl;
        exit(1);
    }
    json j;
    file >> j;

    camera_angle_x = j["camera_angle_x"];
    
    for (auto& frame : j["frames"]) {
        Frame f;
        std::string path = frame["file_path"];
        if (path.find(".png") == std::string::npos) {
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
    std::cout << "Loading " << frames.size() << " images..." << std::endl;
    for (size_t i = 0; i < frames.size(); ++i) {
        int w, h, channels;
        unsigned char* img = stbi_load(frames[i].file_path.c_str(), &w, &h, &channels, 4);
        if (!img) {
            std::cerr << "Failed to load image: " << frames[i].file_path << std::endl;
            exit(1);
        }

        if (i == 0) {
            width = w;
            height = h;
            focal_length = 0.5f * width / tanf(0.5f * camera_angle_x);
        }

        for (int p = 0; p < width * height; ++p) {
            uint8_t r = img[p * 4 + 0];
            uint8_t g = img[p * 4 + 1];
            uint8_t b = img[p * 4 + 2];
            uint8_t a = img[p * 4 + 3];

            // For now, we will store RGBA. We can blend in the kernel.
            h_images_rgba.push_back(r);
            h_images_rgba.push_back(g);
            h_images_rgba.push_back(b);
            h_images_rgba.push_back(a);
        }
        stbi_image_free(img);
    }
    total_rays = width * height * frames.size();
    std::cout << "Loaded " << total_rays << " total rays compactly. Focal length: " << focal_length << std::endl;
}

__global__ void fetchRayChunkKernel(
    int offset, int size, int total_rays,
    int width, int height, float focal_length,
    const float* d_transforms, // [num_images, 16]
    const uint8_t* d_images_rgba, // [num_images, height, width, 4]
    const uint32_t* d_shuffled_indices,
    float3* d_chunk_rays_o, float3* d_chunk_rays_d, float* d_chunk_rgb,
    float3 bg_color
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    if (offset + idx >= total_rays) return;

    uint32_t global_idx = d_shuffled_indices[offset + idx];

    int img_idx = global_idx / (width * height);
    int pixel_idx = global_idx % (width * height);
    int px = pixel_idx % width;
    int py = pixel_idx / width;

    // NeRF Camera space direction (X right, Y up, Z backwards)
    float dir_x = (px - width * 0.5f) / focal_length;
    float dir_y = -(py - height * 0.5f) / focal_length;
    float dir_z = -1.0f;

    // Normalize dir in camera space
    float inv_norm = rsqrtf(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    dir_x *= inv_norm;
    dir_y *= inv_norm;
    dir_z *= inv_norm;

    const float* c2w = &d_transforms[img_idx * 16];

    // C2W * dir
    float world_dir_x = c2w[0]*dir_x + c2w[1]*dir_y + c2w[2]*dir_z;
    float world_dir_y = c2w[4]*dir_x + c2w[5]*dir_y + c2w[6]*dir_z;
    float world_dir_z = c2w[8]*dir_x + c2w[9]*dir_y + c2w[10]*dir_z;

    // C2W * origin (0,0,0,1) scaled to fit in AABB
    float scale = 1.0f;
    float world_o_x = c2w[3] * scale;
    float world_o_y = c2w[7] * scale;
    float world_o_z = c2w[11] * scale;

    d_chunk_rays_o[idx] = make_float3(world_o_x, world_o_y, world_o_z);
    d_chunk_rays_d[idx] = make_float3(world_dir_x, world_dir_y, world_dir_z);

    // Decode 8-bit RGBA and blend with white
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

// We just need indices 0..N-1
__global__ void initIndicesKernel(uint32_t* indices, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        indices[id] = id;
    }
}

void DataLoader::setupCUBShuffle() {
    CUDA_CHECK(cudaMalloc(&d_indices, total_rays * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_shuffled_indices, total_rays * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_random_keys, total_rays * sizeof(float)));

    int blockSize = 256;
    int gridSize = (total_rays + blockSize - 1) / blockSize;
    initIndicesKernel<<<gridSize, blockSize>>>(d_indices, total_rays);
    CUDA_CHECK(cudaDeviceSynchronize());

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    gen = (void*)generator;

    // Determine temp storage size for CUB sort
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_random_keys, d_random_keys, // keys
        d_indices, d_shuffled_indices, // values
        total_rays
    );
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}

void DataLoader::loadDataToGPU() {
    std::cout << "Allocating Compact VRAM Dataset..." << std::endl;
    
    // Store original images compactly (256 MB)
    CUDA_CHECK(cudaMalloc(&d_images_rgba, total_rays * 4 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_images_rgba, h_images_rgba.data(), total_rays * 4 * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Store camera matrices (~6.4 KB)
    std::vector<float> h_transforms(frames.size() * 16);
    for (size_t i = 0; i < frames.size(); ++i) {
        for (int j = 0; j < 16; ++j) {
            h_transforms[i * 16 + j] = frames[i].transform_matrix[j];
        }
    }
    CUDA_CHECK(cudaMalloc(&d_transforms, frames.size() * 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transforms, h_transforms.data(), frames.size() * 16 * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate Tiny Chunk Buffers (1 Million Rays = 12 MB)
    int max_chunk_size = 1024 * 1024; 
    CUDA_CHECK(cudaMalloc(&d_chunk_rays_o, max_chunk_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_chunk_rays_d, max_chunk_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_chunk_rgb_true, max_chunk_size * 3 * sizeof(float)));

    setupCUBShuffle();
    if (is_training) {
        shuffleRays(0);
    } else {
        CUDA_CHECK(cudaMemcpy(d_shuffled_indices, d_indices, total_rays * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Compact Dataset completely loaded to VRAM." << std::endl;
}

void DataLoader::shuffleRays(cudaStream_t stream) {
    curandGenerator_t generator = (curandGenerator_t)gen;
    curandSetStream(generator, stream);
    
    // Generate random floats [0.0, 1.0]
    curandGenerateUniform(generator, d_random_keys, total_rays);

    // Sort indices by the random keys to shuffle them (No more moving full rays!)
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_random_keys, d_random_keys,
        d_indices, d_shuffled_indices,
        total_rays, 0, sizeof(float)*8, stream
    );
}

void DataLoader::fetchRayChunk(int offset, int size, float3 bg_color, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    fetchRayChunkKernel<<<gridSize, blockSize, 0, stream>>>(
        offset, size, total_rays,
        width, height, focal_length,
        d_transforms, d_images_rgba, d_shuffled_indices,
        d_chunk_rays_o, d_chunk_rays_d, d_chunk_rgb_true,
        bg_color
    );
}
