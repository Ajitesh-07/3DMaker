#include "../InstantNerf.h"
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../third_party/stb_image_write.h"

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

    // C2W * dir
    float world_dir_x = c00*dir_x + c01*dir_y + c02*dir_z;
    float world_dir_y = c10*dir_x + c11*dir_y + c12*dir_z;
    float world_dir_z = c20*dir_x + c21*dir_y + c22*dir_z;

    d_rays_o[idx] = make_float3(c03, c13, c23);
    d_rays_d[idx] = make_float3(world_dir_x, world_dir_y, world_dir_z);
}

int main(int argc, char** argv) {
    std::cout << "Loading NeRF Model for Rendering..." << std::endl;
    
    InstantNerf nerf;
    try {
        nerf.load("../benchmarks/saved/model.inerf");
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully!" << std::endl;

    std::cout << "\nGenerating 360 degree video frames..." << std::endl;
    std::filesystem::create_directories("../benchmarks/frames_render");
    
    int video_w = 800; // UHD 4K resolution
    int video_h = 800;
    int video_pixels = video_w * video_h;
    float camera_angle_x = 0.6911112070083618f;
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
    nerf.setProfiling(true);
    for (int frame = 0; frame < 1; ++frame) {
        float angle = frame * (2.0f * 3.14159265f / 120.0f);
        float scale = 1.0f;
        float radius = 4.0311f * scale; 
        float elev = 30.0f * 3.14159265f / 180.0f; // 30 degrees elevation
        
        float pz = radius * sinf(elev);
        float r_xy = radius * cosf(elev);
        float px = r_xy * cosf(angle);
        float py = r_xy * sinf(angle);

        // Z_c points BACKWARDS from camera to origin (since camera looks at -Z)
        float zc_x = px / radius, zc_y = py / radius, zc_z = pz / radius;

        // X_c points RIGHT. World UP is (0,0,1). X_c = normalize(cross(UP, Z_c))
        float xc_x = -zc_y, xc_y = zc_x, xc_z = 0.0f;
        float len_x = sqrtf(xc_x*xc_x + xc_y*xc_y);
        if (len_x > 0.0f) { xc_x /= len_x; xc_y /= len_x; }

        // Y_c points UP. Y_c = cross(Z_c, X_c)
        float yc_x = zc_y * xc_z - zc_z * xc_y;
        float yc_y = zc_z * xc_x - zc_x * xc_z;
        float yc_z = zc_x * xc_y - zc_y * xc_x;

        int blocks = (video_pixels + 255) / 256;
        generate_custom_rays_kernel<<<blocks, 256>>>(
            video_w, video_h, focal_length,
            xc_x, yc_x, zc_x, px,
            xc_y, yc_y, zc_y, py,
            xc_z, yc_z, zc_z, pz,
            (float3*)d_video_rays_o, (float3*)d_video_rays_d
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        nerf.renderImage((float3*)d_video_rays_o, (float3*)d_video_rays_d, video_pixels, d_video_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float_to_byte_kernel<<<blocks, 256>>>(d_video_out, d_video_byte, video_pixels);
        CUDA_CHECK(cudaMemcpy(h_video_byte.data(), d_video_byte, video_pixels * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        
        char filename[256];
        sprintf(filename, "../benchmarks/frames_render/video_frame_%03d.png", frame);
        stbi_write_png(filename, video_w, video_h, 3, h_video_byte.data(), video_w * 3);
        
        if (frame == 0) {
            std::cout << "\n--- Profiling Stats for Frame 0 ---" << std::endl;
            nerf.printStats();
            nerf.setProfiling(false); // disable for remainder
            std::cout << "-----------------------------------\n" << std::endl;
        }

        if (frame % 10 == 0) std::cout << "Rendered " << frame << "/120 frames..." << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_video_rays_o));
    CUDA_CHECK(cudaFree(d_video_rays_d));
    CUDA_CHECK(cudaFree(d_video_out));
    CUDA_CHECK(cudaFree(d_video_byte));

    // std::cout << "Creating MP4 with FFmpeg..." << std::endl;
    // system("ffmpeg -y -framerate 30 -i ../benchmarks/frames_render/video_frame_%03d.png -c:v libx264 -pix_fmt yuv420p ../benchmarks/rendered_video.mp4");

    // std::cout << "Render Complete! Saved as ../benchmarks/rendered_video.mp4" << std::endl;
    return 0;
}
