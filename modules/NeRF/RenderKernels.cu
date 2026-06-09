#include "RenderKernels.h"
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void custom_rays_kernel(
    int width, int height, float focal_length, 
    float c00, float c01, float c02, float c03,
    float c10, float c11, float c12, float c13,
    float c20, float c21, float c22, float c23,
    float3* rays_o, float3* rays_d) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int x = idx % width;
    int y = idx / width;

    float cx = width * 0.5f;
    float cy = height * 0.5f;

    // NeRF Camera space direction (X right, Y up, Z backwards)
    float dir_x = (x - cx) / focal_length;
    float dir_y = -(y - cy) / focal_length;
    float dir_z = -1.0f;

    // Direction in world space
    float dx = c00 * dir_x + c01 * dir_y + c02 * dir_z;
    float dy = c10 * dir_x + c11 * dir_y + c12 * dir_z;
    float dz = c20 * dir_x + c21 * dir_y + c22 * dir_z;

    // Normalize
    float length = sqrtf(dx*dx + dy*dy + dz*dz);
    rays_d[idx] = make_float3(dx/length, dy/length, dz/length);
    rays_o[idx] = make_float3(c03, c13, c23);
}

__global__ void float_to_byte_rgba_kernel(const float* rgb_float, uint8_t* rgba_byte, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixels) {
        float r = fmaxf(0.0f, fminf(1.0f, rgb_float[idx * 3 + 0]));
        float g = fmaxf(0.0f, fminf(1.0f, rgb_float[idx * 3 + 1]));
        float b = fmaxf(0.0f, fminf(1.0f, rgb_float[idx * 3 + 2]));
        
        rgba_byte[idx * 4 + 0] = (uint8_t)(r * 255.0f);
        rgba_byte[idx * 4 + 1] = (uint8_t)(g * 255.0f);
        rgba_byte[idx * 4 + 2] = (uint8_t)(b * 255.0f);
        rgba_byte[idx * 4 + 3] = 255;
    }
}

void wrapper_generate_custom_rays(
    int video_w, int video_h, float focal_length,
    float xc_x, float yc_x, float zc_x, float px,
    float xc_y, float yc_y, float zc_y, float py,
    float xc_z, float yc_z, float zc_z, float pz,
    float3* d_rays_o, float3* d_rays_d,
    cudaStream_t stream)
{
    int pixels = video_w * video_h;
    int blocks = (pixels + 255) / 256;
    custom_rays_kernel<<<blocks, 256, 0, stream>>>(
        video_w, video_h, focal_length,
        xc_x, yc_x, zc_x, px,
        xc_y, yc_y, zc_y, py,
        xc_z, yc_z, zc_z, pz,
        d_rays_o, d_rays_d
    );
}

void wrapper_float_to_byte(
    const float* d_rgb_in,
    uint8_t* d_rgba_out,
    int num_pixels,
    cudaStream_t stream)
{
    int blocks = (num_pixels + 255) / 256;
    float_to_byte_rgba_kernel<<<blocks, 256, 0, stream>>>(d_rgb_in, d_rgba_out, num_pixels);
}
