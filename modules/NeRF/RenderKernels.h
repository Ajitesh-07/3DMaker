#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void wrapper_generate_custom_rays(
    int video_w, int video_h, float focal_length,
    float xc_x, float yc_x, float zc_x, float px,
    float xc_y, float yc_y, float zc_y, float py,
    float xc_z, float yc_z, float zc_z, float pz,
    float3* d_rays_o, float3* d_rays_d,
    cudaStream_t stream = 0
);

void wrapper_float_to_byte(
    const float* d_rgb_in,
    uint8_t* d_rgba_out,
    int num_pixels,
    cudaStream_t stream = 0
);
