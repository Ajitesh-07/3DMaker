#include "InstantNerf.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void render_rays_kernel(
    const uint32_t num_rays,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_sigma,
    const float* __restrict__ rgb_output,
    const float* __restrict__ rgb_true,
    float* __restrict__ final_rgb,   
    float* __restrict__ final_depth,
    float* __restrict__ phi_out,
    float3 bg_color
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;

    uint32_t offset = ray_offsets[r];
    uint32_t count = num_steps[r];

    float T = 1.0f;
    float r_c = 0.0f, g_c = 0.0f, b_c = 0.0f;
    float depth = 0.0f;

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];
        
        // compute delta_t
        float delta_t = 0.0f;
        if (i < count - 1) {
            delta_t = t_sorted[idx + 1] - t;
        } else {
            delta_t = 1e-3f; // some default small value for the last sample
        }

        // clamp delta_t just in case
        if (delta_t < 0.0f) delta_t = 0.0f;

        float sigma = density_sigma[idx];
        // Relu for density is already handled if MLP outputs non-negative, 
        // but since we did expf(), sigma is strictly positive.
        float alpha = 1.0f - expf(-sigma * delta_t);
        float weight = alpha * T;

        r_c += weight * rgb_output[idx * 3 + 0];
        g_c += weight * rgb_output[idx * 3 + 1];
        b_c += weight * rgb_output[idx * 3 + 2];
        
        depth += weight * t;

        T *= (1.0f - alpha);
        
        if (T < 1e-4f) break; // Early stopping
    }

    // Composite with background color using remaining transmittance
    r_c += T * bg_color.x;
    g_c += T * bg_color.y;
    b_c += T * bg_color.z;

    // Write output
    final_rgb[r * 3 + 0] = r_c;
    final_rgb[r * 3 + 1] = g_c;
    final_rgb[r * 3 + 2] = b_c;
    final_depth[r] = depth;

    if (rgb_true != nullptr && phi_out != nullptr) {
        phi_out[r * 3 + 0] = 2.0f * (r_c - rgb_true[r * 3 + 0]);
        phi_out[r * 3 + 1] = 2.0f * (g_c - rgb_true[r * 3 + 1]);
        phi_out[r * 3 + 2] = 2.0f * (b_c - rgb_true[r * 3 + 2]);
    }
}

__global__ void render_rays_distortion_kernel(
    const uint32_t num_rays,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_sigma,
    const float* __restrict__ rgb_output,
    const float* __restrict__ rgb_true,
    float* __restrict__ final_rgb,   
    float* __restrict__ final_depth,
    float* __restrict__ phi_out,
    float* __restrict__ dw_out,
    float lambda_dist,
    float3 bg_color
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;

    uint32_t offset = ray_offsets[r];
    uint32_t count = num_steps[r];

    float T = 1.0f;
    float r_c = 0.0f, g_c = 0.0f, b_c = 0.0f;
    float depth = 0.0f;

    float c = 2.0f / 3;

    float dw_global_sum = 0.0f;
    float dwm_global_sum = 0.0f;

    float dw_sum = 0.0f;
    float dwm_sum = 0.0f;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];

        float delta_t = 0.0f;
        if (i < count - 1) {
            delta_t = t_sorted[idx + 1] - t;
        } else {
            delta_t = 1e-3f;
        }

        if (delta_t < 0.0f) delta_t = 0.0f;
        
        float m = t + delta_t / 2;

        float sigma = density_sigma[idx];
        float alpha = 1.0f - expf(-sigma * delta_t);
        float weight = alpha * T;

        r_c += weight * rgb_output[idx * 3 + 0];
        g_c += weight * rgb_output[idx * 3 + 1];
        b_c += weight * rgb_output[idx * 3 + 2];
        
        depth += weight * t;

        T *= (1.0f - alpha);
        dw_global_sum += weight;
        dwm_global_sum += weight*m;

        dw_out[idx] = weight;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];

        float delta_t = 0.0f;
        if (i < count - 1) {
            delta_t = t_sorted[idx + 1] - t;
        } else {
            delta_t = 1e-3f;
        }

        if (delta_t < 0.0f) delta_t = 0.0f;

        float crrWeight = dw_out[idx];
        float m = t + delta_t / 2;

        float dl_bilinear = m * (2 * dw_sum - dw_global_sum) + (dwm_global_sum - 2*dwm_sum);
        dw_out[idx] = lambda_dist*(2*(dl_bilinear) + c * crrWeight * delta_t);

        dw_sum += crrWeight;
        dwm_sum += crrWeight * m;
    }

    r_c += T * bg_color.x;
    g_c += T * bg_color.y;
    b_c += T * bg_color.z;

    // Write output
    final_rgb[r * 3 + 0] = r_c;
    final_rgb[r * 3 + 1] = g_c;
    final_rgb[r * 3 + 2] = b_c;
    final_depth[r] = depth;

    if (rgb_true != nullptr && phi_out != nullptr) {
        phi_out[r * 3 + 0] = 2.0f * (r_c - rgb_true[r * 3 + 0]);
        phi_out[r * 3 + 1] = 2.0f * (g_c - rgb_true[r * 3 + 1]);
        phi_out[r * 3 + 2] = 2.0f * (b_c - rgb_true[r * 3 + 2]);
    }

}

extern "C" void launchVolumeRendering(
    const uint32_t num_rays,
    const uint32_t* d_ray_offsets,
    const uint32_t* d_num_steps,
    const float* d_t_sorted,
    const float* d_density_sigma,
    const float* d_rgb_output,
    const float* d_rgb_true,
    float* d_render_rgb,
    float* d_render_depth,
    float* d_phi_out,
    float3 bg_color,
    cudaStream_t stream
) {
    if (num_rays == 0) return;
    
    constexpr int BS = 256;
    int gs = (num_rays + BS - 1) / BS;
    
    render_rays_kernel<<<gs, BS, 0, stream>>>(
        num_rays,
        d_ray_offsets,
        d_num_steps,
        d_t_sorted,
        d_density_sigma,
        d_rgb_output,
        d_rgb_true,
        d_render_rgb,
        d_render_depth,
        d_phi_out,
        bg_color
    );
}
