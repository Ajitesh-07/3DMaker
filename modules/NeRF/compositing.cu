#include "InstantNerf.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void render_rays_kernel(
    const uint32_t num_rays,
    const uint32_t base,
    const uint32_t raysDone,
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

    uint32_t offset = ray_offsets[r + raysDone] - base;
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

__device__ __forceinline__ float disparity_warp(float t, float t_near, float inv_g_range) {
    return (1.0f - t_near / fmaxf(t, 1e-4f)) * inv_g_range;
}

__global__ void render_rays_distortion_kernel(
    const uint32_t num_rays,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_sigma,
    const float* __restrict__ depth_chunk,
    const float* __restrict__ sigma_chunk,
    const float* __restrict__ rgb_output,
    const float* __restrict__ rgb_true,
    float* __restrict__ final_rgb,   
    float* __restrict__ final_depth,
    float* __restrict__ phi_out,
    float* __restrict__ dw_out,
    float* __restrict weight_sum,
    float lambda_dist,
    float lambda_depth,
    float3 bg_color
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;

    uint32_t offset = ray_offsets[r];
    uint32_t count = num_steps[r];
    float d_prior     = depth_chunk ? depth_chunk[r] : -1.0f;   // null = depth supervision off -> gated out below
    float sigma_prior = sigma_chunk ? sigma_chunk[r] : -1.0f;

    float depth_weight_loss = (d_prior >= 0) ? lambda_depth : 0.0f; 

    float t_near    = (count > 0) ? fmaxf(t_sorted[offset], 1e-4f) : 1.0f;
    float t_far_ray = (count > 0) ? t_sorted[offset + count - 1] : 1.0f;
    float g_far = 1.0f - t_near / fmaxf(t_far_ray, t_near + 1e-4f);
    float inv_g_range = (g_far > 1e-6f) ? (1.0f / g_far) : 0.0f;

    float T = 1.0f;
    float r_c = 0.0f, g_c = 0.0f, b_c = 0.0f;
    float depth = 0.0f;

    float c = 2.0f / 3;

    float dw_global_sum = 0.0f;
    float dwm_global_sum = 0.0f;

    float dw_sum = 0.0f;
    float dwm_sum = 0.0f;
    float depth_loss_sum = 0.0f;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];

        float t_next = (i < count - 1) ? t_sorted[idx + 1] : (t + 1e-3f);
        float delta_t = t_next - t;
        if (delta_t < 0.0f) { delta_t = 0.0f; t_next = t; }

        float s      = disparity_warp(t,      t_near, inv_g_range);
        float s_next = disparity_warp(t_next, t_near, inv_g_range);
        float m = 0.5f * (s + s_next);

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

        float tx = (t - d_prior) / sigma_prior;
        float depthWeight = expf(-0.5f * tx*tx)*delta_t;
        depth_loss_sum += depthWeight;

        dw_out[idx] = weight;
    }

    depth_loss_sum = 1.0f / fmaxf(depth_loss_sum, 1e-8f);

    float weight_g_sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];

        float t_next = (i < count - 1) ? t_sorted[idx + 1] : (t + 1e-3f);
        float delta_t = t_next - t;
        if (delta_t < 0.0f) { delta_t = 0.0f; t_next = t; }

        float s      = disparity_warp(t,      t_near, inv_g_range);
        float s_next = disparity_warp(t_next, t_near, inv_g_range);
        float m = 0.5f * (s + s_next);
        float delta_s = s_next - s;
        if (delta_s < 0.0f) delta_s = 0.0f;

        float crrWeight = dw_out[idx];

        float tx = (t - d_prior) / sigma_prior;
        float depthWeight = expf(-0.5f * tx*tx)*delta_t * depth_loss_sum;
        float depthLoss = -depthWeight * (1.0f / fmaxf(crrWeight, 1e-6)); 

        float dl_bilinear = m * (2 * dw_sum - dw_global_sum) + (dwm_global_sum - 2*dwm_sum);
        dw_out[idx] = lambda_dist*(2*(dl_bilinear) + c * crrWeight * delta_s) + depth_weight_loss*depthLoss;
        weight_g_sum += crrWeight*(lambda_dist*(2*(dl_bilinear) + c * crrWeight * delta_s) + depth_weight_loss*depthLoss);

        dw_sum += crrWeight;
        dwm_sum += crrWeight * m;
    }

    weight_sum[r] = weight_g_sum;

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
    const float* d_depth_chunk,
    const float* d_sigma_chunk,
    const float* d_rgb_output,
    const float* d_rgb_true,
    float* d_render_rgb,
    float* d_render_depth,
    float* d_phi_out,
    float* d_dw_out,
    float* d_weight_sum,
    float lambda_dist,
    float lambda_depth,
    float3 bg_color,
    uint32_t base,
    uint32_t raysDone,
    cudaStream_t stream
) {
    if (num_rays == 0) return;
    
    constexpr int BS = 256;
    int gs = (num_rays + BS - 1) / BS;

    if (d_dw_out == nullptr) {
        render_rays_kernel<<<gs, BS, 0, stream>>>(
            num_rays,
            base,
            raysDone,
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
    } else {
        render_rays_distortion_kernel<<<gs, BS, 0, stream>>>(
            num_rays,
            d_ray_offsets,
            d_num_steps,
            d_t_sorted,
            d_density_sigma,
            d_depth_chunk,
            d_sigma_chunk,
            d_rgb_output,
            d_rgb_true,
            d_render_rgb,
            d_render_depth,
            d_phi_out,
            d_dw_out,
            d_weight_sum,
            lambda_dist,
            lambda_depth,
            bg_color
        );
    }
}
