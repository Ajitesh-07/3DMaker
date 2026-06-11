#include <cuda_runtime.h>
#include <stdint.h>
#include "InstantNerf.h"
#include <curand_kernel.h>
#include <cub/cub.cuh>

__device__ __forceinline__ void store_float4(float4* addr, float x, float y, float z, float w) {
    #ifndef __INTELLISENSE__
    asm volatile(
        "st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(addr), "f"(x), "f"(y), "f"(z), "f"(w)
        : "memory"
    );
    #endif
}

__global__ void compute_real_sh_degree_3_kernel(
    const uint32_t total_hits,
    const float3* __restrict__ directions, // Normalized ray directions [total_hits]
    float* __restrict__ out_sh             // Output feature matrix [total_hits * 16]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_hits) return;

    // Load direction for this specific hit
    float3 d = directions[idx];
    float x = d.x;
    float y = d.y;
    float z = d.z;

    // Precompute squares
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    // Compute all 16 coefficients directly into registers
    float sh0 = 0.28209479f;
    float sh1 = -0.48860251f * y;
    float sh2 =  0.48860251f * z;
    float sh3 = -0.48860251f * x;

    float sh4 =  1.09254843f * x * y;
    float sh5 = -1.09254843f * y * z;
    float sh6 =  0.31539156f * (3.0f * z2 - 1.0f);
    float sh7 = -1.09254843f * x * z;

    float sh8 =  0.54627421f * (x2 - y2);
    float sh9 = -0.59004359f * y * (3.0f * x2 - y2);
    float sh10 =  2.89061144f * x * y * z;
    float sh11 = -0.45704580f * y * (5.0f * z2 - 1.0f);
     
    float sh12 =  0.37317633f * z * (5.0f * z2 - 3.0f);
    float sh13 = -0.45704580f * x * (5.0f * z2 - 1.0f);
    float sh14 =  1.44530572f * z * (x2 - y2);
    float sh15 = -0.59004359f * x * (x2 - 3.0f * y2);

    float4* out_ptr_vec = reinterpret_cast<float4*>(&out_sh[idx * 16]);

    store_float4(&out_ptr_vec[0], sh0, sh1,  sh2,  sh3);
    store_float4(&out_ptr_vec[1], sh4, sh5,  sh6,  sh7);
    store_float4(&out_ptr_vec[2], sh8, sh9,  sh10, sh11);
    store_float4(&out_ptr_vec[3], sh12,sh13, sh14, sh15);
}

void launchComputeRealSHDegree3(
    const uint32_t total_hits,
    const float3* d_directions,
    float* d_out_sh,
    cudaStream_t stream
) {
    if (total_hits == 0) return;

    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (total_hits + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch execution
    compute_real_sh_degree_3_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        total_hits,
        d_directions,
        d_out_sh
    );
}

__global__ void fill_float_kernel(float* __restrict__ data, float value, uint32_t count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    data[i] = value;
}

__global__ void compute_ray_aabb_inv_kernel(
    const uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3 aabb_min,
    const float3 aabb_max,
    float3* __restrict__ rays_d_inv,
    float* __restrict__ nears,
    float* __restrict__ fars
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    float3 o = rays_o[i];
    float3 d = rays_d[i];

    if (fabsf(d.x) < 1e-8f) d.x = 1e-8f;
    if (fabsf(d.y) < 1e-8f) d.y = 1e-8f;
    if (fabsf(d.z) < 1e-8f) d.z = 1e-8f;

    float3 d_inv;
    d_inv.x = 1.0f / d.x;
    d_inv.y = 1.0f / d.y;
    d_inv.z = 1.0f / d.z;
    rays_d_inv[i] = d_inv;

    float t1x = (aabb_min.x - o.x) * d_inv.x;
    float t2x = (aabb_max.x - o.x) * d_inv.x;
    if (t1x > t2x) { float tmp = t1x; t1x = t2x; t2x = tmp; }

    float t1y = (aabb_min.y - o.y) * d_inv.y;
    float t2y = (aabb_max.y - o.y) * d_inv.y;
    if (t1y > t2y) { float tmp = t1y; t1y = t2y; t2y = tmp; }

    float t1z = (aabb_min.z - o.z) * d_inv.z;
    float t2z = (aabb_max.z - o.z) * d_inv.z;
    if (t1z > t2z) { float tmp = t1z; t1z = t2z; t2z = tmp; }

    float t_enter = fmaxf(fmaxf(t1x, t1y), t1z);
    float t_exit  = fminf(fminf(t2x, t2y), t2z);

    if (t_enter > t_exit || t_exit < 0.0f) {
        nears[i] = 0.0f;
        fars[i]  = 0.0f;
    } else {
        nears[i] = fmaxf(t_enter, 0.0f);
        fars[i]  = t_exit;
    }
}

__global__ void compute_SH_gather(
    const float3* __restrict__ d_chunk_d,
    const uint32_t* __restrict__ d_ray_indices,
    const int offset,
    const int batchSize,
    const float densityBias,
    float* __restrict__ d_density_out,
    half* __restrict__ d_color_in,
    float* __restrict__ d_density_sigma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;
    
    float3 d =  d_chunk_d[d_ray_indices[idx + offset]];
    float* densityOut = d_density_out + idx*16;

    float x = d.x;
    float y = d.y;
    float z = d.z;

    // Precompute squares
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    // Compute all 16 coefficients directly into registers
    float sh0 = 0.28209479f;
    float sh1 = -0.48860251f * y;
    float sh2 =  0.48860251f * z;
    float sh3 = -0.48860251f * x;

    float sh4 =  1.09254843f * x * y;
    float sh5 = -1.09254843f * y * z;
    float sh6 =  0.31539156f * (3.0f * z2 - 1.0f);
    float sh7 = -1.09254843f * x * z;

    float sh8 =  0.54627421f * (x2 - y2);
    float sh9 = -0.59004359f * y * (3.0f * x2 - y2);
    float sh10 =  2.89061144f * x * y * z;
    float sh11 = -0.45704580f * y * (5.0f * z2 - 1.0f);
     
    float sh12 =  0.37317633f * z * (5.0f * z2 - 3.0f);
    float sh13 = -0.45704580f * x * (5.0f * z2 - 1.0f);
    float sh14 =  1.44530572f * z * (x2 - y2);
    float sh15 = -0.59004359f * x * (x2 - 3.0f * y2);

    // Vectorized loads of density outputs (4x float4 = 16 floats)
    const float4* densityOut_vec = reinterpret_cast<const float4*>(densityOut);
    float4 d03   = densityOut_vec[0];
    float4 d47   = densityOut_vec[1];
    float4 d811  = densityOut_vec[2];
    float4 d1215 = densityOut_vec[3];

    // Sanitize the raw density logits before they (a) drive sigma via expf and
    // (b) get packed into fp16 as the color-MLP input. On dense real-world
    // scenes a logit can blow past the fp16 range; left unchecked it becomes an
    // Inf/NaN color input that taints the loss. Clamp to a safe fp16-representable
    // band and scrub NaNs so neither path can carry a non-finite value forward.
    auto sanitize_logit = [](float v) -> float {
        v = fmaxf(-30000.0f, fminf(30000.0f, v));
        return (v != v) ? 0.0f : v;
    };
    d03.x = sanitize_logit(d03.x); d03.y = sanitize_logit(d03.y);
    d03.z = sanitize_logit(d03.z); d03.w = sanitize_logit(d03.w);
    d47.x = sanitize_logit(d47.x); d47.y = sanitize_logit(d47.y);
    d47.z = sanitize_logit(d47.z); d47.w = sanitize_logit(d47.w);
    d811.x = sanitize_logit(d811.x); d811.y = sanitize_logit(d811.y);
    d811.z = sanitize_logit(d811.z); d811.w = sanitize_logit(d811.w);
    d1215.x = sanitize_logit(d1215.x); d1215.y = sanitize_logit(d1215.y);
    d1215.z = sanitize_logit(d1215.z); d1215.w = sanitize_logit(d1215.w);

    d_density_sigma[idx] = expf(fminf(d03.x - densityBias, 8.0f));

    // Pack 16 floats into 4 float4s (8 halfs per float4)
    __half2 h01 = __floats2half2_rn(d03.x, d03.y);
    __half2 h23 = __floats2half2_rn(d03.z, d03.w);
    __half2 h45 = __floats2half2_rn(d47.x, d47.y);
    __half2 h67 = __floats2half2_rn(d47.z, d47.w);
    float4 packed0_7 = make_float4(
        __int_as_float(*(uint32_t*)&h01),
        __int_as_float(*(uint32_t*)&h23),
        __int_as_float(*(uint32_t*)&h45),
        __int_as_float(*(uint32_t*)&h67)
    );

    __half2 h89 = __floats2half2_rn(d811.x, d811.y);
    __half2 h1011 = __floats2half2_rn(d811.z, d811.w);
    __half2 h1213 = __floats2half2_rn(d1215.x, d1215.y);
    __half2 h1415 = __floats2half2_rn(d1215.z, d1215.w);
    float4 packed8_15 = make_float4(
        __int_as_float(*(uint32_t*)&h89),
        __int_as_float(*(uint32_t*)&h1011),
        __int_as_float(*(uint32_t*)&h1213),
        __int_as_float(*(uint32_t*)&h1415)
    );

    __half2 hs01 = __floats2half2_rn(sh0, sh1);
    __half2 hs23 = __floats2half2_rn(sh2, sh3);
    __half2 hs45 = __floats2half2_rn(sh4, sh5);
    __half2 hs67 = __floats2half2_rn(sh6, sh7);
    float4 packed16_23 = make_float4(
        __int_as_float(*(uint32_t*)&hs01),
        __int_as_float(*(uint32_t*)&hs23),
        __int_as_float(*(uint32_t*)&hs45),
        __int_as_float(*(uint32_t*)&hs67)
    );

    __half2 hs89 = __floats2half2_rn(sh8, sh9);
    __half2 hs1011 = __floats2half2_rn(sh10, sh11);
    __half2 hs1213 = __floats2half2_rn(sh12, sh13);
    __half2 hs1415 = __floats2half2_rn(sh14, sh15);
    float4 packed24_31 = make_float4(
        __int_as_float(*(uint32_t*)&hs89),
        __int_as_float(*(uint32_t*)&hs1011),
        __int_as_float(*(uint32_t*)&hs1213),
        __int_as_float(*(uint32_t*)&hs1415)
    );

    float4* d_color_in_vec = reinterpret_cast<float4*>(&d_color_in[idx * 32]);
    store_float4(&d_color_in_vec[0], packed0_7.x, packed0_7.y, packed0_7.z, packed0_7.w);
    store_float4(&d_color_in_vec[1], packed8_15.x, packed8_15.y, packed8_15.z, packed8_15.w);
    store_float4(&d_color_in_vec[2], packed16_23.x, packed16_23.y, packed16_23.z, packed16_23.w);
    store_float4(&d_color_in_vec[3], packed24_31.x, packed24_31.y, packed24_31.z, packed24_31.w);
}

__global__ void gather_ith_data(
    const uint32_t num_rays,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float3* __restrict__ chunk_d,
    const float* __restrict__ mlp_positions,
    const uint32_t depth_index,
    uint32_t* __restrict__ active_ray_indices,
    float* __restrict__ gathered_positions,
    float3* __restrict__ gathered_directions,
    uint32_t* __restrict__ out_count
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;

    if (depth_index < num_steps[r]) {
        uint32_t insert_idx = atomicAdd(out_count, 1);
        uint32_t hit_idx = ray_offsets[r] + depth_index;

        active_ray_indices[insert_idx] = r;
        gathered_positions[insert_idx * 3 + 0] = mlp_positions[hit_idx * 3 + 0];
        gathered_positions[insert_idx * 3 + 1] = mlp_positions[hit_idx * 3 + 1];
        gathered_positions[insert_idx * 3 + 2] = mlp_positions[hit_idx * 3 + 2];
        gathered_directions[insert_idx] = chunk_d[r];

    }
}

__global__ void compute_color_grad(
    const uint32_t numRays,
    const uint32_t initialOffset,
    const uint32_t* __restrict__ num_steps,
    const uint32_t* __restrict__ ray_offsets,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_mlp_out,
    const float* __restrict__ color_mlp_out,
    const float* __restrict__ phi_chunk,
    const float* __restrict__ final_rgb,
    half* __restrict__ custom_color_grad,
    float* __restrict__ tmp_dsigma,
    float loss_scale,
    const float densityBias,
    const uint32_t totalRaysBatch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    int globalIdx = idx + initialOffset;

    float currentT = 1.0f;
    float3 currentRgb = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t rayIdx = ray_offsets[globalIdx];
    float t = t_sorted[rayIdx];
    uint32_t maxDepth = num_steps[globalIdx];

    float phi_r = phi_chunk[globalIdx*3];
    float phi_g = phi_chunk[globalIdx*3 + 1];
    float phi_b = phi_chunk[globalIdx*3 + 2];

    float final_r = final_rgb[globalIdx*3];
    float final_g = final_rgb[globalIdx*3 + 1];
    float final_b = final_rgb[globalIdx*3 + 2];

    float delta_t;

    uint32_t start_hit_chunk = ray_offsets[initialOffset]; 

    for (int depth_idx = 0; depth_idx < maxDepth; depth_idx++) {
        uint32_t global_hit_idx = rayIdx + depth_idx;
        int i = global_hit_idx - start_hit_chunk;

        float t_current = t_sorted[global_hit_idx];
        if (depth_idx == maxDepth - 1) {
            delta_t = 1e-3f;
        } else {
            delta_t = fmaxf(t_sorted[global_hit_idx + 1] - t_current, 0.0f);
        }


        float sigma = expf(fminf(density_mlp_out[i*16] - densityBias, 8.0f));
        float alpha = 1.0f - expf(-sigma*delta_t);
        float weight = currentT * alpha;

        float c_r = color_mlp_out[i*3 + 0];
        float c_g = color_mlp_out[i*3 + 1];
        float c_b = color_mlp_out[i*3 + 2];

        currentRgb.x += weight * c_r;
        currentRgb.y += weight * c_g;
        currentRgb.z += weight * c_b;

        float batch_scale = 1.0f / (float)totalRaysBatch;

        float cg_r = phi_r * weight * c_r * (1.0f - c_r) * loss_scale * batch_scale;
        float cg_g = phi_g * weight * c_g * (1.0f - c_g) * loss_scale * batch_scale;
        float cg_b = phi_b * weight * c_b * (1.0f - c_b) * loss_scale * batch_scale;

        cg_r = fmaxf(-65504.0f, fminf(65504.0f, cg_r));
        cg_g = fmaxf(-65504.0f, fminf(65504.0f, cg_g));
        cg_b = fmaxf(-65504.0f, fminf(65504.0f, cg_b));
        if (cg_r != cg_r || cg_g != cg_g || cg_b != cg_b) {
            cg_r = 0.0f; cg_g = 0.0f; cg_b = 0.0f;
        }

        custom_color_grad[i * 3 + 0] = __float2half(cg_r);
        custom_color_grad[i * 3 + 1] = __float2half(cg_g);
        custom_color_grad[i * 3 + 2] = __float2half(cg_b);

        float suff_r = final_r - currentRgb.x;
        float suff_g = final_g - currentRgb.y;
        float suff_b = final_b - currentRgb.z;

        float ds_r = currentT * c_r * (1.0f - alpha) - suff_r;
        float ds_g = currentT * c_g * (1.0f - alpha) - suff_g;
        float ds_b = currentT * c_b * (1.0f - alpha) - suff_b;

        float d_sigma_i = delta_t * (ds_r * phi_r + ds_g * phi_g + ds_b * phi_b);
        float ds = d_sigma_i * sigma * loss_scale * batch_scale;
        ds = fmaxf(-65504.0f, fminf(65504.0f, ds));
        if (ds != ds) {
            ds = 0.0f;
        }
        tmp_dsigma[i] = ds;
        currentT = currentT * (1.0f -  alpha);
    }
}

__global__ void update_compute_grad(
    const uint32_t batch_size,
    const uint32_t* __restrict__ active_ray_indices,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const uint32_t depth_index,
    const float* __restrict__ density_mlp_out,
    const float* __restrict__ color_mlp_out,
    const float* __restrict__ phi_chunk,
    const float* __restrict__ final_rgb,
    float* __restrict__ current_T,
    float* __restrict__ current_rgb,
    half* __restrict__ custom_color_grad,
    float* __restrict__ tmp_dsigma,
    float loss_scale,
    const float densityBias
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    uint32_t r = active_ray_indices[i];
    uint32_t hit_idx = ray_offsets[r] + depth_index;
    float t = t_sorted[hit_idx];

    float delta_t;
    if (depth_index < num_steps[r] - 1) {
        delta_t = fmaxf(t_sorted[hit_idx + 1] - t, 0.0f);
        if (delta_t == 0.0f) delta_t = 1e-3f;
    } else {
        delta_t = 1e-3f;
    }

    float sigma = expf(fminf(density_mlp_out[i*16] - densityBias, 8.0f));
    float alpha = 1.0f - expf(-sigma*delta_t);

    float T = current_T[r];

    float c_r = color_mlp_out[i * 3 + 0];
    float c_g = color_mlp_out[i * 3 + 1];
    float c_b = color_mlp_out[i * 3 + 2];
    
    float acc_r = current_rgb[r * 3 + 0];
    float acc_g = current_rgb[r * 3 + 1];
    float acc_b = current_rgb[r * 3 + 2];
    
    float weight = T * alpha;
    
    current_rgb[r * 3 + 0] = acc_r + weight * c_r;
    current_rgb[r * 3 + 1] = acc_g + weight * c_g;
    current_rgb[r * 3 + 2] = acc_b + weight * c_b;

    float phi_r = phi_chunk[r * 3 + 0];
    float phi_g = phi_chunk[r * 3 + 1];
    float phi_b = phi_chunk[r * 3 + 2];

    float cg_r = phi_r * weight * c_r * (1.0f - c_r) * loss_scale;
    float cg_g = phi_g * weight * c_g * (1.0f - c_g) * loss_scale;
    float cg_b = phi_b * weight * c_b * (1.0f - c_b) * loss_scale;
    cg_r = fmaxf(-65504.0f, fminf(65504.0f, cg_r));
    cg_g = fmaxf(-65504.0f, fminf(65504.0f, cg_g));
    cg_b = fmaxf(-65504.0f, fminf(65504.0f, cg_b));
    if (cg_r != cg_r) cg_r = 0.0f;
    if (cg_g != cg_g) cg_g = 0.0f;
    if (cg_b != cg_b) cg_b = 0.0f;
    custom_color_grad[i * 3 + 0] = __float2half(cg_r);
    custom_color_grad[i * 3 + 1] = __float2half(cg_g);
    custom_color_grad[i * 3 + 2] = __float2half(cg_b);
    
    float suff_r = final_rgb[r * 3 + 0] - current_rgb[r * 3 + 0];
    float suff_g = final_rgb[r * 3 + 1] - current_rgb[r * 3 + 1];
    float suff_b = final_rgb[r * 3 + 2] - current_rgb[r * 3 + 2];
    
    // Ti*ci*(1-alpha_i) - Suffix
    float ds_r = T * c_r * (1.0f - alpha) - suff_r;
    float ds_g = T * c_g * (1.0f - alpha) - suff_g;
    float ds_b = T * c_b * (1.0f - alpha) - suff_b;
    
    float d_sigma_i = delta_t * (ds_r * phi_r + ds_g * phi_g + ds_b * phi_b);
    float ds = d_sigma_i * sigma * loss_scale;
    ds = fmaxf(-65504.0f, fminf(65504.0f, ds));
    if (ds != ds) ds = 0.0f;
    tmp_dsigma[i] = ds;
    current_T[r] = T * (1.0f - alpha);
}

__device__ inline __half2 sanitize_half2(__half2 v) {
    float2 f = __half22float2(v);
    f.x = fmaxf(-65504.0f, fminf(65504.0f, f.x));
    if (f.x != f.x) f.x = 0.0f;
    f.y = fmaxf(-65504.0f, fminf(65504.0f, f.y));
    if (f.y != f.y) f.y = 0.0f;
    return __floats2half2_rn(f.x, f.y);
}

__global__ void compute_density_grad(
    const uint32_t batch_size,
    const float* __restrict__ tmp_dsigma,
    half* __restrict__ custom_dx_out,
    half* __restrict__ custom_density_grad
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    const float4* custom_dx_out_vec = reinterpret_cast<const float4*>(custom_dx_out + i * 32);
    float4 vec0 = custom_dx_out_vec[0];
    float4 vec1 = custom_dx_out_vec[1];

    __half2 h01 = *reinterpret_cast<__half2*>(&vec0.x);
    __half2 h23 = *reinterpret_cast<__half2*>(&vec0.y);
    __half2 h45 = *reinterpret_cast<__half2*>(&vec0.z);
    __half2 h67 = *reinterpret_cast<__half2*>(&vec0.w);
    
    __half2 h89   = *reinterpret_cast<__half2*>(&vec1.x);
    __half2 h1011 = *reinterpret_cast<__half2*>(&vec1.y);
    __half2 h1213 = *reinterpret_cast<__half2*>(&vec1.z);
    __half2 h1415 = *reinterpret_cast<__half2*>(&vec1.w);

    float2 f01 = __half22float2(h01);
    f01.x += tmp_dsigma[i];
    f01.x = fmaxf(-65504.0f, fminf(65504.0f, f01.x));
    if (f01.x != f01.x) f01.x = 0.0f;
    f01.y = fmaxf(-65504.0f, fminf(65504.0f, f01.y));
    if (f01.y != f01.y) f01.y = 0.0f;
    h01 = __floats2half2_rn(f01.x, f01.y);

    h23 = sanitize_half2(h23);
    h45 = sanitize_half2(h45);
    h67 = sanitize_half2(h67);
    h89 = sanitize_half2(h89);
    h1011 = sanitize_half2(h1011);
    h1213 = sanitize_half2(h1213);
    h1415 = sanitize_half2(h1415);

    float4 out0 = make_float4(
        __int_as_float(*(uint32_t*)&h01),
        __int_as_float(*(uint32_t*)&h23),
        __int_as_float(*(uint32_t*)&h45),
        __int_as_float(*(uint32_t*)&h67)
    );
    
    float4 out1 = make_float4(
        __int_as_float(*(uint32_t*)&h89),
        __int_as_float(*(uint32_t*)&h1011),
        __int_as_float(*(uint32_t*)&h1213),
        __int_as_float(*(uint32_t*)&h1415)
    );

    float4* custom_density_grad_vec = reinterpret_cast<float4*>(custom_density_grad + i * 16);
    store_float4(&custom_density_grad_vec[0], out0.x, out0.y, out0.z, out0.w);
    store_float4(&custom_density_grad_vec[1], out1.x, out1.y, out1.z, out1.w);
}

__global__ void sampleAABB(
    float* __restrict__ samples,
    float3 aabbMin,
    float3 aabbMax,
    int N,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float4 rnd = curand_uniform4(&state);
    
    float4* samples_vec = reinterpret_cast<float4*>(samples);
    store_float4(&samples_vec[idx], rnd.x, rnd.y, rnd.z, 0.0f);
}

__device__ inline float atomicMaxFloatFast(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    return __int_as_float(atomicMax(address_as_i, __float_as_int(val)));
}

__global__ void updateTmpGrid(
    const float* __restrict__ samples,
    const float* __restrict__ density_out,
    float* __restrict__ tmpGrid,
    float densityBias,
    float3 aabbMin,
    float3 aabbMax,
    uint3 gridResolution,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float4* samples_vec = reinterpret_cast<const float4*>(samples);
    float4 p_val = samples_vec[idx];

    float3 p_norm = make_float3(
        p_val.x,
        p_val.y,
        p_val.z
    );

    int ix = min(max((int)floorf(p_norm.x * gridResolution.x), 0), (int)gridResolution.x - 1);
    int iy = min(max((int)floorf(p_norm.y * gridResolution.y), 0), (int)gridResolution.y - 1);
    int iz = min(max((int)floorf(p_norm.z * gridResolution.z), 0), (int)gridResolution.z - 1);

    uint32_t gridIdx = ix + iy * gridResolution.x + iz * gridResolution.x * gridResolution.y;

    float densityValue = density_out[idx*16];
    float sigma = expf(fminf(densityValue - densityBias, 8.0f));

    atomicMaxFloatFast(&tmpGrid[gridIdx], sigma);   
}

__global__ void updateMasterGrid(
    const float* __restrict__ tmpGrid,
    float* __restrict__ masterGrid,
    float decayValue,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float prevVal = masterGrid[idx];
    float crrDensity = tmpGrid[idx];

    float val = (prevVal < 0.f) ? prevVal : fmaxf(prevVal * decayValue, crrDensity);
    masterGrid[idx] = val;
}

void inline computeSum(float* d_sum, void* d_temp_storage, size_t temp_storage_bytes, float* d_in, int num_items) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
}

__global__ void updateOccupancyGrid(
    uint32_t* d_occupancy_grid,
    const float* d_master_grid,
    float* d_sum,
    int total_cells
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float mean_density = (*d_sum) / static_cast<float>(total_cells);
    
    bool is_active = false;
    if (cell_idx < total_cells) {
        float threshold = 0.01f;
        is_active = (d_master_grid[cell_idx] > threshold);
    }

    unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, is_active);

    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        int word_idx = cell_idx / 32;
        d_occupancy_grid[word_idx] = warp_mask;
    }
}

__global__ void bitfield_max_pool(
    const uint8_t* __restrict__ level_in,
    uint32_t* __restrict__ level_out_32,
    uint3 res_out
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_cells = res_out.x * res_out.y * res_out.z;
    
    bool is_occupied = false;

    if (out_idx < total_out_cells) {
        int out_x = out_idx % res_out.x;
        int out_y = (out_idx / res_out.x) % res_out.y;
        int out_z = out_idx / (res_out.x * res_out.y);

        int in_x = out_x * 2;
        int in_y = out_y * 2;
        int in_z = out_z * 2;
        
        int res_in_x = res_out.x * 2;
        int res_in_y = res_out.y * 2;

        #pragma unroll
        for (int dz = 0; dz < 2; ++dz) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int in_idx = (in_z + dz) * (res_in_x * res_in_y) + 
                                 (in_y + dy) * res_in_x + 
                                 (in_x + dx);

                    int byte_idx = in_idx >> 3;
                    int bit_idx  = in_idx & 7;
                    
                    bool bit = (level_in[byte_idx] >> bit_idx) & 1;
                    is_occupied = is_occupied || bit; 
                }
            }
        }
    }

    unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, is_occupied);

    int lane_id = threadIdx.x % 32;
    if (lane_id == 0 && out_idx < total_out_cells) {
        if (total_out_cells >= 32) {
            int word_idx = out_idx / 32;
            level_out_32[word_idx] = warp_mask;
        } else {
            uint8_t* level_out_8 = reinterpret_cast<uint8_t*>(level_out_32);
            int num_bytes = (total_out_cells + 7) / 8;
            for (int i = 0; i < num_bytes; ++i) {
                level_out_8[i] = (warp_mask >> (i * 8)) & 0xFF;
            }
        }
    }
}

__global__ void extractActiveCells(
    const uint8_t* __restrict__ occupancyGrid,
    int* __restrict__ activeCellIndices,
    int* __restrict__ numActiveCells,
    int totalCells
) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= totalCells) return;

    int byteIdx = cellIdx / 8;
    int bitIdx  = cellIdx % 8;

    bool isOccupied = (occupancyGrid[byteIdx] >> bitIdx) & 1;

    if (isOccupied) {
        int writePos = atomicAdd(numActiveCells, 1);
        activeCellIndices[writePos] = cellIdx;
    }
}

__global__ void sampleOccupiedAABB(
    float* __restrict__ samples,
    const int* __restrict__ activeCellIndices,
    const int* __restrict__ numActiveCells,
    float3 aabbMin,
    float3 aabbMax,
    uint3 gridResolution,
    int N,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int totalActive = *numActiveCells;
    if (totalActive == 0) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float4 rnd = curand_uniform4(&state);

    int listIdx = min((int)(rnd.w * totalActive), totalActive - 1);
    int cell1D = activeCellIndices[listIdx];

    int gridX = cell1D % gridResolution.x;
    int gridY = (cell1D / gridResolution.x) % gridResolution.y;
    int gridZ = cell1D / (gridResolution.x * gridResolution.y);

    float3 cellSizeNorm = make_float3(
        1.0f / (float)gridResolution.x,
        1.0f / (float)gridResolution.y,
        1.0f / (float)gridResolution.z
    );
    float3 cellMinNorm = make_float3(
        gridX * cellSizeNorm.x,
        gridY * cellSizeNorm.y,
        gridZ * cellSizeNorm.z
    );

    float4* samples_vec = reinterpret_cast<float4*>(samples);
    store_float4(&samples_vec[idx], cellMinNorm.x + rnd.x * cellSizeNorm.x, cellMinNorm.y + rnd.y * cellSizeNorm.y, cellMinNorm.z + rnd.z * cellSizeNorm.z, 0.0f);
}
