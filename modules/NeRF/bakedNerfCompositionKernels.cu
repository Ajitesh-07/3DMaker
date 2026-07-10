#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include "BakedNerf.h"
#include "bakedNerfKernelDef.cuh"

template <int VIEW_FEATURES>
__global__ void interpolateRenderRays(
    const int numRays,
    const uint32_t raysDone,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_sigma,
    const float* __restrict__ rgb_output,
    const half* __restrict__  sigma_master,
    const float* __restrict__ diffuse_master,
    const float* __restrict__ features_master,
    const float* __restrict__ student_frac,
    int* __restrict__ student_rows_out,
    half* __restrict__ student_sigma,
    float* __restrict__ student_weights,
    half* __restrict__ student_point_data,
    float* __restrict__ point_teacher_rgb,
    float3 bg_color,
    int m_pad
){
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRays) return;

    const float3* diffuse_master_v3 = reinterpret_cast<const float3*>(diffuse_master);
    uint32_t offset = ray_offsets[r + raysDone];
    uint32_t count = num_steps[r];

    float T = 1.0f;
    float r_c = 0.0f, g_c = 0.0f, b_c = 0.0f;
    float depth = 0.0f;

    float sT = 1.0f;
    float sr_c = 0.0f, sg_c = 0.0f, sb_c = 0.0f;

    int student_rows_idx[8];
    float3 frac;
    float interpolated_features[VIEW_FEATURES];
    float sf_acc[VIEW_FEATURES];
    #pragma unroll
    for (int j = 0; j < VIEW_FEATURES; j++) sf_acc[j] = 0.0f;

    for(uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];

        loadVec4(&student_rows_out[idx*8], student_rows_idx);
        loadVec4(&student_rows_out[idx*8 + 4], student_rows_idx + 4);

        frac.x = student_frac[idx*3 + 0];
        frac.y = student_frac[idx*3 + 1];
        frac.z = student_frac[idx*3 + 2];

        float3 d000 = student_rows_idx[0] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[0]];
        float3 d100 = student_rows_idx[1] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[1]];
        float3 c00  = lerp_float3(d000, d100, frac.x);

        float3 d010 = student_rows_idx[2] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[2]];
        float3 d110 = student_rows_idx[3] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[3]];
        float3 c10  = lerp_float3(d010, d110, frac.x);

        float3 c0 = lerp_float3(c00, c10, frac.y);

        float3 d001 = student_rows_idx[4] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[4]];
        float3 d101 = student_rows_idx[5] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[5]];
        float3 c01  = lerp_float3(d001, d101, frac.x);

        float3 d011 = student_rows_idx[6] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[6]];
        float3 d111 = student_rows_idx[7] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[7]];
        float3 c11  = lerp_float3(d011, d111, frac.x);

        float3 c1 = lerp_float3(c01, c11, frac.y);
        float3 final_diffuse = lerp_float3(c0, c1, frac.z);

        if constexpr (VIEW_FEATURES == 4) {
            float vA[4], vB[4]; 
            
            if (student_rows_idx[0] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[0]*4], vA);
            if (student_rows_idx[1] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[1]*4], vB);
            float4 c00_f = lerp_arrays(vA, vB, frac.x);

            if (student_rows_idx[2] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[2]*4], vA);
            if (student_rows_idx[3] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[3]*4], vB);
            float4 c10_f = lerp_arrays(vA, vB, frac.x);

            float4 c0_f = lerp_float4(c00_f, c10_f, frac.y);

            if (student_rows_idx[4] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[4]*4], vA);
            if (student_rows_idx[5] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[5]*4], vB);
            float4 c01_f = lerp_arrays(vA, vB, frac.x);

            if (student_rows_idx[6] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[6]*4], vA);
            if (student_rows_idx[7] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[7]*4], vB);
            float4 c11_f = lerp_arrays(vA, vB, frac.x);

            float4 c1_f = lerp_float4(c01_f, c11_f, frac.y);

            float4 final_res = lerp_float4(c0_f, c1_f, frac.z);
            
            interpolated_features[0] = final_res.x;
            interpolated_features[1] = final_res.y;
            interpolated_features[2] = final_res.z;
            interpolated_features[3] = final_res.w;
        } else {
            #pragma unroll
            for(int j = 0; j < VIEW_FEATURES; j++) {
                float v000 = student_rows_idx[0] == -1 ? 0.0f : features_master[student_rows_idx[0]*VIEW_FEATURES + j];
                float v100 = student_rows_idx[1] == -1 ? 0.0f : features_master[student_rows_idx[1]*VIEW_FEATURES + j];
                float v010 = student_rows_idx[2] == -1 ? 0.0f : features_master[student_rows_idx[2]*VIEW_FEATURES + j];
                float v110 = student_rows_idx[3] == -1 ? 0.0f : features_master[student_rows_idx[3]*VIEW_FEATURES + j];
                
                float v001 = student_rows_idx[4] == -1 ? 0.0f : features_master[student_rows_idx[4]*VIEW_FEATURES + j];
                float v101 = student_rows_idx[5] == -1 ? 0.0f : features_master[student_rows_idx[5]*VIEW_FEATURES + j];
                float v011 = student_rows_idx[6] == -1 ? 0.0f : features_master[student_rows_idx[6]*VIEW_FEATURES + j];
                float v111 = student_rows_idx[7] == -1 ? 0.0f : features_master[student_rows_idx[7]*VIEW_FEATURES + j];

                float c00_f = fmaf(frac.x, v100 - v000, v000);
                float c10_f = fmaf(frac.x, v110 - v010, v010);
                float c01_f = fmaf(frac.x, v101 - v001, v001);
                float c11_f = fmaf(frac.x, v111 - v011, v011);

                float c0_f = fmaf(frac.y, c10_f - c00_f, c00_f);
                float c1_f = fmaf(frac.y, c11_f - c01_f, c01_f);

                interpolated_features[j] = fmaf(frac.z, c1_f - c0_f, c0_f);
            }
        }

        float s000 = student_rows_idx[0] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[0]]);
        float s100 = student_rows_idx[1] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[1]]);
        float s010 = student_rows_idx[2] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[2]]);
        float s110 = student_rows_idx[3] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[3]]);

        float s001 = student_rows_idx[4] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[4]]);
        float s101 = student_rows_idx[5] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[5]]);
        float s011 = student_rows_idx[6] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[6]]);
        float s111 = student_rows_idx[7] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[7]]);

        float c00_s = fmaf(frac.x, s100 - s000, s000);
        float c10_s = fmaf(frac.x, s110 - s010, s010);
        float c01_s = fmaf(frac.x, s101 - s001, s001);
        float c11_s = fmaf(frac.x, s111 - s011, s011);
        
        float c0_s = fmaf(frac.y, c10_s - c00_s, c00_s);
        float c1_s = fmaf(frac.y, c11_s - c01_s, c01_s);
        float final_sigma_float = fmaf(frac.z, c1_s - c0_s, c0_s);

        float delta_t = 0.0f;
        if (i < count - 1) {
            delta_t = t_sorted[idx + 1] - t;
        } else {
            delta_t = 1e-3f; 
        }

        float t_sigma = density_sigma[idx];
        float t_alpha = 1.0f - expf(-t_sigma * delta_t);
        float t_weight = t_alpha * T;

        r_c += t_weight * rgb_output[idx * 3 + 0];
        g_c += t_weight * rgb_output[idx * 3 + 1];
        b_c += t_weight * rgb_output[idx * 3 + 2];

        depth += t_weight * t;
        T *= (1.0f - t_alpha);

        float s_sigma_safe = fmaxf(0.0f, final_sigma_float); 
        float s_alpha = 1.0f - expf(-s_sigma_safe * delta_t);
        float s_weight = s_alpha * sT;

        sr_c += s_weight * final_diffuse.x;
        sg_c += s_weight * final_diffuse.y;
        sb_c += s_weight * final_diffuse.z;
        
        sT *= (1.0f - s_alpha);

        student_sigma[idx] = __float2half(final_sigma_float);
        student_weights[idx] = s_weight;

        uint32_t data_offset = idx * (19 + VIEW_FEATURES + m_pad);

        student_point_data[data_offset + 0] = __float2half(sr_c);
        student_point_data[data_offset + 1] = __float2half(sg_c);
        student_point_data[data_offset + 2] = __float2half(sb_c);

        #pragma unroll
        for(int j = 0; j < VIEW_FEATURES; j++) {
            sf_acc[j] += s_weight * interpolated_features[j];
            student_point_data[data_offset + 3 + j] = __float2half(sf_acc[j]);
        }

        point_teacher_rgb[idx*3 + 0] = r_c;
        point_teacher_rgb[idx*3 + 1] = g_c;
        point_teacher_rgb[idx*3 + 2] = b_c;
    }
}


template <int VIEW_FEATURES>
__global__ void interpolateStudentRenderRays(
    const int numRays,
    const uint32_t base,
    const uint32_t raysDone,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_sigma,
    const half* __restrict__  sigma_master,
    const float* __restrict__ diffuse_master,
    const float* __restrict__ features_master,
    const float* __restrict__ student_frac,
    int* __restrict__ student_rows_out,
    half* __restrict__ student_sigma,
    float* __restrict__ student_weights,
    half* __restrict__ student_point_data,
    float3 bg_color,
    int m_pad
){
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRays) return;

    const float3* diffuse_master_v3 = reinterpret_cast<const float3*>(diffuse_master);
    uint32_t offset = ray_offsets[r + raysDone] - base;
    uint32_t count = num_steps[r];

    float sT = 1.0f;
    float sr_c = 0.0f, sg_c = 0.0f, sb_c = 0.0f;

    int student_rows_idx[8];
    float3 frac;
    float interpolated_features[VIEW_FEATURES];
    float sf_acc[VIEW_FEATURES];
    #pragma unroll
    for (int j = 0; j < VIEW_FEATURES; j++) sf_acc[j] = 0.0f;

    for(uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];

        loadVec4(&student_rows_out[idx*8], student_rows_idx);
        loadVec4(&student_rows_out[idx*8 + 4], student_rows_idx + 4);

        frac.x = student_frac[idx*3 + 0];
        frac.y = student_frac[idx*3 + 1];
        frac.z = student_frac[idx*3 + 2];

        float3 d000 = student_rows_idx[0] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[0]];
        float3 d100 = student_rows_idx[1] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[1]];
        float3 c00  = lerp_float3(d000, d100, frac.x);

        float3 d010 = student_rows_idx[2] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[2]];
        float3 d110 = student_rows_idx[3] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[3]];
        float3 c10  = lerp_float3(d010, d110, frac.x);

        float3 c0 = lerp_float3(c00, c10, frac.y);

        float3 d001 = student_rows_idx[4] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[4]];
        float3 d101 = student_rows_idx[5] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[5]];
        float3 c01  = lerp_float3(d001, d101, frac.x);

        float3 d011 = student_rows_idx[6] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[6]];
        float3 d111 = student_rows_idx[7] == -1 ? make_float3(0.0f, 0.0f, 0.0f) : diffuse_master_v3[student_rows_idx[7]];
        float3 c11  = lerp_float3(d011, d111, frac.x);

        float3 c1 = lerp_float3(c01, c11, frac.y);
        float3 final_diffuse = lerp_float3(c0, c1, frac.z);

        if constexpr (VIEW_FEATURES == 4) {
            float vA[4], vB[4]; 
            
            if (student_rows_idx[0] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[0]*4], vA);
            if (student_rows_idx[1] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[1]*4], vB);
            float4 c00_f = lerp_arrays(vA, vB, frac.x);

            if (student_rows_idx[2] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[2]*4], vA);
            if (student_rows_idx[3] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[3]*4], vB);
            float4 c10_f = lerp_arrays(vA, vB, frac.x);

            float4 c0_f = lerp_float4(c00_f, c10_f, frac.y);

            if (student_rows_idx[4] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[4]*4], vA);
            if (student_rows_idx[5] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[5]*4], vB);
            float4 c01_f = lerp_arrays(vA, vB, frac.x);

            if (student_rows_idx[6] == -1) { vA[0] = 0; vA[1] = 0; vA[2] = 0; vA[3] = 0; } else loadVec4f(&features_master[student_rows_idx[6]*4], vA);
            if (student_rows_idx[7] == -1) { vB[0] = 0; vB[1] = 0; vB[2] = 0; vB[3] = 0; } else loadVec4f(&features_master[student_rows_idx[7]*4], vB);
            float4 c11_f = lerp_arrays(vA, vB, frac.x);

            float4 c1_f = lerp_float4(c01_f, c11_f, frac.y);

            float4 final_res = lerp_float4(c0_f, c1_f, frac.z);
            
            interpolated_features[0] = final_res.x;
            interpolated_features[1] = final_res.y;
            interpolated_features[2] = final_res.z;
            interpolated_features[3] = final_res.w;
        } else {
            #pragma unroll
            for(int j = 0; j < VIEW_FEATURES; j++) {
                float v000 = student_rows_idx[0] == -1 ? 0.0f : features_master[student_rows_idx[0]*VIEW_FEATURES + j];
                float v100 = student_rows_idx[1] == -1 ? 0.0f : features_master[student_rows_idx[1]*VIEW_FEATURES + j];
                float v010 = student_rows_idx[2] == -1 ? 0.0f : features_master[student_rows_idx[2]*VIEW_FEATURES + j];
                float v110 = student_rows_idx[3] == -1 ? 0.0f : features_master[student_rows_idx[3]*VIEW_FEATURES + j];
                
                float v001 = student_rows_idx[4] == -1 ? 0.0f : features_master[student_rows_idx[4]*VIEW_FEATURES + j];
                float v101 = student_rows_idx[5] == -1 ? 0.0f : features_master[student_rows_idx[5]*VIEW_FEATURES + j];
                float v011 = student_rows_idx[6] == -1 ? 0.0f : features_master[student_rows_idx[6]*VIEW_FEATURES + j];
                float v111 = student_rows_idx[7] == -1 ? 0.0f : features_master[student_rows_idx[7]*VIEW_FEATURES + j];

                float c00_f = fmaf(frac.x, v100 - v000, v000);
                float c10_f = fmaf(frac.x, v110 - v010, v010);
                float c01_f = fmaf(frac.x, v101 - v001, v001);
                float c11_f = fmaf(frac.x, v111 - v011, v011);

                float c0_f = fmaf(frac.y, c10_f - c00_f, c00_f);
                float c1_f = fmaf(frac.y, c11_f - c01_f, c01_f);

                interpolated_features[j] = fmaf(frac.z, c1_f - c0_f, c0_f);
            }
        }

        float s000 = student_rows_idx[0] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[0]]);
        float s100 = student_rows_idx[1] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[1]]);
        float s010 = student_rows_idx[2] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[2]]);
        float s110 = student_rows_idx[3] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[3]]);

        float s001 = student_rows_idx[4] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[4]]);
        float s101 = student_rows_idx[5] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[5]]);
        float s011 = student_rows_idx[6] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[6]]);
        float s111 = student_rows_idx[7] == -1 ? 0.0f : __half2float(sigma_master[student_rows_idx[7]]);

        float c00_s = fmaf(frac.x, s100 - s000, s000);
        float c10_s = fmaf(frac.x, s110 - s010, s010);
        float c01_s = fmaf(frac.x, s101 - s001, s001);
        float c11_s = fmaf(frac.x, s111 - s011, s011);
        
        float c0_s = fmaf(frac.y, c10_s - c00_s, c00_s);
        float c1_s = fmaf(frac.y, c11_s - c01_s, c01_s);
        float final_sigma_float = fmaf(frac.z, c1_s - c0_s, c0_s);

        float delta_t = 0.0f;
        if (i < count - 1) {
            delta_t = t_sorted[idx + 1] - t;
        } else {
            delta_t = 1e-3f; 
        }

        float s_sigma_safe = fmaxf(0.0f, final_sigma_float); 
        float s_alpha = 1.0f - expf(-s_sigma_safe * delta_t);
        float s_weight = s_alpha * sT;

        sr_c += s_weight * final_diffuse.x;
        sg_c += s_weight * final_diffuse.y;
        sb_c += s_weight * final_diffuse.z;
        
        sT *= (1.0f - s_alpha);

        student_sigma[idx] = __float2half(final_sigma_float);
        student_weights[idx] = s_weight;

        #pragma unroll
        for(int j = 0; j < VIEW_FEATURES; j++) {
            sf_acc[j] += s_weight * interpolated_features[j];
        }
    }

    uint32_t data_offset = r * (19 + VIEW_FEATURES + m_pad);

    student_point_data[data_offset + 0] = __float2half(sr_c + sT * bg_color.x);
    student_point_data[data_offset + 1] = __float2half(sg_c + sT * bg_color.y);
    student_point_data[data_offset + 2] = __float2half(sb_c + sT * bg_color.z);

    #pragma unroll
    for(int j = 0; j < VIEW_FEATURES; j++) {
        student_point_data[data_offset + 3 + j] = __float2half(sf_acc[j]);
    }
}

void launch_interpolate_render_rays(
    int view_features,
    const int numRays,
    const uint32_t base,
    const uint32_t raysDone,
    const uint32_t* ray_offsets,
    const uint32_t* num_steps,
    const float* t_sorted,
    const float* density_sigma,
    const float* rgb_output,
    const half* sigma_master,
    const float* diffuse_master,
    const float* features_master,
    const float* student_frac,
    int* student_rows_out,
    half* student_sigma,
    float* student_weights,
    half* student_point_data,
    float* point_teacher_rgb,
    float3 bg_color,
    int m_pad,
    cudaStream_t stream
) {
    if (numRays <= 0) return;

    constexpr int THREADS_PER_BLOCK = 256;
    int blocks = (numRays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


    #define LAUNCH_KERNEL(N)                                                     \
        do {                                                                     \
            if (point_teacher_rgb != nullptr) {                                  \
                interpolateRenderRays<N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>( \
                    numRays, raysDone, ray_offsets, num_steps, t_sorted,         \
                    density_sigma, rgb_output, sigma_master, diffuse_master,     \
                    features_master, student_frac, student_rows_out, student_sigma, \
                    student_weights, student_point_data, point_teacher_rgb, bg_color, m_pad \
                );                                                               \
            } else {                                                             \
                interpolateStudentRenderRays<N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>( \
                    numRays, base, raysDone, ray_offsets, num_steps, t_sorted,   \
                    density_sigma, sigma_master, diffuse_master,                 \
                    features_master, student_frac, student_rows_out, student_sigma, \
                    student_weights, student_point_data, bg_color, m_pad         \
                );                                                               \
            }                                                                    \
        } while(0)
    

    switch (view_features) {
        case 1: LAUNCH_KERNEL(1); break;
        case 2: LAUNCH_KERNEL(2); break;
        case 3: LAUNCH_KERNEL(3); break;
        case 4: LAUNCH_KERNEL(4); break;
        case 5: LAUNCH_KERNEL(5); break;
        case 6: LAUNCH_KERNEL(6); break;
        default:
            std::cerr << "Error: VIEW_FEATURES = " << view_features << " is not supported." << std::endl;
            throw std::runtime_error("Unsupported VIEW_FEATURES dimension.");
    }

    #undef LAUNCH_KERNEL
}