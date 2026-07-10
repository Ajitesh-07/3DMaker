#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include "BakedNerf.h"
#include "bakedNerfKernelDef.cuh"

__global__ void baked_compute_ray_aabb_inv_kernel(
    const uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3 aabb_min,
    const float3 aabb_max,
    const int num_cascades,
    float3* __restrict__ rays_d_inv,
    float* __restrict__ nears,
    float* __restrict__ fars
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    float scale = exp2f((float)(num_cascades - 1));
    float3 min_bound = make_float3(
        aabb_min.x * scale,
        aabb_min.y * scale,
        aabb_min.z * scale
    );

    float3 max_bound = make_float3(
        aabb_max.x * scale,
        aabb_max.y * scale,
        aabb_max.z * scale
    );


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

    float t1x = (min_bound.x - o.x) * d_inv.x;
    float t2x = (max_bound.x - o.x) * d_inv.x;
    if (t1x > t2x) { float tmp = t1x; t1x = t2x; t2x = tmp; }

    float t1y = (min_bound.y - o.y) * d_inv.y;
    float t2y = (max_bound.y - o.y) * d_inv.y;
    if (t1y > t2y) { float tmp = t1y; t1y = t2y; t2y = tmp; }

    float t1z = (min_bound.z - o.z) * d_inv.z;
    float t2z = (max_bound.z - o.z) * d_inv.z;
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

__global__ void baked_compute_SH_gather(
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

    const float4* densityOut_vec = reinterpret_cast<const float4*>(densityOut);
    float4 d03   = densityOut_vec[0];
    float4 d47   = densityOut_vec[1];
    float4 d811  = densityOut_vec[2];
    float4 d1215 = densityOut_vec[3];

    auto sanitize_logit = [](float v) -> float {
        v = fmaxf(-64.0f, fminf(64.0f, v));
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



__global__ void compute_SH_student_gather(
    const float3* __restrict__ d_chunk_d,
    const uint32_t* __restrict__ d_ray_indices,
    half* __restrict__ d_student_point_data,
    const int offset,
    const int viewFeatures,
    const int pad,
    int totalHits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalHits) return;

    int total = 3 + viewFeatures + pad + 16;
    int sh_offset = (idx * total) + 3 + viewFeatures + pad; 
    
    float3 d = d_chunk_d[d_ray_indices[idx + offset]];

    float x = d.x;
    float y = d.y;
    float z = d.z;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    float sh0  =  0.28209479f;
    float sh1  = -0.48860251f * y;
    float sh2  =  0.48860251f * z;
    float sh3  = -0.48860251f * x;

    float sh4  =  1.09254843f * x * y;
    float sh5  = -1.09254843f * y * z;
    float sh6  =  0.31539156f * (3.0f * z2 - 1.0f);
    float sh7  = -1.09254843f * x * z;

    float sh8  =  0.54627421f * (x2 - y2);
    float sh9  = -0.59004359f * y * (3.0f * x2 - y2);
    float sh10 =  2.89061144f * x * y * z;
    float sh11 = -0.45704580f * y * (5.0f * z2 - 1.0f);
     
    float sh12 =  0.37317633f * z * (5.0f * z2 - 3.0f);
    float sh13 = -0.45704580f * x * (5.0f * z2 - 1.0f);
    float sh14 =  1.44530572f * z * (x2 - y2);
    float sh15 = -0.59004359f * x * (x2 - 3.0f * y2);

    __half2 hs01 = __floats2half2_rn(sh0, sh1);
    __half2 hs23 = __floats2half2_rn(sh2, sh3);
    __half2 hs45 = __floats2half2_rn(sh4, sh5);
    __half2 hs67 = __floats2half2_rn(sh6, sh7);
    
    float4 packed0_7 = make_float4(
        __int_as_float(*(uint32_t*)&hs01),
        __int_as_float(*(uint32_t*)&hs23),
        __int_as_float(*(uint32_t*)&hs45),
        __int_as_float(*(uint32_t*)&hs67)
    );

    __half2 hs89 = __floats2half2_rn(sh8, sh9);
    __half2 hs1011 = __floats2half2_rn(sh10, sh11);
    __half2 hs1213 = __floats2half2_rn(sh12, sh13);
    __half2 hs1415 = __floats2half2_rn(sh14, sh15);
    
    float4 packed8_15 = make_float4(
        __int_as_float(*(uint32_t*)&hs89),
        __int_as_float(*(uint32_t*)&hs1011),
        __int_as_float(*(uint32_t*)&hs1213),
        __int_as_float(*(uint32_t*)&hs1415)
    );

    float4* dest = reinterpret_cast<float4*>(&d_student_point_data[sh_offset]);
    
    store_float4(&dest[0], packed0_7.x, packed0_7.y, packed0_7.z, packed0_7.w);
    store_float4(&dest[1], packed8_15.x, packed8_15.y, packed8_15.z, packed8_15.w);
}



__global__ void compute_SH_student_deferred(
    const float3* __restrict__ d_chunk_d,
    half* __restrict__ d_student_point_data,
    const int viewFeatures,
    const int pad,
    const int numRays
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    int total = 3 + viewFeatures + pad + 16;
    int sh_offset = (idx * total) + 3 + viewFeatures + pad; 
    
    float3 d = d_chunk_d[idx];

    float x = d.x;
    float y = d.y;
    float z = d.z;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    float sh0  =  0.28209479f;
    float sh1  = -0.48860251f * y;
    float sh2  =  0.48860251f * z;
    float sh3  = -0.48860251f * x;

    float sh4  =  1.09254843f * x * y;
    float sh5  = -1.09254843f * y * z;
    float sh6  =  0.31539156f * (3.0f * z2 - 1.0f);
    float sh7  = -1.09254843f * x * z;

    float sh8  =  0.54627421f * (x2 - y2);
    float sh9  = -0.59004359f * y * (3.0f * x2 - y2);
    float sh10 =  2.89061144f * x * y * z;
    float sh11 = -0.45704580f * y * (5.0f * z2 - 1.0f);
     
    float sh12 =  0.37317633f * z * (5.0f * z2 - 3.0f);
    float sh13 = -0.45704580f * x * (5.0f * z2 - 1.0f);
    float sh14 =  1.44530572f * z * (x2 - y2);
    float sh15 = -0.59004359f * x * (x2 - 3.0f * y2);

    __half2 hs01 = __floats2half2_rn(sh0, sh1);
    __half2 hs23 = __floats2half2_rn(sh2, sh3);
    __half2 hs45 = __floats2half2_rn(sh4, sh5);
    __half2 hs67 = __floats2half2_rn(sh6, sh7);
    
    float4 packed0_7 = make_float4(
        __int_as_float(*(uint32_t*)&hs01),
        __int_as_float(*(uint32_t*)&hs23),
        __int_as_float(*(uint32_t*)&hs45),
        __int_as_float(*(uint32_t*)&hs67)
    );

    __half2 hs89 = __floats2half2_rn(sh8, sh9);
    __half2 hs1011 = __floats2half2_rn(sh10, sh11);
    __half2 hs1213 = __floats2half2_rn(sh12, sh13);
    __half2 hs1415 = __floats2half2_rn(sh14, sh15);
    
    float4 packed8_15 = make_float4(
        __int_as_float(*(uint32_t*)&hs89),
        __int_as_float(*(uint32_t*)&hs1011),
        __int_as_float(*(uint32_t*)&hs1213),
        __int_as_float(*(uint32_t*)&hs1415)
    );

    float4* dest = reinterpret_cast<float4*>(&d_student_point_data[sh_offset]);
    
    store_float4(&dest[0], packed0_7.x, packed0_7.y, packed0_7.z, packed0_7.w);
    store_float4(&dest[1], packed8_15.x, packed8_15.y, packed8_15.z, packed8_15.w);
}

void launch_compute_SH_student_deferred(
    const float3* d_chunk_d,
    half* d_student_point_data,
    const int viewFeatures,
    const int pad,
    const int numRays,
    cudaStream_t stream
) {
    if (numRays <= 0) return;
    constexpr int BS = 256;
    int gs = (numRays + BS - 1) / BS;
    compute_SH_student_deferred<<<gs, BS, 0, stream>>>(
        d_chunk_d, d_student_point_data, viewFeatures, pad, numRays
    );
}

__global__ void buildInvCellMap(
    const int* __restrict__ cellIds,
    int* __restrict__ cellSlots,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int gIdx = cellIds[idx];
    cellSlots[gIdx] = idx;
}

__global__ void accumulateColor(
    float* __restrict__ d_color_out,
    float* __restrict__ d_color_sum,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    d_color_sum[idx*3 + 0] += d_color_out[idx*3 + 0];
    d_color_sum[idx*3 + 1] += d_color_out[idx*3 + 1];
    d_color_sum[idx*3 + 2] += d_color_out[idx*3 + 2];

}

__global__ void fillVertexGrid(
    float* __restrict__ density_out,
    float* __restrict__ color_sum,
    half* __restrict__  masterSigmaGrid,
    float* __restrict__ masterColorGrid,
    float invNum,
    float densityBias,
    int offset,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float logit = density_out[idx*16];
    float sigma = expf(fminf(logit - densityBias, 8.0f));

    masterColorGrid[(offset+idx)*3 + 0] = color_sum[idx*3 + 0] * invNum;
    masterColorGrid[(offset+idx)*3 + 1] = color_sum[idx*3 + 1] * invNum;
    masterColorGrid[(offset+idx)*3 + 2] = color_sum[idx*3 + 2] * invNum;
    masterSigmaGrid[offset+idx] = __float2half(sigma);
}

__global__ void k_genSubVoxelPositions(
    const int* __restrict__ cellIds, int numBlocks, int B,
    uint3 gridRes, float3 aabbMin, float3 aabbMax,
    float* __restrict__ outPos)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int per = B * B * B;
    if (tid >= numBlocks * per) return;

    int sub = tid % per;
    int bi  = tid / per;
    int cell = cellIds[bi];

    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = cell / G;
    int local   = cell % G;
    int gx = local % gridRes.x;
    int gy = (local / gridRes.x) % gridRes.y;
    int gz = local / (gridRes.x * gridRes.y);

    int sx = sub % B;
    int sy = (sub / B) % B;
    int sz = sub / (B * B);

    float scale = exp2f((float)cascade);
    float3 cmin = make_float3(aabbMin.x * scale, aabbMin.y * scale, aabbMin.z * scale);
    float3 cmax = make_float3(aabbMax.x * scale, aabbMax.y * scale, aabbMax.z * scale);
    float3 csz  = make_float3((cmax.x - cmin.x) / gridRes.x, (cmax.y - cmin.y) / gridRes.y, (cmax.z - cmin.z) / gridRes.z);

    float invB = 1.0f / (float)(B - 1);
    float3 pos = make_float3(
        cmin.x + (gx + sx * invB) * csz.x,
        cmin.y + (gy + sy * invB) * csz.y,
        cmin.z + (gz + sz * invB) * csz.z);

    pos = bake_contract_pos(pos, aabbMin, aabbMax);
    
    float4* out_pos_ptr = reinterpret_cast<float4*>(&outPos[tid*4]);

    store_float4(out_pos_ptr, pos.x, pos.y, pos.z, 0.0f);
}

__global__ void k_buildSubVoxelMasks(
    const float* __restrict__ logits, const int* __restrict__ cellIds,
    int numBlocks, int B, uint3 gridRes,
    float densityBias, float minDensityThreshold, float baseVoxel,
    uint64_t* __restrict__ masksOut,
    unsigned long long* __restrict__ survivorTotal,
    unsigned long long* __restrict__ fillHist,
    uint32_t* __restrict__ uniqueBitset,           // nullable: 1 bit / GLOBAL vertex that SURVIVES sigma
    uint32_t* __restrict__ allBitset)              // nullable: 1 bit / GLOBAL vertex of an occupied voxel (no sigma filter)
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= numBlocks) return;

    int per = B * B * B;
    int G = gridRes.x * gridRes.y * gridRes.z;
    int cell    = cellIds[bi];
    int cascade = cell / G;
    int local   = cell % G;
    int gx = local % gridRes.x;
    int gy = (local / gridRes.x) % gridRes.y;
    int gz = local / (gridRes.x * gridRes.y);
    float cascadeThresh = (minDensityThreshold / baseVoxel) / exp2f((float)cascade);

    const long long Lx = (long long)gridRes.x * (B - 1) + 1;
    const long long Ly = (long long)gridRes.y * (B - 1) + 1;
    const long long Lz = (long long)gridRes.z * (B - 1) + 1;

    uint64_t mask = 0;
    for (int s = 0; s < per; ++s) {
        float logit = logits[bi * per + s];
        float sigma = expf(fminf(logit - densityBias, 8.0f));
        bool surv = sigma > cascadeThresh;
        if (surv) mask |= (1ull << s);
        if (uniqueBitset || allBitset) {
            int sx = s % B, sy = (s / B) % B, sz = s / (B * B);
            long long Vx = (long long)gx * (B - 1) + sx;
            long long Vy = (long long)gy * (B - 1) + sy;
            long long Vz = (long long)gz * (B - 1) + sz;
            long long vidx = (long long)cascade * (Lx * Ly * Lz) + Vz * (Lx * Ly) + Vy * Lx + Vx;
            uint32_t word = (uint32_t)(vidx >> 5), bit = 1u << (uint32_t)(vidx & 31);
            if (allBitset) atomicOr(&allBitset[word], bit);                 // every vertex of an occupied voxel
            if (surv && uniqueBitset) atomicOr(&uniqueBitset[word], bit);   // only sigma-surviving vertices
        }
    }
    if (masksOut) masksOut[bi] = mask;
    int popc = __popcll(mask);
    atomicAdd(survivorTotal, (unsigned long long)popc);
    atomicAdd(&fillHist[popc], 1ULL);
}

__global__ void buildSubVoxelMasks(
    const float* __restrict__ logits,
    const int* __restrict__ cellIds,
    int numBlocks, int B,
    uint3 gridRes,
    float densityBias, float minDensityThreshold,
    float baseVoxel,
    uint32_t* __restrict__ masksOut,
    uint64_t* __restrict__ subVoxelMask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;

    int per = B*B*B;
    int bminusOne = B-1;
    int bminusOneSquared = bminusOne * bminusOne;
    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = cellIds[idx] / G;
    float cascadeThresh = (minDensityThreshold / baseVoxel) / exp2f((float)cascade);

    uint32_t mask = 0;
    uint64_t subMask = 0;
    for (int i = 0; i < per; i++) {
        float logit = logits[idx * per + i];
        float sigma = expf(fminf(logit - densityBias, 8.0f));

        int x = i % B;
        int y = (i / B) % B;
        int z = i / (B * B);
        int idx2 = x + y*B + z*B*B;

        if (sigma > cascadeThresh) subMask |= (1ull << idx2);

        #pragma unroll
        for(int dx = -1; dx < 1; dx++) {
            #pragma unroll
            for(int dy = -1; dy < 1; dy++) {
                #pragma unroll
                for(int dz = -1; dz < 1; dz++) {
                    int clampedX = clamp_int(x+dx, 0, bminusOne-1);
                    int clampedY = clamp_int(y+dy, 0, bminusOne-1);
                    int clampedZ = clamp_int(z+dz, 0, bminusOne-1);

                    int index = clampedX + (clampedY * bminusOne) + (clampedZ * bminusOneSquared);

                    if (sigma > cascadeThresh) mask |= (1ull << index);
                }
            }
        }
    }

    masksOut[idx] = mask;
    subVoxelMask[idx] = subMask;
}   

__global__ void buildGrid(
    const int* __restrict__ cellIds,
    const int* __restrict__ cellSlots,
    const uint64_t* __restrict__ subVoxelMask,
    const int numBlocks,
    const int offset,
    const int B,
    const uint3 gridRes, 
    const float3 aabbMin, 
    const float3 aabbMax,
    float* __restrict__ outPos,
    uint32_t* __restrict__ blocksIds,
    int* globalCounter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int per = B * B * B;

    bool valid = tid < (numBlocks * per);
    float3 pos = make_float3(0.0f, 0.0f, 0.0f);
    bool owns = false;
    int bi = 0;
    int sub = 0;
    int gx = 0, gy = 0, gz = 0, sx = 0, sy = 0, sz = 0;
    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = 0;

    bool sigmaValid = false;

    if (valid) {
        sub = tid % per;
        bi  = tid / per;
        int cell = cellIds[bi];

        cascade = cell / G;
        int local   = cell % G;
        gx = local % gridRes.x;
        gy = (local / gridRes.x) % gridRes.y;
        gz = local / (gridRes.x * gridRes.y);

        sx = sub % B;
        sy = (sub / B) % B;
        sz = sub / (B * B);
        
        uint64_t cellMask = subVoxelMask[bi];
        sigmaValid = (cellMask >> sub) & 1;

        float scale = exp2f((float)cascade);
        float3 cmin = make_float3(aabbMin.x * scale, aabbMin.y * scale, aabbMin.z * scale);
        float3 cmax = make_float3(aabbMax.x * scale, aabbMax.y * scale, aabbMax.z * scale);
        float3 csz  = make_float3((cmax.x - cmin.x) / gridRes.x, (cmax.y - cmin.y) / gridRes.y, (cmax.z - cmin.z) / gridRes.z);

        float invB = 1.0f / (float)(B - 1);
        pos = make_float3(
            cmin.x + (gx + sx * invB) * csz.x,
            cmin.y + (gy + sy * invB) * csz.y,
            cmin.z + (gz + sz * invB) * csz.z);

        pos = bake_contract_pos(pos, aabbMin, aabbMax);

        int Lx = gridRes.x*(B-1) + 1;
        int Ly = gridRes.y*(B-1) + 1;
        int Lz = gridRes.z*(B-1) + 1;
        int vx = gx * (B-1) + sx;
        int vy = gy * (B-1) + sy; 
        int vz = gz * (B-1) + sz; 

        int coords = cascade * (Lx*Ly*Lz) + (vx + vy*Lx + vz*Lx*Ly);
        int nx = (sx == 0) && (gx > 0);
        int px = (sx == B - 1) && (gx < gridRes.x - 1);
        
        int ny = (sy == 0) && (gy > 0);
        int py = (sy == B - 1) && (gy < gridRes.y - 1);
        
        int nz = (sz == 0) && (gz > 0);
        int pz = (sz == B - 1) && (gz < gridRes.z - 1);

        owns = true;

        if (owns && nx) {                                  
            for (int oy = -ny; oy <= py && owns; ++oy)
                for (int oz = -nz; oz <= pz && owns; ++oz)
                    if (cellSlots[cascade*G + (gx-1) + (gy+oy)*gridRes.x + (gz+oz)*gridRes.x*gridRes.y] != -1)
                        owns = false;
        }
        if (owns && ny) {                                  
            for (int oz = -nz; oz <= pz && owns; ++oz)
                if (cellSlots[cascade*G + gx + (gy-1)*gridRes.x + (gz+oz)*gridRes.x*gridRes.y] != -1)
                    owns = false;
        }
        if (owns && nz) {
            if (cellSlots[cascade*G + gx + gy*gridRes.x + (gz-1)*gridRes.x*gridRes.y] != -1)
                owns = false;
        }
    }

    bool claim = sigmaValid && owns;
    unsigned int claim_mask = __ballot_sync(0xFFFFFFFFu, claim);

    if (claim) {
        int leader     = __ffs(claim_mask) - 1;
        int warp_count = __popc(claim_mask);
        int baseIdx    = 0;
        if (lane_id == leader) baseIdx = atomicAdd(globalCounter, warp_count);
        baseIdx = __shfl_sync(claim_mask, baseIdx, leader);
        int idx = baseIdx + __popc(claim_mask & ((1u << lane_id) - 1));

        float4* dst = reinterpret_cast<float4*>(&outPos[(idx - offset)*4]);
        store_float4(dst, pos.x, pos.y, pos.z, 0.f);

        int lox=(sx==0&&gx>0)?-1:0,  hox=(sx==B-1&&gx<gridRes.x-1)?1:0;
        int loy=(sy==0&&gy>0)?-1:0,  hoy=(sy==B-1&&gy<gridRes.y-1)?1:0;
        int loz=(sz==0&&gz>0)?-1:0,  hoz=(sz==B-1&&gz<gridRes.z-1)?1:0;

        for (int oz=loz; oz<=hoz; ++oz)
        for (int oy=loy; oy<=hoy; ++oy)
        for (int ox=lox; ox<=hox; ++ox) {
            int ncell = cascade*G + (gx+ox) + (gy+oy)*gridRes.x + (gz+oz)*gridRes.x*gridRes.y;
            int nbi = cellSlots[ncell];
            if (nbi == -1) continue;
            int nsx = (ox==0)?sx:(ox>0?0:B-1);
            int nsy = (oy==0)?sy:(oy>0?0:B-1);
            int nsz = (oz==0)?sz:(oz>0?0:B-1);
            blocksIds[nbi*per + nsx + nsy*B + nsz*B*B] = idx;
        }
    }
}


__global__ void compute_residual_and_loss_grad(
    int total_hits,
    int padded_b_size,
    int view_features,
    int m_pad,
    const half* __restrict__ student_point_data,
    const float* __restrict__ teacher_rgb,
    float* __restrict__ student_rgb_chunk,
    half* __restrict__ phi_chunk
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_b_size) return;
    
    if (idx >= total_hits) {
        phi_chunk[idx * 3 + 0] = __float2half(0.0f);
        phi_chunk[idx * 3 + 1] = __float2half(0.0f);
        phi_chunk[idx * 3 + 2] = __float2half(0.0f);
        return;
    }
    
    int data_offset = idx * (19 + view_features + m_pad);
    float sr = __half2float(student_point_data[data_offset + 0]);
    float sg = __half2float(student_point_data[data_offset + 1]);
    float sb = __half2float(student_point_data[data_offset + 2]);
    
    float mr = student_rgb_chunk[idx * 3 + 0];
    float mg = student_rgb_chunk[idx * 3 + 1];
    float mb = student_rgb_chunk[idx * 3 + 2];
    
    float cr = fminf(fmaxf(sr + mr, 0.0f), 1.0f);
    float cg = fminf(fmaxf(sg + mg, 0.0f), 1.0f);
    float cb = fminf(fmaxf(sb + mb, 0.0f), 1.0f);
    
    student_rgb_chunk[idx * 3 + 0] = cr;
    student_rgb_chunk[idx * 3 + 1] = cg;
    student_rgb_chunk[idx * 3 + 2] = cb;
    
    float tr = teacher_rgb[idx * 3 + 0];
    float tg = teacher_rgb[idx * 3 + 1];
    float tb = teacher_rgb[idx * 3 + 2];
    
    phi_chunk[idx * 3 + 0] = __float2half(2.0f * (cr - tr));
    phi_chunk[idx * 3 + 1] = __float2half(2.0f * (cg - tg));
    phi_chunk[idx * 3 + 2] = __float2half(2.0f * (cb - tb));
}


__global__ void compute_residual(
    int padded_b_size,
    int view_features,
    int m_pad,
    const half* __restrict__ student_point_data,
    float* __restrict__ student_rgb_chunk
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_b_size) return;

    int data_offset = idx * (19 + view_features + m_pad);
    float sr = __half2float(student_point_data[data_offset + 0]);
    float sg = __half2float(student_point_data[data_offset + 1]);
    float sb = __half2float(student_point_data[data_offset + 2]);
    
    float mr = student_rgb_chunk[idx * 3 + 0];
    float mg = student_rgb_chunk[idx * 3 + 1];
    float mb = student_rgb_chunk[idx * 3 + 2];
    
    float cr = fminf(fmaxf(sr + mr, 0.0f), 1.0f);
    float cg = fminf(fmaxf(sg + mg, 0.0f), 1.0f);
    float cb = fminf(fmaxf(sb + mb, 0.0f), 1.0f);
    
    student_rgb_chunk[idx * 3 + 0] = cr;
    student_rgb_chunk[idx * 3 + 1] = cg;
    student_rgb_chunk[idx * 3 + 2] = cb;
}

void launch_compute_residual_and_loss_grad(
    int total_hits,
    int padded_b_size,
    int view_features,
    int m_pad,
    const half* student_point_data,
    const float* teacher_rgb,
    float* student_rgb_chunk,
    half* phi_chunk,
    cudaStream_t stream
) {
    if (padded_b_size <= 0) return;
    int threads = 256;
    int blocks = (padded_b_size + threads - 1) / threads;
    compute_residual_and_loss_grad<<<blocks, threads, 0, stream>>>(
        total_hits, padded_b_size, view_features, m_pad, student_point_data, teacher_rgb, student_rgb_chunk, phi_chunk
    );
}

__global__ void compute_ray_gradient_sums(
    int num_rays,
    int view_features,
    int m_pad,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const half* __restrict__ phi_chunk,
    const half* __restrict__ dx_out,
    float* __restrict__ out_ray_color_sum,
    float* __restrict__ out_ray_feat_sum
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;
    
    uint32_t offset = ray_offsets[r];
    uint32_t count = num_steps[r];
    
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    
    float sum_feat[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) sum_feat[j] = 0.0f;
    
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        
        float pr = __half2float(phi_chunk[idx * 3 + 0]);
        float pg = __half2float(phi_chunk[idx * 3 + 1]);
        float pb = __half2float(phi_chunk[idx * 3 + 2]);
        
        int dx_idx = idx * (19 + view_features + m_pad);
        float dx_r = __half2float(dx_out[dx_idx + 0]);
        float dx_g = __half2float(dx_out[dx_idx + 1]);
        float dx_b = __half2float(dx_out[dx_idx + 2]);
        
        sum_r += (pr + dx_r);
        sum_g += (pg + dx_g);
        sum_b += (pb + dx_b);
        
        for (int j = 0; j < view_features; j++) {
            float dx_f = __half2float(dx_out[dx_idx + 3 + j]);
            sum_feat[j] += dx_f;
        }
    }
    
    out_ray_color_sum[r * 3 + 0] = sum_r;
    out_ray_color_sum[r * 3 + 1] = sum_g;
    out_ray_color_sum[r * 3 + 2] = sum_b;
    
    for (int j = 0; j < view_features; j++) {
        out_ray_feat_sum[r * view_features + j] = sum_feat[j];
    }
}

__global__ void scatter_gradients_to_embeddings(
    int num_rays,
    int view_features,
    int m_pad,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_sigma,
    const half* __restrict__ phi_chunk,
    const half* __restrict__ dx_out,
    const float* __restrict__ global_ray_color_sum,
    const float* __restrict__ global_ray_feat_sum,
    const float* __restrict__ student_weights,
    const int* __restrict__ student_rows_out,
    const float* __restrict__ student_frac_out,
    int* __restrict__ out_corner_rowids,
    float* __restrict__ out_diffuse_grads,
    float* __restrict__ out_feature_grads
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;
    
    uint32_t offset = ray_offsets[r];
    uint32_t count = num_steps[r];
    if (count == 0) return;
    
    float prefix_r = 0.0f;
    float prefix_g = 0.0f;
    float prefix_b = 0.0f;
    
    float prefix_feat[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) prefix_feat[j] = 0.0f;
    
    float total_r = global_ray_color_sum[r * 3 + 0];
    float total_g = global_ray_color_sum[r * 3 + 1];
    float total_b = global_ray_color_sum[r * 3 + 2];
    
    float total_feat[8];
    for (int j = 0; j < view_features; j++) {
        total_feat[j] = global_ray_feat_sum[r * view_features + j];
    }
    
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = offset + i;
        
        int dx_idx = idx * (19 + view_features + m_pad);
        float dx_r = __half2float(dx_out[dx_idx + 0]);
        float dx_g = __half2float(dx_out[dx_idx + 1]);
        float dx_b = __half2float(dx_out[dx_idx + 2]);
        
        float pr = __half2float(phi_chunk[idx * 3 + 0]);
        float pg = __half2float(phi_chunk[idx * 3 + 1]);
        float pb = __half2float(phi_chunk[idx * 3 + 2]);
        
        float current_grad_r = pr + dx_r;
        float current_grad_g = pg + dx_g;
        float current_grad_b = pb + dx_b;
        
        // Suffix sum = Total sum - Prefix sum
        float suffix_r = total_r - prefix_r;
        float suffix_g = total_g - prefix_g;
        float suffix_b = total_b - prefix_b;
        
        float w_i = student_weights[idx];
        
        // Gradient for color point c_{d,i}: w_i * suffix_sum
        float grad_cd_r = w_i * suffix_r;
        float grad_cd_g = w_i * suffix_g;
        float grad_cd_b = w_i * suffix_b;
        
        // Now update prefix sum for NEXT point
        prefix_r += current_grad_r;
        prefix_g += current_grad_g;
        prefix_b += current_grad_b;
        
        float grad_feat[8];
        for (int j = 0; j < view_features; j++) {
            float dx_f = __half2float(dx_out[dx_idx + 3 + j]);
            float suffix_f = total_feat[j] - prefix_feat[j];
            grad_feat[j] = w_i * suffix_f;
            prefix_feat[j] += dx_f;
        }
        
        // Distribute to 8 corners
        float frac_x = student_frac_out[idx * 3 + 0];
        float frac_y = student_frac_out[idx * 3 + 1];
        float frac_z = student_frac_out[idx * 3 + 2];
        
        float w000 = (1.0f - frac_x) * (1.0f - frac_y) * (1.0f - frac_z);
        float w100 = frac_x * (1.0f - frac_y) * (1.0f - frac_z);
        float w010 = (1.0f - frac_x) * frac_y * (1.0f - frac_z);
        float w110 = frac_x * frac_y * (1.0f - frac_z);
        float w001 = (1.0f - frac_x) * (1.0f - frac_y) * frac_z;
        float w101 = frac_x * (1.0f - frac_y) * frac_z;
        float w011 = (1.0f - frac_x) * frac_y * frac_z;
        float w111 = frac_x * frac_y * frac_z;
        
        float weights[8] = {w000, w100, w010, w110, w001, w101, w011, w111};
    
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int corner_rowid = student_rows_out[idx * 8 + c];
            float w_c = weights[c];
            
            int out_idx = idx * 8 + c;
            out_corner_rowids[out_idx] = corner_rowid;
            out_diffuse_grads[out_idx * 3 + 0] = grad_cd_r * w_c;
            out_diffuse_grads[out_idx * 3 + 1] = grad_cd_g * w_c;
            out_diffuse_grads[out_idx * 3 + 2] = grad_cd_b * w_c;
            
            for (int j = 0; j < view_features; j++) {
                out_feature_grads[out_idx * view_features + j] = grad_feat[j] * w_c;
            }
        }
    }
}

void launch_scatter_gradients_to_embeddings(
    int num_rays,
    int view_features,
    int m_pad,
    const uint32_t* ray_offsets,
    const uint32_t* num_steps,
    const float* t_sorted,
    const float* density_sigma,
    const half* phi_chunk,
    const half* dx_out,
    const float* global_ray_color_sum,
    const float* global_ray_feat_sum,
    const float* student_weights,
    const int* student_rows_out,
    const float* student_frac_out,
    int* out_corner_rowids,
    float* out_diffuse_grads,
    float* out_feature_grads,
    cudaStream_t stream
) {
    if (num_rays <= 0) return;
    int threads = 256;
    int blocks = (num_rays + threads - 1) / threads;
    scatter_gradients_to_embeddings<<<blocks, threads, 0, stream>>>(
        num_rays, view_features, m_pad, ray_offsets, num_steps, t_sorted, density_sigma,
        phi_chunk, dx_out, global_ray_color_sum, global_ray_feat_sum,
        student_weights, student_rows_out, student_frac_out,
        out_corner_rowids, out_diffuse_grads, out_feature_grads
    );
}
