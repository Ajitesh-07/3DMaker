#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include "BakedNerf.h"

__device__ __forceinline__ int clamp_int(int val, int min_val, int max_val) {
    return min(max(val, min_val), max_val);
}

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

__device__ __forceinline__ void loadVec4(const int* __restrict__ ptr, int dst[4]) {
    #ifndef __INTELLISENSE__
    asm volatile(
        "ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "l"(ptr)
    );
    #endif
}

__device__ __forceinline__ void loadVec4f(const float* __restrict__ ptr, float dst[4]) {
    #ifndef __INTELLISENSE__
    asm volatile(
        "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "l"(ptr)
    );
    #endif
}

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

static __device__ __forceinline__ float3 bake_contract_pos(float3 pos, float3 aabb_min, float3 aabb_max) {
    float3 ext = make_float3(aabb_max.x - aabb_min.x, aabb_max.y - aabb_min.y, aabb_max.z - aabb_min.z);
    float3 rel = make_float3((pos.x - aabb_min.x) / ext.x, (pos.y - aabb_min.y) / ext.y, (pos.z - aabb_min.z) / ext.z);
    float3 n   = make_float3(rel.x * 2.0f - 1.0f, rel.y * 2.0f - 1.0f, rel.z * 2.0f - 1.0f);
    float3 c;
    c.x = fabsf(n.x) <= 1.0f ? n.x : (2.0f - 1.0f / fabsf(n.x)) * copysignf(1.0f, n.x);
    c.y = fabsf(n.y) <= 1.0f ? n.y : (2.0f - 1.0f / fabsf(n.y)) * copysignf(1.0f, n.y);
    c.z = fabsf(n.z) <= 1.0f ? n.z : (2.0f - 1.0f / fabsf(n.z)) * copysignf(1.0f, n.z);
    return make_float3(
        fmaxf(0.0f, fminf((c.x + 2.0f) * 0.25f, 1.0f)),
        fmaxf(0.0f, fminf((c.y + 2.0f) * 0.25f, 1.0f)),
        fmaxf(0.0f, fminf((c.z + 2.0f) * 0.25f, 1.0f)));
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

// ============================================================================
// Phase-2 student/teacher ray marcher (analogue of processRaysHitLinear).
//
// For each ray: outer DDA over the occupancy grid (mipmap, per-cascade pyramid) to
// find occupied finest-level voxels; then an inner 3^3 = (B-1)^3 sub-cube DDA within
// each occupied voxel (skipping sub-cubes whose 27-bit mask bit is 0). For every
// occupied sub-cube the ray traverses we emit ONE sample at the mid-t of the segment:
//   - teacher: the contracted landing position (stride-4, ready for queryRadiance)
//   - student: the 8 corner vertex ROWS of that sub-cube (from blockIdx; -1 = pruned)
//              and the 3 trilinear fractions (fx,fy,fz) of the sample within the sub-cube
//              (the gather kernel expands them into the 8 corner weights).
//
// Two passes, like processRaysHitLinear: (A) count samples/ray, prefix-sum, find the
// cutoff ray so total samples <= batchSize, (B) re-march the fitted rays and emit.
// Returns how many rays fit. Needs a persistent dense-cell -> block-index reverseMap
// (== distil's d_cellSlots, which must be kept instead of freed).
// ============================================================================

__device__ __forceinline__ int bake_get_cascade(float3 pos, float3 aabb_min, float3 aabb_max, int num_cascades) {
    float rx = fmaxf(pos.x / aabb_max.x, pos.x / aabb_min.x);
    float ry = fmaxf(pos.y / aabb_max.y, pos.y / aabb_min.y);
    float rz = fmaxf(pos.z / aabb_max.z, pos.z / aabb_min.z);
    float max_scale = fmaxf(rx, fmaxf(ry, rz));
    if (max_scale <= 1.0f) return 0;
    int cascade = (int)ceilf(log2f(max_scale));
    return max(0, min(cascade, num_cascades - 1));
}

__device__ __forceinline__ uint32_t bake_mipmap_offset(uint3 base_res, int target_level) {
    uint32_t offset = 0;
    uint3 res = base_res;
    for (int l = 0; l < target_level; ++l) {
        offset += res.x * res.y * res.z;
        res.x >>= 1; res.y >>= 1; res.z >>= 1;
    }
    return offset;
}

template<bool EMIT>
__global__ void bakedMarchRays(
    uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3* __restrict__ rays_d_inv,
    const float* __restrict__ nears,
    const float* __restrict__ fars,
    const uint8_t*  __restrict__ occupancy_grid,
    const uint32_t* __restrict__ voxelMask27,   // per-block 27-bit sub-cube occupancy (m_voxelMask)
    const uint32_t* __restrict__ vertexRows,    // per-block * B^3 vertex row map (m_blockIdx), 0xFFFFFFFF = pruned
    const int*      __restrict__ reverseMap,    // dense cell (cascade*G+local) -> block index, -1 = empty
    uint3  gridRes, float3 aabbMin, float3 aabbMax,
    int numCascades, int levelsMipmap, int B,
    // EMIT-only outputs:
    const uint32_t* __restrict__ ray_offsets, uint32_t base_offset,
    float* __restrict__ teacher_points_out,     // batchSize*4 (contracted, stride-4)
    int*   __restrict__ student_rows_out,       // batchSize*8 (corner rows, -1 = pruned)
    float* __restrict__ student_frac_out,       // batchSize*3 (fx,fy,fz; expand to 8 weights at gather time)
    uint32_t* __restrict__ ray_indices_out,     // batchSize
    float* __restrict__ t_hits_out,
    // count-only output:
    uint32_t* __restrict__ num_steps_per_ray)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    float3 o = rays_o[i];
    float3 d = rays_d[i];
    float3 d_inv = rays_d_inv[i];
    float t_min = nears[i];
    float t_max_ray = fars[i];

    float3 aabb_extent = make_float3(aabbMax.x - aabbMin.x, aabbMax.y - aabbMin.y, aabbMax.z - aabbMin.z);
    int3 step = make_int3((d.x >= 0.0f) ? 1 : -1, (d.y >= 0.0f) ? 1 : -1, (d.z >= 0.0f) ? 1 : -1);

    const int G       = gridRes.x * gridRes.y * gridRes.z;
    const int per     = B * B * B;          // 64
    const int subRes  = B - 1;              // 3 sub-cubes / axis
    const uint32_t total_mipmap_cells = bake_mipmap_offset(gridRes, levelsMipmap);

    uint32_t hit_count = 0;
    uint32_t out_base  = EMIT ? (ray_offsets[i] - base_offset) : 0u;

    float current_t = t_min;
    int current_level = levelsMipmap - 1;

    while (current_t < t_max_ray && hit_count < MAX_HITS) {
        float3 cp = make_float3(o.x + current_t * d.x, o.y + current_t * d.y, o.z + current_t * d.z);
        int cascade = bake_get_cascade(cp, aabbMin, aabbMax, numCascades);
        float cs = exp2f((float)cascade);

        float3 caMin = make_float3(aabbMin.x * cs, aabbMin.y * cs, aabbMin.z * cs);
        float3 caExt = make_float3(aabb_extent.x * cs, aabb_extent.y * cs, aabb_extent.z * cs);
        uint3  res_l = make_uint3(gridRes.x >> current_level, gridRes.y >> current_level, gridRes.z >> current_level);
        float3 vsz   = make_float3(caExt.x / res_l.x, caExt.y / res_l.y, caExt.z / res_l.z);
        float3 rp    = make_float3((cp.x - caMin.x) / caExt.x, (cp.y - caMin.y) / caExt.y, (cp.z - caMin.z) / caExt.z);
        int3 vi = make_int3(
            max(0, min((int)floorf(rp.x * res_l.x), (int)res_l.x - 1)),
            max(0, min((int)floorf(rp.y * res_l.y), (int)res_l.y - 1)),
            max(0, min((int)floorf(rp.z * res_l.z), (int)res_l.z - 1)));

        uint32_t lvl_off  = bake_mipmap_offset(gridRes, current_level);
        uint32_t casc_off = cascade * total_mipmap_cells;
        uint32_t flat = casc_off + lvl_off + vi.z * (res_l.x * res_l.y) + vi.y * res_l.x + vi.x;
        bool occ = (occupancy_grid[flat >> 3] >> (flat & 7)) & 1;

        // exit-t of the current (level) voxel
        float3 nb = make_float3(
            caMin.x + (vi.x + (step.x > 0 ? 1.0f : 0.0f)) * vsz.x,
            caMin.y + (vi.y + (step.y > 0 ? 1.0f : 0.0f)) * vsz.y,
            caMin.z + (vi.z + (step.z > 0 ? 1.0f : 0.0f)) * vsz.z);
        float3 tmax = make_float3((nb.x - o.x) * d_inv.x, (nb.y - o.y) * d_inv.y, (nb.z - o.z) * d_inv.z);
        float next_t = fminf(fminf(tmax.x, tmax.y), tmax.z);

        if (occ) {
            if (current_level > 0) {
                current_level--;   // refine, do NOT advance current_t
                continue;
            }
            // ---- finest occupied voxel: inner 3^3 sub-cube DDA ----
            int local = vi.z * (gridRes.x * gridRes.y) + vi.y * gridRes.x + vi.x;
            int bi = reverseMap[cascade * G + local];
            if (bi >= 0) {
                uint32_t subMask = voxelMask27[bi];
                float3 vmin = make_float3(caMin.x + vi.x * vsz.x, caMin.y + vi.y * vsz.y, caMin.z + vi.z * vsz.z);
                float3 ssz  = make_float3(vsz.x / subRes, vsz.y / subRes, vsz.z / subRes);
                float voxel_exit = next_t;

                float sub_t = fmaxf(current_t, t_min);
                int guard = 0;
                while (sub_t < voxel_exit && hit_count < MAX_HITS && guard < 3 * subRes) {
                    guard++;
                    float3 sp = make_float3(o.x + sub_t * d.x, o.y + sub_t * d.y, o.z + sub_t * d.z);
                    // sub-cube index from the segment-entry point
                    int si = max(0, min((int)floorf((sp.x - vmin.x) / ssz.x), subRes - 1));
                    int sj = max(0, min((int)floorf((sp.y - vmin.y) / ssz.y), subRes - 1));
                    int sk = max(0, min((int)floorf((sp.z - vmin.z) / ssz.z), subRes - 1));

                    float sbx = vmin.x + (si + (step.x > 0 ? 1.0f : 0.0f)) * ssz.x;
                    float sby = vmin.y + (sj + (step.y > 0 ? 1.0f : 0.0f)) * ssz.y;
                    float sbz = vmin.z + (sk + (step.z > 0 ? 1.0f : 0.0f)) * ssz.z;
                    float sub_next = fminf(fminf((sbx - o.x) * d_inv.x, (sby - o.y) * d_inv.y), (sbz - o.z) * d_inv.z);
                    float seg_exit = fminf(sub_next, voxel_exit);

                    int cubeIdx = si + sj * subRes + sk * subRes * subRes;   // 0..26
                    if ((subMask >> cubeIdx) & 1u) {
                        if (EMIT) {
                            uint32_t w = out_base + hit_count;
                            float t_mid = 0.5f * (sub_t + seg_exit);
                            float3 mp = make_float3(o.x + t_mid * d.x, o.y + t_mid * d.y, o.z + t_mid * d.z);

                            float3 cpos = bake_contract_pos(mp, aabbMin, aabbMax);
                            float4* tp = reinterpret_cast<float4*>(teacher_points_out);
                            store_float4(&tp[w], cpos.x, cpos.y, cpos.z, 0.0f);
                            ray_indices_out[w] = i;
                            t_hits_out[w] = t_mid;

                            // trilinear fraction of the sample within its sub-cube (si,sj,sk);
                            // the gather kernel expands these 3 into the 8 corner weights.
                            student_frac_out[w * 3 + 0] = fminf(fmaxf((mp.x - vmin.x) / ssz.x - (float)si, 0.0f), 1.0f);
                            student_frac_out[w * 3 + 1] = fminf(fmaxf((mp.y - vmin.y) / ssz.y - (float)sj, 0.0f), 1.0f);
                            student_frac_out[w * 3 + 2] = fminf(fmaxf((mp.z - vmin.z) / ssz.z - (float)sk, 0.0f), 1.0f);

                            #pragma unroll
                            for (int c = 0; c < 8; ++c) {
                                int di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
                                int vsub = (si + di) + (sj + dj) * B + (sk + dk) * B * B;   // corner vertex, 0..63
                                student_rows_out[w * 8 + c] = (int)vertexRows[bi * per + vsub];  // 0xFFFFFFFF -> -1
                            }
                        }
                        hit_count++;
                    }
                    sub_t = fmaxf(sub_t + 1e-6f, sub_next + 1e-6f);
                }
            }
        }

        current_t = fmaxf(current_t + 1e-5f, next_t + 1e-6f);
        current_level = levelsMipmap - 1;
    }

    if (!EMIT) num_steps_per_ray[i] = hit_count;
}

void custom_exclusive_sum(const uint32_t* d_in, uint32_t* d_out, uint32_t* d_block_sums, int num_items, cudaStream_t stream);
__global__ void find_cutoff_ray_kernel(
    const uint32_t* __restrict__ ray_offsets, const uint32_t* __restrict__ num_steps,
    uint32_t num_rays, uint32_t batch_size, uint32_t* __restrict__ active_rays_count);

// Returns how many rays fit (their total sample count <= batchSize).
int processBakedRaysLinear(
    uint32_t num_rays,
    const float3* rays_o, const float3* rays_d, const float3* rays_d_inv,
    const float* nears, const float* fars,
    const uint8_t*  occupancy_grid,
    const uint32_t* voxelMask27,
    const uint32_t* vertexRows,
    const int*      reverseMap,
    uint3 gridRes, float3 aabbMin, float3 aabbMax,
    int numCascades, int mipmapLevels, int B,
    int batchSize,
    uint32_t* totalSamples,
    uint32_t* d_active_rays_count,
    uint32_t* d_num_steps,
    uint32_t* d_ray_offsets,
    float*    d_teacher_points_out,   // batchSize*4
    int*      d_student_rows_out,     // batchSize*8
    float*    d_student_frac_out,     // batchSize*3
    uint32_t* d_ray_indices,          // batchSize
    float* d_t_sorted,
    uint32_t* d_block_sums,
    cudaStream_t stream)
{
    constexpr int BS = 256;
    const int gs = (num_rays + BS - 1) / BS;

    // Pass A: count samples per ray.
    bakedMarchRays<false><<<gs, BS, 0, stream>>>(
        num_rays, rays_o, rays_d, rays_d_inv, nears, fars,
        occupancy_grid, voxelMask27, vertexRows, reverseMap,
        gridRes, aabbMin, aabbMax, numCascades, mipmapLevels, B,
        nullptr, 0u, nullptr, nullptr, nullptr, nullptr, nullptr,
        d_num_steps);

    custom_exclusive_sum(d_num_steps, d_ray_offsets, d_block_sums, num_rays, stream);

    uint32_t last_offset = 0, last_count = 0;
    cudaMemcpyAsync(&last_offset, &d_ray_offsets[num_rays - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_count,  &d_num_steps[num_rays - 1],   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (last_offset + last_count == 0) { *totalSamples = 0; return num_rays; }

    // Find the cutoff ray so the fitted total sample count <= batchSize.
    cudaMemsetAsync(d_active_rays_count, 0, sizeof(uint32_t), stream);
    find_cutoff_ray_kernel<<<gs, BS, 0, stream>>>(d_ray_offsets, d_num_steps, num_rays, (uint32_t)batchSize, d_active_rays_count);

    uint32_t raysFit = 0;
    cudaMemcpyAsync(&raysFit, d_active_rays_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (raysFit == 0) { *totalSamples = 0; return 0; }

    uint32_t tot_off = 0, tot_cnt = 0;
    cudaMemcpyAsync(&tot_off, &d_ray_offsets[raysFit - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&tot_cnt, &d_num_steps[raysFit - 1],   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    *totalSamples = tot_off + tot_cnt;

    // Pass B: re-march the fitted rays and emit at their dense offsets.
    const int gs2 = (raysFit + BS - 1) / BS;
    bakedMarchRays<true><<<gs2, BS, 0, stream>>>(
        raysFit, rays_o, rays_d, rays_d_inv, nears, fars,
        occupancy_grid, voxelMask27, vertexRows, reverseMap,
        gridRes, aabbMin, aabbMax, numCascades, mipmapLevels, B,
        d_ray_offsets, 0u,
        d_teacher_points_out, d_student_rows_out, d_student_frac_out, d_ray_indices,
        d_t_sorted, nullptr);

    return (int)raysFit;
}

__device__ __forceinline__ float4 lerp_arrays(const float a[4], const float b[4], float t) {
    return make_float4(
        fmaf(t, b[0] - a[0], a[0]),
        fmaf(t, b[1] - a[1], a[1]),
        fmaf(t, b[2] - a[2], a[2]),
        fmaf(t, b[3] - a[3], a[3])
    );
}

__device__ __forceinline__ float4 lerp_float4(float4 a, float4 b, float t) {
    return make_float4(
        fmaf(t, b.x - a.x, a.x),
        fmaf(t, b.y - a.y, a.y),
        fmaf(t, b.z - a.z, a.z),
        fmaf(t, b.w - a.w, a.w)
    );
}

__device__ __forceinline__ float3 lerp_float3(float3 a, float3 b, float t) {
    return make_float3(
        fmaf(t, b.x - a.x, a.x),
        fmaf(t, b.y - a.y, a.y),
        fmaf(t, b.z - a.z, a.z)
    );
}
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

        float3 d000 = diffuse_master_v3[student_rows_idx[0]];
        float3 d100 = diffuse_master_v3[student_rows_idx[1]];
        float3 c00  = lerp_float3(d000, d100, frac.x);

        float3 d010 = diffuse_master_v3[student_rows_idx[2]];
        float3 d110 = diffuse_master_v3[student_rows_idx[3]];
        float3 c10  = lerp_float3(d010, d110, frac.x);

        float3 c0 = lerp_float3(c00, c10, frac.y);

        float3 d001 = diffuse_master_v3[student_rows_idx[4]];
        float3 d101 = diffuse_master_v3[student_rows_idx[5]];
        float3 c01  = lerp_float3(d001, d101, frac.x);

        float3 d011 = diffuse_master_v3[student_rows_idx[6]];
        float3 d111 = diffuse_master_v3[student_rows_idx[7]];
        float3 c11  = lerp_float3(d011, d111, frac.x);

        float3 c1 = lerp_float3(c01, c11, frac.y);
        float3 final_diffuse = lerp_float3(c0, c1, frac.z);

        if constexpr (VIEW_FEATURES == 4) {
            float vA[4], vB[4]; 
            
            loadVec4f(&features_master[student_rows_idx[0]*4], vA);
            loadVec4f(&features_master[student_rows_idx[1]*4], vB);
            float4 c00_f = lerp_arrays(vA, vB, frac.x);

            loadVec4f(&features_master[student_rows_idx[2]*4], vA);
            loadVec4f(&features_master[student_rows_idx[3]*4], vB);
            float4 c10_f = lerp_arrays(vA, vB, frac.x);

            float4 c0_f = lerp_float4(c00_f, c10_f, frac.y);

            loadVec4f(&features_master[student_rows_idx[4]*4], vA);
            loadVec4f(&features_master[student_rows_idx[5]*4], vB);
            float4 c01_f = lerp_arrays(vA, vB, frac.x);

            loadVec4f(&features_master[student_rows_idx[6]*4], vA);
            loadVec4f(&features_master[student_rows_idx[7]*4], vB);
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
                float v000 = features_master[student_rows_idx[0]*VIEW_FEATURES + j];
                float v100 = features_master[student_rows_idx[1]*VIEW_FEATURES + j];
                float v010 = features_master[student_rows_idx[2]*VIEW_FEATURES + j];
                float v110 = features_master[student_rows_idx[3]*VIEW_FEATURES + j];
                
                float v001 = features_master[student_rows_idx[4]*VIEW_FEATURES + j];
                float v101 = features_master[student_rows_idx[5]*VIEW_FEATURES + j];
                float v011 = features_master[student_rows_idx[6]*VIEW_FEATURES + j];
                float v111 = features_master[student_rows_idx[7]*VIEW_FEATURES + j];

                float c00_f = fmaf(frac.x, v100 - v000, v000);
                float c10_f = fmaf(frac.x, v110 - v010, v010);
                float c01_f = fmaf(frac.x, v101 - v001, v001);
                float c11_f = fmaf(frac.x, v111 - v011, v011);

                float c0_f = fmaf(frac.y, c10_f - c00_f, c00_f);
                float c1_f = fmaf(frac.y, c11_f - c01_f, c01_f);

                interpolated_features[j] = fmaf(frac.z, c1_f - c0_f, c0_f);
            }
        }

        float s000 = __half2float(sigma_master[student_rows_idx[0]]);
        float s100 = __half2float(sigma_master[student_rows_idx[1]]);
        float s010 = __half2float(sigma_master[student_rows_idx[2]]);
        float s110 = __half2float(sigma_master[student_rows_idx[3]]);

        float s001 = __half2float(sigma_master[student_rows_idx[4]]);
        float s101 = __half2float(sigma_master[student_rows_idx[5]]);
        float s011 = __half2float(sigma_master[student_rows_idx[6]]);
        float s111 = __half2float(sigma_master[student_rows_idx[7]]);

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

void launch_interpolate_render_rays(
    int view_features,
    const int numRays,
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
    cudaStream_t stream = nullptr
) {
    if (numRays <= 0) return;

    constexpr int THREADS_PER_BLOCK = 256;
    int blocks = (numRays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    #define LAUNCH_KERNEL(N) \
        interpolateRenderRays<N><<<blocks, THREADS_PER_BLOCK, 0, stream>>>( \
            numRays, raysDone, ray_offsets, num_steps, t_sorted, \
            density_sigma, rgb_output, sigma_master, diffuse_master, \
            features_master, student_frac, student_rows_out, student_sigma, \
            student_weights, student_point_data, point_teacher_rgb, bg_color, m_pad \
        )

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
