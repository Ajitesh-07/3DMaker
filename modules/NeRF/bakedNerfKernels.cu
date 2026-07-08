#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    uint32_t* __restrict__ uniqueBitset)           // nullable: 1 bit / GLOBAL vertex, for the deduped count
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
        if (sigma > cascadeThresh) {
            mask |= (1ull << s);
            if (uniqueBitset) {
                int sx = s % B, sy = (s / B) % B, sz = s / (B * B);
                long long Vx = (long long)gx * (B - 1) + sx;
                long long Vy = (long long)gy * (B - 1) + sy;
                long long Vz = (long long)gz * (B - 1) + sz;
                long long vidx = (long long)cascade * (Lx * Ly * Lz) + Vz * (Lx * Ly) + Vy * Lx + Vx;
                atomicOr(&uniqueBitset[vidx >> 5], 1u << (uint32_t)(vidx & 31));
            }
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
    uint32_t* __restrict__ masksOut
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
    for (int i = 0; i < per; i++) {
        float logit = logits[idx * per + i];
        float sigma = expf(fminf(logit - densityBias, 8.0f));

        int x = i % B;
        int y = (i / B) % B;
        int z = i / (B * B);

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
}   

__global__ void buildGrid(
    const int* __restrict__ cellIds,
    const int* __restrict__ cellSlots,
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

    unsigned int owned_mask = __ballot_sync(0xFFFFFFFFu, owns);

    if (owns) {
        int leader     = __ffs(owned_mask) - 1;
        int warp_count = __popc(owned_mask);
        int baseIdx    = 0;
        if (lane_id == leader) baseIdx = atomicAdd(globalCounter, warp_count);
        baseIdx = __shfl_sync(owned_mask, baseIdx, leader);
        int idx = baseIdx + __popc(owned_mask & ((1u << lane_id) - 1));

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
