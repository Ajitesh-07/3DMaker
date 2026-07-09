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


__global__ void compute_ray_aabb_inv_kernel(
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
        nullptr, 0u, nullptr, nullptr, nullptr, nullptr,
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
        nullptr);

    return (int)raysFit;
}
