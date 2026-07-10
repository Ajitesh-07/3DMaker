#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Device Inline Functions
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

__device__ __forceinline__ float3 bake_contract_pos(float3 pos, float3 aabb_min, float3 aabb_max) {
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

// Global Kernels
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
);

__global__ void baked_compute_SH_gather(
    const float3* __restrict__ d_chunk_d,
    const uint32_t* __restrict__ d_ray_indices,
    const int offset,
    const int batchSize,
    const float densityBias,
    float* __restrict__ d_density_out,
    half* __restrict__ d_color_in,
    float* __restrict__ d_density_sigma
);

__global__ void compute_SH_student_gather(
    const float3* __restrict__ d_chunk_d,
    const uint32_t* __restrict__ d_ray_indices,
    half* __restrict__ d_student_point_data,
    const int offset,
    const int viewFeatures,
    const int pad,
    int totalHits
);

__global__ void compute_SH_student_deferred(
    const float3* __restrict__ d_chunk_d,
    half* __restrict__ d_student_point_data,
    const int viewFeatures,
    const int pad,
    const int numRays
);

__global__ void buildInvCellMap(
    const int* __restrict__ cellIds,
    int* __restrict__ cellSlots,
    int N
);

__global__ void accumulateColor(
    float* __restrict__ d_color_out,
    float* __restrict__ d_color_sum,
    int N
);

__global__ void fillVertexGrid(
    float* __restrict__ density_out,
    float* __restrict__ color_sum,
    half* __restrict__  masterSigmaGrid,
    float* __restrict__ masterColorGrid,
    float invNum,
    float densityBias,
    int offset,
    int N
);

__global__ void k_genSubVoxelPositions(
    const int* __restrict__ cellIds, int numBlocks, int B,
    uint3 gridRes, float3 aabbMin, float3 aabbMax,
    float* __restrict__ outPos);

__global__ void k_buildSubVoxelMasks(
    const float* __restrict__ logits, const int* __restrict__ cellIds,
    int numBlocks, int B, uint3 gridRes,
    float densityBias, float minDensityThreshold, float baseVoxel,
    uint64_t* __restrict__ masksOut,
    unsigned long long* __restrict__ survivorTotal,
    unsigned long long* __restrict__ fillHist,
    uint32_t* __restrict__ uniqueBitset,           // nullable: 1 bit / GLOBAL vertex that SURVIVES sigma
    uint32_t* __restrict__ allBitset);

__global__ void buildSubVoxelMasks(
    const float* __restrict__ logits,
    const int* __restrict__ cellIds,
    int numBlocks, int B,
    uint3 gridRes,
    float densityBias, float minDensityThreshold,
    float baseVoxel,
    uint32_t* __restrict__ masksOut,
    uint64_t* __restrict__ subVoxelMask
);

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
);

__global__ void compute_residual_and_loss_grad(
    int total_hits,
    int padded_b_size,
    int view_features,
    int m_pad,
    const half* __restrict__ student_point_data,
    const float* __restrict__ teacher_rgb,
    float* __restrict__ student_rgb_chunk,
    half* __restrict__ phi_chunk
);

__global__ void compute_residual(
    int padded_b_size,
    int view_features,
    int m_pad,
    const half* __restrict__ student_point_data,
    float* __restrict__ student_rgb_chunk
);

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
);

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
);

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
    uint32_t* __restrict__ num_steps_per_ray);

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
);

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
);

// Host Launch / Process Functions
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
);

void processBakedRaysHitData(
    const uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float* nears,
    const float* fars,
    const uint8_t* occupancy_grid,
    const uint32_t* voxelMask27,
    const uint32_t* vertexRows,
    const int*      reverseMap,
    const uint3 gridRes,
    const float3 aabbMin,
    const float3 aabbMax,
    const int numCascades,
    const int mipmapLevels,
    const int B,
    
    uint32_t* d_num_steps,
    uint32_t* d_ray_offsets,
    uint32_t* d_block_sums,
    cudaStream_t stream
);

int processBakedRaysHitPositions(
    const uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float* nears,
    const float* fars,
    const uint8_t* occupancy_grid,
    const uint32_t* voxelMask27,
    const uint32_t* vertexRows,
    const int*      reverseMap,
    const uint3 grid_resolution,
    const float3 aabb_min,
    const float3 aabb_max,
    const int numCascades,
    const int mipmapLevels,
    const int B,

    const uint32_t rays_done,
    const int batchSize,
    uint32_t* totalHits,
    uint32_t* out_base,
    uint32_t* d_num_steps,
    uint32_t* d_ray_offsets,
    float*    d_teacher_points_out,   // batchSize*4
    int*      d_student_rows_out,     // batchSize*8
    float*    d_student_frac_out,     // batchSize*3
    uint32_t* d_ray_indices,
    uint32_t* d_active_rays_count,
    float* d_t_sorted,
    uint32_t* d_block_sums,
    cudaStream_t stream
);

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
    cudaStream_t stream);

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
    cudaStream_t stream = nullptr
);

void launch_compute_SH_student_deferred(
    const float3* d_chunk_d,
    half* d_student_point_data,
    const int viewFeatures,
    const int pad,
    const int numRays,
    cudaStream_t stream = nullptr
);

void launch_compute_residual_and_loss_grad(
    int total_hits,
    int padded_b_size,
    int view_features,
    int m_pad,
    const half* student_point_data,
    const float* teacher_rgb,
    float* student_rgb_chunk,
    half* phi_chunk,
    cudaStream_t stream = nullptr
);

void launch_bakedRenderRaysFused(
    int view_features,
    uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float* nears,
    const float* fars,
    const uint8_t* occupancy_grid,
    const uint32_t* voxelMask27,
    const uint32_t* vertexRows,
    const int* reverseMap,
    uint3 gridRes, float3 aabbMin, float3 aabbMax,
    int numCascades, int levelsMipmap, int B,
    const half* sigma_master,
    const float3* diffuse_master,
    const float* features_master,
    float3 bg_color,
    half* student_point_data,
    int m_pad,
    cudaStream_t stream = nullptr
);
