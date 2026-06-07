#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system_error.h>
#include <stdint.h>
#include <math.h>
#include <cub/cub.cuh>
#include "InstantNerf.h"

__device__ __forceinline__ void store_uint4(uint4* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    #ifndef __INTELLISENSE__
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w)
        : "memory"
    );
    #endif
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

__device__ __forceinline__ uint32_t expand_bits_3d(uint32_t v) {
    v &= 0x000003ff;                 
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

__device__ __forceinline__ uint32_t morton3d(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits_3d(x) | (expand_bits_3d(y) << 1) | (expand_bits_3d(z) << 2);
}

__device__ inline uint32_t get_mipmap_offset(uint3 base_res, int target_level) {
    uint32_t offset = 0;
    uint3 res = base_res;
    for (int l = 0; l < target_level; ++l) {
        offset += res.x * res.y * res.z;
        res.x >>= 1; // Divide by 2
        res.y >>= 1;
        res.z >>= 1;
    }
    return offset;
}

template<bool COMPUTE_MORTON>
__global__ void march_rays_dda_kernel(
    const uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3* __restrict__ rays_d_inv,
    const float* __restrict__ nears,
    const float* __restrict__ fars,
    const uint8_t* __restrict__ occupancy_grid,
    const uint3    grid_resolution, // This is the base Level 0 resolution (e.g. 128x128x128)
    const float3   aabb_min,
    const float3   aabb_max,
    const int      levelsMipmap,    // NEW PARAMETER
    uint32_t* __restrict__ packed_coords_out, 
    float* __restrict__ t_hits_out,
    uint32_t* __restrict__ num_steps_per_ray
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    float3 o = rays_o[i];
    float3 d = rays_d[i];
    float t_min = nears[i];
    float t_max_ray = fars[i];
    float3 d_inv = rays_d_inv[i];

    float3 aabb_extent = make_float3(aabb_max.x - aabb_min.x, aabb_max.y - aabb_min.y, aabb_max.z - aabb_min.z);

    int3 step = make_int3((d.x >= 0.0f) ? 1 : -1, (d.y >= 0.0f) ? 1 : -1, (d.z >= 0.0f) ? 1 : -1);

    uint32_t hit_count = 0; 
    uint32_t out_base_idx = i * MAX_HITS;
    uint32_t local_coords[4];
    float    local_ts[4];
    uint32_t local_idx = 0;

    float current_t = t_min;
    
    int current_level = levelsMipmap - 1;

    while (current_t < t_max_ray && (hit_count + local_idx) < MAX_HITS) {
        
        uint3 res_l = make_uint3(
            grid_resolution.x >> current_level,
            grid_resolution.y >> current_level,
            grid_resolution.z >> current_level
        );

        float3 voxel_size_l = make_float3(
            aabb_extent.x / res_l.x,
            aabb_extent.y / res_l.y,
            aabb_extent.z / res_l.z
        );

        // 2. Where are we in space right now?
        float3 current_pos = make_float3(o.x + current_t * d.x, o.y + current_t * d.y, o.z + current_t * d.z);
        
        float3 rel_pos = make_float3(
            (current_pos.x - aabb_min.x) / aabb_extent.x,
            (current_pos.y - aabb_min.y) / aabb_extent.y,
            (current_pos.z - aabb_min.z) / aabb_extent.z
        );

        int3 voxel_index = make_int3(
            max(0, min((int)floorf(rel_pos.x * res_l.x), (int)res_l.x - 1)),
            max(0, min((int)floorf(rel_pos.y * res_l.y), (int)res_l.y - 1)),
            max(0, min((int)floorf(rel_pos.z * res_l.z), (int)res_l.z - 1))
        );

        // 3. Read the bit from the specific mipmap level
        uint32_t level_offset = get_mipmap_offset(grid_resolution, current_level);
        uint32_t flat_idx = level_offset + 
                            voxel_index.z * (res_l.x * res_l.y) + 
                            voxel_index.y * res_l.x + 
                            voxel_index.x;

        uint32_t byte_idx = flat_idx >> 3;
        uint32_t bit_idx  = flat_idx & 7;
        bool is_occupied  = (occupancy_grid[byte_idx] >> bit_idx) & 1;

        if (is_occupied) {
            if (current_level > 0) {
                current_level--;
                continue; 
            } else {
                if constexpr (COMPUTE_MORTON) {
                    local_coords[local_idx] = morton3d((uint32_t)voxel_index.x, (uint32_t)voxel_index.y, (uint32_t)voxel_index.z);
                }
                local_ts[local_idx] = current_t;
                local_idx++;

                if (local_idx == 4) {
                    if constexpr (COMPUTE_MORTON) {
                        uint4* out_coords_vec = reinterpret_cast<uint4*>(&packed_coords_out[out_base_idx + hit_count]);
                        store_uint4(out_coords_vec, local_coords[0], local_coords[1], local_coords[2], local_coords[3]);
                    }
                    float4* out_ts_vec = reinterpret_cast<float4*>(&t_hits_out[out_base_idx + hit_count]);
                    store_float4(out_ts_vec, local_ts[0], local_ts[1], local_ts[2], local_ts[3]);

                    hit_count += 4;
                    local_idx = 0;
                }
            }
        }

        float3 next_boundary = make_float3(
            aabb_min.x + (voxel_index.x + (step.x > 0 ? 1.0f : 0.0f)) * voxel_size_l.x,
            aabb_min.y + (voxel_index.y + (step.y > 0 ? 1.0f : 0.0f)) * voxel_size_l.y,
            aabb_min.z + (voxel_index.z + (step.z > 0 ? 1.0f : 0.0f)) * voxel_size_l.z
        );

        float3 t_max_axis = make_float3(
            (next_boundary.x - o.x) * d_inv.x,
            (next_boundary.y - o.y) * d_inv.y,
            (next_boundary.z - o.z) * d_inv.z
        );

        float next_t = fminf(fminf(t_max_axis.x, t_max_axis.y), t_max_axis.z);
        
        current_t = fmaxf(current_t + 1e-5f, next_t + 1e-6f); 

        current_level = levelsMipmap - 1;
    }

    // Flush remaining hits
    for (uint32_t j = 0; j < local_idx; ++j) {
        if constexpr (COMPUTE_MORTON) {
            packed_coords_out[out_base_idx + hit_count + j] = local_coords[j];
        }
        t_hits_out[out_base_idx + hit_count + j] = local_ts[j];
    }
    
    hit_count += local_idx;
    num_steps_per_ray[i] = hit_count;
}

void launchMarchRaysDDA(
    const uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float* nears,
    const float* fars,
    const uint8_t* occupancy_grid,
    const uint3 grid_resolution,
    const float3 aabb_min,
    const float3 aabb_max,
    const int mipmapLevels,
    uint32_t* packed_coords_out,
    float* t_hits_out,
    uint32_t* num_steps_per_ray,
    cudaStream_t stream
) {
    if (num_rays == 0) return;

    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (num_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;

    march_rays_dda_kernel<true><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        num_rays,
        rays_o,
        rays_d,
        rays_d_inv,
        nears,
        fars,
        occupancy_grid,
        grid_resolution,
        aabb_min,
        aabb_max,
        mipmapLevels,
        packed_coords_out,
        t_hits_out,
        num_steps_per_ray
    );
}

// Stage 2: Compaction Kernel
__global__ void compact_hits_kernel(
    const uint32_t num_rays,
    const uint32_t* __restrict__ num_steps_per_ray,
    const uint32_t* __restrict__ ray_offsets,       
    const uint32_t* __restrict__ sparse_morton,     
    uint32_t* __restrict__ dense_morton_keys,       
    uint32_t* __restrict__ dense_sparse_indices     
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    uint32_t count = num_steps_per_ray[i];
    if (count == 0) return;

    uint32_t sparse_start = i * MAX_HITS;
    uint32_t dense_start = ray_offsets[i];

    for (uint32_t j = 0; j < count; ++j) {
        dense_morton_keys[dense_start + j] = sparse_morton[sparse_start + j];
        dense_sparse_indices[dense_start + j] = sparse_start + j;
    }
}

__global__ void generate_mlp_batch_kernel(
    const uint32_t total_hits,
    const uint32_t* __restrict__ sorted_sparse_indices, 
    const float*    __restrict__ sparse_ts,             
    const float3*   __restrict__ rays_o,
    const float3*   __restrict__ rays_d,
    const float3    aabb_min,
    const float3    aabb_max,
    float*          __restrict__ mlp_positions,
    uint32_t*       __restrict__ ray_indices,
    float*          __restrict__ t_sorted
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_hits) return;

    uint32_t sparse_idx = sorted_sparse_indices[idx];

    uint32_t ray_id = sparse_idx / MAX_HITS;

    float t = sparse_ts[sparse_idx];
    float3 o = rays_o[ray_id];
    float3 d = rays_d[ray_id];

    float3 pos;
    pos.x = o.x + t * d.x;
    pos.y = o.y + t * d.y;
    pos.z = o.z + t * d.z;

    // Scale position to [0, 1] using AABB
    pos.x = (pos.x - aabb_min.x) / (aabb_max.x - aabb_min.x);
    pos.y = (pos.y - aabb_min.y) / (aabb_max.y - aabb_min.y);
    pos.z = (pos.z - aabb_min.z) / (aabb_max.z - aabb_min.z);

    // Clamp to [0, 1] just in case of precision issues
    pos.x = fmaxf(0.0f, fminf(pos.x, 1.0f));
    pos.y = fmaxf(0.0f, fminf(pos.y, 1.0f));
    pos.z = fmaxf(0.0f, fminf(pos.z, 1.0f));

    float4* mlp_positions_vec = reinterpret_cast<float4*>(mlp_positions);
    store_float4(&mlp_positions_vec[idx], pos.x, pos.y, pos.z, 0.0f);

    ray_indices[idx] = ray_id;
    t_sorted[idx]    = t;
}

void processRaysChunk(
    const uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float* nears,
    const float* fars,
    const uint8_t* occupancy_grid,
    const uint3 grid_resolution,
    const float3 aabb_min,
    const float3 aabb_max,
    const int mipmapLevels,

    uint32_t* d_sparse_morton,   
    float* d_sparse_ts,          
    uint32_t* d_num_steps,      
    
    uint32_t* d_ray_offsets,
    uint32_t* d_dense_morton_keys_in,
    uint32_t* d_dense_sparse_indices_in,
    uint32_t* d_dense_morton_keys_out,
    uint32_t* d_dense_sparse_indices_out,
    
    float* d_mlp_positions_batch,
    uint32_t* d_ray_indices,
    float* d_t_sorted,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size_rays = (num_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;

    march_rays_dda_kernel<true><<<grid_size_rays, BLOCK_SIZE, 0, stream>>>(
        num_rays,
        rays_o,
        rays_d,
        rays_d_inv,
        nears,
        fars,
        occupancy_grid,
        grid_resolution,
        aabb_min,
        aabb_max,
        mipmapLevels,
        d_sparse_morton,
        d_sparse_ts,
        d_num_steps
    );

    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;

    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_num_steps, d_ray_offsets, num_rays, stream);
    CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_num_steps, d_ray_offsets, num_rays, stream);

    uint32_t last_offset, last_count;
    cudaMemcpyAsync(&last_offset, &d_ray_offsets[num_rays - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_count, &d_num_steps[num_rays - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    uint32_t total_hits = last_offset + last_count;
    if (total_hits == 0) return;

    compact_hits_kernel<<<grid_size_rays, BLOCK_SIZE, 0, stream>>>(
        num_rays, d_num_steps, d_ray_offsets, 
        d_sparse_morton, 
        d_dense_morton_keys_in, d_dense_sparse_indices_in
    );

    cudaFreeAsync(d_temp_storage, stream);
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        d_dense_morton_keys_in, d_dense_morton_keys_out,
        d_dense_sparse_indices_in, d_dense_sparse_indices_out,
        total_hits, 0, sizeof(uint32_t)*8, stream
    );
    
    CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_dense_morton_keys_in, d_dense_morton_keys_out,
        d_dense_sparse_indices_in, d_dense_sparse_indices_out,
        total_hits, 0, sizeof(uint32_t)*8, stream
    );

    const int grid_size_hits = (total_hits + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    generate_mlp_batch_kernel<<<grid_size_hits, BLOCK_SIZE, 0, stream>>>(
        total_hits,
        d_dense_sparse_indices_out,
        d_sparse_ts,
        rays_o, rays_d,
        aabb_min, aabb_max,
        d_mlp_positions_batch,
        d_ray_indices,
        d_t_sorted
    );

    cudaFreeAsync(d_temp_storage, stream);
}

__global__ void compact_hits_linear_kernel(
    const uint32_t num_rays,
    const uint32_t* __restrict__ num_steps_per_ray,
    const uint32_t* __restrict__ ray_offsets,
    uint32_t* __restrict__ dense_sparse_indices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    uint32_t count = num_steps_per_ray[i];
    if (count == 0) return;

    uint32_t sparse_start = i * MAX_HITS;
    uint32_t dense_start  = ray_offsets[i];

    for (uint32_t j = 0; j < count; ++j) {
        dense_sparse_indices[dense_start + j] = sparse_start + j;
    }
}

__global__ void block_prefix_sum_kernel(const uint32_t* d_in, uint32_t* d_out, uint32_t* d_block_sums, int num_items) {
    __shared__ uint32_t temp[1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t val = (gid < num_items) ? d_in[gid] : 0;
    temp[tid] = val;
    __syncthreads();

    // Hillis-Steele inclusive scan
    for (int offset = 1; offset < 1024; offset *= 2) {
        uint32_t prev = 0;
        if (tid >= offset) prev = temp[tid - offset];
        __syncthreads();
        if (tid >= offset) temp[tid] += prev;
        __syncthreads();
    }

    uint32_t excl_val = (tid > 0) ? temp[tid - 1] : 0;
    
    if (gid < num_items) {
        d_out[gid] = excl_val;
    }
    
    if (tid == 1023 && d_block_sums) {
        d_block_sums[blockIdx.x] = temp[1023];
    }
}

__global__ void add_block_sums_kernel(uint32_t* d_out, const uint32_t* d_block_sums, int num_items) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_items && blockIdx.x > 0) {
        d_out[gid] += d_block_sums[blockIdx.x];
    }
}

void custom_exclusive_sum(const uint32_t* d_in, uint32_t* d_out, int num_items, cudaStream_t stream) {
    if (num_items == 0) return;
    int num_blocks = (num_items + 1023) / 1024;
    
    uint32_t* d_block_sums = nullptr;
    if (num_blocks > 1) {
        cudaMallocAsync(&d_block_sums, num_blocks * sizeof(uint32_t), stream);
    }
    
    block_prefix_sum_kernel<<<num_blocks, 1024, 0, stream>>>(d_in, d_out, d_block_sums, num_items);
    
    if (num_blocks > 1) {
        block_prefix_sum_kernel<<<1, 1024, 0, stream>>>(d_block_sums, d_block_sums, nullptr, num_blocks);
        add_block_sums_kernel<<<num_blocks, 1024, 0, stream>>>(d_out, d_block_sums, num_items);
        cudaFreeAsync(d_block_sums, stream);
    }
}
void custom_exclusive_sum(const uint32_t* d_in, uint32_t* d_out, uint32_t* d_block_sums, int num_items, cudaStream_t stream) {
    if (num_items == 0) return;
    int num_blocks = (num_items + 1023) / 1024;
    
    block_prefix_sum_kernel<<<num_blocks, 1024, 0, stream>>>(d_in, d_out, d_block_sums, num_items);
    
    if (num_blocks > 1) {
        block_prefix_sum_kernel<<<1, 1024, 0, stream>>>(d_block_sums, d_block_sums, nullptr, num_blocks);
        add_block_sums_kernel<<<num_blocks, 1024, 0, stream>>>(d_out, d_block_sums, num_items);
    }
}

void processRaysChunkLinear(
    const uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float* nears,
    const float* fars,
    const uint8_t* occupancy_grid,
    const uint3 grid_resolution,
    const float3 aabb_min,
    const float3 aabb_max,
    const int mipmapLevels,

    float* d_sparse_ts,
    uint32_t* d_num_steps,

    uint32_t* d_ray_offsets,
    uint32_t* d_dense_sparse_indices,

    float* d_mlp_positions_batch,
    uint32_t* d_ray_indices,
    float* d_t_sorted,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size_rays = (num_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Stage 1: DDA ray march (no morton computation)
    march_rays_dda_kernel<false><<<grid_size_rays, BLOCK_SIZE, 0, stream>>>(
        num_rays,
        rays_o, rays_d, rays_d_inv,
        nears, fars,
        occupancy_grid,
        grid_resolution, aabb_min, aabb_max,
        mipmapLevels,
        nullptr, d_sparse_ts, d_num_steps
    );

    // Stage 2: Prefix sum for compaction offsets
    custom_exclusive_sum(d_num_steps, d_ray_offsets, num_rays, stream);

    uint32_t last_offset, last_count;
    cudaMemcpyAsync(&last_offset, &d_ray_offsets[num_rays - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_count, &d_num_steps[num_rays - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    uint32_t total_hits = last_offset + last_count;

    if (total_hits == 0) return;

    // Stage 3: Compact (linear — no morton sort)
    compact_hits_linear_kernel<<<grid_size_rays, BLOCK_SIZE, 0, stream>>>(
        num_rays, d_num_steps, d_ray_offsets,
        d_dense_sparse_indices
    );

    // Stage 4: Generate MLP batch positions + ray metadata
    const int grid_size_hits = (total_hits + BLOCK_SIZE - 1) / BLOCK_SIZE;

    generate_mlp_batch_kernel<<<grid_size_hits, BLOCK_SIZE, 0, stream>>>(
        total_hits,
        d_dense_sparse_indices,
        d_sparse_ts,
        rays_o, rays_d,
        aabb_min, aabb_max,
        d_mlp_positions_batch,
        d_ray_indices,
        d_t_sorted
    );
}

__global__ void march_rays_dda_calc_hits(
    const uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3* __restrict__ rays_d_inv,
    const float* __restrict__ nears,
    const float* __restrict__ fars,
    const uint8_t* __restrict__ occupancy_grid,
    const uint3    grid_resolution, // This is the base Level 0 resolution (e.g. 128x128x128)
    const float3   aabb_min,
    const float3   aabb_max,
    const int      levelsMipmap,
    uint32_t* __restrict__ num_steps_per_ray
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    float3 o = rays_o[i];
    float3 d = rays_d[i];
    float t_min = nears[i];
    float t_max_ray = fars[i];
    float3 d_inv = rays_d_inv[i];

    float3 aabb_extent = make_float3(aabb_max.x - aabb_min.x, aabb_max.y - aabb_min.y, aabb_max.z - aabb_min.z);

    int3 step = make_int3((d.x >= 0.0f) ? 1 : -1, (d.y >= 0.0f) ? 1 : -1, (d.z >= 0.0f) ? 1 : -1);

    uint32_t hit_count = 0;
    float current_t = t_min;
    
    int current_level = levelsMipmap - 1;

    while (current_t < t_max_ray && hit_count < MAX_HITS) {
        uint3 res_l = make_uint3(
            grid_resolution.x >> current_level,
            grid_resolution.y >> current_level,
            grid_resolution.z >> current_level
        );

        float3 voxel_size_l = make_float3(
            aabb_extent.x / res_l.x,
            aabb_extent.y / res_l.y,
            aabb_extent.z / res_l.z
        );

        float3 current_pos = make_float3(o.x + current_t * d.x, o.y + current_t * d.y, o.z + current_t * d.z);

        float3 rel_pos = make_float3(
            (current_pos.x - aabb_min.x) / aabb_extent.x,
            (current_pos.y - aabb_min.y) / aabb_extent.y,
            (current_pos.z - aabb_min.z) / aabb_extent.z
        );

        int3 voxel_index = make_int3(
            max(0, min((int)floorf(rel_pos.x * res_l.x), (int)res_l.x - 1)),
            max(0, min((int)floorf(rel_pos.y * res_l.y), (int)res_l.y - 1)),
            max(0, min((int)floorf(rel_pos.z * res_l.z), (int)res_l.z - 1))
        );

        uint32_t level_offset = get_mipmap_offset(grid_resolution, current_level);
        uint32_t flat_idx = level_offset + 
                            voxel_index.z * (res_l.x * res_l.y) + 
                            voxel_index.y * res_l.x + 
                            voxel_index.x;
        
        uint32_t byte_idx = flat_idx >> 3;
        uint32_t bit_idx  = flat_idx & 7;
        bool is_occupied  = (occupancy_grid[byte_idx] >> bit_idx) & 1;

        if (is_occupied) {
            if (current_level > 0) {
                current_level--;
                continue;
            } else hit_count++;
        }

        float3 next_boundary = make_float3(
            aabb_min.x + (voxel_index.x + (step.x > 0 ? 1.0f : 0.0f)) * voxel_size_l.x,
            aabb_min.y + (voxel_index.y + (step.y > 0 ? 1.0f : 0.0f)) * voxel_size_l.y,
            aabb_min.z + (voxel_index.z + (step.z > 0 ? 1.0f : 0.0f)) * voxel_size_l.z
        );

        float3 t_max_axis = make_float3(
            (next_boundary.x - o.x) * d_inv.x,
            (next_boundary.y - o.y) * d_inv.y,
            (next_boundary.z - o.z) * d_inv.z
        );

        float next_t = fminf(fminf(t_max_axis.x, t_max_axis.y), t_max_axis.z);
        
        current_t = fmaxf(current_t + 1e-5f, next_t + 1e-6f); 

        current_level = levelsMipmap - 1;
    }

    num_steps_per_ray[i] = hit_count;
}

__global__ void find_cutoff_ray_kernel(
    const uint32_t* __restrict__ ray_offsets, 
    const uint32_t* __restrict__ num_steps,
    uint32_t num_rays, 
    uint32_t batch_size, 
    uint32_t* __restrict__ active_rays_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_rays) {
        uint32_t hits_inclusive = ray_offsets[i] + num_steps[i];
        
        if (hits_inclusive <= batch_size) {
            bool boundary = true;
            if (i < num_rays - 1) {
                uint32_t next_hits_inclusive = ray_offsets[i + 1] + num_steps[i + 1];
                if (next_hits_inclusive <= batch_size) {
                    boundary = false;
                }
            }
            
            if (boundary) {
                *active_rays_count = i + 1;
            }
        }
    } else if (i == num_rays) { 
        if (num_rays > 0 && ray_offsets[0] + num_steps[0] > batch_size) {
            *active_rays_count = 0;
        }
    }
}

__global__ void march_rays_dda_offset(
    const uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3* __restrict__ rays_d_inv,
    const float* __restrict__ nears,
    const float* __restrict__ fars,
    const uint8_t* __restrict__ occupancy_grid,
    const uint3    grid_resolution,
    const float3   aabb_min,
    const float3   aabb_max,
    const int      levelsMipmap,
    const uint32_t* __restrict__ ray_offsets,
    float* __restrict__ t_hits_out,
    uint32_t* __restrict__ ray_indices_out,
    float* __restrict__ mlp_positions
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    float3 o = rays_o[i];
    float3 d = rays_d[i];
    float t_min = nears[i];
    float t_max_ray = fars[i];
    float3 d_inv = rays_d_inv[i];

    float3 aabb_extent = make_float3(aabb_max.x - aabb_min.x, aabb_max.y - aabb_min.y, aabb_max.z - aabb_min.z);
    int3 step = make_int3((d.x >= 0.0f) ? 1 : -1, (d.y >= 0.0f) ? 1 : -1, (d.z >= 0.0f) ? 1 : -1);

    uint32_t hit_count = 0; 
    uint32_t out_base_idx = ray_offsets[i];
    
    float current_t = t_min;
    int current_level = levelsMipmap - 1;

    while (current_t < t_max_ray && hit_count < MAX_HITS) {
        
        uint3 res_l = make_uint3(
            grid_resolution.x >> current_level,
            grid_resolution.y >> current_level,
            grid_resolution.z >> current_level
        );

        float3 voxel_size_l = make_float3(
            aabb_extent.x / res_l.x,
            aabb_extent.y / res_l.y,
            aabb_extent.z / res_l.z
        );

        float3 current_pos = make_float3(o.x + current_t * d.x, o.y + current_t * d.y, o.z + current_t * d.z);
        
        float3 rel_pos = make_float3(
            (current_pos.x - aabb_min.x) / aabb_extent.x,
            (current_pos.y - aabb_min.y) / aabb_extent.y,
            (current_pos.z - aabb_min.z) / aabb_extent.z
        );

        int3 voxel_index = make_int3(
            max(0, min((int)floorf(rel_pos.x * res_l.x), (int)res_l.x - 1)),
            max(0, min((int)floorf(rel_pos.y * res_l.y), (int)res_l.y - 1)),
            max(0, min((int)floorf(rel_pos.z * res_l.z), (int)res_l.z - 1))
        );

        uint32_t level_offset = get_mipmap_offset(grid_resolution, current_level);
        uint32_t flat_idx = level_offset + 
                            voxel_index.z * (res_l.x * res_l.y) + 
                            voxel_index.y * res_l.x + 
                            voxel_index.x;

        uint32_t byte_idx = flat_idx >> 3;
        uint32_t bit_idx  = flat_idx & 7;
        bool is_occupied  = (occupancy_grid[byte_idx] >> bit_idx) & 1;

        if (is_occupied) {
            if (current_level > 0) {
                current_level--;
                continue; 
            } else {
                uint32_t write_idx = out_base_idx + hit_count;
                
                t_hits_out[write_idx] = current_t;
                ray_indices_out[write_idx] = i; 
                
                float3 pos = make_float3(
                    o.x + current_t * d.x, 
                    o.y + current_t * d.y, 
                    o.z + current_t * d.z
                );
                
                pos.x = fmaxf(0.0f, fminf((pos.x - aabb_min.x) / aabb_extent.x, 1.0f));
                pos.y = fmaxf(0.0f, fminf((pos.y - aabb_min.y) / aabb_extent.y, 1.0f));
                pos.z = fmaxf(0.0f, fminf((pos.z - aabb_min.z) / aabb_extent.z, 1.0f));
                
                float4* mlp_pos_vec = reinterpret_cast<float4*>(mlp_positions);
                store_float4(&mlp_pos_vec[write_idx], pos.x, pos.y, pos.z, 0.0f);
                
                hit_count++;
            }
        }

        float3 next_boundary = make_float3(
            aabb_min.x + (voxel_index.x + (step.x > 0 ? 1.0f : 0.0f)) * voxel_size_l.x,
            aabb_min.y + (voxel_index.y + (step.y > 0 ? 1.0f : 0.0f)) * voxel_size_l.y,
            aabb_min.z + (voxel_index.z + (step.z > 0 ? 1.0f : 0.0f)) * voxel_size_l.z
        );

        float3 t_max_axis = make_float3(
            (next_boundary.x - o.x) * d_inv.x,
            (next_boundary.y - o.y) * d_inv.y,
            (next_boundary.z - o.z) * d_inv.z
        );

        float next_t = fminf(fminf(t_max_axis.x, t_max_axis.y), t_max_axis.z);
        current_t = fmaxf(current_t + 1e-5f, next_t + 1e-6f); 
        current_level = levelsMipmap - 1;
    }
}

int processRaysHitLinear(
    const uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float* nears,
    const float* fars,
    const uint8_t* occupancy_grid,
    const uint3 grid_resolution,
    const float3 aabb_min,
    const float3 aabb_max,
    const int mipmapLevels,
    const int batchSize,
    uint32_t* totalHits,
    uint32_t* d_active_rays_count,

    uint32_t* d_num_steps,
    uint32_t* d_ray_offsets,
    
    float* d_mlp_positions_batch,
    uint32_t* d_ray_indices,
    float* d_t_sorted,
    uint32_t* d_block_sums,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    const int gs = (num_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;
    march_rays_dda_calc_hits<<<gs, BLOCK_SIZE, 0, stream>>>(
        num_rays,
        rays_o,
        rays_d,
        rays_d_inv,
        nears,
        fars,
        occupancy_grid,
        grid_resolution,
        aabb_min,
        aabb_max,
        mipmapLevels,
        d_num_steps
    );

    custom_exclusive_sum(d_num_steps, d_ray_offsets, d_block_sums, num_rays, stream);

    uint32_t last_offset, last_count;
    cudaMemcpyAsync(&last_offset, &d_ray_offsets[num_rays - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_count, &d_num_steps[num_rays - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t total_hits = last_offset + last_count;
    if (total_hits == 0) {
        *totalHits = 0;
        return num_rays;
    }

    cudaMemsetAsync(d_active_rays_count, 0, sizeof(uint32_t), stream);

    find_cutoff_ray_kernel<<<gs, BLOCK_SIZE, 0, stream>>>(
        d_ray_offsets,
        d_num_steps,
        num_rays,
        batchSize,
        d_active_rays_count
    );

    uint32_t h_active_rays_count = 0;
    cudaMemcpyAsync(&h_active_rays_count, d_active_rays_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_active_rays_count == 0) {
        *totalHits = 0;
        return 0;
    }

    uint32_t actual_total_hits;
    cudaMemcpyAsync(&actual_total_hits, &d_ray_offsets[h_active_rays_count - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    uint32_t actual_last_count;
    cudaMemcpyAsync(&actual_last_count, &d_num_steps[h_active_rays_count - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    *totalHits = actual_total_hits + actual_last_count;

    int gs2 = (h_active_rays_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    march_rays_dda_offset<<<gs2, BLOCK_SIZE, 0, stream>>>(
        h_active_rays_count,
        rays_o,
        rays_d,
        rays_d_inv,
        nears,
        fars,
        occupancy_grid,
        grid_resolution,
        aabb_min,
        aabb_max,
        mipmapLevels,
        d_ray_offsets,
        d_t_sorted,
        d_ray_indices,
        d_mlp_positions_batch
    );

    return h_active_rays_count;
}

