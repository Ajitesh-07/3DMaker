#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../InstantNerf.h"

// ============================================================================
//  Helper: check CUDA errors
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " — " << cudaGetErrorString(err) << "\n";            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ============================================================================
//  CPU reference helpers
// ============================================================================
static uint32_t cpu_expand_bits_3d(uint32_t v) {
    v &= 0x000003ffu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

static uint32_t cpu_morton3d(uint32_t x, uint32_t y, uint32_t z) {
    return cpu_expand_bits_3d(x) | (cpu_expand_bits_3d(y) << 1) | (cpu_expand_bits_3d(z) << 2);
}

// ============================================================================
//  CPU reference: Stage 1 — DDA ray marching
// ============================================================================
static void cpu_march_rays_dda(
    uint32_t num_rays,
    const float3* rays_o,
    const float3* rays_d,
    const float3* rays_d_inv,
    const float*  nears,
    const float*  fars,
    const uint8_t* occupancy_grid,
    uint3   grid_resolution,
    float3  aabb_min,
    float3  aabb_max,
    uint32_t* packed_coords_out,   // [num_rays * MAX_HITS]
    float*    t_hits_out,          // [num_rays * MAX_HITS]
    uint32_t* num_steps_per_ray    // [num_rays]
) {
    float3 extent = {aabb_max.x - aabb_min.x,
                     aabb_max.y - aabb_min.y,
                     aabb_max.z - aabb_min.z};
    float3 voxel_size = {extent.x / grid_resolution.x,
                         extent.y / grid_resolution.y,
                         extent.z / grid_resolution.z};

    for (uint32_t i = 0; i < num_rays; ++i) {
        float3 o = rays_o[i];
        float3 d = rays_d[i];
        float3 d_inv_v = rays_d_inv[i];
        float  t_min = nears[i];
        float  t_max_ray = fars[i];

        // entry point
        float3 p_entry = {o.x + t_min * d.x,
                          o.y + t_min * d.y,
                          o.z + t_min * d.z};

        float3 rel_pos = {(p_entry.x - aabb_min.x) / extent.x,
                          (p_entry.y - aabb_min.y) / extent.y,
                          (p_entry.z - aabb_min.z) / extent.z};

        float3 grid_pos = {rel_pos.x * grid_resolution.x,
                           rel_pos.y * grid_resolution.y,
                           rel_pos.z * grid_resolution.z};

        int3 voxel_index;
        voxel_index.x = std::max(0, std::min((int)std::floor(grid_pos.x), (int)grid_resolution.x - 1));
        voxel_index.y = std::max(0, std::min((int)std::floor(grid_pos.y), (int)grid_resolution.y - 1));
        voxel_index.z = std::max(0, std::min((int)std::floor(grid_pos.z), (int)grid_resolution.z - 1));

        int3 step;
        step.x = (d.x >= 0.0f) ? 1 : -1;
        step.y = (d.y >= 0.0f) ? 1 : -1;
        step.z = (d.z >= 0.0f) ? 1 : -1;

        float3 t_delta;
        t_delta.x = std::fabs(voxel_size.x * d_inv_v.x);
        t_delta.y = std::fabs(voxel_size.y * d_inv_v.y);
        t_delta.z = std::fabs(voxel_size.z * d_inv_v.z);

        float3 next_boundary;
        next_boundary.x = aabb_min.x + (voxel_index.x + (step.x > 0 ? 1.0f : 0.0f)) * voxel_size.x;
        next_boundary.y = aabb_min.y + (voxel_index.y + (step.y > 0 ? 1.0f : 0.0f)) * voxel_size.y;
        next_boundary.z = aabb_min.z + (voxel_index.z + (step.z > 0 ? 1.0f : 0.0f)) * voxel_size.z;

        float3 t_max_axis;
        t_max_axis.x = (next_boundary.x - o.x) * d_inv_v.x;
        t_max_axis.y = (next_boundary.y - o.y) * d_inv_v.y;
        t_max_axis.z = (next_boundary.z - o.z) * d_inv_v.z;

        float current_t = t_min;
        uint32_t hit_count = 0;
        uint32_t out_base = i * MAX_HITS;

        while (current_t < t_max_ray && hit_count < MAX_HITS) {
            if (voxel_index.x < 0 || voxel_index.x >= (int)grid_resolution.x ||
                voxel_index.y < 0 || voxel_index.y >= (int)grid_resolution.y ||
                voxel_index.z < 0 || voxel_index.z >= (int)grid_resolution.z) {
                break;
            }

            uint32_t flat_idx = voxel_index.z * (grid_resolution.x * grid_resolution.y) +
                                voxel_index.y * grid_resolution.x +
                                voxel_index.x;

            uint32_t morton_idx = cpu_morton3d((uint32_t)voxel_index.x,
                                              (uint32_t)voxel_index.y,
                                              (uint32_t)voxel_index.z);

            uint32_t byte_idx = flat_idx >> 3;
            uint32_t bit_idx  = flat_idx & 7;
            bool is_occupied  = (occupancy_grid[byte_idx] >> bit_idx) & 1;

            if (is_occupied) {
                packed_coords_out[out_base + hit_count] = morton_idx;
                t_hits_out[out_base + hit_count]        = current_t;
                hit_count++;
            }

            // DDA step
            if (t_max_axis.x < t_max_axis.y) {
                if (t_max_axis.x < t_max_axis.z) {
                    current_t = t_max_axis.x;
                    voxel_index.x += step.x;
                    t_max_axis.x += t_delta.x;
                } else {
                    current_t = t_max_axis.z;
                    voxel_index.z += step.z;
                    t_max_axis.z += t_delta.z;
                }
            } else {
                if (t_max_axis.y < t_max_axis.z) {
                    current_t = t_max_axis.y;
                    voxel_index.y += step.y;
                    t_max_axis.y += t_delta.y;
                } else {
                    current_t = t_max_axis.z;
                    voxel_index.z += step.z;
                    t_max_axis.z += t_delta.z;
                }
            }
        }

        num_steps_per_ray[i] = hit_count;
    }
}

// ============================================================================
//  CPU reference: Stage 2 — compaction
// ============================================================================
static void cpu_compact_hits(
    uint32_t num_rays,
    const uint32_t* num_steps_per_ray,
    const uint32_t* ray_offsets,       // exclusive prefix sum
    const uint32_t* sparse_morton,
    uint32_t* dense_morton_keys,
    uint32_t* dense_sparse_indices
) {
    for (uint32_t i = 0; i < num_rays; ++i) {
        uint32_t count = num_steps_per_ray[i];
        if (count == 0) continue;
        uint32_t sparse_start = i * MAX_HITS;
        uint32_t dense_start  = ray_offsets[i];
        for (uint32_t j = 0; j < count; ++j) {
            dense_morton_keys[dense_start + j]   = sparse_morton[sparse_start + j];
            dense_sparse_indices[dense_start + j] = sparse_start + j;
        }
    }
}

// ============================================================================
//  CPU reference: Stage 3 — generate MLP batch positions + ray metadata
// ============================================================================
static void cpu_generate_mlp_batch(
    uint32_t total_hits,
    const uint32_t* sorted_sparse_indices,
    const float*    sparse_ts,
    const float3*   rays_o,
    const float3*   rays_d,
    float*          mlp_positions,   // [total_hits * 3]
    uint32_t*       ray_indices,     // [total_hits]
    float*          t_sorted         // [total_hits]
) {
    for (uint32_t idx = 0; idx < total_hits; ++idx) {
        uint32_t sparse_idx = sorted_sparse_indices[idx];
        uint32_t ray_id     = sparse_idx / MAX_HITS;

        float t   = sparse_ts[sparse_idx];
        float3 o  = rays_o[ray_id];
        float3 d  = rays_d[ray_id];

        mlp_positions[idx * 3 + 0] = o.x + t * d.x;
        mlp_positions[idx * 3 + 1] = o.y + t * d.y;
        mlp_positions[idx * 3 + 2] = o.z + t * d.z;

        ray_indices[idx] = ray_id;
        t_sorted[idx]    = t;
    }
}

// ============================================================================
//  CPU exclusive prefix-sum (mirrors CUB ExclusiveSum)
// ============================================================================
static void cpu_exclusive_sum(const uint32_t* in, uint32_t* out, uint32_t n) {
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = running;
        running += in[i];
    }
}

// ============================================================================
//  CPU radix sort by key (morton -> sparse_index)
// ============================================================================
static void cpu_sort_by_key(
    const uint32_t* keys_in,
    const uint32_t* vals_in,
    uint32_t* keys_out,
    uint32_t* vals_out,
    uint32_t n
) {
    std::vector<uint32_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0u);
    std::stable_sort(indices.begin(), indices.end(),
        [&](uint32_t a, uint32_t b) { return keys_in[a] < keys_in[b]; });
    for (uint32_t i = 0; i < n; ++i) {
        keys_out[i] = keys_in[indices[i]];
        vals_out[i] = vals_in[indices[i]];
    }
}

// ============================================================================
//  Verification helpers
// ============================================================================
static bool verify_uint32(const char* label, const uint32_t* cpu, const uint32_t* gpu,
                           uint32_t count, int max_prints = 5) {
    int mismatches = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (cpu[i] != gpu[i]) {
            if (mismatches < max_prints)
                std::cout << "  " << label << " mismatch [" << i << "]: CPU=" << cpu[i]
                          << " GPU=" << gpu[i] << "\n";
            mismatches++;
        }
    }
    if (mismatches > 0) {
        std::cout << label << ": FAILED (" << mismatches << " mismatches)\n";
        return false;
    }
    std::cout << label << ": SUCCESS\n";
    return true;
}

static bool verify_float(const char* label, const float* cpu, const float* gpu,
                          uint32_t count, float tol = 1e-5f, int max_prints = 5) {
    float max_err = 0.f;
    int mismatches = 0;
    for (uint32_t i = 0; i < count; ++i) {
        float diff = std::fabs(cpu[i] - gpu[i]);
        if (diff > max_err) max_err = diff;
        if (diff > tol) {
            if (mismatches < max_prints)
                std::cout << "  " << label << " mismatch [" << i << "]: CPU=" << cpu[i]
                          << " GPU=" << gpu[i] << " diff=" << diff << "\n";
            mismatches++;
        }
    }
    if (mismatches > 0) {
        std::cout << label << ": FAILED (max_err=" << max_err << ", mismatches=" << mismatches << ")\n";
        return false;
    }
    std::cout << label << ": SUCCESS (max_err=" << max_err << ")\n";
    return true;
}

// ============================================================================
//  Generate deterministic test scene
// ============================================================================
struct TestScene {
    uint32_t num_rays;
    uint3   grid_resolution;
    float3  aabb_min, aabb_max;

    std::vector<float3> rays_o;
    std::vector<float3> rays_d;
    std::vector<float3> rays_d_inv;
    std::vector<float>  nears;
    std::vector<float>  fars;
    std::vector<uint8_t> occupancy_grid;
};

static TestScene generateTestScene(uint32_t num_rays, uint32_t grid_res, float occupancy_ratio) {
    TestScene s;
    s.num_rays = num_rays;
    s.grid_resolution = {grid_res, grid_res, grid_res};
    s.aabb_min = {-1.0f, -1.0f, -1.0f};
    s.aabb_max = { 1.0f,  1.0f,  1.0f};

    // Occupancy grid (bit-packed)
    uint32_t total_voxels = grid_res * grid_res * grid_res;
    uint32_t grid_bytes   = (total_voxels + 7) / 8;
    s.occupancy_grid.resize(grid_bytes, 0);

    srand(42);
    for (uint32_t v = 0; v < total_voxels; ++v) {
        float r = (float)rand() / RAND_MAX;
        if (r < occupancy_ratio) {
            s.occupancy_grid[v >> 3] |= (1u << (v & 7));
        }
    }

    // Rays: origin at sphere surface pointing inward through the volume
    s.rays_o.resize(num_rays);
    s.rays_d.resize(num_rays);
    s.rays_d_inv.resize(num_rays);
    s.nears.resize(num_rays);
    s.fars.resize(num_rays);

    for (uint32_t i = 0; i < num_rays; ++i) {
        // random direction, origin outside AABB
        float theta = ((float)rand() / RAND_MAX) * 2.0f * 3.14159265f;
        float phi   = ((float)rand() / RAND_MAX) * 3.14159265f;
        float sp    = sinf(phi);

        float3 dir = {sp * cosf(theta), sp * sinf(theta), cosf(phi)};
        float len  = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        dir.x /= len; dir.y /= len; dir.z /= len;

        // origin 3 units from centre, pointing inward
        float3 origin = {-dir.x * 3.0f, -dir.y * 3.0f, -dir.z * 3.0f};

        s.rays_o[i] = origin;
        s.rays_d[i] = dir;
        s.rays_d_inv[i] = {1.0f / (dir.x == 0.f ? 1e-8f : dir.x),
                           1.0f / (dir.y == 0.f ? 1e-8f : dir.y),
                           1.0f / (dir.z == 0.f ? 1e-8f : dir.z)};

        // Compute AABB intersection [t_near, t_far]
        float t_near = -1e30f, t_far = 1e30f;
        for (int a = 0; a < 3; ++a) {
            float o_a = (&origin.x)[a];
            float d_a = (&dir.x)[a];
            float inv = (&s.rays_d_inv[i].x)[a];
            float lo  = (&s.aabb_min.x)[a];
            float hi  = (&s.aabb_max.x)[a];
            float t1  = (lo - o_a) * inv;
            float t2  = (hi - o_a) * inv;
            if (t1 > t2) std::swap(t1, t2);
            t_near = std::max(t_near, t1);
            t_far  = std::min(t_far,  t2);
        }
        t_near = std::max(t_near, 0.0f);

        s.nears[i] = t_near;
        s.fars[i]  = t_far;
    }

    return s;
}

// ============================================================================
//  CORRECTNESS TEST
// ============================================================================
void testProcessRaysCorrectness() {
    std::cout << "\n========================================\n";
    std::cout << "  processRays — Correctness Verification\n";
    std::cout << "========================================\n";

    constexpr uint32_t NUM_RAYS = 4096;
    constexpr uint32_t GRID_RES = 32;
    constexpr float OCCUPANCY   = 0.3f;

    auto scene = generateTestScene(NUM_RAYS, GRID_RES, OCCUPANCY);

    // ---- Allocate sparse buffers (CPU) ----
    std::vector<uint32_t> cpu_sparse_morton(NUM_RAYS * MAX_HITS, 0);
    std::vector<float>    cpu_sparse_ts(NUM_RAYS * MAX_HITS, 0.0f);
    std::vector<uint32_t> cpu_num_steps(NUM_RAYS, 0);

    // ---- Stage 1 CPU reference: DDA march ----
    std::cout << "\n--- Stage 1: march_rays_dda ---\n";
    cpu_march_rays_dda(
        NUM_RAYS,
        scene.rays_o.data(), scene.rays_d.data(), scene.rays_d_inv.data(),
        scene.nears.data(), scene.fars.data(),
        scene.occupancy_grid.data(),
        scene.grid_resolution, scene.aabb_min, scene.aabb_max,
        cpu_sparse_morton.data(), cpu_sparse_ts.data(), cpu_num_steps.data());

    uint32_t cpu_total_hits = 0;
    for (uint32_t i = 0; i < NUM_RAYS; ++i) cpu_total_hits += cpu_num_steps[i];
    std::cout << "  CPU total hits: " << cpu_total_hits << "\n";

    // ---- GPU Stage 1 ----
    float3*   d_rays_o; float3*   d_rays_d; float3*   d_rays_d_inv;
    float*    d_nears;  float*    d_fars;
    uint8_t*  d_occ;
    uint32_t* d_sparse_morton; float* d_sparse_ts; uint32_t* d_num_steps;

    CUDA_CHECK(cudaMalloc(&d_rays_o,        NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d,        NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d_inv,    NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_nears,         NUM_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fars,          NUM_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_occ,           scene.occupancy_grid.size()));
    CUDA_CHECK(cudaMalloc(&d_sparse_morton, NUM_RAYS * MAX_HITS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sparse_ts,     NUM_RAYS * MAX_HITS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num_steps,     NUM_RAYS * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_rays_o,     scene.rays_o.data(),        NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d,     scene.rays_d.data(),        NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d_inv, scene.rays_d_inv.data(),    NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nears,      scene.nears.data(),         NUM_RAYS * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fars,       scene.fars.data(),          NUM_RAYS * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_occ,        scene.occupancy_grid.data(),scene.occupancy_grid.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sparse_morton, 0, NUM_RAYS * MAX_HITS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_sparse_ts,     0, NUM_RAYS * MAX_HITS * sizeof(float)));

    launchMarchRaysDDA(
        NUM_RAYS, d_rays_o, d_rays_d, d_rays_d_inv, d_nears, d_fars,
        d_occ, scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
        d_sparse_morton, d_sparse_ts, d_num_steps, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back & verify stage 1
    std::vector<uint32_t> gpu_num_steps(NUM_RAYS);
    std::vector<uint32_t> gpu_sparse_morton(NUM_RAYS * MAX_HITS);
    std::vector<float>    gpu_sparse_ts(NUM_RAYS * MAX_HITS);

    CUDA_CHECK(cudaMemcpy(gpu_num_steps.data(),     d_num_steps,     NUM_RAYS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sparse_morton.data(), d_sparse_morton, NUM_RAYS * MAX_HITS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sparse_ts.data(),     d_sparse_ts,     NUM_RAYS * MAX_HITS * sizeof(float), cudaMemcpyDeviceToHost));

    uint32_t gpu_total_hits = 0;
    for (uint32_t i = 0; i < NUM_RAYS; ++i) gpu_total_hits += gpu_num_steps[i];
    std::cout << "  GPU total hits: " << gpu_total_hits << "\n";

    verify_uint32("num_steps_per_ray", cpu_num_steps.data(), gpu_num_steps.data(), NUM_RAYS);

    // Compare only the valid entries per ray (first num_steps entries per ray)
    {
        int morton_mismatches = 0, ts_mismatches = 0;
        float max_t_err = 0.f;
        for (uint32_t r = 0; r < NUM_RAYS; ++r) {
            uint32_t count = std::min(cpu_num_steps[r], gpu_num_steps[r]);
            uint32_t base = r * MAX_HITS;
            for (uint32_t j = 0; j < count; ++j) {
                if (cpu_sparse_morton[base + j] != gpu_sparse_morton[base + j])
                    morton_mismatches++;
                float diff = std::fabs(cpu_sparse_ts[base + j] - gpu_sparse_ts[base + j]);
                if (diff > max_t_err) max_t_err = diff;
                if (diff > 1e-5f) ts_mismatches++;
            }
        }
        std::cout << "packed_coords (valid hits): "
                  << (morton_mismatches == 0 ? "SUCCESS" : "FAILED") << " (" << morton_mismatches << " mismatches)\n";
        std::cout << "t_hits (valid hits):        "
                  << (ts_mismatches == 0 ? "SUCCESS" : "FAILED") << " (max_err=" << max_t_err
                  << ", mismatches=" << ts_mismatches << ")\n";
    }

    // ---- Full-pipeline test via processRaysChunk ----
    std::cout << "\n--- Full Pipeline: processRaysChunk ---\n";

    uint32_t* d_ray_offsets;
    uint32_t* d_dense_morton_in;  uint32_t* d_dense_sparse_idx_in;
    uint32_t* d_dense_morton_out; uint32_t* d_dense_sparse_idx_out;
    float*    d_mlp_positions;
    uint32_t* d_ray_indices;
    float*    d_t_sorted;

    uint32_t max_total_hits = NUM_RAYS * MAX_HITS;  // safe upper bound
    CUDA_CHECK(cudaMalloc(&d_ray_offsets,         NUM_RAYS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_morton_in,     max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_sparse_idx_in, max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_morton_out,    max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_sparse_idx_out,max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_mlp_positions,       max_total_hits * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ray_indices,         max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_t_sorted,            max_total_hits * sizeof(float)));

    // Reset sparse buffers before processRaysChunk (it re-runs march internally)
    CUDA_CHECK(cudaMemset(d_sparse_morton, 0, NUM_RAYS * MAX_HITS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_sparse_ts,     0, NUM_RAYS * MAX_HITS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_num_steps,     0, NUM_RAYS * sizeof(uint32_t)));

    processRaysChunk(
        NUM_RAYS,
        d_rays_o, d_rays_d, d_rays_d_inv, d_nears, d_fars,
        d_occ, scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
        d_sparse_morton, d_sparse_ts, d_num_steps,
        d_ray_offsets,
        d_dense_morton_in, d_dense_sparse_idx_in,
        d_dense_morton_out, d_dense_sparse_idx_out,
        d_mlp_positions, d_ray_indices, d_t_sorted, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back GPU results
    CUDA_CHECK(cudaMemcpy(gpu_num_steps.data(),     d_num_steps,     NUM_RAYS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sparse_morton.data(), d_sparse_morton, NUM_RAYS * MAX_HITS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sparse_ts.data(),     d_sparse_ts,     NUM_RAYS * MAX_HITS * sizeof(float), cudaMemcpyDeviceToHost));

    gpu_total_hits = 0;
    for (uint32_t i = 0; i < NUM_RAYS; ++i) gpu_total_hits += gpu_num_steps[i];

    // ---- CPU full pipeline ----
    // Re-run stage 1 to be sure
    std::fill(cpu_sparse_morton.begin(), cpu_sparse_morton.end(), 0u);
    std::fill(cpu_sparse_ts.begin(), cpu_sparse_ts.end(), 0.0f);
    std::fill(cpu_num_steps.begin(), cpu_num_steps.end(), 0u);

    cpu_march_rays_dda(
        NUM_RAYS,
        scene.rays_o.data(), scene.rays_d.data(), scene.rays_d_inv.data(),
        scene.nears.data(), scene.fars.data(),
        scene.occupancy_grid.data(),
        scene.grid_resolution, scene.aabb_min, scene.aabb_max,
        cpu_sparse_morton.data(), cpu_sparse_ts.data(), cpu_num_steps.data());

    cpu_total_hits = 0;
    for (uint32_t i = 0; i < NUM_RAYS; ++i) cpu_total_hits += cpu_num_steps[i];

    // Stage 2 CPU: prefix sum + compaction
    std::vector<uint32_t> cpu_ray_offsets(NUM_RAYS);
    cpu_exclusive_sum(cpu_num_steps.data(), cpu_ray_offsets.data(), NUM_RAYS);

    std::vector<uint32_t> cpu_dense_morton_in(cpu_total_hits);
    std::vector<uint32_t> cpu_dense_sparse_idx_in(cpu_total_hits);
    cpu_compact_hits(NUM_RAYS, cpu_num_steps.data(), cpu_ray_offsets.data(),
                     cpu_sparse_morton.data(),
                     cpu_dense_morton_in.data(), cpu_dense_sparse_idx_in.data());

    // Stage 2 CPU: radix sort
    std::vector<uint32_t> cpu_dense_morton_out(cpu_total_hits);
    std::vector<uint32_t> cpu_dense_sparse_idx_out(cpu_total_hits);
    cpu_sort_by_key(cpu_dense_morton_in.data(), cpu_dense_sparse_idx_in.data(),
                    cpu_dense_morton_out.data(), cpu_dense_sparse_idx_out.data(),
                    cpu_total_hits);

    // Stage 3 CPU: generate MLP positions + ray metadata
    std::vector<float> cpu_mlp_positions(cpu_total_hits * 3);
    std::vector<uint32_t> cpu_ray_indices(cpu_total_hits);
    std::vector<float> cpu_t_sorted(cpu_total_hits);
    cpu_generate_mlp_batch(cpu_total_hits, cpu_dense_sparse_idx_out.data(),
                           cpu_sparse_ts.data(),
                           scene.rays_o.data(), scene.rays_d.data(),
                           cpu_mlp_positions.data(),
                           cpu_ray_indices.data(),
                           cpu_t_sorted.data());

    // Read GPU MLP positions
    std::vector<float> gpu_mlp_positions(gpu_total_hits * 3);
    CUDA_CHECK(cudaMemcpy(gpu_mlp_positions.data(), d_mlp_positions,
                          gpu_total_hits * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  Total hits — CPU: " << cpu_total_hits << " | GPU: " << gpu_total_hits << "\n";

    if (cpu_total_hits == gpu_total_hits && cpu_total_hits > 0) {
        // Verify sorted morton keys
        std::vector<uint32_t> gpu_dense_morton_out_vec(gpu_total_hits);
        CUDA_CHECK(cudaMemcpy(gpu_dense_morton_out_vec.data(), d_dense_morton_out,
                              gpu_total_hits * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        verify_uint32("Sorted morton keys", cpu_dense_morton_out.data(),
                      gpu_dense_morton_out_vec.data(), cpu_total_hits);

        // Verify MLP positions
        verify_float("MLP positions", cpu_mlp_positions.data(),
                     gpu_mlp_positions.data(), cpu_total_hits * 3, 1e-4f);

        // Verify ray_indices
        std::vector<uint32_t> gpu_ray_indices(gpu_total_hits);
        CUDA_CHECK(cudaMemcpy(gpu_ray_indices.data(), d_ray_indices,
                              gpu_total_hits * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        verify_uint32("ray_indices", cpu_ray_indices.data(),
                      gpu_ray_indices.data(), cpu_total_hits);

        // Verify t_sorted
        std::vector<float> gpu_t_sorted(gpu_total_hits);
        CUDA_CHECK(cudaMemcpy(gpu_t_sorted.data(), d_t_sorted,
                              gpu_total_hits * sizeof(float), cudaMemcpyDeviceToHost));
        verify_float("t_sorted", cpu_t_sorted.data(),
                     gpu_t_sorted.data(), cpu_total_hits, 1e-5f);
    } else if (cpu_total_hits != gpu_total_hits) {
        std::cout << "  FAILED: Total hit count mismatch!\n";
    } else {
        std::cout << "  No hits generated (trivial pass).\n";
    }

    // Cleanup
    cudaFree(d_rays_o); cudaFree(d_rays_d); cudaFree(d_rays_d_inv);
    cudaFree(d_nears); cudaFree(d_fars); cudaFree(d_occ);
    cudaFree(d_sparse_morton); cudaFree(d_sparse_ts); cudaFree(d_num_steps);
    cudaFree(d_ray_offsets);
    cudaFree(d_dense_morton_in); cudaFree(d_dense_sparse_idx_in);
    cudaFree(d_dense_morton_out); cudaFree(d_dense_sparse_idx_out);
    cudaFree(d_mlp_positions); cudaFree(d_ray_indices); cudaFree(d_t_sorted);
}

// ============================================================================
//  PERFORMANCE BENCHMARK
// ============================================================================
void testProcessRaysPerformance() {
    std::cout << "\n========================================\n";
    std::cout << "  processRays — Performance Benchmark\n";
    std::cout << "========================================\n";

    // Fixed parameters
    constexpr uint32_t NUM_RAYS       = 65536;   // 64K rays
    constexpr uint32_t GRID_RES       = 128;
    constexpr float    OCCUPANCY      = 0.25f;
    constexpr int      WARMUP_RUNS    = 5;
    constexpr int      BENCHMARK_RUNS = 50;

    auto scene = generateTestScene(NUM_RAYS, GRID_RES, OCCUPANCY);

    std::cout << "\n  Configuration:\n"
              << "    Rays:         " << NUM_RAYS << "\n"
              << "    Grid:         " << GRID_RES << "^3\n"
              << "    Occupancy:    " << (OCCUPANCY * 100.f) << "%\n"
              << "    Warmup runs:  " << WARMUP_RUNS << "\n"
              << "    Timed runs:   " << BENCHMARK_RUNS << "\n\n";

    // ---- Allocate GPU memory ----
    float3*   d_rays_o; float3*   d_rays_d; float3* d_rays_d_inv;
    float*    d_nears;  float*    d_fars;
    uint8_t*  d_occ;
    uint32_t* d_sparse_morton; float* d_sparse_ts; uint32_t* d_num_steps;
    uint32_t* d_ray_offsets;
    uint32_t* d_dense_morton_in;  uint32_t* d_dense_sparse_idx_in;
    uint32_t* d_dense_morton_out; uint32_t* d_dense_sparse_idx_out;
    float*    d_mlp_positions;
    uint32_t* d_ray_indices;
    float*    d_t_sorted;

    uint32_t max_total_hits = NUM_RAYS * MAX_HITS;

    CUDA_CHECK(cudaMalloc(&d_rays_o,         NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d,         NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d_inv,     NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_nears,          NUM_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fars,           NUM_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_occ,            scene.occupancy_grid.size()));
    CUDA_CHECK(cudaMalloc(&d_sparse_morton,  max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sparse_ts,      max_total_hits * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num_steps,      NUM_RAYS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ray_offsets,    NUM_RAYS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_morton_in,     max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_sparse_idx_in, max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_morton_out,    max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dense_sparse_idx_out,max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_mlp_positions,       max_total_hits * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ray_indices,         max_total_hits * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_t_sorted,            max_total_hits * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rays_o,     scene.rays_o.data(),         NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d,     scene.rays_d.data(),         NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d_inv, scene.rays_d_inv.data(),     NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nears,      scene.nears.data(),          NUM_RAYS * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fars,       scene.fars.data(),           NUM_RAYS * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_occ,        scene.occupancy_grid.data(), scene.occupancy_grid.size(),cudaMemcpyHostToDevice));

    // ---- Benchmark: launchMarchRaysDDA (Stage 1 only) ----
    std::cout << "--- Stage 1: launchMarchRaysDDA ---\n";
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            launchMarchRaysDDA(NUM_RAYS, d_rays_o, d_rays_d, d_rays_d_inv,
                               d_nears, d_fars, d_occ,
                               scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
                               d_sparse_morton, d_sparse_ts, d_num_steps, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            launchMarchRaysDDA(NUM_RAYS, d_rays_o, d_rays_d, d_rays_d_inv,
                               d_nears, d_fars, d_occ,
                               scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
                               d_sparse_morton, d_sparse_ts, d_num_steps, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / BENCHMARK_RUNS;

        std::cout << "  Avg time:     " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";
        std::cout << "  Throughput:   " << std::fixed << std::setprecision(2)
                  << (NUM_RAYS / (avg_ms * 1e-3)) / 1e6 << " M rays/sec\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // ---- Benchmark: processRaysChunk (full pipeline) ----
    std::cout << "\n--- Full Pipeline: processRaysChunk ---\n";
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            processRaysChunk(
                NUM_RAYS, d_rays_o, d_rays_d, d_rays_d_inv, d_nears, d_fars,
                d_occ, scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
                d_sparse_morton, d_sparse_ts, d_num_steps,
                d_ray_offsets,
                d_dense_morton_in, d_dense_sparse_idx_in,
                d_dense_morton_out, d_dense_sparse_idx_out,
                d_mlp_positions, d_ray_indices, d_t_sorted, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            processRaysChunk(
                NUM_RAYS, d_rays_o, d_rays_d, d_rays_d_inv, d_nears, d_fars,
                d_occ, scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
                d_sparse_morton, d_sparse_ts, d_num_steps,
                d_ray_offsets,
                d_dense_morton_in, d_dense_sparse_idx_in,
                d_dense_morton_out, d_dense_sparse_idx_out,
                d_mlp_positions, d_ray_indices, d_t_sorted, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / BENCHMARK_RUNS;

        // Get total hits for throughput stat
        std::vector<uint32_t> h_num_steps(NUM_RAYS);
        CUDA_CHECK(cudaMemcpy(h_num_steps.data(), d_num_steps, NUM_RAYS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t total_hits = 0;
        for (uint32_t i = 0; i < NUM_RAYS; ++i) total_hits += h_num_steps[i];

        std::cout << "  Total hits:   " << total_hits << "\n";
        std::cout << "  Avg time:     " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";
        std::cout << "  Throughput:   " << std::fixed << std::setprecision(2)
                  << (NUM_RAYS / (avg_ms * 1e-3)) / 1e6 << " M rays/sec\n";
        std::cout << "  Hit rate:     " << std::fixed << std::setprecision(2)
                  << (total_hits / (avg_ms * 1e-3)) / 1e6 << " M hits/sec\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // ---- Benchmark: processRaysChunkLinear (no morton sort) ----
    std::cout << "\n--- Linear Pipeline: processRaysChunkLinear ---\n";
    {
        uint32_t* d_dense_sparse_idx_linear;
        CUDA_CHECK(cudaMalloc(&d_dense_sparse_idx_linear, max_total_hits * sizeof(uint32_t)));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            processRaysChunkLinear(
                NUM_RAYS, d_rays_o, d_rays_d, d_rays_d_inv, d_nears, d_fars,
                d_occ, scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
                d_sparse_ts, d_num_steps,
                d_ray_offsets,
                d_dense_sparse_idx_linear,
                d_mlp_positions, d_ray_indices, d_t_sorted, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            processRaysChunkLinear(
                NUM_RAYS, d_rays_o, d_rays_d, d_rays_d_inv, d_nears, d_fars,
                d_occ, scene.grid_resolution, scene.aabb_min, scene.aabb_max, /*numCascades*/1, 1,
                d_sparse_ts, d_num_steps,
                d_ray_offsets,
                d_dense_sparse_idx_linear,
                d_mlp_positions, d_ray_indices, d_t_sorted, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / BENCHMARK_RUNS;

        std::vector<uint32_t> h_num_steps(NUM_RAYS);
        CUDA_CHECK(cudaMemcpy(h_num_steps.data(), d_num_steps, NUM_RAYS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t total_hits = 0;
        for (uint32_t i = 0; i < NUM_RAYS; ++i) total_hits += h_num_steps[i];

        std::cout << "  Total hits:   " << total_hits << "\n";
        std::cout << "  Avg time:     " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";
        std::cout << "  Throughput:   " << std::fixed << std::setprecision(2)
                  << (NUM_RAYS / (avg_ms * 1e-3)) / 1e6 << " M rays/sec\n";
        std::cout << "  Hit rate:     " << std::fixed << std::setprecision(2)
                  << (total_hits / (avg_ms * 1e-3)) / 1e6 << " M hits/sec\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(d_dense_sparse_idx_linear);
    }

    // Cleanup
    cudaFree(d_rays_o); cudaFree(d_rays_d); cudaFree(d_rays_d_inv);
    cudaFree(d_nears); cudaFree(d_fars); cudaFree(d_occ);
    cudaFree(d_sparse_morton); cudaFree(d_sparse_ts); cudaFree(d_num_steps);
    cudaFree(d_ray_offsets);
    cudaFree(d_dense_morton_in); cudaFree(d_dense_sparse_idx_in);
    cudaFree(d_dense_morton_out); cudaFree(d_dense_sparse_idx_out);
    cudaFree(d_mlp_positions); cudaFree(d_ray_indices); cudaFree(d_t_sorted);
}

// ============================================================================
//  MAIN
// ============================================================================
int main() {
    std::cout << "============================================\n";
    std::cout << "  processRays.cu — Test & Benchmark Suite\n";
    std::cout << "============================================\n";

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "  Device: " << prop.name << "\n";
    std::cout << "  SM count: " << prop.multiProcessorCount
              << "  Compute: " << prop.major << "." << prop.minor << "\n\n";

    testProcessRaysCorrectness();
    testProcessRaysPerformance();

    std::cout << "\n============================================\n";
    std::cout << "  All tests completed.\n";
    std::cout << "============================================\n";
    return 0;
}
