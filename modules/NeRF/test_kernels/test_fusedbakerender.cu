#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../InstantNerf.h"
#include "../bakedNerfKernelDef.cuh"

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
//  CPU Implementation of bakedRenderRaysFused
// ============================================================================
static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

static float3 lerp_f3(float3 a, float3 b, float t) {
    return make_float3(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t));
}

static int bake_get_cascade_cpu(float3 p, float3 aabbMin, float3 aabbMax, int num_cascades) {
    float3 ext = {aabbMax.x - aabbMin.x, aabbMax.y - aabbMin.y, aabbMax.z - aabbMin.z};
    float max_dist = 0.0f;
    float3 center = {aabbMin.x + ext.x * 0.5f, aabbMin.y + ext.y * 0.5f, aabbMin.z + ext.z * 0.5f};
    max_dist = std::max(max_dist, std::abs(p.x - center.x) / (ext.x * 0.5f));
    max_dist = std::max(max_dist, std::abs(p.y - center.y) / (ext.y * 0.5f));
    max_dist = std::max(max_dist, std::abs(p.z - center.z) / (ext.z * 0.5f));
    
    int cascade = 0;
    while (max_dist > 1.0f && cascade < num_cascades - 1) {
        max_dist *= 0.5f;
        cascade++;
    }
    return cascade;
}

static uint32_t bake_mipmap_offset_cpu(uint3 res, int level) {
    uint32_t offset = 0;
    for (int l = 0; l < level; ++l) {
        uint3 r = {res.x >> l, res.y >> l, res.z >> l};
        offset += r.x * r.y * r.z;
    }
    return offset;
}

static float cpu_half2float(half h) {
    return __half2float(h); // __half2float is supported on host in modern CUDA
}

static void cpu_bakedRenderRaysFused(
    uint32_t num_rays,
    int view_features,
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
    int m_pad
) {
    float3 aabb_extent = {aabbMax.x - aabbMin.x, aabbMax.y - aabbMin.y, aabbMax.z - aabbMin.z};
    
    int G = gridRes.x * gridRes.y * gridRes.z;
    int per = B * B * B;
    int subRes = B - 1;
    uint32_t total_mipmap_cells = bake_mipmap_offset_cpu(gridRes, levelsMipmap);

    int total_features = 3 + view_features + m_pad;

    for (uint32_t i = 0; i < num_rays; ++i) {
        float3 o = rays_o[i];
        float3 d = rays_d[i];
        float3 d_inv = rays_d_inv[i];
        float t_min = nears[i];
        float t_max_ray = fars[i];

        int3 step = {(d.x >= 0.0f) ? 1 : -1, (d.y >= 0.0f) ? 1 : -1, (d.z >= 0.0f) ? 1 : -1};
        float current_t = t_min;
        int current_level = levelsMipmap - 1;

        float T = 1.0f;
        float r_c = 0.0f, g_c = 0.0f, b_c = 0.0f;
        std::vector<float> sf_acc(view_features, 0.0f);

        bool has_pending = false;
    float pending_t_mid = 0.0f;
    float pending_sigma = 0.0f;
    float3 pending_color = {0.0f, 0.0f, 0.0f};
    std::vector<float> pending_features(view_features, 0.0f);

    while (current_t < t_max_ray && T > 1e-4f) {
            float3 cp = {o.x + current_t * d.x, o.y + current_t * d.y, o.z + current_t * d.z};
            int cascade = bake_get_cascade_cpu(cp, aabbMin, aabbMax, numCascades);
            float cs = (float)(1 << cascade);

            float3 caMin = {aabbMin.x * cs, aabbMin.y * cs, aabbMin.z * cs};
            float3 caExt = {aabb_extent.x * cs, aabb_extent.y * cs, aabb_extent.z * cs};
            uint3 res_l = {gridRes.x >> current_level, gridRes.y >> current_level, gridRes.z >> current_level};
            float3 vsz = {caExt.x / res_l.x, caExt.y / res_l.y, caExt.z / res_l.z};
            float3 rp = {(cp.x - caMin.x) / caExt.x, (cp.y - caMin.y) / caExt.y, (cp.z - caMin.z) / caExt.z};

            int3 vi = {
                std::max(0, std::min((int)std::floor(rp.x * res_l.x), (int)res_l.x - 1)),
                std::max(0, std::min((int)std::floor(rp.y * res_l.y), (int)res_l.y - 1)),
                std::max(0, std::min((int)std::floor(rp.z * res_l.z), (int)res_l.z - 1))
            };

            uint32_t lvl_off = bake_mipmap_offset_cpu(gridRes, current_level);
            uint32_t casc_off = cascade * total_mipmap_cells;
            uint32_t flat = casc_off + lvl_off + vi.z * (res_l.x * res_l.y) + vi.y * res_l.x + vi.x;
            bool occ = (occupancy_grid[flat >> 3] >> (flat & 7)) & 1;

            float3 nb = {
                caMin.x + (vi.x + (step.x > 0 ? 1.0f : 0.0f)) * vsz.x,
                caMin.y + (vi.y + (step.y > 0 ? 1.0f : 0.0f)) * vsz.y,
                caMin.z + (vi.z + (step.z > 0 ? 1.0f : 0.0f)) * vsz.z
            };
            
            float tmax_x = (nb.x - o.x) * d_inv.x;
            float tmax_y = (nb.y - o.y) * d_inv.y;
            float tmax_z = (nb.z - o.z) * d_inv.z;
            float next_t = std::min(std::min(tmax_x, tmax_y), tmax_z);

            if (occ) {
                if (current_level > 0) {
                    current_level--;
                    continue;
                }
                int local = vi.z * (gridRes.x * gridRes.y) + vi.y * gridRes.x + vi.x;
                int bi = reverseMap[cascade * G + local];
                if (bi >= 0) {
                    uint32_t subMask = voxelMask27[bi];
                    float3 vmin = {caMin.x + vi.x * vsz.x, caMin.y + vi.y * vsz.y, caMin.z + vi.z * vsz.z};
                    float3 ssz = {vsz.x / subRes, vsz.y / subRes, vsz.z / subRes};
                    float voxel_exit = next_t;

                    float sub_t = std::max(current_t, t_min);
                    int guard = 0;
                    while (sub_t < voxel_exit && T > 1e-4f && guard < 3 * subRes) {
                        guard++;
                        float3 sp = {o.x + sub_t * d.x, o.y + sub_t * d.y, o.z + sub_t * d.z};
                        int si = std::max(0, std::min((int)std::floor((sp.x - vmin.x) / ssz.x), subRes - 1));
                        int sj = std::max(0, std::min((int)std::floor((sp.y - vmin.y) / ssz.y), subRes - 1));
                        int sk = std::max(0, std::min((int)std::floor((sp.z - vmin.z) / ssz.z), subRes - 1));

                        float sbx = vmin.x + (si + (step.x > 0 ? 1.0f : 0.0f)) * ssz.x;
                        float sby = vmin.y + (sj + (step.y > 0 ? 1.0f : 0.0f)) * ssz.y;
                        float sbz = vmin.z + (sk + (step.z > 0 ? 1.0f : 0.0f)) * ssz.z;
                        
                        float sub_next = std::min(std::min((sbx - o.x) * d_inv.x, (sby - o.y) * d_inv.y), (sbz - o.z) * d_inv.z);
                        float seg_exit = std::min(sub_next, voxel_exit);

                        int cubeIdx = si + sj * subRes + sk * subRes * subRes;
                        if ((subMask >> cubeIdx) & 1u) {
                            float t_mid = 0.5f * (sub_t + seg_exit);
                            
                            float3 mp = {o.x + t_mid * d.x, o.y + t_mid * d.y, o.z + t_mid * d.z};
                            
                            float fx = std::min(std::max((mp.x - vmin.x) / ssz.x - (float)si, 0.0f), 1.0f);
                            float fy = std::min(std::max((mp.y - vmin.y) / ssz.y - (float)sj, 0.0f), 1.0f);
                            float fz = std::min(std::max((mp.z - vmin.z) / ssz.z - (float)sk, 0.0f), 1.0f);

                            int row[8];
                            for (int c = 0; c < 8; ++c) {
                                int di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
                                int vsub = (si + di) + (sj + dj) * B + (sk + dk) * B * B;
                                row[c] = (int)vertexRows[bi * per + vsub];
                            }

                            float s[8];
                            for(int c=0; c<8; ++c) s[c] = row[c] == -1 ? 0.0f : cpu_half2float(sigma_master[row[c]]);
                            
                            float s00 = s[0] + fx * (s[1] - s[0]);
                            float s10 = s[2] + fx * (s[3] - s[2]);
                            float s01 = s[4] + fx * (s[5] - s[4]);
                            float s11 = s[6] + fx * (s[7] - s[6]);
                            float s0  = s00 + fy * (s10 - s00);
                            float s1  = s01 + fy * (s11 - s01);
                            float sigma = s0 + fz * (s1 - s0);
                            float s_sigma_safe = std::max(0.0f, sigma);

                            if (has_pending) {
                                float delta_t = t_mid - pending_t_mid;
                                float alpha = 1.0f - std::exp(-pending_sigma * delta_t);
                                float weight = alpha * T;

                                r_c += weight * pending_color.x;
                                g_c += weight * pending_color.y;
                                b_c += weight * pending_color.z;

                                for (int j = 0; j < view_features; j++) {
                                    sf_acc[j] += weight * pending_features[j];
                                }
                                
                                T *= (1.0f - alpha);
                                if (T <= 1e-4f) break;
                            }

                            float3 c[8];
                            for(int k=0; k<8; ++k) c[k] = row[k] == -1 ? make_float3(0,0,0) : diffuse_master[row[k]];
                            
                            float3 c00 = lerp_f3(c[0], c[1], fx);
                            float3 c10 = lerp_f3(c[2], c[3], fx);
                            float3 c01 = lerp_f3(c[4], c[5], fx);
                            float3 c11 = lerp_f3(c[6], c[7], fx);
                            float3 cx0 = lerp_f3(c00, c10, fy);
                            float3 cx1 = lerp_f3(c01, c11, fy);
                            
                            pending_color = lerp_f3(cx0, cx1, fz);
                            pending_t_mid = t_mid;
                            pending_sigma = s_sigma_safe;

                            float w[8];
                            w[0] = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
                            w[1] = fx * (1.0f - fy) * (1.0f - fz);
                            w[2] = (1.0f - fx) * fy * (1.0f - fz);
                            w[3] = fx * fy * (1.0f - fz);
                            w[4] = (1.0f - fx) * (1.0f - fy) * fz;
                            w[5] = fx * (1.0f - fy) * fz;
                            w[6] = (1.0f - fx) * fy * fz;
                            w[7] = fx * fy * fz;

                            std::fill(pending_features.begin(), pending_features.end(), 0.0f);
                            for (int k = 0; k < 8; ++k) {
                                if (row[k] != -1) {
                                    float wk = w[k];
                                    int base_idx = row[k] * view_features;
                                    for (int j = 0; j < view_features; j++) {
                                        pending_features[j] += wk * features_master[base_idx + j];
                                    }
                                }
                            }

                            has_pending = true;
                        }
                        sub_t = std::max(sub_t + 1e-6f, sub_next + 1e-6f);
                    }
                }
            }
            current_t = std::max(current_t + 1e-5f, next_t + 1e-6f);
            current_level = levelsMipmap - 1;
        }

        if (has_pending && T > 1e-4f) {
            float delta_t = 1e-3f;
            float alpha = 1.0f - std::exp(-pending_sigma * delta_t);
            float weight = alpha * T;

            r_c += weight * pending_color.x;
            g_c += weight * pending_color.y;
            b_c += weight * pending_color.z;

            for (int j = 0; j < view_features; j++) {
                sf_acc[j] += weight * pending_features[j];
            }
            T *= (1.0f - alpha);
        }

        r_c += T * bg_color.x;
        g_c += T * bg_color.y;
        b_c += T * bg_color.z;

        int sh_offset = (i * (total_features + 16)) + total_features;
        int data_offset = i * (total_features + 16);

        student_point_data[data_offset + 0] = __float2half(r_c);
        student_point_data[data_offset + 1] = __float2half(g_c);
        student_point_data[data_offset + 2] = __float2half(b_c);

        for (int j = 0; j < view_features; j++) {
            student_point_data[data_offset + 3 + j] = __float2half(sf_acc[j]);
        }
        for (int j = 0; j < m_pad; j++) {
            student_point_data[data_offset + 3 + view_features + j] = __float2half(0.0f);
        }

        float x = d.x;
        float y = d.y;
        float z = d.z;

        float x2 = x * x;
        float y2 = y * y;
        float z2 = z * z;

        float sh[16];
        sh[0]  =  0.28209479f;
        sh[1]  = -0.48860251f * y;
        sh[2]  =  0.48860251f * z;
        sh[3]  = -0.48860251f * x;
        sh[4]  =  1.09254843f * x * y;
        sh[5]  = -1.09254843f * y * z;
        sh[6]  =  0.31539156f * (3.0f * z2 - 1.0f);
        sh[7]  = -1.09254843f * x * z;
        sh[8]  =  0.54627421f * (x2 - y2);
        sh[9]  = -0.59004359f * y * (3.0f * x2 - y2);
        sh[10] =  2.89061144f * x * y * z;
        sh[11] = -0.45704580f * y * (5.0f * z2 - 1.0f);
        sh[12] =  0.37317633f * z * (5.0f * z2 - 3.0f);
        sh[13] = -0.45704580f * x * (5.0f * z2 - 1.0f);
        sh[14] =  1.44530572f * z * (x2 - y2);
        sh[15] = -0.59004359f * x * (x2 - 3.0f * y2);

        for (int j = 0; j < 16; j++) {
            student_point_data[sh_offset + j] = __float2half(sh[j]);
        }
    }
}

// ============================================================================
//  Test Setup and Verification
// ============================================================================
bool verify_half_array(const char* label, const half* cpu, const half* gpu, uint32_t count, float tol = 1e-3f, int max_prints = 10) {
    float max_err = 0.f;
    int mismatches = 0;
    for (uint32_t i = 0; i < count; ++i) {
        float f_cpu = cpu_half2float(cpu[i]);
        float f_gpu = cpu_half2float(gpu[i]);
        float diff = std::fabs(f_cpu - f_gpu);
        
        // Handle NaN/Inf
        if (std::isnan(f_cpu) || std::isnan(f_gpu) || std::isinf(f_cpu) || std::isinf(f_gpu)) {
             diff = (f_cpu == f_gpu) ? 0.0f : 999.0f;
        }

        if (diff > max_err) max_err = diff;
        if (diff > tol) {
            if (mismatches < max_prints)
                std::cout << "  " << label << " mismatch [" << i << "]: CPU=" << f_cpu
                          << " GPU=" << f_gpu << " diff=" << diff << "\n";
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

int main() {
    std::cout << "Comparing CPU vs GPU Fused Render Kernel...\n" << std::endl;

    constexpr uint32_t NUM_RAYS = 1024;
    constexpr uint32_t GRID_RES = 32;
    constexpr int NUM_CASCADES = 1;
    constexpr int LEVELS_MIPMAP = 1;
    constexpr int B = 3; // SPARSE_B
    constexpr int VIEW_FEATURES = 2; // e.g. 2
    constexpr int M_PAD = 3; // (8 - ((3 + 2) % 8)) % 8 = 3
    constexpr int TOTAL_FEATURES = 3 + VIEW_FEATURES + M_PAD + 16;
    constexpr int TOTAL_VOXELS = GRID_RES * GRID_RES * GRID_RES;
    constexpr int PER_BLOCK = B * B * B;

    uint3 gridRes = {GRID_RES, GRID_RES, GRID_RES};
    float3 aabbMin = {-1.0f, -1.0f, -1.0f};
    float3 aabbMax = { 1.0f,  1.0f,  1.0f};
    float3 bg_color = {1.0f, 1.0f, 1.0f}; // white background

    // CPU allocations
    std::vector<float3> h_rays_o(NUM_RAYS);
    std::vector<float3> h_rays_d(NUM_RAYS);
    std::vector<float3> h_rays_d_inv(NUM_RAYS);
    std::vector<float>  h_nears(NUM_RAYS, 0.0f);
    std::vector<float>  h_fars(NUM_RAYS, 10.0f);
    
    std::vector<uint8_t> h_occ((TOTAL_VOXELS + 7) / 8, 0);
    std::vector<int> h_reverseMap(TOTAL_VOXELS, -1);
    
    int num_active_blocks = 0;
    
    // Fill random rays
    for (uint32_t i = 0; i < NUM_RAYS; ++i) {
        h_rays_o[i] = make_float3(((float)rand()/RAND_MAX - 0.5f)*3.0f, ((float)rand()/RAND_MAX - 0.5f)*3.0f, -2.0f);
        float3 dir = {((float)rand()/RAND_MAX - 0.5f)*0.2f, ((float)rand()/RAND_MAX - 0.5f)*0.2f, 1.0f};
        float len = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
        dir.x /= len; dir.y /= len; dir.z /= len;
        h_rays_d[i] = dir;
        h_rays_d_inv[i] = make_float3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);
        h_nears[i] = 0.01f;
        h_fars[i] = 5.0f;
    }

    // Fill grid (20% occupancy)
    for (int v = 0; v < TOTAL_VOXELS; ++v) {
        if ((float)rand() / RAND_MAX < 0.2f) {
            h_occ[v >> 3] |= (1u << (v & 7));
            h_reverseMap[v] = num_active_blocks++;
        }
    }

    std::vector<uint32_t> h_voxelMask27(num_active_blocks);
    std::vector<uint32_t> h_vertexRows(num_active_blocks * PER_BLOCK);
    
    int num_vertices = 0;
    for (int b = 0; b < num_active_blocks; ++b) {
        // Random 27-bit mask
        h_voxelMask27[b] = rand() & ((1<<27) - 1);
        for (int p = 0; p < PER_BLOCK; ++p) {
            if ((float)rand() / RAND_MAX < 0.8f) {
                h_vertexRows[b * PER_BLOCK + p] = num_vertices++;
            } else {
                h_vertexRows[b * PER_BLOCK + p] = -1;
            }
        }
    }

    std::vector<half>   h_sigma_master(num_vertices);
    std::vector<float3> h_diffuse_master(num_vertices);
    std::vector<float>  h_features_master(num_vertices * VIEW_FEATURES);

    for (int v = 0; v < num_vertices; ++v) {
        h_sigma_master[v] = __float2half(((float)rand() / RAND_MAX) * 50.0f); // Random density up to 50
        h_diffuse_master[v] = make_float3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        for (int f = 0; f < VIEW_FEATURES; ++f) {
            h_features_master[v * VIEW_FEATURES + f] = (float)rand() / RAND_MAX - 0.5f;
        }
    }

    std::vector<half> h_student_point_data_cpu(NUM_RAYS * TOTAL_FEATURES, __float2half(0.0f));
    std::vector<half> h_student_point_data_gpu(NUM_RAYS * TOTAL_FEATURES, __float2half(0.0f));

    // Run CPU Oracle
    std::cout << "Running CPU oracle..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_bakedRenderRaysFused(
        NUM_RAYS, VIEW_FEATURES,
        h_rays_o.data(), h_rays_d.data(), h_rays_d_inv.data(),
        h_nears.data(), h_fars.data(),
        h_occ.data(), h_voxelMask27.data(), h_vertexRows.data(), h_reverseMap.data(),
        gridRes, aabbMin, aabbMax, NUM_CASCADES, LEVELS_MIPMAP, B,
        h_sigma_master.data(), h_diffuse_master.data(), h_features_master.data(),
        bg_color, h_student_point_data_cpu.data(), M_PAD
    );
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms\n";

    // Allocate GPU
    float3 *d_rays_o, *d_rays_d, *d_rays_d_inv;
    float *d_nears, *d_fars;
    uint8_t *d_occ;
    uint32_t *d_voxelMask27, *d_vertexRows;
    int *d_reverseMap;
    half *d_sigma_master;
    float3 *d_diffuse_master;
    float *d_features_master;
    half *d_student_point_data_gpu;

    CUDA_CHECK(cudaMalloc(&d_rays_o, NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d, NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rays_d_inv, NUM_RAYS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_nears, NUM_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fars, NUM_RAYS * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_occ, h_occ.size()));
    CUDA_CHECK(cudaMalloc(&d_reverseMap, h_reverseMap.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_voxelMask27, h_voxelMask27.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vertexRows, h_vertexRows.size() * sizeof(uint32_t)));
    
    CUDA_CHECK(cudaMalloc(&d_sigma_master, h_sigma_master.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_diffuse_master, h_diffuse_master.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_features_master, h_features_master.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_student_point_data_gpu, h_student_point_data_gpu.size() * sizeof(half)));

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_rays_o, h_rays_o.data(), NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d, h_rays_d.data(), NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rays_d_inv, h_rays_d_inv.data(), NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nears, h_nears.data(), NUM_RAYS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fars, h_fars.data(), NUM_RAYS * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_occ, h_occ.data(), h_occ.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reverseMap, h_reverseMap.data(), h_reverseMap.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_voxelMask27, h_voxelMask27.data(), h_voxelMask27.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vertexRows, h_vertexRows.data(), h_vertexRows.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_sigma_master, h_sigma_master.data(), h_sigma_master.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diffuse_master, h_diffuse_master.data(), h_diffuse_master.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_features_master, h_features_master.data(), h_features_master.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Run GPU 
    std::cout << "Running GPU kernel..." << std::endl;
    launch_bakedRenderRaysFused(
        VIEW_FEATURES,
        NUM_RAYS,
        d_rays_o, d_rays_d, d_rays_d_inv,
        d_nears, d_fars,
        d_occ, d_voxelMask27, d_vertexRows, d_reverseMap,
        gridRes, aabbMin, aabbMax, NUM_CASCADES, LEVELS_MIPMAP, B,
        d_sigma_master, d_diffuse_master, d_features_master,
        bg_color, d_student_point_data_gpu, M_PAD, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_student_point_data_gpu.data(), d_student_point_data_gpu, h_student_point_data_gpu.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare
    std::cout << "\n======================================================\n";
    verify_half_array("student_point_data (rgb + features + SH)", h_student_point_data_cpu.data(), h_student_point_data_gpu.data(), h_student_point_data_cpu.size(), 5e-3f, 20);
    std::cout << "======================================================\n";

    // Free
    cudaFree(d_rays_o); cudaFree(d_rays_d); cudaFree(d_rays_d_inv);
    cudaFree(d_nears); cudaFree(d_fars);
    cudaFree(d_occ); cudaFree(d_reverseMap); cudaFree(d_voxelMask27); cudaFree(d_vertexRows);
    cudaFree(d_sigma_master); cudaFree(d_diffuse_master); cudaFree(d_features_master);
    cudaFree(d_student_point_data_gpu);

    return 0;
}
