template <int VIEW_FEATURES>
__global__ void bakedRenderRaysFused(
    uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3* __restrict__ rays_d_inv,
    const float* __restrict__ nears,
    const float* __restrict__ fars,
    const uint8_t*  __restrict__ occupancy_grid,
    const uint32_t* __restrict__ voxelMask27,
    const uint32_t* __restrict__ vertexRows,
    const int*      __restrict__ reverseMap,
    uint3  gridRes, float3 aabbMin, float3 aabbMax,
    int numCascades, int levelsMipmap, int B,
    const half* __restrict__ sigma_master,
    const float3* __restrict__ diffuse_master,
    const float* __restrict__ features_master,
    float3 bg_color,
    half* __restrict__ student_point_data,
    int m_pad
) {
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
    const int per     = B * B * B;
    const int subRes  = B - 1;
    const uint32_t total_mipmap_cells = bake_mipmap_offset(gridRes, levelsMipmap);

    float current_t = t_min;
    int current_level = levelsMipmap - 1;

    float T = 1.0f;
    float r_c = 0.0f, g_c = 0.0f, b_c = 0.0f;
    float sf_acc[VIEW_FEATURES];
    #pragma unroll
    for (int j = 0; j < VIEW_FEATURES; j++) sf_acc[j] = 0.0f;

    while (current_t < t_max_ray && T > 1e-4f) {
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

        float3 nb = make_float3(
            caMin.x + (vi.x + (step.x > 0 ? 1.0f : 0.0f)) * vsz.x,
            caMin.y + (vi.y + (step.y > 0 ? 1.0f : 0.0f)) * vsz.y,
            caMin.z + (vi.z + (step.z > 0 ? 1.0f : 0.0f)) * vsz.z);
        float3 tmax = make_float3((nb.x - o.x) * d_inv.x, (nb.y - o.y) * d_inv.y, (nb.z - o.z) * d_inv.z);
        float next_t = fminf(fminf(tmax.x, tmax.y), tmax.z);

        if (occ) {
            if (current_level > 0) {
                current_level--;
                continue;
            }
            int local = vi.z * (gridRes.x * gridRes.y) + vi.y * gridRes.x + vi.x;
            int bi = reverseMap[cascade * G + local];
            if (bi >= 0) {
                uint32_t subMask = voxelMask27[bi];
                float3 vmin = make_float3(caMin.x + vi.x * vsz.x, caMin.y + vi.y * vsz.y, caMin.z + vi.z * vsz.z);
                float3 ssz  = make_float3(vsz.x / subRes, vsz.y / subRes, vsz.z / subRes);
                float voxel_exit = next_t;

                float sub_t = fmaxf(current_t, t_min);
                int guard = 0;
                while (sub_t < voxel_exit && T > 1e-4f && guard < 3 * subRes) {
                    guard++;
                    float3 sp = make_float3(o.x + sub_t * d.x, o.y + sub_t * d.y, o.z + sub_t * d.z);
                    int si = max(0, min((int)floorf((sp.x - vmin.x) / ssz.x), subRes - 1));
                    int sj = max(0, min((int)floorf((sp.y - vmin.y) / ssz.y), subRes - 1));
                    int sk = max(0, min((int)floorf((sp.z - vmin.z) / ssz.z), subRes - 1));

                    float sbx = vmin.x + (si + (step.x > 0 ? 1.0f : 0.0f)) * ssz.x;
                    float sby = vmin.y + (sj + (step.y > 0 ? 1.0f : 0.0f)) * ssz.y;
                    float sbz = vmin.z + (sk + (step.z > 0 ? 1.0f : 0.0f)) * ssz.z;
                    float sub_next = fminf(fminf((sbx - o.x) * d_inv.x, (sby - o.y) * d_inv.y), (sbz - o.z) * d_inv.z);
                    float seg_exit = fminf(sub_next, voxel_exit);

                    int cubeIdx = si + sj * subRes + sk * subRes * subRes;
                    if ((subMask >> cubeIdx) & 1u) {
                        float t_mid = 0.5f * (sub_t + seg_exit);
                        float dt = seg_exit - sub_t;
                        
                        float3 mp = make_float3(o.x + t_mid * d.x, o.y + t_mid * d.y, o.z + t_mid * d.z);
                        
                        float fx = fminf(fmaxf((mp.x - vmin.x) / ssz.x - (float)si, 0.0f), 1.0f);
                        float fy = fminf(fmaxf((mp.y - vmin.y) / ssz.y - (float)sj, 0.0f), 1.0f);
                        float fz = fminf(fmaxf((mp.z - vmin.z) / ssz.z - (float)sk, 0.0f), 1.0f);

                        int row[8];
                        #pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            int di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
                            int vsub = (si + di) + (sj + dj) * B + (sk + dk) * B * B;
                            row[c] = (int)vertexRows[bi * per + vsub];
                        }

                        float s[8];
                        #pragma unroll
                        for(int c=0; c<8; ++c) s[c] = row[c] == -1 ? 0.0f : __half2float(sigma_master[row[c]]);
                        
                        float s00 = s[0] + fx * (s[1] - s[0]);
                        float s10 = s[2] + fx * (s[3] - s[2]);
                        float s01 = s[4] + fx * (s[5] - s[4]);
                        float s11 = s[6] + fx * (s[7] - s[6]);
                        float s0  = s00 + fy * (s10 - s00);
                        float s1  = s01 + fy * (s11 - s01);
                        float sigma = s0 + fz * (s1 - s0);

                        if (sigma > 0.0f) {
                            float alpha = 1.0f - expf(-sigma * dt);
                            float weight = alpha * T;

                            float3 c[8];
                            #pragma unroll
                            for(int k=0; k<8; ++k) c[k] = row[k] == -1 ? make_float3(0,0,0) : diffuse_master[row[k]];
                            
                            float3 c00 = lerp_float3(c[0], c[1], fx);
                            float3 c10 = lerp_float3(c[2], c[3], fx);
                            float3 c01 = lerp_float3(c[4], c[5], fx);
                            float3 c11 = lerp_float3(c[6], c[7], fx);
                            float3 cx0 = lerp_float3(c00, c10, fy);
                            float3 cx1 = lerp_float3(c01, c11, fy);
                            float3 color = lerp_float3(cx0, cx1, fz);

                            r_c += weight * color.x;
                            g_c += weight * color.y;
                            b_c += weight * color.z;

                            float f[8];
                            for (int j = 0; j < VIEW_FEATURES; j++) {
                                #pragma unroll
                                for(int k=0; k<8; ++k) f[k] = row[k] == -1 ? 0.0f : features_master[row[k] * VIEW_FEATURES + j];
                                
                                float f00 = f[0] + fx * (f[1] - f[0]);
                                float f10 = f[2] + fx * (f[3] - f[2]);
                                float f01 = f[4] + fx * (f[5] - f[4]);
                                float f11 = f[6] + fx * (f[7] - f[6]);
                                float fx0 = f00 + fy * (f10 - f00);
                                float fx1 = f01 + fy * (f11 - f01);
                                float feat = fx0 + fz * (fx1 - fx0);
                                sf_acc[j] += weight * feat;
                            }
                            
                            T *= (1.0f - alpha);
                        }
                    }
                    sub_t = fmaxf(sub_t + 1e-6f, sub_next + 1e-6f);
                }
            }
        }
        current_t = fmaxf(current_t + 1e-5f, next_t + 1e-6f);
        current_level = levelsMipmap - 1;
    }

    r_c += T * bg_color.x;
    g_c += T * bg_color.y;
    b_c += T * bg_color.z;

    int total_features = 3 + VIEW_FEATURES + m_pad;
    int sh_offset = (i * (total_features + 16)) + total_features;
    int data_offset = i * (total_features + 16);

    student_point_data[data_offset + 0] = __float2half(r_c);
    student_point_data[data_offset + 1] = __float2half(g_c);
    student_point_data[data_offset + 2] = __float2half(b_c);

    for (int j = 0; j < VIEW_FEATURES; j++) {
        student_point_data[data_offset + 3 + j] = __float2half(sf_acc[j]);
    }
    for (int j = 0; j < m_pad; j++) {
        student_point_data[data_offset + 3 + VIEW_FEATURES + j] = __float2half(0.0f);
    }

    // SH evaluation (optimized)
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

    float4* dest = reinterpret_cast<float4*>(&student_point_data[sh_offset]);
    store_float4(&dest[0], packed0_7.x, packed0_7.y, packed0_7.z, packed0_7.w);
    store_float4(&dest[1], packed8_15.x, packed8_15.y, packed8_15.z, packed8_15.w);
}

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
    cudaStream_t stream
) {
    if (num_rays == 0) return;
    int BS = 256;
    int gs = (num_rays + BS - 1) / BS;

    switch (view_features) {
        case 6: bakedRenderRaysFused<6><<<gs, BS, 0, stream>>>(num_rays, rays_o, rays_d, rays_d_inv, nears, fars, occupancy_grid, voxelMask27, vertexRows, reverseMap, gridRes, aabbMin, aabbMax, numCascades, levelsMipmap, B, sigma_master, diffuse_master, features_master, bg_color, student_point_data, m_pad); break;
        case 5: bakedRenderRaysFused<5><<<gs, BS, 0, stream>>>(num_rays, rays_o, rays_d, rays_d_inv, nears, fars, occupancy_grid, voxelMask27, vertexRows, reverseMap, gridRes, aabbMin, aabbMax, numCascades, levelsMipmap, B, sigma_master, diffuse_master, features_master, bg_color, student_point_data, m_pad); break;
        case 4: bakedRenderRaysFused<4><<<gs, BS, 0, stream>>>(num_rays, rays_o, rays_d, rays_d_inv, nears, fars, occupancy_grid, voxelMask27, vertexRows, reverseMap, gridRes, aabbMin, aabbMax, numCascades, levelsMipmap, B, sigma_master, diffuse_master, features_master, bg_color, student_point_data, m_pad); break;
        case 3: bakedRenderRaysFused<3><<<gs, BS, 0, stream>>>(num_rays, rays_o, rays_d, rays_d_inv, nears, fars, occupancy_grid, voxelMask27, vertexRows, reverseMap, gridRes, aabbMin, aabbMax, numCascades, levelsMipmap, B, sigma_master, diffuse_master, features_master, bg_color, student_point_data, m_pad); break;
        case 2: bakedRenderRaysFused<2><<<gs, BS, 0, stream>>>(num_rays, rays_o, rays_d, rays_d_inv, nears, fars, occupancy_grid, voxelMask27, vertexRows, reverseMap, gridRes, aabbMin, aabbMax, numCascades, levelsMipmap, B, sigma_master, diffuse_master, features_master, bg_color, student_point_data, m_pad); break;
        case 1: bakedRenderRaysFused<1><<<gs, BS, 0, stream>>>(num_rays, rays_o, rays_d, rays_d_inv, nears, fars, occupancy_grid, voxelMask27, vertexRows, reverseMap, gridRes, aabbMin, aabbMax, numCascades, levelsMipmap, B, sigma_master, diffuse_master, features_master, bg_color, student_point_data, m_pad); break;
        default: printf("Error: Unsupported view_features %d in launch_bakedRenderRaysFused\n", view_features); break;
    }
}
