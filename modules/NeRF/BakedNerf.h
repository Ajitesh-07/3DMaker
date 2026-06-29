#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstdint>
#include "InstantNerf.h"
#include "DeviceBuffer.h"

#define SPARSE_B 4

struct BakeOptions {
    uint3 voxelGridResolution = make_uint3(512, 512, 512);  // must equal gridResolution * SPARSE_B
    int   viewFeatures        = 4;
    int   numDiffuseDirs      = 16;
    float sigmaThreshold      = -1.0f;  // <0: inherit the teacher's minDensityThreshold; >=0: bake density threshold
    int   queryBatch          = 1 << 20;

    int   fineTuneSteps       = 2000;
    float learningRate        = 1e-2f;
    int   deferredHidden      = 32;
    int   deferredLayers      = 3;
};

class BakedNerf {
public:
    BakedNerf() = default;
    ~BakedNerf();

    BakedNerf(const BakedNerf&) = delete;
    BakedNerf& operator=(const BakedNerf&) = delete;

    void init(const BakeOptions& opts);
    void distil(InstantNerf& teacher);

    void renderImage(
        const float3* d_rays_o,
        const float3* d_rays_d,
        uint32_t numRays,
        float* d_rgb_out = nullptr,
        cudaStream_t stream = 0
    );

    void save(const std::string& file);
    void load(const std::string& file);

private:
    int  K()            const { return m_opts.viewFeatures; }
    int  vecLen()       const { return 4 + m_opts.viewFeatures; }     // sigma(1)+diffuse(3)+feat(K)
    int  deferredInDim()const { return 3 + m_opts.viewFeatures + 16; }
    void allocRenderScratch();

    BakeOptions m_opts;

    DeviceBuffer<uint8_t> m_occupancyGrid{0};
    float                 m_bakeThreshold = 0.0f;

    DeviceBuffer<uint64_t> m_voxelMask{0};
    DeviceBuffer<float>    m_voxelGrid{0};
    uint32_t               m_numBlocks = 0;
    uint32_t               m_numVoxels = 0;

    TinyMLP*            m_deferredMLP = nullptr;
    bool                m_deferredTrained = false;

    NerfOptions         m_teacherOpts;

    DeviceBuffer<float3>   d_rays_d_inv{0};
    DeviceBuffer<float>    d_nears{0}, d_fars{0};
    DeviceBuffer<uint32_t> d_num_steps{0}, d_ray_offsets{0}, d_ray_indices{0};
    DeviceBuffer<uint32_t> d_block_sums{0}, d_active_rays_count{0};
    DeviceBuffer<float>    d_positions{0}, d_t_sorted{0};
    DeviceBuffer<float>    d_sigma{0}, d_diffuse{0}, d_feat{0};
    DeviceBuffer<float>    d_acc_diffuse{0}, d_acc_feat{0}, d_depth{0};
    DeviceBuffer<half>     d_deferred_in{0};
    DeviceBuffer<float>    d_specular{0}, d_final_rgb{0};
    bool m_scratchReady = false;
};
