#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstdint>
#include "../TinyMLP/EmbeddingTable.h"
#include "InstantNerf.h"
#include "DeviceBuffer.h"

#define SPARSE_B 4

struct BakeOptions {
    uint3 voxelGridResolution = make_uint3(512, 512, 512);  // must equal gridResolution * SPARSE_B
    int   viewFeatures        = 4;
    int   numDiffuseDirs      = 16;
    float sigmaThreshold      = -1.0f;  // <0: inherit the teacher's minDensityThreshold; >=0: bake density threshold
    int   queryBatch          = 1 << 20;
    int   bakeDiffuseN        = 32;

    int   fineTuneSteps       = 2000;     // joint-fit steps; 0 = structure + sigma only (census mode)
    float learningRate        = 1e-2f;    // EmbeddingTable row-Adagrad lr (all 7 channels)
    float mlpLearningRate     = 1e-3f;    // Adam lr for the shared deferred MLP g
    int   fitRaysPerStep      = 1 << 13;  // rays marched per joint-fit step
    int   deferredHidden      = 32;
    int   deferredLayers      = 3;
};

// Census stats produced by diagonstic() — no structure is built, just measured.
struct BakeDiagnostics {
    float threshold = 0.0f;                        // sigma/density threshold used
    std::vector<int> perCascadeFilled;             // occupied base cells, one entry per cascade
    long long totalFilled = 0;                     // occupied base cells across all cascades (= numBlocks)
    long long candidateSubVoxels = 0;              // totalFilled * SPARSE_B^3
    long long survivors = 0;                       // surviving vertices, WITH per-voxel boundary duplication (= numVoxels)
    long long uniqueSurvivors = 0;                 // deduped surviving vertices (shared boundary vertices counted once)
    std::vector<unsigned long long> fillHist;      // [SPARSE_B^3 + 1] per-block occupied-sub-voxel histogram
};

class BakedNerf {
public:
    BakedNerf() = default;
    ~BakedNerf();

    BakedNerf(const BakedNerf&) = delete;
    BakedNerf& operator=(const BakedNerf&) = delete;

    void init(const BakeOptions& opts);
    void diagonstic(InstantNerf& teacher);
    void distil(InstantNerf& teacher, cudaStream_t stream);

    // census results from distil: Tier-1 occupied blocks / Tier-2 surviving sub-voxels
    uint32_t numBlocks() const { return m_numBlocks; }
    uint32_t numVoxels() const { return m_numVoxels; }
    const BakeDiagnostics& diagnostics() const { return m_diag; }

    void renderImage(
        const float3* d_rays_o,
        const float3* d_rays_d,
        uint32_t numRays,
        float* d_rgb_out = nullptr,
        cudaStream_t stream = 0
    );

    // diagnostic: teacher radiance at every marched sample x baked sigma weights.
    // Isolates geometry/weight errors (foggy here = sigma pipeline) from appearance
    // errors (crisp here = payload colors / fit are the gap).
    void renderImageTeacherColor(
        InstantNerf& teacher,
        const float3* d_rays_o,
        const float3* d_rays_d,
        uint32_t numRays,
        float* d_rgb_out
    );

    void save(const std::string& file);
    void load(const std::string& file);

private:
    int  K()            const { return m_opts.viewFeatures; }
    int  vecLen()       const { return 4 + m_opts.viewFeatures; }     // sigma(1)+diffuse(3)+feat(K)
    int  deferredInDim()const { return 3 + m_opts.viewFeatures + 16; }
    void allocRenderScratch();
    void fitDeferred(InstantNerf& teacher);   // Phase 2: distil view-dependent color -> features + deferred MLP

    std::vector<float3> m_viewDirs;

    BakeOptions m_opts;

    DeviceBuffer<uint8_t> m_occupancyGrid{0};
    float                 m_bakeThreshold = 0.0f;

    DeviceBuffer<uint32_t> m_voxelMask{0};
    DeviceBuffer<half> m_voxelSigma{0};
    DeviceBuffer<uint32_t> m_blockIdx{0};
    EmbeddingTable* m_voxelDiffuse = nullptr;
    EmbeddingTable* m_voxelFeatures = nullptr;

    uint32_t               m_numBlocks = 0;
    uint32_t               m_numVoxels = 0;
    BakeDiagnostics        m_diag;

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
    DeviceBuffer<float>    d_sh_rays{0};          // per-pixel SH(ray_dir) for deferred shading
    DeviceBuffer<float>    d_T_pix{0};            // per-pixel final transmittance
    size_t                 m_scratchRays = 0;     // capacity (in rays) of the deferred render scratch
    bool m_scratchReady = false;
};
