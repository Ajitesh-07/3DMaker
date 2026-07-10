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
#include "Timers.h"

#define SPARSE_B 4

struct BakeOptions {
    uint3 voxelGridResolution = make_uint3(512, 512, 512);  // must equal gridResolution * SPARSE_B
    int   viewFeatures        = 4;
    int   numDiffuseDirs      = 16;
    float sigmaThreshold      = -1.0f;  // <0: inherit the teacher's minDensityThreshold; >=0: bake density threshold
    int   queryBatch          = 1 << 20;
    int   bakeDiffuseN        = 32;
    int   jointFitBatch       = 32 * 1024;

    int   fineTuneSteps       = 2000;     // joint-fit steps; 0 = structure + sigma only (census mode)
    float learningRate        = 1e-2f;    // EmbeddingTable row-Adagrad lr (all 7 channels)
    float mlpLearningRate     = 1e-3f;    // Adam lr for the shared deferred MLP g
    int   deferredHidden      = 32;
    int   deferredLayers      = 3;

    bool isProfiling = false;
};

struct BakeDiagnostics {
    float threshold = 0.0f;                        // sigma/density threshold used
    std::vector<int> perCascadeFilled;             // occupied base cells, one entry per cascade
    long long totalFilled = 0;                     // occupied base cells across all cascades (= numBlocks)
    long long candidateSubVoxels = 0;              // totalFilled * SPARSE_B^3
    long long survivors = 0;                       // surviving vertices, WITH per-voxel boundary duplication (= numVoxels)
    long long uniqueSurvivors = 0;                 // deduped surviving vertices (shared boundary vertices counted once)
    long long uniqueAll = 0;                        // deduped vertices of occupied voxels, NO sigma filter (= what distil stores)
    std::vector<unsigned long long> fillHist;      // [SPARSE_B^3 + 1] per-block occupied-sub-voxel histogram
};

struct BakedRenderingBuffer {
    DeviceBuffer<int> d_cellSlots{0};

    DeviceBuffer<float3> d_rays_d_inv_chunk{0};
    DeviceBuffer<float> d_nears_chunk{0};
    DeviceBuffer<float> d_fars_chunk{0};
    DeviceBuffer<uint32_t> d_active_rays_count{0};
    DeviceBuffer<uint32_t> d_ray_offsets{0};
    DeviceBuffer<uint32_t> d_ray_indices{0};
    DeviceBuffer<uint32_t> d_num_steps{0};
    DeviceBuffer<uint32_t> d_block_sums{0};
    DeviceBuffer<float>   d_teacher_points_out{0};
    DeviceBuffer<float>  d_student_frac_out{0};
    DeviceBuffer<int>    d_student_rows_out{0};

    DeviceBuffer<float> d_density_out{0};
    DeviceBuffer<half> d_color_input{0};
    DeviceBuffer<float> d_density_sigma{0};
    DeviceBuffer<float> d_rgb_output{0};
    DeviceBuffer<float> d_t_sorted{0};
    DeviceBuffer<half> d_student_sigma{0};
    DeviceBuffer<half> d_student_point_data{0};
    DeviceBuffer<half> d_deferred_backward{0};
    DeviceBuffer<float> d_render_rgb_chunk{0};
    DeviceBuffer<float> d_student_rgb_chunk{0};
    DeviceBuffer<float> d_student_weights{0};
    DeviceBuffer<half> d_phi_chunk{0};
    DeviceBuffer<float> d_ray_color_sum{0};
    DeviceBuffer<float> d_ray_feat_sum{0};
    DeviceBuffer<int> d_corner_rowids{0};
    DeviceBuffer<float> d_diffuse_grads{0};
    DeviceBuffer<float> d_feature_grads{0};
};

class BakedGPUStats {
public:
    MetricTracker processRaysTime;
    MetricTracker densityInference;
    MetricTracker teacherSHGather; 
    MetricTracker colorInference;
    MetricTracker interpolateRender;
    MetricTracker studentSHGather;
    MetricTracker deferredZeroGrad;
    MetricTracker deferredFwd;
    MetricTracker residualLossGrad;
    MetricTracker deferredBwd;
    MetricTracker gradsSum;
    MetricTracker embeddingGrads;
    MetricTracker embeddingTableStep;
    MetricTracker mlpStep;

    MetricTracker renderGetOffsets;

    MetricTracker renderProcessRay;
    MetricTracker renderInterpolateRays;
    MetricTracker renderSHGather;
    MetricTracker renderDeferredInference;
    MetricTracker renderGetResidual;
    MetricTracker renderAsyncCpy;

    MetricGroup renderHotLoop;


    int totalEvents = 0;
    std::vector<PendingTimer> pendingTimers;

    void init() {
        renderHotLoop.add(renderProcessRay);
        renderHotLoop.add(renderInterpolateRays);
        renderHotLoop.add(renderSHGather);
        renderHotLoop.add(renderDeferredInference);
        renderHotLoop.add(renderGetResidual);
        renderHotLoop.add(renderAsyncCpy);
    }

    void resolvePendingTimers() {
        if (pendingTimers.empty()) return;
        totalEvents = pendingTimers.size();
        cudaEventSynchronize(pendingTimers.back().stop);

        for (auto& timer : pendingTimers) {
            float ms = 0;
            cudaEventElapsedTime(&ms, timer.start, timer.stop);
            timer.tracker->update(ms);

            cudaEventDestroy(timer.start);
            cudaEventDestroy(timer.stop);
        }
        pendingTimers.clear();
    }

    void reset() {
        processRaysTime.reset();
        densityInference.reset();
        teacherSHGather.reset();
        colorInference.reset();
        interpolateRender.reset();
        studentSHGather.reset();
        deferredZeroGrad.reset();
        deferredFwd.reset();
        residualLossGrad.reset();
        deferredBwd.reset();
        gradsSum.reset();
        embeddingGrads.reset();
        embeddingTableStep.reset();
        mlpStep.reset();

        renderGetOffsets.reset();
        renderProcessRay.reset();
        renderInterpolateRays.reset();
        renderSHGather.reset();
        renderDeferredInference.reset();
        renderGetResidual.reset();
        renderAsyncCpy.reset();


        pendingTimers.clear();
        totalEvents = 0;
    }
};

class BakedNerf {
public:
    BakedNerf() = default;
    ~BakedNerf();

    BakedNerf(const BakedNerf&) = delete;
    BakedNerf& operator=(const BakedNerf&) = delete;

    void init(const BakeOptions& opts);
    void diagonstic(InstantNerf& teacher);
    void bakeGeometry(InstantNerf& teacher, cudaStream_t stream);
    void jointFit(
        InstantNerf& teacher, 
        const float3* d_rays_o,
        const float3* d_rays_d,
        int numRays,
        int& trainSteps,
        cudaStream_t stream);


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


    void resetStats();
    void printStats();

    void save(const std::string& file);
    void load(const std::string& file);

private:
    int  K()            const { return m_opts.viewFeatures; }
    int  vecLen()       const { return 4 + m_opts.viewFeatures; }     // sigma(1)+diffuse(3)+feat(K)
    int  deferredInDim()const { return 3 + m_opts.viewFeatures + 16 + m_pad; }
    void allocRenderScratch();

    std::vector<float3> m_viewDirs;
    int m_pad;

    BakeOptions m_opts;

    DeviceBuffer<uint8_t> m_occupancyGrid{0};
    float                 m_bakeThreshold = 0.0f;

    DeviceBuffer<uint64_t> m_subVoxelMask{0};
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

    BakedRenderingBuffer m_render_buffers;
    BakedGPUStats m_profile_stats;

    bool m_baked = false;

    template <typename F>
    __forceinline__ void measure(
        cudaStream_t stream, 
        MetricTracker& tracker,
        F&& func
    ) {
    if (!m_opts.isProfiling) {
        func();
        return;
    } else {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
        func();
        cudaEventRecord(stop, stream);

        m_profile_stats.pendingTimers.push_back({start, stop, &tracker});
    }
    }
};
