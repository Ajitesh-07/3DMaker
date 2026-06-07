#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "DeviceBuffer.h"
#include "Timers.h"

#define CUDA_CHECK(call)                                                   \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            fflush(stderr);                                                     \
            throw std::runtime_error(cudaGetErrorString(_e));                   \
        }                                                                       \
    } while (0)

#define MAX_HITS 128

// Forward declaration
class TinyMLPHashGrid;
class TinyMLP;

struct NerfOptions {
    uint3 gridResolution = make_uint3(128, 128, 128);
    float3  aabbMin         = make_float3(-1.0f, -1.0f, -1.0f);
    float3  aabbMax         = make_float3( 1.0f,  1.0f,  1.0f);
    int levelsMipmap = 4;

    int densityHiddenDim = 64;
    int densityNumLayers = 2;

    int colorHiddenDim = 64;
    int colorNumLayers = 2;

    int hashTableSize = 1 << 19;
    int numLevels = 16;
    float growthFactor = 1.3819f;
    int baseResolution = 16;
    int featuresPerLevel = 2;

    int batchSize = 256 * 1024;
    int rayChunkSize = 256 * 1024;

    // Optimization hyperparameters
    float learningRate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float lossScale = 1.0f; // backward kernel uses FP16 shmem — keep at 1.0

    // Background color for volume rendering compositing
    float3 bgColor = make_float3(1.0f, 1.0f, 1.0f); // white for nerf_synthetic
    float minDensityThreshold = 0.01;
    float decayValue = 0.95f;
    float densityBias = 1.0f;

    bool isProfiling = false;
};

struct RenderingBuffers {
    DeviceBuffer<float3> d_rays_d_inv_chunk{0};
    DeviceBuffer<float> d_nears_chunk{0};
    DeviceBuffer<float> d_fars_chunk{0};
    DeviceBuffer<uint32_t> d_block_sums{0};
    DeviceBuffer<uint32_t> d_active_rays_count{0};
    DeviceBuffer<float> d_sparse_ts{0};
    DeviceBuffer<uint32_t> d_num_steps{0};
    DeviceBuffer<uint32_t> d_ray_offsets{0};
    DeviceBuffer<uint32_t> d_dense_sparse_indices{0};
    DeviceBuffer<float> d_mlp_positions{0};
    DeviceBuffer<uint32_t> d_ray_indices{0};
    DeviceBuffer<float> d_t_sorted{0};

    DeviceBuffer<float> d_density_out{0};
    DeviceBuffer<half> d_color_input{0};
    DeviceBuffer<float> d_color_output{0};
    DeviceBuffer<float> d_density_sigma{0};
    DeviceBuffer<float> d_rgb_output{0};
    DeviceBuffer<float> d_render_rgb_chunk{0};
    DeviceBuffer<float> d_render_depth_chunk{0};
    DeviceBuffer<float> d_phi_chunk{0};

    DeviceBuffer<uint32_t> d_out_count{0};
    DeviceBuffer<float> d_batch_position{0};
    DeviceBuffer<float3> d_batch_direction{0};
    DeviceBuffer<uint32_t> d_active_ray_indices{0};

    DeviceBuffer<float> d_current_t{0};
    DeviceBuffer<float> d_current_rgb{0};
    DeviceBuffer<half> d_custom_color_grad{0};
    DeviceBuffer<half> d_custom_density_grad{0};
    DeviceBuffer<float> d_tmpsigma{0};
    DeviceBuffer<half> d_color_dx_out{0};

    DeviceBuffer<float> d_occupancy_samples{0};
    DeviceBuffer<float> d_tmp_grid{0};
    DeviceBuffer<float> d_sum{0};

    DeviceBuffer<int> d_activeCellIndices{0};
    DeviceBuffer<int> d_numActiveCells{0};

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    std::vector<uint32_t> h_ray_offsets;
};

class GPUStats {
public:
    MetricTracker processRaysTime;
    MetricTracker inferenceDensityFwd;
    MetricTracker inferenceGatherSH;
    MetricTracker inferenceColorFwd;
    MetricTracker volumeRendering;
    MetricTracker fillFloatKernel;
    MetricTracker trainGatherTime;
    MetricTracker trainZeroGrad;
    MetricTracker trainDensityFwd;
    MetricTracker trainComputeSH;
    MetricTracker trainColorFwd;
    MetricTracker trainColorGrad;
    MetricTracker trainColorBwd;
    MetricTracker trainDensityGrad;
    MetricTracker trainDensityBwd;
    MetricTracker trainColorOpt;
    MetricTracker trainDensityOpt;

    MetricTracker earlyOccupancySampleTime;
    MetricTracker earlyOccupancyDensityFwd;
    MetricTracker earlyOccupancyUpdateTmpGrid;
    MetricTracker earlyOccupancyUpdateMasterGrid;
    MetricTracker earlyOccupancyComputeSum;
    MetricTracker earlyOccupancyUpdateBitgrid;
    MetricTracker earlyOccupancyUpdateMipmap;

    MetricTracker lateOccupancySampleNonUniformTime;
    MetricTracker lateOccupancySampleUniformTime;
    MetricTracker lateOccupancyDensityFwd;
    MetricTracker lateOccupancyUpdateTmpGrid;
    MetricTracker lateOccupancyUpdateMasterGrid;
    MetricTracker lateOccupancyComputeSum;
    MetricTracker lateOccupancyUpdateBitgrid;
    MetricTracker lateOccupancyUpdateMipmap;

    
    MetricGroup inferenceTime;
    MetricGroup trainTime;
    MetricGroup rayChunkTime;
    MetricGroup lateOccupancyInferenceTime;
    MetricGroup lateOccupancyUpdateTime;
    MetricGroup earlyOccupancyInferenceTime;
    MetricGroup earlyOccupancyUpdateTime;

    std::vector<PendingTimer> pendingTimers;
    int totalEvents = 0;

    void init() {
        inferenceTime.add(inferenceDensityFwd);
        inferenceTime.add(inferenceGatherSH);
        inferenceTime.add(inferenceColorFwd);

        earlyOccupancyInferenceTime.add(earlyOccupancyDensityFwd);
        earlyOccupancyInferenceTime.add(earlyOccupancyUpdateTmpGrid);

        earlyOccupancyUpdateTime.add(earlyOccupancySampleTime);
        earlyOccupancyUpdateTime.add(earlyOccupancyInferenceTime);
        earlyOccupancyUpdateTime.add(earlyOccupancyUpdateMasterGrid);
        earlyOccupancyUpdateTime.add(earlyOccupancyComputeSum);
        earlyOccupancyUpdateTime.add(earlyOccupancyUpdateBitgrid);
        earlyOccupancyUpdateTime.add(earlyOccupancyUpdateMipmap);
        
        lateOccupancyInferenceTime.add(lateOccupancyDensityFwd);
        lateOccupancyInferenceTime.add(lateOccupancyUpdateTmpGrid);

        lateOccupancyUpdateTime.add(lateOccupancySampleUniformTime);
        lateOccupancyUpdateTime.add(lateOccupancySampleNonUniformTime);
        lateOccupancyUpdateTime.add(lateOccupancyInferenceTime);
        lateOccupancyUpdateTime.add(lateOccupancyUpdateMasterGrid);
        lateOccupancyUpdateTime.add(lateOccupancyComputeSum);
        lateOccupancyUpdateTime.add(lateOccupancyUpdateBitgrid);
        lateOccupancyUpdateTime.add(lateOccupancyUpdateMipmap);

        trainTime.add(trainGatherTime);
        trainTime.add(trainDensityFwd);
        trainTime.add(trainComputeSH);
        trainTime.add(trainColorFwd);
        trainTime.add(trainColorGrad);
        trainTime.add(trainColorBwd);
        trainTime.add(trainDensityGrad);
        trainTime.add(trainDensityBwd);
        
        rayChunkTime.add(trainZeroGrad);
        rayChunkTime.add(earlyOccupancyUpdateTime);
        rayChunkTime.add(lateOccupancyUpdateTime);
        rayChunkTime.add(processRaysTime);
        rayChunkTime.add(inferenceTime);
        rayChunkTime.add(volumeRendering);
        rayChunkTime.add(fillFloatKernel);
        rayChunkTime.add(trainTime);
        rayChunkTime.add(trainColorOpt);
        rayChunkTime.add(trainDensityOpt);
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
        inferenceDensityFwd.reset();
        inferenceGatherSH.reset();
        inferenceColorFwd.reset();
        volumeRendering.reset();
        fillFloatKernel.reset();
        trainGatherTime.reset();
        trainZeroGrad.reset();
        trainDensityFwd.reset();
        trainComputeSH.reset();
        trainColorFwd.reset();
        trainColorGrad.reset();
        trainColorBwd.reset();
        trainDensityGrad.reset();
        trainDensityBwd.reset();
        trainColorOpt.reset();
        trainDensityOpt.reset();

        earlyOccupancySampleTime.reset();
        earlyOccupancyDensityFwd.reset();
        earlyOccupancyUpdateTmpGrid.reset();
        earlyOccupancyUpdateMasterGrid.reset();
        earlyOccupancyComputeSum.reset();
        earlyOccupancyUpdateBitgrid.reset();

        lateOccupancySampleNonUniformTime.reset();
        lateOccupancySampleUniformTime.reset();
        lateOccupancyDensityFwd.reset();
        lateOccupancyUpdateTmpGrid.reset();
        lateOccupancyUpdateMasterGrid.reset();
        lateOccupancyComputeSum.reset();
        lateOccupancyUpdateBitgrid.reset();

        pendingTimers.clear();
        totalEvents = 0;
    }
};

class InstantNerf {
public:
    InstantNerf() = default;
    ~InstantNerf();

    InstantNerf(const InstantNerf&) = delete;
    InstantNerf& operator=(const InstantNerf&) = delete;

    void init(const NerfOptions& opts);
    void printStats();
    void resetStats();
    void setLearningRate(float lr) { m_opts.learningRate = lr; }
    void setBgColor(float3 c) { m_opts.bgColor = c; }
    void setProfiling(bool p) { m_opts.isProfiling = p; }

    void trainWithRays(
        const float3* d_rays_o,
        const float3* d_rays_d,
        const float* d_rgb_true,
        uint32_t numRays,
        int& trainStepCount,
        uint32_t* hitCounts,
        float* d_rgb_out = nullptr,
        cudaStream_t stream = 0
    );

    void trainWithRaysHit(
        const float3* d_rays_o,
        const float3* d_rays_d,
        const float* d_rgb_true,
        uint32_t numRays,
        int& trainStepCount,
        uint32_t* hitCounts,
        float* d_rgb_out,
        cudaStream_t stream
    );

    void renderImage(
        const float3* d_rays_o,
        const float3* d_rays_d,
        uint32_t numRays,
        float* d_rgb_out = nullptr,
        cudaStream_t stream = 0
    );

    void save(const std::string& filename);
    void load(const std::string& filename);

private:
    void initRenderBuffers();
    void freeBuffers();
    void earlyOccupancyGridUpdate(cudaStream_t stream = 0);
    void lateOccupancyGridUpdate(cudaStream_t stream = 0);
    void buildMipmaps(cudaStream_t stream, int level0_cells);

    NerfOptions m_opts;
    RenderingBuffers m_render_buffers;
    TinyMLP* m_colorMLP = nullptr;
    TinyMLPHashGrid* m_densityMLP = nullptr;
    GPUStats m_profile_stats;

    DeviceBuffer<uint8_t> d_occupancyGrid{0};
    DeviceBuffer<float> d_masterOccupancyGrid{0}; 

    bool m_traininit = false;
    bool m_renderinit = false;
    int m_trainSteps = 0;

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

// kernels

extern "C" void launchMarchRaysDDA(
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
);

extern "C" void processRaysChunk(
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
);

extern "C" void processRaysChunkLinear(
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
);

extern "C" int processRaysHitLinear(
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
);

extern "C" void launchComputeRealSHDegree3(
    const uint32_t total_hits,
    const float3* d_directions,
    float* d_out_sh,
    cudaStream_t stream = 0
);

extern "C" void launchVolumeRendering(
    const uint32_t num_rays,
    const uint32_t* d_ray_offsets,
    const uint32_t* d_num_steps,
    const float* d_t_sorted,
    const float* d_density_sigma,
    const float* d_rgb_output,
    const float* d_rgb_true,
    float* d_render_rgb,
    float* d_render_depth,
    float* d_phi_out,
    float3 bg_color = make_float3(1.0f, 1.0f, 1.0f),
    cudaStream_t stream = 0
);