#include "InstantNerf.h"
#include "otherKernels.cu"
#include "../TinyMLP/TinyMLPHashGrid.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <chrono>

// Total cells of one cascade's pyramid (level-0 + every mip level) = the per-cascade
// stride 'S' used by the bitgrid layout the marcher reads (cascade*S + level_offset + voxel).
static inline int pyramidCellsPerCascade(uint3 res, int levels) {
    int s = 0;
    for (int l = 0; l < levels; ++l) {
        s += (int)(res.x * res.y * res.z);
        res = make_uint3(res.x / 2, res.y / 2, res.z / 2);
    }
    return s;
}

std::vector<uint32_t> get_chunk_boundaries_cpu(
    std::vector<uint32_t>& h_ray_offsets, 
    uint32_t* d_ray_offsets, 
    uint32_t num_rays, 
    uint32_t totalHits,
    uint32_t allowedSize, 
    cudaStream_t stream
) {
    std::vector<uint32_t> boundaries;
    if (num_rays == 0) {
        boundaries.push_back(0);
        return boundaries;
    }

    cudaError_t err = cudaMemcpyAsync(h_ray_offsets.data(), d_ray_offsets, num_rays * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR in h_ray_offsets memcpy: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);

    uint32_t last_val = 0;
    cudaMemcpy(&last_val, d_ray_offsets + num_rays - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t current_ray_idx = 0;
    
    while (current_ray_idx < num_rays) {
        if (totalHits - h_ray_offsets[current_ray_idx] <= allowedSize) {
            boundaries.push_back(num_rays);
            break;
        }

        uint32_t target = h_ray_offsets[current_ray_idx] + allowedSize;
        
        auto it = std::upper_bound(h_ray_offsets.data() + current_ray_idx, h_ray_offsets.data() + num_rays, target);
        uint32_t next_ray_idx = std::distance(h_ray_offsets.data(), it) - 1;

        if (next_ray_idx == current_ray_idx) {
            next_ray_idx++;
        }
        
        boundaries.push_back(next_ray_idx);
        current_ray_idx = next_ray_idx;
    }
    
    return boundaries;
}

InstantNerf::~InstantNerf() {
    if (m_colorMLP) {
        delete m_colorMLP;
        m_colorMLP = nullptr;
    }
    if (m_densityMLP) {
        delete m_densityMLP;
        m_densityMLP = nullptr;
    }

    cudaFree(m_render_buffers.d_temp_storage);
}

void InstantNerf::resetStats() {
    m_profile_stats.reset();
}

void InstantNerf::printStats() {
    if (!m_opts.isProfiling) {
        printf("Profiling is disabled.\n");
        return;
    }

    printf("Total Events: %d\n", m_profile_stats.totalEvents);

    auto printTracker = [](const char* name, const MetricTracker& t, int indent) {
        if (t.count == 0) return;
        for (int i = 0; i < indent; i++) printf("    ");
        printf("%s (Total: %.3f ms, Avg: %.3f ms, Min: %.3f, Max: %.3f, Std: %.3f, Count: %llu)\n", name, t.totalMs, t.getAverage(), t.getMin(), t.getMax(), t.getStdDev(), (unsigned long long)t.count);
    };

    auto printGroup = [&](const char* name, const MetricGroup& g, int indent) {
        uint64_t count = g.getCount();
        if (count == 0) return;
        for (int i = 0; i < indent; i++) printf("    ");
        printf("%s (Total: %.3f ms, Avg: %.3f ms, Count: %llu)\n", name, g.getTotalMs(), g.getAverage(), (unsigned long long)count);
    };

    printGroup("rayChunkTime", m_profile_stats.rayChunkTime, 0);
    printTracker("processRaysTime", m_profile_stats.processRaysTime, 1);
    
    printGroup("inferenceTime", m_profile_stats.inferenceTime, 1);
    printTracker("inferenceDensityFwd", m_profile_stats.inferenceDensityFwd, 2);
    printTracker("inferenceGatherSH", m_profile_stats.inferenceGatherSH, 2);
    printTracker("inferenceColorFwd", m_profile_stats.inferenceColorFwd, 2);

    printTracker("volumeRendering", m_profile_stats.volumeRendering, 1);
    printTracker("fillFloatKernel", m_profile_stats.fillFloatKernel, 1);
    printTracker("trainZeroGrad", m_profile_stats.trainZeroGrad, 1);

    printGroup("earlyOccupancyGridUpdate", m_profile_stats.earlyOccupancyUpdateTime, 1);
    printTracker("earlyOccupancySampleTime", m_profile_stats.earlyOccupancySampleTime, 2);
    printGroup("earlyOccupancyInferenceTime", m_profile_stats.earlyOccupancyInferenceTime, 2);
    printTracker("earlyOccupancyDensityFwd", m_profile_stats.earlyOccupancyDensityFwd, 3);
    printTracker("earlyOccupancyUpdateTmpGrid", m_profile_stats.earlyOccupancyUpdateTmpGrid, 3);
    printTracker("earlyOccupancyUpdateMasterGrid", m_profile_stats.earlyOccupancyUpdateMasterGrid, 2);
    printTracker("earlyOccupancyComputeSum", m_profile_stats.earlyOccupancyComputeSum, 2);
    printTracker("earlyOccupancyUpdateBitgrid", m_profile_stats.earlyOccupancyUpdateBitgrid, 2);
    printTracker("earlyOccupancyUpdateMipmap", m_profile_stats.earlyOccupancyUpdateMipmap, 2);

    printGroup("lateOccupancyGridUpdate", m_profile_stats.lateOccupancyUpdateTime, 1);
    printTracker("lateOccupancySampleUniformTime", m_profile_stats.lateOccupancySampleUniformTime, 2);
    printTracker("lateOccupancySampleNonUniformTime", m_profile_stats.lateOccupancySampleNonUniformTime, 2);
    printGroup("lateOccupancyInferenceTime", m_profile_stats.lateOccupancyInferenceTime, 2);
    printTracker("lateOccupancyDensityFwd", m_profile_stats.lateOccupancyDensityFwd, 3);
    printTracker("lateOccupancyUpdateTmpGrid", m_profile_stats.lateOccupancyUpdateTmpGrid, 3);
    printTracker("lateOccupancyUpdateMasterGrid", m_profile_stats.lateOccupancyUpdateMasterGrid, 2);
    printTracker("lateOccupancyComputeSum", m_profile_stats.lateOccupancyComputeSum, 2);
    printTracker("lateOccupancyUpdateBitgrid", m_profile_stats.lateOccupancyUpdateBitgrid, 2);
    printTracker("lateOccupancyUpdateMipmap", m_profile_stats.lateOccupancyUpdateMipmap, 2);

    printGroup("trainTime", m_profile_stats.trainTime, 1);
    printTracker("trainGatherTime", m_profile_stats.trainGatherTime, 2);
    printTracker("trainDensityFwd", m_profile_stats.trainDensityFwd, 2);
    printTracker("trainComputeSH", m_profile_stats.trainComputeSH, 2);
    printTracker("trainColorFwd", m_profile_stats.trainColorFwd, 2);
    printTracker("trainColorGrad", m_profile_stats.trainColorGrad, 2);
    printTracker("trainColorBwd", m_profile_stats.trainColorBwd, 2);
    printTracker("trainDensityGrad", m_profile_stats.trainDensityGrad, 2);
    printTracker("trainDensityBwd", m_profile_stats.trainDensityBwd, 2);

    printTracker("trainColorOpt", m_profile_stats.trainColorOpt, 1);
    printTracker("trainDensityOpt", m_profile_stats.trainDensityOpt, 1);
}

void InstantNerf::initRenderBuffers() 
{
    if (!m_renderinit) {
        fprintf(stderr, "[InstantNerf] Error: init() must be called before initRenderBuffers()\n");
        return;

    }

    if (m_opts.rayChunkSize % 16 != 0) {
        fprintf(stderr, "[InstantNerf] Error: rayChunkSize must be a multiple of 16");
        return;
    }

    m_render_buffers.d_rays_d_inv_chunk = DeviceBuffer<float3>(m_opts.rayChunkSize);
    m_render_buffers.d_nears_chunk = DeviceBuffer<float>(m_opts.rayChunkSize);
    m_render_buffers.d_fars_chunk = DeviceBuffer<float>(m_opts.rayChunkSize);
    m_render_buffers.d_block_sums = DeviceBuffer<uint32_t>((m_opts.rayChunkSize + 1023) / 1024);
    m_render_buffers.d_active_rays_count = DeviceBuffer<uint32_t>(1);
    m_render_buffers.d_num_steps = DeviceBuffer<uint32_t>(m_opts.rayChunkSize);
    m_render_buffers.d_ray_offsets = DeviceBuffer<uint32_t>(m_opts.rayChunkSize);
    int gridCells = m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z; // G: one cascade
    int masterGridCells = m_opts.numCascades * gridCells;                                          // full persistent grid
    uint32_t max_density_out = std::max({(uint32_t)(16 * m_opts.batchSize), (uint32_t)(16 * gridCells), (uint32_t)(16 * m_opts.renderBatchSize)});
    m_render_buffers.d_density_out = DeviceBuffer<float>(max_density_out);
    m_render_buffers.d_render_rgb_chunk = DeviceBuffer<float>(3 * m_opts.rayChunkSize);
    m_render_buffers.d_render_depth_chunk = DeviceBuffer<float>(m_opts.rayChunkSize);
    m_render_buffers.d_phi_chunk = DeviceBuffer<float>(3 * m_opts.rayChunkSize);
    m_render_buffers.d_out_count = DeviceBuffer<uint32_t>(1);
    m_render_buffers.d_custom_color_grad = DeviceBuffer<half>(3 * m_opts.batchSize);
    m_render_buffers.d_custom_density_grad = DeviceBuffer<half>(16 * m_opts.batchSize);
    m_render_buffers.d_tmpsigma = DeviceBuffer<float>(m_opts.batchSize);
    m_render_buffers.d_color_dx_out = DeviceBuffer<half>(32 * m_opts.batchSize);
    m_render_buffers.d_dw_out = DeviceBuffer<float>(m_opts.batchSize);
    m_render_buffers.d_weight_sum = DeviceBuffer<float>(m_opts.rayChunkSize);
    
    m_render_buffers.d_occupancy_samples = DeviceBuffer<float>(4 * gridCells); // one cascade chunk
    m_render_buffers.d_tmp_grid = DeviceBuffer<float>(masterGridCells);
    m_render_buffers.d_sum = DeviceBuffer<float>(1);
    
    m_render_buffers.d_numActiveCells = DeviceBuffer<int>(1);
    m_render_buffers.d_activeCellIndices = DeviceBuffer<int>(masterGridCells);
    
    cub::DeviceReduce::Sum(m_render_buffers.d_temp_storage, m_render_buffers.temp_storage_bytes, d_masterOccupancyGrid.data(), m_render_buffers.d_sum.data(), masterGridCells);
    cudaMalloc(&m_render_buffers.d_temp_storage, m_render_buffers.temp_storage_bytes);
    
    m_render_buffers.h_ray_offsets.reserve(m_opts.rayChunkSize);
    
    
    if (m_memMode == TRAINING) {
        m_render_buffers.d_color_input = DeviceBuffer<half>(32 * m_opts.batchSize);
        m_render_buffers.d_color_output = DeviceBuffer<float>(3 * m_opts.batchSize);
        m_render_buffers.d_mlp_positions = DeviceBuffer<float>(m_opts.batchSize * 4);
        m_render_buffers.d_ray_indices = DeviceBuffer<uint32_t>(m_opts.batchSize);
        m_render_buffers.d_t_sorted = DeviceBuffer<float>(m_opts.batchSize);
        m_render_buffers.d_density_sigma = DeviceBuffer<float>(m_opts.batchSize);
        m_render_buffers.d_rgb_output = DeviceBuffer<float>(m_opts.batchSize * 3);
        uint32_t totalHits = m_opts.rayChunkSize * MAX_HITS;
        m_colorMLP->switchToTrainingMode();
        m_densityMLP->switchToTrainingMode();
    } else if (m_memMode == INFERENCE) {

        uint32_t sizing = m_opts.renderBatchSize;

        if (m_opts.legacyRenderFlag) {
            sizing = m_opts.rayChunkSize * MAX_HITS;
        }

        m_render_buffers.d_color_input = DeviceBuffer<half>(32 * m_opts.renderBatchSize);
        m_render_buffers.d_color_output = DeviceBuffer<float>(3 * m_opts.renderBatchSize);
        m_render_buffers.d_mlp_positions = DeviceBuffer<float>(sizing * 4);
        m_render_buffers.d_ray_indices = DeviceBuffer<uint32_t>(sizing);
        m_render_buffers.d_t_sorted = DeviceBuffer<float>(sizing);
        m_render_buffers.d_density_sigma = DeviceBuffer<float>(sizing);
        m_render_buffers.d_rgb_output = DeviceBuffer<float>(sizing * 3);
        m_render_buffers.d_sparse_ts = DeviceBuffer<float>(sizing);
        m_render_buffers.d_dense_sparse_indices = DeviceBuffer<uint32_t>(sizing);
        
        m_colorMLP->switchToInferenceMode();
        m_densityMLP->switchToInferenceMode();
    } 

    m_traininit = true;
} 



void InstantNerf::freeBuffers() {
    m_render_buffers.d_rays_d_inv_chunk = DeviceBuffer<float3>(0);
    m_render_buffers.d_nears_chunk = DeviceBuffer<float>(0);
    m_render_buffers.d_fars_chunk = DeviceBuffer<float>(0);
    m_render_buffers.d_sparse_ts = DeviceBuffer<float>(0);
    m_render_buffers.d_num_steps = DeviceBuffer<uint32_t>(0);
    m_render_buffers.d_ray_offsets = DeviceBuffer<uint32_t>(0);
    m_render_buffers.d_dense_sparse_indices = DeviceBuffer<uint32_t>(0);
    m_render_buffers.d_mlp_positions = DeviceBuffer<float>(0);
    m_render_buffers.d_ray_indices = DeviceBuffer<uint32_t>(0);
    m_render_buffers.d_t_sorted = DeviceBuffer<float>(0);
    m_render_buffers.d_density_out = DeviceBuffer<float>(0);
    m_render_buffers.d_color_input = DeviceBuffer<half>(0);
    m_render_buffers.d_color_output = DeviceBuffer<float>(0);
    m_render_buffers.d_density_sigma = DeviceBuffer<float>(0);
    m_render_buffers.d_rgb_output = DeviceBuffer<float>(0);
    m_render_buffers.d_render_rgb_chunk = DeviceBuffer<float>(0);
    m_render_buffers.d_render_depth_chunk = DeviceBuffer<float>(0);
    m_render_buffers.d_phi_chunk = DeviceBuffer<float>(0);
    m_render_buffers.d_out_count = DeviceBuffer<uint32_t>(0);
    m_render_buffers.d_active_ray_indices = DeviceBuffer<uint32_t>(0);
    m_render_buffers.d_batch_position = DeviceBuffer<float>(0);
    m_render_buffers.d_batch_direction = DeviceBuffer<float3>(0);
    m_render_buffers.d_current_t = DeviceBuffer<float>(0);
    m_render_buffers.d_current_rgb = DeviceBuffer<float>(0);
    m_render_buffers.d_custom_color_grad = DeviceBuffer<half>(0);
    m_render_buffers.d_custom_density_grad = DeviceBuffer<half>(0);
    m_render_buffers.d_tmpsigma = DeviceBuffer<float>(0);
    m_render_buffers.d_color_dx_out = DeviceBuffer<half>(0);
    m_render_buffers.d_occupancy_samples = DeviceBuffer<float>(0);
    m_render_buffers.d_tmp_grid = DeviceBuffer<float>(0);
    m_render_buffers.d_sum = DeviceBuffer<float>(0);
    m_render_buffers.d_numActiveCells = DeviceBuffer<int>(0);
    m_render_buffers.d_activeCellIndices = DeviceBuffer<int>(0);

    if (m_render_buffers.d_temp_storage) {
        cudaFree(m_render_buffers.d_temp_storage);
        m_render_buffers.d_temp_storage = nullptr;
    }
    m_render_buffers.h_ray_offsets.clear();
    m_render_buffers.h_ray_offsets.shrink_to_fit();

    m_traininit = false;
}

void InstantNerf::init(const NerfOptions& opts, MemoryMode memMode) {
    m_opts = opts;
    m_memMode = memMode;

    int occupancyGridCells = 0;
    for (int j = 0; j < opts.numCascades; j++) {
        uint3 res = make_uint3(opts.gridResolution.x, opts.gridResolution.y, opts.gridResolution.z);
        for (int i = 0; i < opts.levelsMipmap; i++) {
            occupancyGridCells += res.x * res.y * res.z;
            res = make_uint3(res.x / 2, res.y / 2, res.z / 2);
        }
    }
    
    int occupancyGridBytes = (occupancyGridCells + 7)/ 8;
    int masterGridCells = opts.numCascades * (opts.gridResolution.x * opts.gridResolution.y * opts.gridResolution.z);

    d_occupancyGrid = DeviceBuffer<uint8_t>(occupancyGridBytes);
    d_occupancyGrid.fill(0);
    d_masterOccupancyGrid = DeviceBuffer<float>(masterGridCells);

    {
        constexpr int BS = 256;
        int gs = (masterGridCells + BS - 1) / BS;
        fill_float_kernel<<<gs, BS>>>(d_masterOccupancyGrid.data(), 0.0f, masterGridCells);
    }

    MLPGridOptions densityOpts;
    densityOpts.vectorDim = 4;
    densityOpts.hiddenDim = opts.densityHiddenDim;
    densityOpts.outputDim = 16;
    densityOpts.numLayers = opts.densityNumLayers;
    densityOpts.activationType = ACT_RELU;
    densityOpts.tableSize = opts.hashTableSize;
    densityOpts.numLevels = opts.numLevels;
    densityOpts.b = opts.growthFactor;
    densityOpts.lowestSize = opts.baseResolution;
    densityOpts.featuresLevel = opts.featuresPerLevel;

    int densityInferBatch = std::max((int)(opts.gridResolution.x * opts.gridResolution.y * opts.gridResolution.z), opts.renderBatchSize);
    m_densityMLP = new TinyMLPHashGrid(densityOpts, opts.batchSize, densityInferBatch);

    MLPOption colorOpts;
    colorOpts.inputDim = 32;
    colorOpts.hiddenDim = opts.colorHiddenDim;
    colorOpts.outputDim = 3;
    colorOpts.numLayers = opts.colorNumLayers;
    colorOpts.activationType = ACT_RELU;
    colorOpts.outputActivation = OUT_ACT_SIGMOID;

    m_colorMLP = new TinyMLP(colorOpts, opts.batchSize, opts.renderBatchSize);

    m_profile_stats.init();
    m_profile_stats.pendingTimers.reserve(100000);
    m_renderinit = true;

    initRenderBuffers();
    earlyOccupancyGridUpdate();
}

void InstantNerf::setMemoryMode(MemoryMode mode) {
    if (!m_traininit) return;
    if (m_memMode == mode) return;
    if (m_opts.renderBatchSize == m_opts.batchSize && !m_opts.legacyRenderFlag) return;

    m_memMode = mode;

    initRenderBuffers();
    
    if (m_memMode == TRAINING) {
        m_colorMLP->switchToTrainingMode();
        m_densityMLP->switchToTrainingMode();
    } else if (m_memMode == INFERENCE) {
        m_colorMLP->switchToInferenceMode();
        m_densityMLP->switchToInferenceMode();
    }
}

void InstantNerf::trainWithRaysHit(
    const float3* d_rays_o,
    const float3* d_rays_d,
    const float* d_rays_depth,
    const float* d_rays_sigma,
    const float* d_rgb_true,
    uint32_t numRays,
    int& trainStepCount,
    uint32_t* hitCounts,
    float* d_rgb_out,
    cudaStream_t stream
) {
    if(!m_traininit) {
        fprintf(stderr, "[InstantNerf] Error: init() must be called before trainWithRaysHit()\n");
        return;
    }

    uint32_t raysDone = 0;
    int i = 0;

    while (raysDone < numRays) {
        uint32_t currentChunkRaysUpperBound = min(m_opts.rayChunkSize, numRays - raysDone);
        const float3* chunk_o = d_rays_o + raysDone;
        const float3* chunk_d = d_rays_d + raysDone;
        const float* chunk_depth = d_rays_depth + raysDone;
        const float* chunk_sigma = d_rays_sigma + raysDone;

        int currentChunkRays;
        uint32_t totalHits;
        measure(stream, m_profile_stats.processRaysTime, [&]() {
        constexpr int BS = 256;
        int gs = (currentChunkRaysUpperBound + BS - 1) / BS;
        compute_ray_aabb_inv_kernel<<<gs, BS, 0, stream>>>(
            currentChunkRaysUpperBound,
            chunk_o, chunk_d, m_opts.aabbMin, m_opts.aabbMax,
            m_opts.numCascades,
            m_render_buffers.d_rays_d_inv_chunk.data(),
            m_render_buffers.d_nears_chunk.data(),
            m_render_buffers.d_fars_chunk.data()
        );

        currentChunkRays = processRaysHitLinear(
            currentChunkRaysUpperBound,
            chunk_o, chunk_d,
            m_render_buffers.d_rays_d_inv_chunk.data(),
            m_render_buffers.d_nears_chunk.data(),
            m_render_buffers.d_fars_chunk.data(),
            d_occupancyGrid.data(),
            m_opts.gridResolution,
            m_opts.aabbMin, m_opts.aabbMax,
            m_opts.numCascades,
            m_opts.levelsMipmap,
            m_opts.samplesPerVoxel,
            m_opts.batchSize,
            &totalHits,
            m_render_buffers.d_active_rays_count.data(),
            m_render_buffers.d_num_steps.data(),
            m_render_buffers.d_ray_offsets.data(),
            m_render_buffers.d_mlp_positions.data(),
            m_render_buffers.d_ray_indices.data(),
            m_render_buffers.d_t_sorted.data(),
            m_render_buffers.d_block_sums.data(),
            stream
        );
        });

        if (currentChunkRays == 0) {
            fprintf(stderr, "Error: Batch size is too small to fit even a single ray! Increase batchSize.\n");
            return;
        }

        if (hitCounts != nullptr) {
            hitCounts[i] = totalHits;
        }
        uint32_t padded_b_size = (totalHits + 15) & ~15;

        measure(stream, m_profile_stats.inferenceDensityFwd, [&](){
        m_densityMLP->inference(m_render_buffers.d_mlp_positions.data(), m_render_buffers.d_density_out.data(), padded_b_size, stream);
        });

        measure(stream, m_profile_stats.inferenceGatherSH, [&]()
        {
            constexpr int BS = 256;
            int gs = (totalHits + BS - 1) / BS;

            compute_SH_gather<<<gs, BS, 0, stream>>>(
                chunk_d, m_render_buffers.d_ray_indices.data(), 
                0, totalHits, m_opts.densityBias,
                m_render_buffers.d_density_out.data(), 
                m_render_buffers.d_color_input.data(),
                m_render_buffers.d_density_sigma.data()
            );
        });

        measure(stream, m_profile_stats.inferenceColorFwd, [&](){
        m_colorMLP->inference(
            m_render_buffers.d_color_input.data(),
            m_render_buffers.d_rgb_output.data(),
            padded_b_size,
            stream
        );
        });

        measure(stream, m_profile_stats.volumeRendering, [&](){
        launchVolumeRendering(
            currentChunkRays,
            m_render_buffers.d_ray_offsets.data(),
            m_render_buffers.d_num_steps.data(),
            m_render_buffers.d_t_sorted.data(),
            m_render_buffers.d_density_sigma.data(),
            chunk_depth, chunk_sigma,
            m_render_buffers.d_rgb_output.data(),
            d_rgb_true ? (d_rgb_true + raysDone * 3) : nullptr,
            m_render_buffers.d_render_rgb_chunk.data(),
            m_render_buffers.d_render_depth_chunk.data(),
            m_render_buffers.d_phi_chunk.data(),
            m_render_buffers.d_dw_out.data(),
            m_render_buffers.d_weight_sum.data(),
            m_opts.lambdaDist,
            m_opts.lambdaDepth,
            m_opts.bgColor, 0, 0,
            stream
        );
        });

        if (d_rgb_out != nullptr) {
            cudaMemcpyAsync(d_rgb_out + raysDone * 3, m_render_buffers.d_render_rgb_chunk.data(), currentChunkRays * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        if (m_trainSteps % 16 == 0) {
            if (m_trainSteps < 256) {
                earlyOccupancyGridUpdate(stream);
            } else {
                lateOccupancyGridUpdate(stream);
            }
        }

        measure(stream, m_profile_stats.trainZeroGrad, [&](){
            m_densityMLP->zero_grad(stream);
            m_colorMLP->zero_grad(stream);
        });

        if (totalHits > 0) {
            cudaMemsetAsync(m_render_buffers.d_custom_color_grad.data(), 0, padded_b_size * 3 * sizeof(half), stream);
            cudaMemsetAsync(m_render_buffers.d_custom_density_grad.data(), 0, padded_b_size * 16 * sizeof(half), stream);
            
            measure(stream, m_profile_stats.trainDensityFwd, [&](){
            m_densityMLP->forward(m_render_buffers.d_mlp_positions.data(), m_render_buffers.d_density_out.data(), padded_b_size, stream);
            });

            constexpr int BS = 256;
            int gs = (totalHits + BS - 1) / BS;
            measure(stream, m_profile_stats.trainComputeSH, [&](){
            compute_SH_gather<<<gs, BS, 0, stream>>>(
                chunk_d, m_render_buffers.d_ray_indices.data(),
                0, totalHits, m_opts.densityBias, m_render_buffers.d_density_out.data(),
                m_render_buffers.d_color_input.data(),
                m_render_buffers.d_density_sigma.data()
            );
            });

            measure(stream, m_profile_stats.trainColorFwd, [&](){
            m_colorMLP->forward(m_render_buffers.d_color_input.data(), m_render_buffers.d_color_output.data(), padded_b_size, stream);
            });

            int gs_color = (currentChunkRays + 255) / 256;
            measure(stream, m_profile_stats.trainColorGrad, [&](){
            compute_color_grad<<<gs_color, 256, 0, stream>>>(
                currentChunkRays,
                0,
                m_render_buffers.d_num_steps.data(),
                m_render_buffers.d_ray_offsets.data(),
                m_render_buffers.d_t_sorted.data(),
                m_render_buffers.d_density_out.data(),
                m_render_buffers.d_color_output.data(),
                m_render_buffers.d_phi_chunk.data(),
                m_render_buffers.d_render_rgb_chunk.data(),
                m_render_buffers.d_dw_out.data(),
                m_render_buffers.d_weight_sum.data(),
                m_render_buffers.d_custom_color_grad.data(),
                  m_render_buffers.d_tmpsigma.data(),
                  m_opts.lossScale,
                  m_opts.densityBias,
                  numRays
              );
            });

            measure(stream, m_profile_stats.trainColorBwd, [&](){
            m_colorMLP->backward(m_render_buffers.d_custom_color_grad.data(), m_render_buffers.d_color_dx_out.data(), padded_b_size, stream);
            });

            int gs_density = (totalHits + 255) / 256;
            measure(stream, m_profile_stats.trainDensityGrad, [&](){
            compute_density_grad<<<gs_density, 256, 0, stream>>>(
                totalHits, 
                m_render_buffers.d_tmpsigma.data(), 
                m_render_buffers.d_color_dx_out.data(), 
                m_render_buffers.d_custom_density_grad.data()
            );
            });

            measure(stream, m_profile_stats.trainDensityBwd, [&](){
            m_densityMLP->backward(m_render_buffers.d_custom_density_grad.data(), padded_b_size, stream);
            });
        }

        measure(stream, m_profile_stats.trainColorOpt, [&](){
        m_colorMLP->step(m_opts.learningRate, m_opts.beta1, m_opts.beta2, m_opts.epsilon, m_opts.lossScale, stream);
        });

        measure(stream, m_profile_stats.trainDensityOpt, [&](){
        m_densityMLP->step(m_opts.learningRate, m_opts.beta1, m_opts.beta2, m_opts.epsilon, m_opts.lossScale, stream);
        });

        trainStepCount++;
        m_trainSteps++;

        raysDone += currentChunkRays;
        i++;
    }

    if (m_opts.isProfiling) {
        m_profile_stats.resolvePendingTimers();
    }
}

void InstantNerf::renderImage(
    const float3* d_rays_o,
    const float3* d_rays_d,
    uint32_t numRays,
    float* d_rgb_out,
    cudaStream_t stream
) {
    if(!m_renderinit) {
        fprintf(stderr, "[InstantNerf] Error: init() must be called before renderImage()\n");
        return;
    }

    if (!m_opts.legacyRenderFlag) {
        fprintf(stderr, "[InstantNerf] Error: renderImage() requires legacyRenderFlag=true; use renderImageHit() instead.\n");
        return;
    }

    for (int offset = 0; offset < numRays; offset += m_opts.rayChunkSize) {
        int currentChunkRays = std::min(m_opts.rayChunkSize, (int)numRays - offset);
        const float3* chunk_o = d_rays_o + offset;
        const float3* chunk_d = d_rays_d + offset;

        measure(stream, m_profile_stats.processRaysTime, [&](){
            constexpr int BS = 256;
            int gs = (currentChunkRays + BS - 1) / BS;

            compute_ray_aabb_inv_kernel<<<gs, BS, 0, stream>>>(
                currentChunkRays,
                chunk_o,
                chunk_d,
                m_opts.aabbMin, m_opts.aabbMax, m_opts.numCascades,
                m_render_buffers.d_rays_d_inv_chunk.data(),
                m_render_buffers.d_nears_chunk.data(),
                m_render_buffers.d_fars_chunk.data()
            );

            processRaysChunkLinear(
                currentChunkRays,
                chunk_o, chunk_d, m_render_buffers.d_rays_d_inv_chunk.data(),
                m_render_buffers.d_nears_chunk.data(),
                m_render_buffers.d_fars_chunk.data(),
                d_occupancyGrid.data(),
                m_opts.gridResolution,
                m_opts.aabbMin, m_opts.aabbMax, 
                m_opts.numCascades, m_opts.levelsMipmap,
                m_opts.samplesPerVoxel,
                m_render_buffers.d_sparse_ts.data(), 
                m_render_buffers.d_num_steps.data(),
                m_render_buffers.d_ray_offsets.data(),
                m_render_buffers.d_dense_sparse_indices.data(),
                m_render_buffers.d_mlp_positions.data(), 
                m_render_buffers.d_ray_indices.data(), 
                m_render_buffers.d_t_sorted.data(),
                stream
            );
        });
        
        uint32_t totalHits = 0;
        uint32_t highestHits = 0;
        
        std::vector<uint32_t> h_num_steps(currentChunkRays);
        m_render_buffers.d_num_steps.copyHost(h_num_steps.data(), currentChunkRays);

        for (uint32_t i = 0; i < currentChunkRays; ++i) {
            totalHits += h_num_steps[i];
            if(h_num_steps[i] > highestHits) highestHits = h_num_steps[i];
        }
        
        for (uint32_t b_offset = 0; b_offset < totalHits; b_offset += m_opts.batchSize) {
            uint32_t b_size = std::min(static_cast<uint32_t>(m_opts.batchSize), totalHits - b_offset);
            uint32_t padded_b_size = (b_size + 15) & ~15;

            measure(stream, m_profile_stats.inferenceDensityFwd, [&](){
            m_densityMLP->inference(m_render_buffers.d_mlp_positions.data() + b_offset * 4, m_render_buffers.d_density_out.data(), padded_b_size, stream);
            });

            measure(stream, m_profile_stats.inferenceGatherSH, [&](){
                constexpr int BS = 256;
                int gs = (b_size + BS - 1) / BS;

                compute_SH_gather<<<gs, BS, 0, stream>>>(
                    chunk_d, m_render_buffers.d_ray_indices.data(), 
                    b_offset, b_size, m_opts.densityBias,
                    m_render_buffers.d_density_out.data(), 
                    m_render_buffers.d_color_input.data(),
                    m_render_buffers.d_density_sigma.data() + b_offset
                );
            });

            measure(stream, m_profile_stats.inferenceColorFwd, [&](){
            m_colorMLP->inference(
                m_render_buffers.d_color_input.data(),
                m_render_buffers.d_rgb_output.data() + b_offset * 3,
                padded_b_size,
                stream
            );
            });
        }

        measure(stream, m_profile_stats.volumeRendering, [&](){
        launchVolumeRendering(
            currentChunkRays,
            m_render_buffers.d_ray_offsets.data(),
            m_render_buffers.d_num_steps.data(),
            m_render_buffers.d_t_sorted.data(),
            m_render_buffers.d_density_sigma.data(),
            nullptr, nullptr,
            m_render_buffers.d_rgb_output.data(),
            nullptr,
            m_render_buffers.d_render_rgb_chunk.data(),
            m_render_buffers.d_render_depth_chunk.data(),
            m_render_buffers.d_phi_chunk.data(),
            nullptr, nullptr, 0, 0,
            m_opts.bgColor, 0, 0,
            stream
        );
        });

        if (d_rgb_out != nullptr) {
            cudaMemcpyAsync(d_rgb_out + offset * 3, m_render_buffers.d_render_rgb_chunk.data(), currentChunkRays * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
    }

    if (m_opts.isProfiling) {
        m_profile_stats.resolvePendingTimers();
    }
}

void InstantNerf::renderImageHit(
    const float3* d_rays_o,
    const float3* d_rays_d,
    uint32_t numRays,
    float* d_rgb_out,
    cudaStream_t stream
) {
    if(!m_traininit) {
        fprintf(stderr, "[InstantNerf] Error: init() must be called before trainWithRaysHit()\n");
        return;
    }

    int hitBatch = (m_memMode == INFERENCE) ? m_opts.renderBatchSize : m_opts.batchSize;

    for (uint32_t ray_offset = 0; ray_offset < numRays; ray_offset += m_opts.rayChunkSize) {
        uint32_t currentChunkRays = min(m_opts.rayChunkSize, numRays - ray_offset);
        const float3* chunk_o = d_rays_o + ray_offset;
        const float3* chunk_d = d_rays_d + ray_offset;

        constexpr int BS = 256;
        int gs = (currentChunkRays + BS - 1) / BS;

        measure(stream, m_profile_stats.processRaysTime, [&]() {
        compute_ray_aabb_inv_kernel<<<gs, BS, 0, stream>>>(
            currentChunkRays,
            chunk_o, chunk_d, m_opts.aabbMin, m_opts.aabbMax, m_opts.numCascades,
            m_render_buffers.d_rays_d_inv_chunk.data(),
            m_render_buffers.d_nears_chunk.data(),
            m_render_buffers.d_fars_chunk.data()
        );

        processRaysHitData(
            currentChunkRays,
            chunk_o, chunk_d,
            m_render_buffers.d_rays_d_inv_chunk.data(),
            m_render_buffers.d_nears_chunk.data(),
            m_render_buffers.d_fars_chunk.data(),
            d_occupancyGrid.data(),
            m_opts.gridResolution,
            m_opts.aabbMin, m_opts.aabbMax,
            m_opts.numCascades,
            m_opts.levelsMipmap,
            m_opts.samplesPerVoxel,
            m_render_buffers.d_num_steps.data(),
            m_render_buffers.d_ray_offsets.data(),
            m_render_buffers.d_block_sums.data(),
            stream
        );
        });
        
        uint32_t raysDone = 0;
        int i = 0;
        while (raysDone < currentChunkRays) {

            uint32_t totalHits;
            uint32_t outBase;

            int crrRays = 0;
            measure(stream, m_profile_stats.processRaysTime, [&]() {
            crrRays = processRaysHitPositions(
                currentChunkRays,
                chunk_o, chunk_d,
                m_render_buffers.d_rays_d_inv_chunk.data(),
                m_render_buffers.d_nears_chunk.data(),
                m_render_buffers.d_fars_chunk.data(),
                d_occupancyGrid.data(),
                m_opts.gridResolution,
                m_opts.aabbMin, m_opts.aabbMax,
                m_opts.numCascades,
                m_opts.levelsMipmap,
                m_opts.samplesPerVoxel,
                raysDone,
                hitBatch,
                &totalHits,
                &outBase,
                m_render_buffers.d_num_steps.data(),
                m_render_buffers.d_ray_offsets.data(),
                m_render_buffers.d_mlp_positions.data(),
                m_render_buffers.d_ray_indices.data(),
                m_render_buffers.d_active_rays_count.data(),
                m_render_buffers.d_t_sorted.data(),
                m_render_buffers.d_block_sums.data(),
                stream
            );
            });

            if (crrRays == 0) {
                fprintf(stderr, "Error: Batch size is too small to fit even a single ray! Increase batchSize.\n");
                return; // Or throw an exception to escape the infinite loop
            }
            uint32_t padded_b_size = (totalHits + 15) & ~15;
            
            measure(stream, m_profile_stats.inferenceDensityFwd, [&](){
                m_densityMLP->inference(m_render_buffers.d_mlp_positions.data(), m_render_buffers.d_density_out.data(), padded_b_size, stream);
            });
            
            measure(stream, m_profile_stats.inferenceGatherSH, [&]()
            {
                constexpr int BS = 256;
                int gs = (totalHits + BS - 1) / BS;
                
                compute_SH_gather<<<gs, BS, 0, stream>>>(
                    chunk_d + raysDone, m_render_buffers.d_ray_indices.data(), 
                    0, totalHits, m_opts.densityBias,
                    m_render_buffers.d_density_out.data(), 
                    m_render_buffers.d_color_input.data(),
                    m_render_buffers.d_density_sigma.data()
                );
            });
            
            measure(stream, m_profile_stats.inferenceColorFwd, [&](){
                m_colorMLP->inference(
                    m_render_buffers.d_color_input.data(),
                    m_render_buffers.d_rgb_output.data(),
                    padded_b_size,
                    stream
                );
            });
            
            measure(stream, m_profile_stats.volumeRendering, [&](){
                launchVolumeRendering(
                    crrRays,
                    m_render_buffers.d_ray_offsets.data(),
                    m_render_buffers.d_num_steps.data() + raysDone,
                    m_render_buffers.d_t_sorted.data(),
                    m_render_buffers.d_density_sigma.data(),
                    nullptr, nullptr,
                    m_render_buffers.d_rgb_output.data(),
                    nullptr,
                    m_render_buffers.d_render_rgb_chunk.data(),
                    m_render_buffers.d_render_depth_chunk.data(),
                    m_render_buffers.d_phi_chunk.data(),
                    nullptr, nullptr, 0, 0,
                    m_opts.bgColor,
                    outBase, raysDone,
                    stream
                );
            });
            
            if (d_rgb_out != nullptr) {
                cudaMemcpyAsync(d_rgb_out + (ray_offset + raysDone) * 3, m_render_buffers.d_render_rgb_chunk.data(), crrRays * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
            raysDone += crrRays;
        }
    }

    if (m_opts.isProfiling) {
        m_profile_stats.resolvePendingTimers();
    }


}

void InstantNerf::lateOccupancyGridUpdate(cudaStream_t stream) {
    int G = m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z;
    int N = m_opts.numCascades * G;
    int S = pyramidCellsPerCascade(m_opts.gridResolution, m_opts.levelsMipmap);
    int halfN = N / 2;

    constexpr int BS = 256;
    int gs = (N + BS - 1) / BS;

    cudaMemsetAsync(m_render_buffers.d_tmp_grid.data(), 0, N * sizeof(float), stream);

    // Build the active-cell list once over the full grid (it is constant during this update).
    cudaMemsetAsync(m_render_buffers.d_numActiveCells.data(), 0, sizeof(int), stream);
    extractActiveCells<<<gs, BS, 0, stream>>>(
        d_occupancyGrid.data(),
        m_render_buffers.d_activeCellIndices.data(),
        m_render_buffers.d_numActiveCells.data(),
        N, G, S
    );

    // Split the half-N sample budget across numCascades chunks so the transient density/sample
    // buffers stay sized for a single cascade. Each chunk draws half occupied-biased + half
    // uniform (global) samples — matching the original sampling ratios in aggregate.
    int chunks = m_opts.numCascades;
    int perChunk = halfN / chunks;            // ~ G/2
    int occPerChunk = perChunk / 2;
    int uniPerChunk = perChunk - occPerChunk;

    for (int ch = 0; ch < chunks; ++ch) {
        unsigned long long seedBase = (unsigned long long)m_trainSteps * (2ULL * chunks) + 2ULL * ch;

        int occGs = (occPerChunk + BS - 1) / BS;
        measure(stream, m_profile_stats.lateOccupancySampleNonUniformTime, [&](){
        sampleOccupiedAABB<<<occGs, BS, 0, stream>>>(
            m_render_buffers.d_occupancy_samples.data(),
            m_render_buffers.d_activeCellIndices.data(),
            m_render_buffers.d_numActiveCells.data(),
            m_opts.aabbMin, m_opts.aabbMax, m_opts.gridResolution,
            occPerChunk, seedBase
        );
        });

        int uniGs = (uniPerChunk + BS - 1) / BS;
        measure(stream, m_profile_stats.lateOccupancySampleUniformTime, [&](){
        sampleAABB<<<uniGs, BS, 0, stream>>>(
            m_render_buffers.d_occupancy_samples.data() + occPerChunk * 4,
            m_opts.aabbMin, m_opts.aabbMax, m_opts.gridResolution,
            uniPerChunk, N, 0, seedBase + 1ULL
        );
        });

        int chunkCount = occPerChunk + uniPerChunk;
        int paddedBatchSize = (chunkCount + 15) & ~15;

        measure(stream, m_profile_stats.lateOccupancyDensityFwd, [&](){
        m_densityMLP->inference(m_render_buffers.d_occupancy_samples.data(), m_render_buffers.d_density_out.data(), paddedBatchSize, stream);
        });

        measure(stream, m_profile_stats.lateOccupancyUpdateTmpGrid, [&](){
        updateTmpGrid<<<(chunkCount + BS - 1) / BS, BS, 0, stream>>>(
            m_render_buffers.d_occupancy_samples.data(),
            m_render_buffers.d_density_out.data(),
            m_render_buffers.d_tmp_grid.data(),
            m_opts.densityBias,
            m_opts.aabbMin,
            m_opts.aabbMax,
            m_opts.gridResolution,
            chunkCount
        );
        });
    }

    measure(stream, m_profile_stats.lateOccupancyUpdateMasterGrid, [&](){
    updateMasterGrid<<<gs, BS, 0, stream>>>(m_render_buffers.d_tmp_grid.data(), d_masterOccupancyGrid.data(), m_opts.decayValue, N);
    });

    measure(stream, m_profile_stats.lateOccupancyComputeSum, [&](){
    computeSum(m_render_buffers.d_sum.data(), m_render_buffers.d_temp_storage, m_render_buffers.temp_storage_bytes, d_masterOccupancyGrid.data(), N);
    });

    uint32_t* d_occupancy_grid_32 = reinterpret_cast<uint32_t*>(d_occupancyGrid.data());

    measure(stream, m_profile_stats.lateOccupancyUpdateBitgrid, [&](){
    updateOccupancyGrid<<<gs, BS, 0, stream>>>(
        d_occupancy_grid_32,
        d_masterOccupancyGrid.data(),
        m_render_buffers.d_sum.data(),
        N, G, S,
        m_opts.minDensityThreshold,
        (m_opts.aabbMax.x - m_opts.aabbMin.x) / (float)m_opts.gridResolution.x
    );
    });

    measure(stream, m_profile_stats.lateOccupancyUpdateMipmap, [&](){
        buildMipmaps(stream, N);
    });

}

void InstantNerf::earlyOccupancyGridUpdate(cudaStream_t stream) {
    int G = m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z;
    int N = m_opts.numCascades * G;
    int S = pyramidCellsPerCascade(m_opts.gridResolution, m_opts.levelsMipmap);
    constexpr int BS = 256;
    int gs = (N + BS - 1) / BS;

    cudaMemsetAsync(m_render_buffers.d_tmp_grid.data(), 0, N * sizeof(float), stream);

    // Sample + evaluate density one cascade at a time so the transient buffers stay sized for a
    // single cascade (G cells). Each cascade deterministically covers its own G level-0 cells
    // (dense range [c*G, c*G + G)); cell ranges are disjoint so the shared tmp grid is safe.
    int cgs = (G + BS - 1) / BS;
    int paddedBatchSize = (G + 15) & ~15;
    for (int c = 0; c < m_opts.numCascades; ++c) {
        int cellOffset = c * G;

        measure(stream, m_profile_stats.earlyOccupancySampleTime, [&](){
        sampleAABB<<<cgs, BS, 0, stream>>>(
            m_render_buffers.d_occupancy_samples.data(),
            m_opts.aabbMin, m_opts.aabbMax, m_opts.gridResolution,
            G, G, cellOffset, (unsigned long long)m_trainSteps + (unsigned long long)c);
        });

        measure(stream, m_profile_stats.earlyOccupancyDensityFwd, [&](){
        m_densityMLP->inference(m_render_buffers.d_occupancy_samples.data(), m_render_buffers.d_density_out.data(), paddedBatchSize, stream);
        });

        measure(stream, m_profile_stats.earlyOccupancyUpdateTmpGrid, [&](){
        updateTmpGrid<<<cgs, BS, 0, stream>>>(
            m_render_buffers.d_occupancy_samples.data(),
            m_render_buffers.d_density_out.data(),
            m_render_buffers.d_tmp_grid.data(),
            m_opts.densityBias,
            m_opts.aabbMin,
            m_opts.aabbMax,
            m_opts.gridResolution,
            G
        );
        });
    }

    measure(stream, m_profile_stats.earlyOccupancyUpdateMasterGrid, [&](){
    updateMasterGrid<<<gs, BS, 0, stream>>>(m_render_buffers.d_tmp_grid.data(), d_masterOccupancyGrid.data(), m_opts.decayValue, N);
    });

    measure(stream, m_profile_stats.earlyOccupancyComputeSum, [&](){
    computeSum(m_render_buffers.d_sum.data(), m_render_buffers.d_temp_storage, m_render_buffers.temp_storage_bytes, d_masterOccupancyGrid.data(), N);
    });

    uint32_t* d_occupancy_grid_32 = reinterpret_cast<uint32_t*>(d_occupancyGrid.data());

    measure(stream, m_profile_stats.earlyOccupancyUpdateBitgrid, [&](){
    updateOccupancyGrid<<<gs, BS, 0, stream>>>(
        d_occupancy_grid_32,
        d_masterOccupancyGrid.data(),
        m_render_buffers.d_sum.data(),
        N, G, S,
        m_opts.minDensityThreshold,
        (m_opts.aabbMax.x - m_opts.aabbMin.x) / (float)m_opts.gridResolution.x
    );
    });

    measure(stream, m_profile_stats.earlyOccupancyUpdateMipmap, [&](){
        buildMipmaps(stream, N);
    });
}

void InstantNerf::buildMipmaps(cudaStream_t stream, int level0_cells) {
    // Each cascade owns a contiguous per-cascade pyramid region [c*S, c*S + S) in the bitgrid,
    // where S = level-0 + all mip levels of one cascade. updateOccupancyGrid has already written
    // every cascade's level-0 bits; here we build the higher mip levels independently per cascade.
    // No pre-zeroing is needed: every mip word is fully overwritten by bitfield_max_pool, since
    // each level's cell count is a multiple of 32 for the supported resolutions.
    int S = pyramidCellsPerCascade(m_opts.gridResolution, m_opts.levelsMipmap);  // cells per cascade pyramid
    size_t cascade_stride_bytes = (size_t)S / 8;  // S is a multiple of 32 -> exact

    for (int c = 0; c < m_opts.numCascades; ++c) {
        uint8_t* current_level_ptr = d_occupancyGrid.data() + c * cascade_stride_bytes;
        uint3 current_res = m_opts.gridResolution;

        for (int l = 0; l < m_opts.levelsMipmap - 1; ++l) {
            uint3 next_res = make_uint3(current_res.x / 2, current_res.y / 2, current_res.z / 2);
            int next_cells = next_res.x * next_res.y * next_res.z;
            uint32_t current_level_bytes = (current_res.x * current_res.y * current_res.z) / 8;
            uint8_t* next_level_ptr = current_level_ptr + current_level_bytes;

            int threads = 256;
            int blocks = (next_cells + threads - 1) / threads;

            bitfield_max_pool<<<blocks, threads, 0, stream>>>(
                current_level_ptr,
                reinterpret_cast<uint32_t*>(next_level_ptr),
                next_res
            );

            current_level_ptr = next_level_ptr;
            current_res = next_res;
        }
    }
}

void InstantNerf::save(const std::string& filename) {
    if(!m_renderinit) throw std::runtime_error("InstantNerf not initialized, cannot save.");
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open file for saving: " + filename);

    const char magic[6] = {'I','N','E','R','F','1'};
    out.write(magic, 6);

    out.write(reinterpret_cast<const char*>(&m_opts), sizeof(NerfOptions));

    int hashGridElements = m_opts.numLevels * m_opts.hashTableSize * m_opts.featuresPerLevel;
    std::vector<float> h_hash(hashGridElements);
    m_densityMLP->saveHashgrid(h_hash.data());
    out.write(reinterpret_cast<const char*>(h_hash.data()), hashGridElements * sizeof(float));

    int d_in = m_opts.numLevels * m_opts.featuresPerLevel;
    int d_out = 16;
    int d_hid = m_opts.densityHiddenDim;
    int numLayers = m_opts.densityNumLayers;
    int d_total_weights = 0;
    int d_total_biases = 0;
    for(int l=0; l<numLayers; l++) {
        int in_d = (l == 0) ? d_in : d_hid;
        int out_d = (l == numLayers - 1) ? d_out : d_hid;
        d_total_weights += in_d * out_d;
        d_total_biases += out_d;
    }
    std::vector<float> h_density_w(d_total_weights);
    std::vector<float> h_density_b(d_total_biases);
    m_densityMLP->saveWeights(h_density_w.data(), h_density_b.data());
    out.write(reinterpret_cast<const char*>(h_density_w.data()), d_total_weights * sizeof(float));
    out.write(reinterpret_cast<const char*>(h_density_b.data()), d_total_biases * sizeof(float));

    int c_in = 32;
    int c_out = 16;
    int c_hid = m_opts.colorHiddenDim;
    int c_numLayers = m_opts.colorNumLayers;
    int c_total_weights = 0;
    int c_total_biases = 0;
    for(int l=0; l<c_numLayers; l++) {
        int in_d = (l == 0) ? c_in : c_hid;
        int out_d = (l == c_numLayers - 1) ? c_out : c_hid;
        c_total_weights += in_d * out_d;
        c_total_biases += out_d;
    }
    std::vector<float> h_color_w(c_total_weights);
    std::vector<float> h_color_b(c_total_biases);
    m_colorMLP->saveWeights(h_color_w.data(), h_color_b.data());
    out.write(reinterpret_cast<const char*>(h_color_w.data()), c_total_weights * sizeof(float));
    out.write(reinterpret_cast<const char*>(h_color_b.data()), c_total_biases * sizeof(float));

    // Cascade-aware sizing: must match init() exactly. The occupancy grids span all
    // numCascades cascades; sizing by a single cascade silently drops the outer ones
    // (background renders empty after a reload).
    int occupancyGridCells = 0;
    for (int j = 0; j < m_opts.numCascades; j++) {
        uint3 res = make_uint3(m_opts.gridResolution.x, m_opts.gridResolution.y, m_opts.gridResolution.z);
        for (int i = 0; i < m_opts.levelsMipmap; i++) {
            occupancyGridCells += res.x * res.y * res.z;
            res = make_uint3(res.x / 2, res.y / 2, res.z / 2);
        }
    }
    int occupancyGridBytes = (occupancyGridCells + 7)/ 8;
    int masterGridCells = m_opts.numCascades * (m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z);

    std::vector<float> h_masterOccupancyGrid(masterGridCells);
    d_masterOccupancyGrid.copyHost(h_masterOccupancyGrid.data(), masterGridCells);
    out.write(reinterpret_cast<const char*>(h_masterOccupancyGrid.data()), masterGridCells * sizeof(float));

    std::vector<uint8_t> h_occupancyGrid(occupancyGridBytes);
    d_occupancyGrid.copyHost(h_occupancyGrid.data(), occupancyGridBytes);
    out.write(reinterpret_cast<const char*>(h_occupancyGrid.data()), occupancyGridBytes * sizeof(uint8_t));
}

void InstantNerf::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file for loading: " + filename);

    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 6) != "INERF1") throw std::runtime_error("Invalid INERF file format");

    NerfOptions loadedOpts;
    in.read(reinterpret_cast<char*>(&loadedOpts), sizeof(NerfOptions));
    
    init(loadedOpts);
    
    int hashGridElements = m_opts.numLevels * m_opts.hashTableSize * m_opts.featuresPerLevel;
    std::vector<float> h_hash(hashGridElements);
    in.read(reinterpret_cast<char*>(h_hash.data()), hashGridElements * sizeof(float));
    m_densityMLP->loadHashgrid(h_hash.data());

    int d_in = m_opts.numLevels * m_opts.featuresPerLevel;
    int d_out = 16;
    int d_hid = m_opts.densityHiddenDim;
    int numLayers = m_opts.densityNumLayers;
    int d_total_weights = 0;
    int d_total_biases = 0;
    for(int l=0; l<numLayers; l++) {
        int in_d = (l == 0) ? d_in : d_hid;
        int out_d = (l == numLayers - 1) ? d_out : d_hid;
        d_total_weights += in_d * out_d;
        d_total_biases += out_d;
    }
    std::vector<float> h_density_w(d_total_weights);
    std::vector<float> h_density_b(d_total_biases);
    in.read(reinterpret_cast<char*>(h_density_w.data()), d_total_weights * sizeof(float));
    in.read(reinterpret_cast<char*>(h_density_b.data()), d_total_biases * sizeof(float));
    m_densityMLP->loadWeights(h_density_w.data(), h_density_b.data());

    int c_in = 32;
    int c_out = 16;
    int c_hid = m_opts.colorHiddenDim;
    int c_numLayers = m_opts.colorNumLayers;
    int c_total_weights = 0;
    int c_total_biases = 0;
    for(int l=0; l<c_numLayers; l++) {
        int in_d = (l == 0) ? c_in : c_hid;
        int out_d = (l == c_numLayers - 1) ? c_out : c_hid;
        c_total_weights += in_d * out_d;
        c_total_biases += out_d;
    }
    std::vector<float> h_color_w(c_total_weights);
    std::vector<float> h_color_b(c_total_biases);
    in.read(reinterpret_cast<char*>(h_color_w.data()), c_total_weights * sizeof(float));
    in.read(reinterpret_cast<char*>(h_color_b.data()), c_total_biases * sizeof(float));
    m_colorMLP->loadWeights(h_color_w.data(), h_color_b.data());

    // Cascade-aware sizing: must match init()/save() exactly, or the outer cascades'
    // occupancy grids are read short and left empty (blank background on reload).
    int occupancyGridCells = 0;
    for (int j = 0; j < m_opts.numCascades; j++) {
        uint3 res = make_uint3(m_opts.gridResolution.x, m_opts.gridResolution.y, m_opts.gridResolution.z);
        for (int i = 0; i < m_opts.levelsMipmap; i++) {
            occupancyGridCells += res.x * res.y * res.z;
            res = make_uint3(res.x / 2, res.y / 2, res.z / 2);
        }
    }
    int occupancyGridBytes = (occupancyGridCells + 7)/ 8;
    int masterGridCells = m_opts.numCascades * (m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z);

    std::vector<float> h_masterOccupancyGrid(masterGridCells);
    in.read(reinterpret_cast<char*>(h_masterOccupancyGrid.data()), masterGridCells * sizeof(float));
    cudaMemcpy(d_masterOccupancyGrid.data(), h_masterOccupancyGrid.data(), masterGridCells * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<uint8_t> h_occupancyGrid(occupancyGridBytes);
    in.read(reinterpret_cast<char*>(h_occupancyGrid.data()), occupancyGridBytes * sizeof(uint8_t));
    cudaMemcpy(d_occupancyGrid.data(), h_occupancyGrid.data(), occupancyGridBytes * sizeof(uint8_t), cudaMemcpyHostToDevice);
}
