#include "InstantNerf.h"
#include "otherKernels.cu"
#include "../TinyMLP/TinyMLPHashGrid.h"
#include <algorithm>

void sampleNonUniform(
    float* d_samples,
    uint8_t* d_occupancy_grid,
    int* d_activeCellIndices,
    int* d_numActiveCells,
    int totalCells,
    float3 aabbMin,
    float3 aabbMax,
    uint3 gridResolution,
    int N,
    unsigned long long seed = 42ULL,
    cudaStream_t stream = 0
) {
    cudaMemsetAsync(d_numActiveCells, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (totalCells + threads - 1) / threads;
    extractActiveCells<<<blocks, threads, 0, stream>>>(
        d_occupancy_grid, d_activeCellIndices, d_numActiveCells, totalCells
    );

    blocks = (N + threads - 1) / threads;
    sampleOccupiedAABB<<<blocks, threads, 0, stream>>>(
        d_samples, d_activeCellIndices, d_numActiveCells, 
        aabbMin, aabbMax, gridResolution, N, seed
    );
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

    printGroup("lateOccupancyGridUpdate", m_profile_stats.lateOccupancyUpdateTime, 1);
    printTracker("lateOccupancySampleUniformTime", m_profile_stats.lateOccupancySampleUniformTime, 2);
    printTracker("lateOccupancySampleNonUniformTime", m_profile_stats.lateOccupancySampleNonUniformTime, 2);
    printGroup("lateOccupancyInferenceTime", m_profile_stats.lateOccupancyInferenceTime, 2);
    printTracker("lateOccupancyDensityFwd", m_profile_stats.lateOccupancyDensityFwd, 3);
    printTracker("lateOccupancyUpdateTmpGrid", m_profile_stats.lateOccupancyUpdateTmpGrid, 3);
    printTracker("lateOccupancyUpdateMasterGrid", m_profile_stats.lateOccupancyUpdateMasterGrid, 2);
    printTracker("lateOccupancyComputeSum", m_profile_stats.lateOccupancyComputeSum, 2);
    printTracker("lateOccupancyUpdateBitgrid", m_profile_stats.lateOccupancyUpdateBitgrid, 2);

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
    if (!m_initialized) {
        fprintf(stderr, "[InstantNerf] Error: init() must be called before initRenderBuffers()\n");
        return;
    }
    m_render_buffers.d_rays_d_inv_chunk = DeviceBuffer<float3>(m_opts.rayChunkSize);
    m_render_buffers.d_nears_chunk = DeviceBuffer<float>(m_opts.rayChunkSize);
    m_render_buffers.d_fars_chunk = DeviceBuffer<float>(m_opts.rayChunkSize);

    if (m_opts.rayChunkSize % 16 != 0) {
        fprintf(stderr, "[InstantNerf] Error: rayChunkSize must be a multiple of 16");
        return;
    }

    uint32_t totalHits = m_opts.rayChunkSize * MAX_HITS;

    m_render_buffers.d_sparse_ts = DeviceBuffer<float>(totalHits);
    m_render_buffers.d_num_steps = DeviceBuffer<uint32_t>(m_opts.rayChunkSize);
    m_render_buffers.d_ray_offsets = DeviceBuffer<uint32_t>(m_opts.rayChunkSize);
    m_render_buffers.d_dense_sparse_indices = DeviceBuffer<uint32_t>(totalHits);
    m_render_buffers.d_mlp_positions = DeviceBuffer<float>(totalHits * 4);
    m_render_buffers.d_ray_indices = DeviceBuffer<uint32_t>(totalHits);
    m_render_buffers.d_t_sorted = DeviceBuffer<float>(totalHits);
    uint32_t occupancyGridCells = m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z;
    uint32_t max_density_out = std::max((uint32_t)(16 * m_opts.batchSize), (uint32_t)(16 * occupancyGridCells));
    m_render_buffers.d_density_out = DeviceBuffer<float>(max_density_out);
    m_render_buffers.d_color_input = DeviceBuffer<half>(32 * m_opts.batchSize);
    m_render_buffers.d_color_output = DeviceBuffer<float>(3 * m_opts.batchSize);
    m_render_buffers.d_density_sigma = DeviceBuffer<float>(totalHits);
    m_render_buffers.d_rgb_output = DeviceBuffer<float>(totalHits * 3);

    m_render_buffers.d_render_rgb_chunk = DeviceBuffer<float>(3 * m_opts.rayChunkSize);
    m_render_buffers.d_render_depth_chunk = DeviceBuffer<float>(m_opts.rayChunkSize);
    m_render_buffers.d_phi_chunk = DeviceBuffer<float>(3 * m_opts.rayChunkSize);
    m_render_buffers.d_out_count = DeviceBuffer<uint32_t>(1);
    m_render_buffers.d_active_ray_indices = DeviceBuffer<uint32_t>(m_opts.batchSize);
    m_render_buffers.d_batch_position = DeviceBuffer<float>(3 * m_opts.batchSize);
    m_render_buffers.d_batch_direction = DeviceBuffer<float3>(m_opts.batchSize);

    m_render_buffers.d_current_t = DeviceBuffer<float>(m_opts.rayChunkSize);
    m_render_buffers.d_current_rgb = DeviceBuffer<float>(3 * m_opts.rayChunkSize);
    m_render_buffers.d_custom_color_grad = DeviceBuffer<half>(3 * m_opts.batchSize);
    m_render_buffers.d_custom_density_grad = DeviceBuffer<half>(16 * m_opts.batchSize);
    m_render_buffers.d_tmpsigma = DeviceBuffer<float>(m_opts.batchSize);
    m_render_buffers.d_color_dx_out = DeviceBuffer<half>(32 * m_opts.batchSize);
    
    m_render_buffers.d_occupancy_samples = DeviceBuffer<float>(4 * m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z);
    m_render_buffers.d_tmp_grid = DeviceBuffer<float>(m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z);
    m_render_buffers.d_sum = DeviceBuffer<float>(1);

    m_render_buffers.d_numActiveCells = DeviceBuffer<int>(1);
    m_render_buffers.d_activeCellIndices = DeviceBuffer<int>(m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z);

    cub::DeviceReduce::Sum(m_render_buffers.d_temp_storage, m_render_buffers.temp_storage_bytes, d_masterOccupancyGrid.data(), m_render_buffers.d_sum.data(), m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z);
    cudaMalloc(&m_render_buffers.d_temp_storage, m_render_buffers.temp_storage_bytes);

    m_render_buffers.h_ray_offsets.reserve(m_opts.rayChunkSize);
}

void InstantNerf::init(const NerfOptions& opts) {
    m_opts = opts;
    
    int occupancyGridCells = opts.gridResolution.x * opts.gridResolution.y * opts.gridResolution.z;
    int occupancyGridBytes = (occupancyGridCells + 7)/ 8;

    d_occupancyGrid = DeviceBuffer<uint8_t>(occupancyGridBytes);
    d_occupancyGrid.fill(0);
    d_masterOccupancyGrid = DeviceBuffer<float>(occupancyGridCells);

    {
        constexpr int BS = 256;
        int gs = (occupancyGridCells + BS - 1) / BS;
        fill_float_kernel<<<gs, BS>>>(d_masterOccupancyGrid.data(), 0.0f, occupancyGridCells);
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

    m_densityMLP = new TinyMLPHashGrid(densityOpts, opts.batchSize, opts.gridResolution.x * opts.gridResolution.y * opts.gridResolution.z);

    MLPOption colorOpts;
    colorOpts.inputDim = 32;
    colorOpts.hiddenDim = opts.colorHiddenDim;
    colorOpts.outputDim = 3;
    colorOpts.numLayers = opts.colorNumLayers;
    colorOpts.activationType = ACT_RELU;
    colorOpts.outputActivation = OUT_ACT_SIGMOID;

    m_colorMLP = new TinyMLP(colorOpts, opts.batchSize);

    m_profile_stats.init();
    m_profile_stats.pendingTimers.reserve(100000);
    m_initialized = true;

    initRenderBuffers();
    earlyOccupancyGridUpdate();
}

void InstantNerf::trainWithRays(
    const float3* d_rays_o,
    const float3* d_rays_d,
    const float* d_rgb_true,
    uint32_t numRays,
    int& trainStepCount,
    float* d_rgb_out,
    cudaStream_t stream
) { 
    if(!m_initialized) {
        fprintf(stderr, "[InstantNerf] Error: init() must be called before trainWithRays()\n");
        return;
    }

    for (int offset = 0; offset < numRays; offset += m_opts.rayChunkSize) {
        int currentChunkRays = std::min(m_opts.rayChunkSize, (int)numRays - offset);
        const float3* chunk_o = d_rays_o + offset;
        const float3* chunk_d = d_rays_d + offset;

        {
            constexpr int BS = 256;
            int gs = (currentChunkRays + BS - 1) / BS;

            measure(stream, m_profile_stats.processRaysTime, [&]() {
            compute_ray_aabb_inv_kernel<<<gs, BS, 0, stream>>>(
                currentChunkRays,
                chunk_o,
                chunk_d,
                m_opts.aabbMin, m_opts.aabbMax,
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
        }
        
        uint32_t totalHits = 0;
        uint32_t highestHits = 0;
        
        std::vector<uint32_t> h_num_steps(currentChunkRays);
        m_render_buffers.d_num_steps.copyHost(h_num_steps.data(), currentChunkRays);

        for (uint32_t i = 0; i < currentChunkRays; ++i) {
            totalHits += h_num_steps[i];
            if(h_num_steps[i] > highestHits) highestHits = h_num_steps[i];
        }

        std::vector<uint32_t> boundaries = get_chunk_boundaries_cpu(m_render_buffers.h_ray_offsets, m_render_buffers.d_ray_offsets.data(), currentChunkRays, totalHits, m_opts.batchSize, stream);

        for (uint32_t b_offset = 0; b_offset < totalHits; b_offset += m_opts.batchSize) {
            uint32_t b_size = std::min(static_cast<uint32_t>(m_opts.batchSize), totalHits - b_offset);
            uint32_t padded_b_size = (b_size + 15) & ~15;

            measure(stream, m_profile_stats.inferenceDensityFwd, [&](){
            m_densityMLP->inference(m_render_buffers.d_mlp_positions.data() + b_offset * 4, m_render_buffers.d_density_out.data(), padded_b_size, stream);
            });

            measure(stream, m_profile_stats.inferenceGatherSH, [&]()
            {
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
            m_render_buffers.d_rgb_output.data(),
            d_rgb_true ? (d_rgb_true + offset * 3) : nullptr,
            m_render_buffers.d_render_rgb_chunk.data(),
            m_render_buffers.d_render_depth_chunk.data(),
            m_render_buffers.d_phi_chunk.data(),
            m_opts.bgColor,
            stream
        );
        });

        measure(stream, m_profile_stats.fillFloatKernel, [&](){
        fill_float_kernel<<<(currentChunkRays + 255)/256, 256, 0, stream>>>(m_render_buffers.d_current_t.data(), 1.0f, currentChunkRays);
        fill_float_kernel<<<(currentChunkRays * 3 + 255)/256, 256, 0, stream>>>(m_render_buffers.d_current_rgb.data(), 0.0f, currentChunkRays * 3);
        });

        if (d_rgb_out != nullptr) {
            cudaMemcpyAsync(d_rgb_out + offset * 3, m_render_buffers.d_render_rgb_chunk.data(), currentChunkRays * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        
        if (m_trainSteps < 256) {
            earlyOccupancyGridUpdate(stream);
        } else {
            lateOccupancyGridUpdate(stream);
        }
        
        measure(stream, m_profile_stats.trainZeroGrad, [&](){
            m_densityMLP->zero_grad(stream);
            m_colorMLP->zero_grad(stream);
        });
        
        for (int step = 0; step < boundaries.size(); step++) {
            uint32_t start_ray = (step == 0) ? 0 : boundaries[step - 1];
            uint32_t end_ray   = boundaries[step];
            uint32_t start_hit = m_render_buffers.h_ray_offsets[start_ray];
            uint32_t end_hit;
            if (end_ray == currentChunkRays) {
                end_hit = totalHits;
            } else {
                end_hit = m_render_buffers.h_ray_offsets[end_ray];
            }

            uint32_t chunk_hits = end_hit - start_hit;
            uint32_t padded_batch_size = (chunk_hits + 15) & ~15;
            float* chunk_mlp_positions = m_render_buffers.d_mlp_positions.data() + (start_hit * 4);

            cudaMemsetAsync(m_render_buffers.d_custom_color_grad.data(), 0, padded_batch_size * 3 * sizeof(half), stream);
            cudaMemsetAsync(m_render_buffers.d_custom_density_grad.data(), 0, padded_batch_size * 16 * sizeof(half), stream);

            measure(stream, m_profile_stats.trainDensityFwd, [&](){
            m_densityMLP->forward(chunk_mlp_positions, m_render_buffers.d_density_out.data(), padded_batch_size, stream);
            });

            constexpr int BS = 256;
            int gs = (chunk_hits + BS - 1) / BS;
            measure(stream, m_profile_stats.trainComputeSH, [&](){
            compute_SH_gather<<<gs, BS, 0, stream>>>(
                chunk_d, m_render_buffers.d_ray_indices.data(),
                start_hit, chunk_hits, m_opts.densityBias, m_render_buffers.d_density_out.data(),
                m_render_buffers.d_color_input.data(),
                m_render_buffers.d_density_sigma.data()
            );
            });

            measure(stream, m_profile_stats.trainColorFwd, [&](){
            m_colorMLP->forward(m_render_buffers.d_color_input.data(), m_render_buffers.d_color_output.data(), padded_batch_size, stream);
            });

            int ray_count = end_ray - start_ray;
            int gs_color = (ray_count + 255) / 256;
            measure(stream, m_profile_stats.trainColorGrad, [&](){
            compute_color_grad<<<gs_color, 256, 0, stream>>>(
                end_ray - start_ray,
                start_ray,
                m_render_buffers.d_num_steps.data(),
                m_render_buffers.d_ray_offsets.data(),
                m_render_buffers.d_t_sorted.data(),
                m_render_buffers.d_density_out.data(),
                m_render_buffers.d_color_output.data(),
                m_render_buffers.d_phi_chunk.data(),
                m_render_buffers.d_render_rgb_chunk.data(),
                m_render_buffers.d_custom_color_grad.data(),
                m_render_buffers.d_tmpsigma.data(),
                m_opts.lossScale,
                m_opts.densityBias
            );
            });

            measure(stream, m_profile_stats.trainColorBwd, [&](){
            m_colorMLP->backward(m_render_buffers.d_custom_color_grad.data(), m_render_buffers.d_color_dx_out.data(), padded_batch_size, stream);
            });

            int gs_density = (chunk_hits + 255) / 256;
            measure(stream, m_profile_stats.trainDensityGrad, [&](){
            compute_density_grad<<<gs_density, 256, 0, stream>>>(
                chunk_hits, 
                m_render_buffers.d_tmpsigma.data(), 
                m_render_buffers.d_color_dx_out.data(), 
                m_render_buffers.d_custom_density_grad.data()
            );
            });

            measure(stream, m_profile_stats.trainDensityBwd, [&](){
            m_densityMLP->backward(m_render_buffers.d_custom_density_grad.data(), padded_batch_size, stream);
            });
        }

        trainStepCount++;
        m_trainSteps++;

        measure(stream, m_profile_stats.trainColorOpt, [&](){
        m_colorMLP->step(m_opts.learningRate, m_opts.beta1, m_opts.beta2, m_opts.epsilon, m_opts.lossScale, stream);
        });

        measure(stream, m_profile_stats.trainDensityOpt, [&](){
        m_densityMLP->step(m_opts.learningRate, m_opts.beta1, m_opts.beta2, m_opts.epsilon, m_opts.lossScale, stream);
        });


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
    if(!m_initialized) {
        fprintf(stderr, "[InstantNerf] Error: init() must be called before trainWithRays()\n");
        return;
    }

    for (int offset = 0; offset < numRays; offset += m_opts.rayChunkSize) {
        int currentChunkRays = std::min(m_opts.rayChunkSize, (int)numRays - offset);
        const float3* chunk_o = d_rays_o + offset;
        const float3* chunk_d = d_rays_d + offset;

        {
            constexpr int BS = 256;
            int gs = (currentChunkRays + BS - 1) / BS;

            compute_ray_aabb_inv_kernel<<<gs, BS, 0, stream>>>(
                currentChunkRays,
                chunk_o,
                chunk_d,
                m_opts.aabbMin, m_opts.aabbMax,
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
                m_render_buffers.d_sparse_ts.data(), 
                m_render_buffers.d_num_steps.data(),
                m_render_buffers.d_ray_offsets.data(),
                m_render_buffers.d_dense_sparse_indices.data(),
                m_render_buffers.d_mlp_positions.data(), 
                m_render_buffers.d_ray_indices.data(), 
                m_render_buffers.d_t_sorted.data(),
                stream
            );
        }
        
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

            m_densityMLP->inference(m_render_buffers.d_mlp_positions.data() + b_offset * 4, m_render_buffers.d_density_out.data(), padded_b_size, stream);

            {
                constexpr int BS = 256;
                int gs = (b_size + BS - 1) / BS;

                compute_SH_gather<<<gs, BS, 0, stream>>>(
                    chunk_d, m_render_buffers.d_ray_indices.data(), 
                    b_offset, b_size, m_opts.densityBias,
                    m_render_buffers.d_density_out.data(), 
                    m_render_buffers.d_color_input.data(),
                    m_render_buffers.d_density_sigma.data() + b_offset
                );
            }

            m_colorMLP->inference(
                m_render_buffers.d_color_input.data(),
                m_render_buffers.d_rgb_output.data() + b_offset * 3,
                padded_b_size,
                stream
            );
        }

        launchVolumeRendering(
            currentChunkRays,
            m_render_buffers.d_ray_offsets.data(),
            m_render_buffers.d_num_steps.data(),
            m_render_buffers.d_t_sorted.data(),
            m_render_buffers.d_density_sigma.data(),
            m_render_buffers.d_rgb_output.data(),
            nullptr,
            m_render_buffers.d_render_rgb_chunk.data(),
            m_render_buffers.d_render_depth_chunk.data(),
            m_render_buffers.d_phi_chunk.data(),
            m_opts.bgColor,
            stream
        );

        if (d_rgb_out != nullptr) {
            cudaMemcpyAsync(d_rgb_out + offset * 3, m_render_buffers.d_render_rgb_chunk.data(), currentChunkRays * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
    }
}

void InstantNerf::lateOccupancyGridUpdate(cudaStream_t stream) {
    int N = m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z;
    int halfN = N / 2;

    constexpr int BS = 256;
    int gs = (N + BS - 1) / BS;

    cudaMemsetAsync(m_render_buffers.d_tmp_grid.data(), 0, N * sizeof(float), stream);

    int occupiedSamples = halfN / 2;
    int uniformSamples = halfN - occupiedSamples;

    measure(stream, m_profile_stats.lateOccupancySampleNonUniformTime, [&](){
    sampleNonUniform(
        m_render_buffers.d_occupancy_samples.data(),
        d_occupancyGrid.data(),
        m_render_buffers.d_activeCellIndices.data(),
        m_render_buffers.d_numActiveCells.data(),
        N,
        m_opts.aabbMin,
        m_opts.aabbMax,
        m_opts.gridResolution,
        occupiedSamples,
        (unsigned long long)m_trainSteps,
        stream
    );
    });

    int uniformGs = (uniformSamples + BS - 1) / BS;
    measure(stream, m_profile_stats.lateOccupancySampleUniformTime, [&](){
    sampleAABB<<<uniformGs, BS, 0, stream>>>(
        m_render_buffers.d_occupancy_samples.data() + occupiedSamples * 4,
        m_opts.aabbMin, m_opts.aabbMax, uniformSamples,
        (unsigned long long)(m_trainSteps + 1)
    );
    });

    int paddedBatchSize = (halfN + 15) & ~15;

    measure(stream, m_profile_stats.lateOccupancyDensityFwd, [&](){
    m_densityMLP->inference(m_render_buffers.d_occupancy_samples.data(), m_render_buffers.d_density_out.data(), paddedBatchSize, stream);
    });

    measure(stream, m_profile_stats.lateOccupancyUpdateTmpGrid, [&](){
    updateTmpGrid<<<(halfN + BS - 1) / BS, BS, 0, stream>>>(
        m_render_buffers.d_occupancy_samples.data(),
        m_render_buffers.d_density_out.data(),
        m_render_buffers.d_tmp_grid.data(),
        m_opts.densityBias,
        m_opts.aabbMin,
        m_opts.aabbMax,
        m_opts.gridResolution,
        halfN
    );
    });

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
        N
    );
    });
}

void InstantNerf::earlyOccupancyGridUpdate(cudaStream_t stream) {
    int N = m_opts.gridResolution.x * m_opts.gridResolution.y * m_opts.gridResolution.z;
    constexpr int BS = 256;
    int gs = (N + BS - 1) / BS;

    cudaMemsetAsync(m_render_buffers.d_tmp_grid.data(), 0, N * sizeof(float), stream);

    measure(stream, m_profile_stats.earlyOccupancySampleTime, [&](){
    sampleAABB<<<gs, BS, 0, stream>>>(m_render_buffers.d_occupancy_samples.data(), m_opts.aabbMin, m_opts.aabbMax, N, (unsigned long long)m_trainSteps);
    });

    int paddedBatchSize = (N + 15) & ~15;

    measure(stream, m_profile_stats.earlyOccupancyDensityFwd, [&](){
    m_densityMLP->inference(m_render_buffers.d_occupancy_samples.data(), m_render_buffers.d_density_out.data(), paddedBatchSize, stream);
    });

    measure(stream, m_profile_stats.earlyOccupancyUpdateTmpGrid, [&](){
    updateTmpGrid<<<(N + BS - 1) / BS, BS, 0, stream>>>(
        m_render_buffers.d_occupancy_samples.data(),
        m_render_buffers.d_density_out.data(),
        m_render_buffers.d_tmp_grid.data(),
        m_opts.densityBias,
        m_opts.aabbMin,
        m_opts.aabbMax,
        m_opts.gridResolution,
        N
    );
    });

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
        N
    );
    });
}
