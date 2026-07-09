#include "BakedNerf.h"
#include "bakedNerfKernels.cu"
#undef CUDA_CHECK
#include "../TinyMLP/TinyMLP.h"
#include "../TinyMLP/EmbeddingTable.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stdexcept>

void generateFibonacciSphere(int N, std::vector<float3>& directions) {
    if (N <= 0) return;
    directions.reserve(N);

    constexpr float PI = 3.14159265358979323846f;

    const float phi = static_cast<float>(PI) * (3.0f - std::sqrt(5.0f));

    for (int i = 0; i < N; ++i) {
        float y = (N == 1) ? 1.0f : 1.0f - (static_cast<float>(i) / static_cast<float>(N - 1)) * 2.0f; 
        
        float r = std::sqrt(std::max(0.0f, 1.0f - y * y)); 
        
        float theta = phi * static_cast<float>(i);
        
        float x = std::cos(theta) * r;
        float z = std::sin(theta) * r;
        directions.push_back({x, y, z});
    }
}


BakedNerf::~BakedNerf() {
    delete m_deferredMLP;
    delete m_voxelDiffuse;
    delete m_voxelFeatures;
}

void BakedNerf::init(const BakeOptions& opts) {
    m_opts = opts;

    MLPOption deferredOpts;
    deferredOpts.activationType = ACT_RELU;
    deferredOpts.outputActivation = OUT_ACT_NONE;  

    deferredOpts.inputDim = deferredInDim();          
    deferredOpts.hiddenDim = m_opts.deferredHidden;
    deferredOpts.outputDim = 3;
    deferredOpts.numLayers = m_opts.deferredLayers;

    m_deferredMLP = new TinyMLP(deferredOpts, 1 << 20, 1 << 20);

    generateFibonacciSphere(m_opts.bakeDiffuseN, m_viewDirs);
}

void BakedNerf::diagonstic(InstantNerf& teacher) {
    m_teacherOpts = teacher.options();

    const uint3 gr = m_teacherOpts.gridResolution;
    if (m_opts.voxelGridResolution.x != gr.x * SPARSE_B ||
        m_opts.voxelGridResolution.y != gr.y * SPARSE_B ||
        m_opts.voxelGridResolution.z != gr.z * SPARSE_B) {
        throw std::runtime_error(
            "BakedNerf::distil: voxelGridResolution must equal gridResolution * SPARSE_B");
    }

    m_bakeThreshold = (m_opts.sigmaThreshold >= 0.0f) ? m_opts.sigmaThreshold
                                                      : m_teacherOpts.minDensityThreshold;
    m_diag = BakeDiagnostics{};
    m_diag.threshold = m_bakeThreshold;
    m_occupancyGrid = DeviceBuffer<uint8_t>(teacher.occupancyBytes());
    teacher.buildOccupancyBitgrid(m_occupancyGrid.data(), m_bakeThreshold);

    uint8_t* cpuGrid = new uint8_t[teacher.occupancyBytes()];
    m_occupancyGrid.copyHost(cpuGrid, teacher.occupancyBytes());

    int levels = m_teacherOpts.levelsMipmap;
    uint3 res = m_teacherOpts.gridResolution;
    int cascadeOffset = 0;
    int baseElems = (int)(res.x * res.y * res.z) / 8;
    for (int i = 0; i < levels; i++) {
        cascadeOffset += (int)(res.x * res.y * res.z);
        res = make_uint3(res.x / 2, res.y / 2, res.z / 2);
    }

    cascadeOffset /= 8;

    int cascades = m_teacherOpts.numCascades;
    int G = (int)(m_teacherOpts.gridResolution.x * m_teacherOpts.gridResolution.y * m_teacherOpts.gridResolution.z);
    long long perCascadeVoxels = (long long)baseElems * 8;   // 128^3 base cells per cascade
    int numFilledVoxels = 0;
    int offset = 0;

    // Collect occupied base-level cells (dense ids = cascade*G + local) while we count them.
    std::vector<int> occupiedCells;
    occupiedCells.reserve(1 << 21);

    printf("threshold %.4f :\n", m_bakeThreshold);
    for (int i = 0; i < cascades; i++) {
        int cascadeFilled = 0;
        for (int j = 0; j < baseElems; j++) {
            uint8_t byte = cpuGrid[offset + j];
            for (int b = 0; b < 8; ++b) {
                if ((byte >> b) & 1) {
                    cascadeFilled++;
                    occupiedCells.push_back(i * G + j * 8 + b);   // local = j*8 + b
                }
            }
        }
        printf("  cascade %d : filled %8d / %lld (%6.3f%%)\n",
               i, cascadeFilled, perCascadeVoxels,
               100.0 * cascadeFilled / (double)perCascadeVoxels);
        m_diag.perCascadeFilled.push_back(cascadeFilled);
        numFilledVoxels += cascadeFilled;
        offset += cascadeOffset;        // full-pyramid bytes per cascade = stride between cascades
    }

    long long totalBaseVoxels = perCascadeVoxels * cascades;   // 128^3 * cascades base cells
    printf("  TOTAL     : filled %8d / %lld (%6.3f%%)\n",
           numFilledVoxels, totalBaseVoxels,
           100.0 * numFilledVoxels / (double)totalBaseVoxels);
    m_diag.totalFilled = numFilledVoxels;   // == occupied blocks

    delete[] cpuGrid;

    const int B = SPARSE_B;
    const int per = B * B * B;                       // 64
    const long long numBlocks = (long long)occupiedCells.size();
    m_numBlocks = (uint32_t)numBlocks;
    m_numVoxels = 0;
    if (numBlocks == 0) { printf("[sub-voxel] no occupied blocks\n"); return; }

    DeviceBuffer<int> d_cellIds((size_t)numBlocks);
    cudaMemcpy(d_cellIds.data(), occupiedCells.data(), (size_t)numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    const int chunkBlocks = 1 << 15;                 // 32768 blocks -> 2,097,152 sub-voxels / chunk
    
    size_t n      = (size_t)chunkBlocks * per;
    size_t padded = (n + 15) & ~15;          // round ROWS up to a multiple of 16

    DeviceBuffer<float> d_pos(padded * 4);   // INPUT: stride-4, padded — this is what the kernel over-reads
    DeviceBuffer<float> d_logit(padded);     // OUTPUT: one logit/row; padded because you pass `padded` as n
    DeviceBuffer<unsigned long long> d_survivors(1);
    DeviceBuffer<unsigned long long> d_hist((size_t)per + 1);
    d_survivors.fill(0);
    d_hist.fill(0);

    // Deduped-vertex count via a global-lattice bitset: 1 bit per physical vertex, so shared
    // boundary vertices (voxel g's s=B-1 face == neighbour g+1's s=0) collapse to one bit and
    // popcount = how many vertices dedup would actually keep. ~28 MB for a 128^3/4-cascade grid.
    const long long Lx = (long long)m_teacherOpts.gridResolution.x * (B - 1) + 1;
    const long long Ly = (long long)m_teacherOpts.gridResolution.y * (B - 1) + 1;
    const long long Lz = (long long)m_teacherOpts.gridResolution.z * (B - 1) + 1;
    const long long totalVertBits = (long long)cascades * Lx * Ly * Lz;
    const size_t bitsetWords = (size_t)((totalVertBits + 31) / 32);
    DeviceBuffer<uint32_t> d_uniqueBitset(bitsetWords);   // sigma-surviving vertices
    DeviceBuffer<uint32_t> d_allBitset(bitsetWords);      // all vertices of occupied voxels (what distil stores)
    d_uniqueBitset.fill(0);
    d_allBitset.fill(0);

    const float baseVoxel = (m_teacherOpts.aabbMax.x - m_teacherOpts.aabbMin.x) / (float)m_teacherOpts.gridResolution.x;
    constexpr int BS = 256;

    for (long long c0 = 0; c0 < numBlocks; c0 += chunkBlocks) {
        int blocksThis = (int)std::min((long long)chunkBlocks, numBlocks - c0);
        int nSub = blocksThis * per;

        int gsPos = (nSub + BS - 1) / BS;
        k_genSubVoxelPositions<<<gsPos, BS>>>(
            d_cellIds.data() + c0, blocksThis, B,
            m_teacherOpts.gridResolution, m_teacherOpts.aabbMin, m_teacherOpts.aabbMax,
            d_pos.data());

        teacher.queryDensityLogit(d_pos.data(), nSub, d_logit.data());

        int gsThr = (blocksThis + BS - 1) / BS;
        k_buildSubVoxelMasks<<<gsThr, BS>>>(
            d_logit.data(), d_cellIds.data() + c0, blocksThis, B,
            m_teacherOpts.gridResolution, m_teacherOpts.densityBias, m_bakeThreshold, baseVoxel,
            nullptr,   // diagnostic pass: don't store the Tier-2 mask structure, only count survivors/hist
            d_survivors.data(), d_hist.data(), d_uniqueBitset.data(), d_allBitset.data());
    }
    cudaDeviceSynchronize();

    unsigned long long survivors = 0;
    std::vector<unsigned long long> hist((size_t)per + 1, 0);
    d_survivors.copyHost(&survivors, 1);
    d_hist.copyHost(hist.data(), (size_t)per + 1);
    m_numVoxels = (uint32_t)survivors;

    // popcount the dedup bitsets -> unique surviving vertices, and unique vertices of occupied voxels
    std::vector<uint32_t> hUniqueBits(bitsetWords), hAllBits(bitsetWords);
    d_uniqueBitset.copyHost(hUniqueBits.data(), bitsetWords);
    d_allBitset.copyHost(hAllBits.data(), bitsetWords);
    unsigned long long uniqueSurvivors = 0, uniqueAll = 0;
    for (uint32_t w : hUniqueBits) { while (w) { w &= w - 1; ++uniqueSurvivors; } }
    for (uint32_t w : hAllBits)    { while (w) { w &= w - 1; ++uniqueAll; } }

    const long long candidateSubVoxels = numBlocks * per;
    m_diag.candidateSubVoxels = candidateSubVoxels;
    m_diag.survivors          = (long long)survivors;
    m_diag.uniqueSurvivors    = (long long)uniqueSurvivors;
    m_diag.uniqueAll          = (long long)uniqueAll;
    m_diag.fillHist           = hist;
    auto bucket = [&](int lo, int hi){ unsigned long long s = 0; for (int k = lo; k <= hi; ++k) s += hist[k]; return s; };

    printf("\n[sub-voxel sigma survival @ B=%d, threshold %.4f]\n", B, m_bakeThreshold);
    printf("  occupied blocks       : %lld\n", numBlocks);
    printf("  candidate sub-voxels  : %lld (blocks x %d)\n", candidateSubVoxels, per);
    printf("  surviving sub-voxels  : %llu  (%.3f%% of candidates, mean %.2f/%d per block)\n",
           survivors, 100.0 * survivors / (double)candidateSubVoxels,
           (double)survivors / (double)numBlocks, per);
    printf("  per-block fill (blocks with N occupied sub-voxels):\n");
    printf("    N=0     : %10llu  (blocks whose 64 sub-centers all missed the bar)\n", hist[0]);
    printf("    N=1-8   : %10llu\n", bucket(1, 8));
    printf("    N=9-16  : %10llu\n", bucket(9, 16));
    printf("    N=17-32 : %10llu\n", bucket(17, 32));
    printf("    N=33-48 : %10llu\n", bucket(33, 48));
    printf("    N=49-63 : %10llu\n", bucket(49, 63));
    printf("    N=64    : %10llu  (fully dense blocks)\n", hist[64]);
    printf("  payload @13B          : %.1f MB   | fit masters+S @32B (<=): %.1f MB\n",
           survivors * 13.0 / (1024.0 * 1024.0), survivors * 32.0 / (1024.0 * 1024.0));
    printf("  unique vertices (dedup): %llu  (%.2fx fewer, saves %.1f%%  ->  payload @13B %.1f MB, fit @32B %.1f MB)\n",
           uniqueSurvivors,
           (double)survivors / (double)std::max(uniqueSurvivors, 1ull),
           100.0 * (1.0 - (double)uniqueSurvivors / (double)std::max(survivors, 1ull)),
           uniqueSurvivors * 13.0 / (1024.0 * 1024.0),
           uniqueSurvivors * 32.0 / (1024.0 * 1024.0));
    printf("  unique vertices (NO sigma filter, = what distil stores): %llu  (%.2fx more than surviving; payload @13B %.1f MB, fit @32B %.1f MB)\n",
           uniqueAll,
           (double)uniqueAll / (double)std::max(uniqueSurvivors, 1ull),
           uniqueAll * 13.0 / (1024.0 * 1024.0),
           uniqueAll * 32.0 / (1024.0 * 1024.0));

}

void BakedNerf::distil(InstantNerf& teacher,
    const float3* d_rays_o,
    const float3* d_rays_d,
    int numRays,
    cudaStream_t stream
) {
    m_teacherOpts = teacher.options();
    m_bakeThreshold = (m_opts.sigmaThreshold >= 0.0f) ? m_opts.sigmaThreshold
                                                      : m_teacherOpts.minDensityThreshold;

    m_occupancyGrid = DeviceBuffer<uint8_t>(teacher.occupancyBytes());
    teacher.buildOccupancyBitgrid(m_occupancyGrid.data(), m_bakeThreshold);

    int levels = m_teacherOpts.levelsMipmap;
    uint3 res = m_teacherOpts.gridResolution;
    int cascadeOffset = 0;
    int baseElems = (int)(res.x * res.y * res.z) / 8;
    for (int i = 0; i < levels; i++) {
        cascadeOffset += (int)(res.x * res.y * res.z);
        res = make_uint3(res.x / 2, res.y / 2, res.z / 2);
    }

    cascadeOffset /= 8;

    int cascades = m_teacherOpts.numCascades;
    int G = (int)(m_teacherOpts.gridResolution.x * m_teacherOpts.gridResolution.y * m_teacherOpts.gridResolution.z);
    long long perCascadeVoxels = (long long)baseElems * 8;   // 128^3 base cells per cascade
    int offset = 0;
    int numFilledVoxels = 0;
    
    uint8_t* cpuGrid = new uint8_t[teacher.occupancyBytes()];
    m_occupancyGrid.copyHost(cpuGrid, teacher.occupancyBytes());

    std::vector<int> occupiedCells;
    occupiedCells.reserve(1 << 21);

    for (int i = 0; i < cascades; i++) {
        int cascadeFilled = 0;
        for (int j = 0; j < baseElems; j++) {
            uint8_t byte = cpuGrid[offset + j];
            for (int b = 0; b < 8; ++b) {
                if ((byte >> b) & 1) {
                    cascadeFilled++;
                    occupiedCells.push_back(i * G + j * 8 + b);   // local = j*8 + b
                }
            }
        }
        numFilledVoxels += cascadeFilled;
        offset += cascadeOffset;
    }
    
    const long long numBlocks = occupiedCells.size();
    m_numBlocks = numBlocks;
    delete[] cpuGrid;

    m_voxelMask = DeviceBuffer<uint32_t>(numBlocks);
    m_subVoxelMask = DeviceBuffer<uint64_t>(numBlocks);

    DeviceBuffer<int> d_cellIds(numBlocks);
    cudaMemcpy(d_cellIds.data(), occupiedCells.data(), (size_t)numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    DeviceBuffer<int> d_cellSlots(cascades*G);
    d_cellSlots.fill(-1);
    
    constexpr int BS =  256;
    int gs = (numBlocks + BS - 1) / BS;
    buildInvCellMap<<<gs, BS, 0, stream>>>(d_cellIds.data(), d_cellSlots.data(), numBlocks);

    int numSubVoxels = SPARSE_B - 1;
    int vertices = SPARSE_B * SPARSE_B * SPARSE_B;
    const float baseVoxel = (m_teacherOpts.aabbMax.x - m_teacherOpts.aabbMin.x) / (float)m_teacherOpts.gridResolution.x;
    const int blocksPerChunk = 32 * 1024;
    int n = blocksPerChunk * vertices;
    n = (n + 15) & ~15;

    DeviceBuffer<float> d_pos(n*4);
    DeviceBuffer<float> d_out(n); 

    for(int i = 0; i < numBlocks; i += blocksPerChunk) {
        int blocksThis = (int)std::min((long long)blocksPerChunk, numBlocks - i);
        int nSub = blocksThis * vertices;

        int gsPos = (nSub + BS - 1) / BS;
        k_genSubVoxelPositions<<<gsPos, BS>>>(
            d_cellIds.data() + i, blocksThis, SPARSE_B,
            m_teacherOpts.gridResolution, m_teacherOpts.aabbMin, m_teacherOpts.aabbMax,
            d_pos.data());

        teacher.queryDensityLogit(d_pos.data(), nSub, d_out.data());

        int gsThr = (blocksThis + BS - 1) / BS;
        buildSubVoxelMasks<<<gsThr, BS>>>(
            d_out.data(), d_cellIds.data()+i, blocksThis, SPARSE_B, m_teacherOpts.gridResolution,
            m_teacherOpts.densityBias, m_bakeThreshold, baseVoxel, m_voxelMask.data()+i,
            m_subVoxelMask.data()+i
        );
    }

    int lastUniqueVertices = 0;
    DeviceBuffer<int> d_globalCounter(1);
    d_globalCounter.fill(0);
    m_blockIdx = DeviceBuffer<uint32_t>(m_numBlocks*vertices);
    m_blockIdx.fill(-1);
    d_pos.fill(0);
    d_out = DeviceBuffer<float>(n*16);

    for(int i = 0; i < numBlocks; i += blocksPerChunk) {
        int blocksThis = (int)std::min((long long)blocksPerChunk, numBlocks - i);
        int nSub = blocksThis * vertices;

        int gsPos = (nSub + BS - 1) / BS;

        buildGrid<<<gsPos, BS, 0, stream>>>(
            d_cellIds.data()+i,
            d_cellSlots.data(),
            m_subVoxelMask.data()+i,
            blocksThis, lastUniqueVertices, SPARSE_B, m_teacherOpts.gridResolution,
            m_teacherOpts.aabbMin, m_teacherOpts.aabbMax, d_pos.data(), m_blockIdx.data(), d_globalCounter.data()
        );

        d_globalCounter.copyHost(&lastUniqueVertices, 1);
    }

    int uniqueVoxels;
    d_globalCounter.copyHost(&uniqueVoxels, 1);
    m_numVoxels = (uint32_t)uniqueVoxels;

    m_voxelSigma = DeviceBuffer<half>(uniqueVoxels);

    std::vector<float> lrDiffuse(3, m_opts.learningRate);
    std::vector<float> lrFeat(4, m_opts.learningRate);

    EmbeddingTableOption diffuseOpts;
    diffuseOpts.rows = uniqueVoxels;
    diffuseOpts.num_features = 3;
    diffuseOpts.lr = lrDiffuse.data();

    EmbeddingTableOption featOpts;
    featOpts.rows = uniqueVoxels;
    featOpts.num_features = 4;
    featOpts.lr = lrFeat.data();

    m_voxelDiffuse = new EmbeddingTable(diffuseOpts, 2048*1024);
    m_voxelFeatures = new EmbeddingTable(featOpts, 2048*1024);

    DeviceBuffer<float> d_color_sum_out(n*3);
    d_color_sum_out.fill(0);
    DeviceBuffer<float> d_color_out(n*3);

    lastUniqueVertices = 0;
    int currentUniqueVertices = 0;
    d_globalCounter.fill(0);

    for(int i = 0; i < numBlocks; i += blocksPerChunk) {
        int blocksThis = (int)std::min((long long)blocksPerChunk, numBlocks - i);
        int nSub = blocksThis * vertices;

        int gsPos = (nSub + BS - 1) / BS;

        buildGrid<<<gsPos, BS, 0, stream>>>(
            d_cellIds.data()+i,
            d_cellSlots.data(),
            m_subVoxelMask.data()+i,
            blocksThis, lastUniqueVertices, SPARSE_B, m_teacherOpts.gridResolution,
            m_teacherOpts.aabbMin, m_teacherOpts.aabbMax, d_pos.data(), m_blockIdx.data(), d_globalCounter.data()
        );

        d_globalCounter.copyHost(&currentUniqueVertices, 1);

        int rawUnique    = currentUniqueVertices - lastUniqueVertices;
        int paddedUnique = (rawUnique + 15) & ~15;
        teacher.queryFullDensity(d_pos.data(), paddedUnique, d_out.data());

        d_color_sum_out.fill(0);

        int gs = (paddedUnique + BS - 1) / BS;
        for (int d = 0; d < m_opts.bakeDiffuseN; d++) {
            teacher.queryColor(d_out.data(), paddedUnique, m_viewDirs[d], d_color_out.data(), stream);

            accumulateColor<<<gs, BS, 0, stream>>>(d_color_out.data(), d_color_sum_out.data(), paddedUnique);
        }

        int gsFill = (rawUnique + BS - 1) / BS;
        fillVertexGrid<<<gsFill, BS, 0, stream>>>(
            d_out.data(), d_color_sum_out.data(),
            m_voxelSigma.data(),
            m_voxelDiffuse->masterWeights(),
            1.0f / m_opts.bakeDiffuseN,
            m_teacherOpts.densityBias,
            lastUniqueVertices, rawUnique
        );
        d_globalCounter.copyHost(&lastUniqueVertices, 1);
    }

    
    // PART 1 DONE
    m_render_buffers.d_cellSlots = std::move(d_cellSlots);
    m_render_buffers.d_rays_d_inv_chunk = DeviceBuffer<float3>(m_teacherOpts.rayChunkSize);
    m_render_buffers.d_nears_chunk = DeviceBuffer<float>(m_teacherOpts.rayChunkSize);
    m_render_buffers.d_fars_chunk = DeviceBuffer<float>(m_teacherOpts.rayChunkSize);
    m_render_buffers.d_block_sums = DeviceBuffer<uint32_t>((m_teacherOpts.rayChunkSize + 1023) / 1024);
    m_render_buffers.d_active_rays_count = DeviceBuffer<uint32_t>(1);
    m_render_buffers.d_num_steps = DeviceBuffer<uint32_t>(m_teacherOpts.rayChunkSize);
    m_render_buffers.d_ray_offsets = DeviceBuffer<uint32_t>(m_teacherOpts.rayChunkSize);
    m_render_buffers.d_ray_indices = DeviceBuffer<uint32_t>(m_opts.jointFitBatch);
    m_render_buffers.d_teacher_points_out = DeviceBuffer<float>(m_opts.jointFitBatch);
    m_render_buffers.d_student_frac_out = DeviceBuffer<float>(m_opts.jointFitBatch * 3);
    m_render_buffers.d_student_rows_out = DeviceBuffer<int>(m_opts.jointFitBatch * 8);

    uint32_t raysDone = 0;

    while(raysDone < numRays) {
        uint32_t currentChunkRaysUpperBound = min(m_teacherOpts.rayChunkSize, numRays - raysDone);
        const float3* chunk_o = d_rays_o + raysDone;
        const float3* chunk_d = d_rays_d + raysDone;

        int currentChunkRays;
        uint32_t totalHits;

        constexpr int BS = 256;
        int gs = (currentChunkRaysUpperBound + BS - 1) / BS;
        compute_ray_aabb_inv_kernel<<<gs, BS, 0, stream>>>(
            currentChunkRaysUpperBound,
            chunk_o, chunk_d, m_teacherOpts.aabbMin, m_teacherOpts.aabbMax,
            m_teacherOpts.numCascades,
            m_render_buffers.d_rays_d_inv_chunk.data(),
            m_render_buffers.d_nears_chunk.data(),
            m_render_buffers.d_fars_chunk.data()
        );

        currentChunkRays = processBakedRaysLinear(
            currentChunkRaysUpperBound,
            chunk_o, chunk_d,
            m_render_buffers.d_rays_d_inv_chunk.data(),
            m_render_buffers.d_nears_chunk.data(),
            m_render_buffers.d_fars_chunk.data(),
            m_occupancyGrid.data(),
            m_voxelMask.data(),
            m_blockIdx.data(),
            m_render_buffers.d_cellSlots.data(),
            m_teacherOpts.gridResolution,
            m_teacherOpts.aabbMin, m_teacherOpts.aabbMax,
            m_teacherOpts.numCascades,
            m_teacherOpts.levelsMipmap,
            SPARSE_B, m_opts.jointFitBatch, &totalHits,
            m_render_buffers.d_active_rays_count.data(),
            m_render_buffers.d_num_steps.data(),
            m_render_buffers.d_ray_offsets.data(),
            m_render_buffers.d_teacher_points_out.data(),
            m_render_buffers.d_student_rows_out.data(),
            m_render_buffers.d_student_frac_out.data(),
            m_render_buffers.d_ray_indices.data(),
            m_render_buffers.d_block_sums.data(),
            stream
        );

        if (currentChunkRays == 0) {
            fprintf(stderr, "Error: Batch size is too small to fit even a single ray! Increase batchSize.\n");
            return;
        }

        uint32_t padded_b_size = (totalHits + 15) & ~15;
        // teacher.m_densityMLP->inference(m_render_buffers.d_teacher_points_out.data(), m_render_buffers.d_density_out.data(), padded_b_size, stream);
    }
}
