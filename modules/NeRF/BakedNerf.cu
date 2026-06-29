#include "BakedNerf.h"
#undef CUDA_CHECK
#include "../TinyMLP/TinyMLP.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <bit>
#include <cmath>
#include <algorithm>
#include <stdexcept>

BakedNerf::~BakedNerf() {
    delete m_deferredMLP;
}

int countOnes(uint8_t n) {
    int count = 0;
    while (n) {
        n &= (n - 1);
        count++;
    }
    return count;
}

// mip-360 contraction, copied from processRays.cu/otherKernels.cu (those are static there).
// Maps a metric/cascade-space position to the contracted [0,1]^3 the teacher MLP expects.
static __device__ __forceinline__ float3 bake_contract_pos(float3 pos, float3 aabb_min, float3 aabb_max) {
    float3 ext = make_float3(aabb_max.x - aabb_min.x, aabb_max.y - aabb_min.y, aabb_max.z - aabb_min.z);
    float3 rel = make_float3((pos.x - aabb_min.x) / ext.x, (pos.y - aabb_min.y) / ext.y, (pos.z - aabb_min.z) / ext.z);
    float3 n   = make_float3(rel.x * 2.0f - 1.0f, rel.y * 2.0f - 1.0f, rel.z * 2.0f - 1.0f);
    float3 c;
    c.x = fabsf(n.x) <= 1.0f ? n.x : (2.0f - 1.0f / fabsf(n.x)) * copysignf(1.0f, n.x);
    c.y = fabsf(n.y) <= 1.0f ? n.y : (2.0f - 1.0f / fabsf(n.y)) * copysignf(1.0f, n.y);
    c.z = fabsf(n.z) <= 1.0f ? n.z : (2.0f - 1.0f / fabsf(n.z)) * copysignf(1.0f, n.z);
    return make_float3(
        fmaxf(0.0f, fminf((c.x + 2.0f) * 0.25f, 1.0f)),
        fmaxf(0.0f, fminf((c.y + 2.0f) * 0.25f, 1.0f)),
        fmaxf(0.0f, fminf((c.z + 2.0f) * 0.25f, 1.0f)));
}

// For a chunk of occupied blocks (dense cell ids = cascade*G + local), emit the contracted
// position of each of the B^3 sub-voxel CENTERS. Output index = blockInChunk*B^3 + sub, matching
// the order queryDensityLogit consumes and k_thresholdSubVoxels reads.
__global__ void k_genSubVoxelPositions(
    const int* __restrict__ cellIds, int numBlocks, int B,
    uint3 gridRes, float3 aabbMin, float3 aabbMax,
    float* __restrict__ outPos)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int per = B * B * B;
    if (tid >= numBlocks * per) return;

    int sub = tid % per;
    int bi  = tid / per;
    int cell = cellIds[bi];

    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = cell / G;
    int local   = cell % G;
    int gx = local % gridRes.x;
    int gy = (local / gridRes.x) % gridRes.y;
    int gz = local / (gridRes.x * gridRes.y);

    int sx = sub % B;
    int sy = (sub / B) % B;
    int sz = sub / (B * B);

    float scale = exp2f((float)cascade);
    float3 cmin = make_float3(aabbMin.x * scale, aabbMin.y * scale, aabbMin.z * scale);
    float3 cmax = make_float3(aabbMax.x * scale, aabbMax.y * scale, aabbMax.z * scale);
    float3 csz  = make_float3((cmax.x - cmin.x) / gridRes.x, (cmax.y - cmin.y) / gridRes.y, (cmax.z - cmin.z) / gridRes.z);

    float invB = 1.0f / (float)B;
    float3 pos = make_float3(
        cmin.x + (gx + (sx + 0.5f) * invB) * csz.x,
        cmin.y + (gy + (sy + 0.5f) * invB) * csz.y,
        cmin.z + (gz + (sz + 0.5f) * invB) * csz.z);

    pos = bake_contract_pos(pos, aabbMin, aabbMax);
    outPos[tid * 3 + 0] = pos.x;
    outPos[tid * 3 + 1] = pos.y;
    outPos[tid * 3 + 2] = pos.z;
}

// Threshold each sub-voxel's sigma against the SAME per-cascade opacity bar the occupancy grid uses,
// count survivors per block, accumulate the global survivor total and a per-block-fill histogram.
__global__ void k_thresholdSubVoxels(
    const float* __restrict__ logits, const int* __restrict__ cellIds,
    int numBlocks, int B, uint3 gridRes,
    float densityBias, float minDensityThreshold, float baseVoxel,
    unsigned long long* __restrict__ survivorTotal,
    unsigned long long* __restrict__ fillHist)   // [B^3 + 1] bins
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= numBlocks) return;

    int per = B * B * B;
    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = cellIds[bi] / G;
    float cascadeThresh = (minDensityThreshold / baseVoxel) / exp2f((float)cascade);

    int popc = 0;
    for (int s = 0; s < per; ++s) {
        float logit = logits[bi * per + s];
        float sigma = expf(fminf(logit - densityBias, 8.0f));
        if (sigma > cascadeThresh) popc++;
    }
    atomicAdd(survivorTotal, (unsigned long long)popc);
    atomicAdd(&fillHist[popc], 1ULL);
}

void BakedNerf::init(const BakeOptions& opts) {
    m_opts = opts;

    MLPOption deferredOpts;
    deferredOpts.activationType = ACT_RELU;
    deferredOpts.outputActivation = OUT_ACT_SIGMOID;
    deferredOpts.inputDim = vecLen();
    deferredOpts.hiddenDim = m_opts.deferredHidden;
    deferredOpts.outputDim = 3;
    deferredOpts.numLayers = m_opts.deferredLayers;

    m_deferredMLP = new TinyMLP(deferredOpts, m_opts.queryBatch, m_opts.queryBatch);
}

void BakedNerf::distil(InstantNerf& teacher) {
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
    m_occupancyGrid = DeviceBuffer<uint8_t>(teacher.occupancyBytes());
    teacher.buildOccupancyBitgrid(m_occupancyGrid.data(), m_bakeThreshold);

    uint8_t* cpuGrid = new uint8_t[teacher.occupancyBytes()];
    m_occupancyGrid.copyHost(cpuGrid, teacher.occupancyBytes());

    int levels = m_teacherOpts.levelsMipmap;
    uint3 res = m_teacherOpts.gridResolution;
    int cascadeOffset = 0;
    int baseElems = (int)(res.x * res.y * res.z) / 8;
    for(int i = 0; i < levels; i++) {
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
    for(int i = 0; i < cascades; i++){
        int cascadeFilled = 0;
        for(int j = 0; j < baseElems; j++) {
            uint8_t byte = cpuGrid[offset+j];
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
        numFilledVoxels += cascadeFilled;
        offset += cascadeOffset;        // full-pyramid bytes per cascade = stride between cascades
    }

    long long totalBaseVoxels = perCascadeVoxels * cascades;   // 128^3 * cascades base cells
    printf("  TOTAL     : filled %8d / %lld (%6.3f%%)\n",
           numFilledVoxels, totalBaseVoxels,
           100.0 * numFilledVoxels / (double)totalBaseVoxels);

    delete[] cpuGrid;

    // ---- Tier-2: how many of each occupied block's B^3 sub-voxel centers actually carry sigma
    //      above the per-cascade bar? Query the teacher density MLP at the contracted sub-centers. ----
    const int B = SPARSE_B;
    const int per = B * B * B;                       // 64
    const long long numBlocks = (long long)occupiedCells.size();
    if (numBlocks == 0) { printf("[sub-voxel] no occupied blocks\n"); return; }

    DeviceBuffer<int> d_cellIds((size_t)numBlocks);
    cudaMemcpy(d_cellIds.data(), occupiedCells.data(), (size_t)numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    const int chunkBlocks = 1 << 15;                 // 32768 blocks -> 2,097,152 sub-voxels / chunk
    DeviceBuffer<float> d_pos((size_t)chunkBlocks * per * 3);
    DeviceBuffer<float> d_logit((size_t)chunkBlocks * per);
    DeviceBuffer<unsigned long long> d_survivors(1);
    DeviceBuffer<unsigned long long> d_hist((size_t)per + 1);
    d_survivors.fill(0);
    d_hist.fill(0);

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
        k_thresholdSubVoxels<<<gsThr, BS>>>(
            d_logit.data(), d_cellIds.data() + c0, blocksThis, B,
            m_teacherOpts.gridResolution, m_teacherOpts.densityBias, m_bakeThreshold, baseVoxel,
            d_survivors.data(), d_hist.data());
    }
    cudaDeviceSynchronize();

    unsigned long long survivors = 0;
    std::vector<unsigned long long> hist((size_t)per + 1, 0);
    d_survivors.copyHost(&survivors, 1);
    d_hist.copyHost(hist.data(), (size_t)per + 1);

    const long long candidateSubVoxels = numBlocks * per;
    auto bucket = [&](int lo, int hi){ unsigned long long s = 0; for (int k = lo; k <= hi; ++k) s += hist[k]; return s; };

    printf("\n[sub-voxel sigma survival @ B=%d, threshold %.4f]\n", B, m_bakeThreshold);
    printf("  occupied blocks      : %lld\n", numBlocks);
    printf("  candidate sub-voxels : %lld (blocks x %d)\n", candidateSubVoxels, per);
    printf("  surviving sub-voxels : %llu  (%.3f%% of candidates, mean %.2f/%d per block)\n",
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
}
