#include "BakedNerf.h"
#undef CUDA_CHECK
#include "../TinyMLP/TinyMLP.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <bit>
#include <cmath>
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
    long long perCascadeVoxels = (long long)baseElems * 8;   // 128^3 base cells per cascade
    int numFilledVoxels = 0;
    int offset = 0;
    printf("threshold %.4f :\n", m_bakeThreshold);
    for(int i = 0; i < cascades; i++){
        int cascadeFilled = 0;
        for(int j = 0; j < baseElems; j++) {
            cascadeFilled += countOnes(cpuGrid[offset+j]);
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
}
