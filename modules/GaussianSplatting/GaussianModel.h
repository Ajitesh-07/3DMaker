#pragma once

#include "GsCommon.h"

// Raw device pointers to one set of per-Gaussian arrays (SoA), passed by value to kernels.
// The same layout is used for parameters, gradients and both Adam moments.
struct GaussianParamsView {
    float3* means;          // [capacity]
    float3* logScales;      // [capacity]            activation: exp
    float4* rotations;      // [capacity] (w,x,y,z)  activation: normalize
    float*  opacityLogits;  // [capacity]            activation: sigmoid
    float3* shDC;           // [capacity]
    float3* shRest;         // [capacity * GS_SH_REST], coefficient-major per Gaussian
};

// Fixed-capacity Gaussian store. Everything is allocated once at `capacity`, which doubles as
// the hard count cap (VRAM safety rail); densify/prune only moves `count` within it.
class GaussianModel {
public:
    explicit GaussianModel(uint32_t capacity);

    GaussianModel(const GaussianModel&) = delete;
    GaussianModel& operator=(const GaussianModel&) = delete;

    uint32_t capacity() const { return m_capacity; }
    uint32_t count() const { return m_count; }
    void     setCount(uint32_t n);

    GaussianParamsView params() { return m_params.view(); }
    GaussianParamsView grads()  { return m_grads.view(); }
    GaussianParamsView adamM()  { return m_adamM.view(); }
    GaussianParamsView adamV()  { return m_adamV.view(); }

    void zeroGrads(cudaStream_t stream = 0);

    // Params + grads + 2 Adam moments = 4 * 236 B per Gaussian of capacity.
    size_t bytesAllocated() const { return (size_t)m_capacity * GS_FLOATS_PER_GAUSSIAN * sizeof(float) * 4; }

    // TODO(M0): initFromPoints(positions, colors, n) — needs SfM points in the normalized frame
    // (colmap2nerf.py does not export them yet).

private:
    struct Buffers {
        DeviceBuffer<float3> means{0};
        DeviceBuffer<float3> logScales{0};
        DeviceBuffer<float4> rotations{0};
        DeviceBuffer<float>  opacityLogits{0};
        DeviceBuffer<float3> shDC{0};
        DeviceBuffer<float3> shRest{0};

        void allocate(uint32_t capacity);
        void zero(uint32_t count, cudaStream_t stream);
        GaussianParamsView view();
    };

    uint32_t m_capacity = 0;
    uint32_t m_count    = 0;

    Buffers m_params;
    Buffers m_grads;
    Buffers m_adamM;
    Buffers m_adamV;
};
