#include "GaussianModel.h"

#include <stdexcept>
#include <string>

GaussianModel::GaussianModel(uint32_t capacity) : m_capacity(capacity) {
    m_params.allocate(capacity);
    m_grads.allocate(capacity);
    m_adamM.allocate(capacity);
    m_adamV.allocate(capacity);

    m_adamM.zero(capacity, 0);
    m_adamV.zero(capacity, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GaussianModel::setCount(uint32_t n) {
    if (n > m_capacity) {
        throw std::runtime_error("GaussianModel::setCount: " + std::to_string(n) +
                                 " exceeds capacity " + std::to_string(m_capacity));
    }
    m_count = n;
}

void GaussianModel::zeroGrads(cudaStream_t stream) {
    m_grads.zero(m_count, stream);
}

void GaussianModel::Buffers::allocate(uint32_t capacity) {
    means         = DeviceBuffer<float3>(capacity);
    logScales     = DeviceBuffer<float3>(capacity);
    rotations     = DeviceBuffer<float4>(capacity);
    opacityLogits = DeviceBuffer<float>(capacity);
    shDC          = DeviceBuffer<float3>(capacity);
    shRest        = DeviceBuffer<float3>((size_t)capacity * GS_SH_REST);
}

void GaussianModel::Buffers::zero(uint32_t count, cudaStream_t stream) {
    if (count == 0) return;
    CUDA_CHECK(cudaMemsetAsync(means.data(),         0, count * sizeof(float3), stream));
    CUDA_CHECK(cudaMemsetAsync(logScales.data(),     0, count * sizeof(float3), stream));
    CUDA_CHECK(cudaMemsetAsync(rotations.data(),     0, count * sizeof(float4), stream));
    CUDA_CHECK(cudaMemsetAsync(opacityLogits.data(), 0, count * sizeof(float),  stream));
    CUDA_CHECK(cudaMemsetAsync(shDC.data(),          0, count * sizeof(float3), stream));
    CUDA_CHECK(cudaMemsetAsync(shRest.data(),        0, (size_t)count * GS_SH_REST * sizeof(float3), stream));
}

GaussianParamsView GaussianModel::Buffers::view() {
    return { means.data(), logScales.data(), rotations.data(),
             opacityLogits.data(), shDC.data(), shRest.data() };
}
