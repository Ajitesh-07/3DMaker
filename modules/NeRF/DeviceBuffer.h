#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <algorithm>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                   \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            std::abort();                                                       \
        }                                                                       \
    } while (0)
#endif

template <typename T>
class DeviceBuffer {
private:
    T* d_ptr = nullptr;
    size_t m_size = 0;

public:
    DeviceBuffer(size_t size) : m_size(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (d_ptr) {
            cudaError_t err = cudaFree(d_ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error during cudaFree at %s:%d: %s\n", 
                        __FILE__, __LINE__, cudaGetErrorString(err));
            }
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (d_ptr) {
                cudaFree(d_ptr); 
            }
            d_ptr = other.d_ptr;
            m_size = other.m_size;
            other.d_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    T* data() {
        return d_ptr;
    } 

    size_t size() {
        return m_size;
    }

    void fill(int value) { 
        if (d_ptr && m_size > 0) {
            cudaError_t err = cudaMemset(d_ptr, value, m_size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA memset failed: ") + cudaGetErrorString(err));
            }
        }
    }

    void copyHost(T* h_ptr, size_t num_elements) {
        if (!d_ptr || num_elements == 0) return;

        size_t sizeCopy = std::min(num_elements, m_size);

        cudaError_t err = cudaMemcpy(h_ptr, d_ptr, sizeCopy * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memcpy to host failed: ") + cudaGetErrorString(err));
        }
    }
};
