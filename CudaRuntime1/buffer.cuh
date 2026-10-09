#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cuda_runtime.h>

template<typename T>
class DeviceBuffer {
private:
    T* data_ = nullptr;
    size_t size_ = 0;

public:
    explicit DeviceBuffer(size_t size)
        : size_(size)
    {
        if (size == 0)
            throw std::runtime_error("Size must be greater than 0");

        if (size > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::overflow_error("DeviceBuffer allocation overflow");
        
        cudaError_t err = cudaMalloc(
            &data_,
            size * sizeof(T)
        );

        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    DeviceBuffer() noexcept = default;

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaFree(data_);
            }

            data_ = other.data_;
            size_ = other.size_;

            other.data_ = nullptr;
            other.size_ = 0;
        }

        return *this;
    }

    T* data() noexcept {
        return data_;
    }

    const T* data() const noexcept {
        return data_;
    }

    size_t size() const noexcept {
        return size_;
    }
};

