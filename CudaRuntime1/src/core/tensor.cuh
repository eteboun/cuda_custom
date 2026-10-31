#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include "utils.h"

#include <cuda_runtime.h>
#include "buffer.cuh"

constexpr int MAX_DIMS = 8;

struct BuildConfig {
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    size_t numel;
    int ndim;
};

struct DeviceTensorView {
    float* data;
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    int ndim;
};

struct ConstDeviceTensorView {
    const float* data;
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    int ndim;
};

class DeviceTensor {
private:
    DeviceBuffer<float> buffer;
    BuildConfig _config{};

    static BuildConfig _cfg(
        const std::vector<size_t>& newShape
    ) {
        if (newShape.empty()) {
            throw std::invalid_argument(
                "Tensor shape cannot be empty."
            );
        }

        if (newShape.size() > MAX_DIMS) {
            throw std::invalid_argument(
                "Tensor exceeds maximum number of dimensions."
            );
        }

        for (size_t dim : newShape) {
            if (dim == 0) {
                throw std::invalid_argument(
                    "Tensor dimensions cannot be zero."
                );
            }
        }

        BuildConfig cfg{};
        cfg.ndim = static_cast<int>(newShape.size());

        size_t stride = 1;

        for (size_t i = newShape.size(); i-- > 0;) {
            if (stride > std::numeric_limits<size_t>::max() / newShape[i]) {
                throw std::overflow_error("Tensor size overflow");
            }

            cfg.shape[i] = newShape[i];
            cfg.strides[i] = stride;
            stride *= newShape[i];
        }

        cfg.numel = stride;
        return cfg;
    }

    void build(
        const std::vector<float>& hostData,
        const std::vector<size_t>& newShape
    ) {
        BuildConfig cfg = _cfg(newShape);

        if (cfg.numel != hostData.size()) {
            throw std::invalid_argument(
                "Shape and data size do not match."
            );
        }

        DeviceBuffer<float> newBuffer(cfg.numel);

        cudaError_t err = cudaMemcpy(
            newBuffer.data(),
            hostData.data(),
            cfg.numel * sizeof(float),
            cudaMemcpyHostToDevice
        );

        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        _config = cfg;
        buffer = std::move(newBuffer);
    }

    void build(
        const std::vector<size_t>& newShape
    ) {
        BuildConfig cfg = _cfg(newShape);

        DeviceBuffer<float> newBuffer(cfg.numel);

        _config = cfg;
        buffer = std::move(newBuffer);
    }

public:
    explicit DeviceTensor(
        const std::vector<size_t>& newShape
    ) {
        build(newShape);
    }

    DeviceTensor(
        const std::vector<float>& hostData,
        const std::vector<size_t>& newShape
    ) {
        build(hostData, newShape);
    }

    ConstDeviceTensorView view() const {
        ConstDeviceTensorView v{};

        v.data = buffer.data();
        v.ndim = _config.ndim;

        for (int i = 0; i < v.ndim; ++i) {
            v.shape[i] = _config.shape[i];
            v.strides[i] = _config.strides[i];
        }

        return v;
    }

    DeviceTensorView view() {
        DeviceTensorView v{};

        v.data = buffer.data();
        v.ndim = _config.ndim;

        for (int i = 0; i < v.ndim; ++i) {
            v.shape[i] = _config.shape[i];
            v.strides[i] = _config.strides[i];
        }

        return v;
    }

    std::vector<float> toHost() const {
        std::vector<float> result(_config.numel);

        cudaError_t err = cudaMemcpy(
            result.data(),
            buffer.data(),
            _config.numel * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (err != cudaSuccess) {
            throw std::runtime_error(
                cudaGetErrorString(err)
            );
        }

        return result;
    }
};
