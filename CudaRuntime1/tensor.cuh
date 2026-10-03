#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cuda_runtime.h>
#include "utils.h"

constexpr int MAX_DIMS = 8;

struct DeviceTensorView {
    float* data;
    int shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    int ndim;
};

class DeviceTensor {
private:
    float* data;
    size_t numel;
    std::vector<int> shape;
    std::vector<size_t> strides;

    void setStrides() {
        strides.resize(shape.size());

        size_t stride = 1;

        for (size_t i = shape.size(); i-- > 0;) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    static void validateShape(const std::vector<int>& shape) {
        if (shape.size() > MAX_DIMS) {
            throw std::invalid_argument(
                "Tensor exceeds maximum number of dimensions."
            );
        }

        for (int dim : shape) {
            if (dim < 0) {
                throw std::invalid_argument(
                    "Tensor dimensions cannot be negative."
                );
            }
        }
    }

    void validateNewShape(const std::vector<int>& newShape) const {
        validateShape(newShape);

        if (calcNumel(newShape) != numel) {
            throw std::invalid_argument(
                "New shape must have the same number of elements."
            );
        }
    }

    void clear() {
        numel = 0;
        data = nullptr;
        shape.clear();
        strides.clear();
    }

public:
    DeviceTensor(const std::vector<int>& shape) {
        validateShape(shape);

        this->shape = shape;
        numel = calcNumel(shape);
        setStrides();

        data = nullptr;

        if (numel > 0) {
            cudaError_t err = cudaMalloc(
                &data,
                numel * sizeof(float)
            );

            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cudaGetErrorString(err)
                );
            }

            err = cudaMemset(
                data,
                0,
                numel * sizeof(float)
            );

            if (err != cudaSuccess) {
                cudaFree(data);
                data = nullptr;

                throw std::runtime_error(
                    cudaGetErrorString(err)
                );
            }
        }
    }

    DeviceTensor(
        const std::vector<float>& hostData,
        const std::vector<int>& shape
    ) : DeviceTensor(shape) {

        if (numel != hostData.size()) {
            throw std::invalid_argument(
                "Shape and data size do not match."
            );
        }

        cudaError_t err = cudaMemcpy(
            data,
            hostData.data(),
            numel * sizeof(float),
            cudaMemcpyHostToDevice
        );

        if (err != cudaSuccess) {
            cudaFree(data);
            data = nullptr;

            throw std::runtime_error(
                cudaGetErrorString(err)
            );
        }
    }

    DeviceTensorView view() const {
        DeviceTensorView v;

        v.data = data;
        v.ndim = static_cast<int>(shape.size());

        for (int i = 0; i < v.ndim; ++i) {
            v.shape[i] = shape[i];
            v.strides[i] = strides[i];
        }

        return v;
    }

    std::vector<float> toHost() const {
        std::vector<float> result(numel);

        cudaError_t err = cudaMemcpy(
            result.data(),
            data,
            numel * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (err != cudaSuccess) {
            throw std::runtime_error(
                cudaGetErrorString(err)
            );
        }

        return result;
    }

    void reshape(const std::vector<int>& newShape) {
        validateNewShape(newShape);

        shape = newShape;
        setStrides();
    }

    int getNdim() const {
        return static_cast<int>(shape.size());
    }

    const std::vector<int>& getShape() const {
        return shape;
    }

    const std::vector<size_t>& getStrides() const {
        return strides;
    }

    size_t getNumel() const {
        return numel;
    }

    void showTabular() {
        if (shape.size() != 2) {
            return;
        }

        const std::vector<float> hostData = toHost();

        const int rows = shape[0];
        const int cols = shape[1];

        for (int row = 0; row < rows; ++row) {
            std::cout << "[ ";

            for (int col = 0; col < cols; ++col) {
                std::cout << hostData[row * cols + col];

                if (col < cols - 1) {
                    std::cout << ", ";
                }
            }

            std::cout << " ]\n";
        }
    }

    ~DeviceTensor() {
        cudaFree(data);
    }

    DeviceTensor& operator=(const DeviceTensor&) = delete;
    DeviceTensor(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&& other) noexcept {
        data = other.data;
        numel = other.numel;
        shape = std::move(other.shape);
        strides = std::move(other.strides);

        other.clear();
    }
    DeviceTensor& operator=(DeviceTensor&& other) noexcept {
        if (this != &other) {
            cudaFree(data);

            data = other.data;
            numel = other.numel;
            shape = std::move(other.shape);
            strides = std::move(other.strides);

            other.clear();
        }

        return *this;
    }
};
