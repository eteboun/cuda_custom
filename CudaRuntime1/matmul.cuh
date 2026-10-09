#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "tensor.cuh"

struct BenchmarkResult {
    float avgMs;
    double gflops;
};

// CPU reference implementation (matmul.cu). Row-major A (M x K), B (K x N).
std::vector<float> cpuMatmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int K,
    int N
);

// Times one cpuMatmul run and writes its output to C.
BenchmarkResult benchmarkCpuMatMul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int K,
    int N,
    std::vector<float>& C
);

// Times cublasSgemm on the same tensors (matmul.cu) and writes its output
// to C (host) for correctness checks.
BenchmarkResult benchmarkCublasMatMul(
    const DeviceTensor& A,
    const DeviceTensor& B,
    std::vector<float>& C,
    int warmupIters = 3,
    int timedIters = 10
);

// Owns a cudaEvent_t so it is destroyed even if an exception is thrown.
class CudaEvent {
private:
    cudaEvent_t event;

public:
    CudaEvent() {
        cudaError_t err = cudaEventCreate(&event);

        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaEventCreate failed: ") +
                cudaGetErrorString(err)
            );
        }
    }

    ~CudaEvent() {
        cudaEventDestroy(event);
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const {
        return event;
    }
};
