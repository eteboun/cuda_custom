#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "tensor.cuh"
#include "matmul_simt.cuh"

template<int TILE_M, int TILE_N, int TILE_K, int BLOCK_X, int BLOCK_Y>
DeviceTensor hostMatmul(
    const DeviceTensor& A,
    const DeviceTensor& B
) {
    if (A.getShape().size() != 2 || B.getShape().size() != 2) {
        throw std::invalid_argument(
            "matmul requires both tensors to be 2-dimensional"
        );
    }

    if (A.getShape()[1] != B.getShape()[0]) {
        throw std::invalid_argument(
            "matmul dimension mismatch: A.shape[1] must equal B.shape[0]"
        );
    }

    int M = A.getShape()[0];
    int N = B.getShape()[1];

    DeviceTensor C({ M, N });

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    deviceMatMul<TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y>
        <<<grid, block >>> (
            A.view(),
            B.view(),
            C.view()
            );

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("matmul kernel launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("matmul kernel execution failed: ") +
            cudaGetErrorString(err)
        );
    }

    return C;
}

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

template<int TILE_M, int TILE_N, int TILE_K, int BLOCK_X, int BLOCK_Y>
BenchmarkResult benchmarkMatMul(
    const DeviceTensor& A,
    const DeviceTensor& B,
    int warmupIters = 3,
    int timedIters = 10
) {
    if (A.getShape().size() != 2 || B.getShape().size() != 2) {
        throw std::invalid_argument(
            "matmul requires both tensors to be 2-dimensional"
        );
    }

    if (A.getShape()[1] != B.getShape()[0]) {
        throw std::invalid_argument(
            "matmul dimension mismatch: A.shape[1] must equal B.shape[0]"
        );
    }

    if (timedIters <= 0) {
        throw std::invalid_argument(
            "benchmark requires at least one timed iteration"
        );
    }

    int M = A.getShape()[0];
    int K = A.getShape()[1];
    int N = B.getShape()[1];

    DeviceTensor C({ M, N });

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    // Warm-up launches: absorb one-time costs (module load, caches).
    for (int i = 0; i < warmupIters; i++) {
        deviceMatMul<TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y>
            <<<grid, block >>> (
                A.view(),
                B.view(),
                C.view()
                );
    }

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("matmul warm-up launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("matmul warm-up execution failed: ") +
            cudaGetErrorString(err)
        );
    }

    CudaEvent start;
    CudaEvent stop;

    cudaEventRecord(start.get());

    for (int i = 0; i < timedIters; i++) {
        deviceMatMul<TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y>
            <<<grid, block >>> (
                A.view(),
                B.view(),
                C.view()
                );
    }

    cudaEventRecord(stop.get());

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("matmul benchmark launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    err = cudaEventSynchronize(stop.get());

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("matmul benchmark execution failed: ") +
            cudaGetErrorString(err)
        );
    }

    float totalMs = 0.0f;
    cudaEventElapsedTime(&totalMs, start.get(), stop.get());

    BenchmarkResult result;
    result.avgMs = totalMs / timedIters;

    // Each output element needs K multiplies and K adds.
    double flops = 2.0 * M * N * K;
    result.gflops = flops / (result.avgMs * 1e-3) / 1e9;

    return result;
}
