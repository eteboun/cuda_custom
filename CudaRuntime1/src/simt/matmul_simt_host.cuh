#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "tensor.cuh"
#include "matmul.cuh"
#include "matmul_simt.cuh"

template<int TILE_M, int TILE_N, int TILE_K, int BLOCK_X, int BLOCK_Y>
DeviceTensor hostSimtMatmul(
    const DeviceTensor& A,
    const DeviceTensor& B
) {
    ConstDeviceTensorView a = A.view();
    ConstDeviceTensorView b = B.view();

    if (a.ndim != 2 || b.ndim != 2) {
        throw std::invalid_argument(
            "matmul requires both tensors to be 2-dimensional"
        );
    }

    if (a.shape[1] != b.shape[0]) {
        throw std::invalid_argument(
            "matmul dimension mismatch: A.shape[1] must equal B.shape[0]"
        );
    }

    size_t M = a.shape[0];
    size_t N = b.shape[1];

    DeviceTensor C({ M, N });

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    deviceMatMul<TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y>
        <<<grid, block>>> (
            a,
            b,
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

template<int TILE_M, int TILE_N, int TILE_K, int BLOCK_X, int BLOCK_Y>
BenchmarkResult benchmarkMatMul(
    const DeviceTensor& A,
    const DeviceTensor& B,
    int warmupIters = 3,
    int timedIters = 10
) {
    ConstDeviceTensorView a = A.view();
    ConstDeviceTensorView b = B.view();

    if (a.ndim != 2 || b.ndim != 2) {
        throw std::invalid_argument(
            "matmul requires both tensors to be 2-dimensional"
        );
    }

    if (a.shape[1] != b.shape[0]) {
        throw std::invalid_argument(
            "matmul dimension mismatch: A.shape[1] must equal B.shape[0]"
        );
    }

    if (timedIters <= 0) {
        throw std::invalid_argument(
            "benchmark requires at least one timed iteration"
        );
    }

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

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
                a,
                b,
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
                a,
                b,
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
