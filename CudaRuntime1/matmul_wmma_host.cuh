#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "buffer.cuh"
#include "tensor.cuh"
#include "matmul.cuh"
#include "matmul_wmma.cuh"

template<int WARP_ROWS, int WARP_COLS>
DeviceTensor hostWmmaMatmul(
    const DeviceTensor& A,
    const DeviceTensor& B
) {
    ConstDeviceTensorView a = A.view();
    ConstDeviceTensorView b = B.view();

    if (a.ndim != 2 ||
        b.ndim != 2) {
        throw std::invalid_argument(
            "Both A and B must be 2D tensors."
        );
    }

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    if (K != b.shape[0]) {
        throw std::invalid_argument(
            "Inner dimensions of A and B must match."
        );
    }

    size_t Mp = ((M + 15) / 16) * 16;
    size_t Kp = ((K + 15) / 16) * 16;
    size_t Np = ((N + 15) / 16) * 16;

    DeviceBuffer<half> Ap(Mp * Kp);
    DeviceBuffer<half> Bp(Kp * Np);
    DeviceBuffer<float> Cp(Mp * Np);

    constexpr int THREADS = 256;
    constexpr int BLOCKS = 256;

    preProcess <<<BLOCKS, THREADS>>>(
        a.data,
        b.data,
        Ap.data(), Bp.data(),
        M, N, K,
        Mp, Np, Kp
        );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    constexpr int WARPS_PER_BLOCK =
        WARP_ROWS * WARP_COLS;

    dim3 block(32, WARPS_PER_BLOCK);

    dim3 grid(
        (Np + WARP_COLS * 16 - 1) / (WARP_COLS * 16),
        (Mp + WARP_ROWS * 16 - 1) / (WARP_ROWS * 16)
    );

    wmmaMatmul<WARP_ROWS, WARP_COLS><<<grid, block>>>(
        Ap.data(), Bp.data(), Cp.data(),
        Mp, Np, Kp
        );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    DeviceTensor C({ M, N });

    postProcess <<<BLOCKS, THREADS>>>(
        Cp.data(), C.view().data,
        M, N, Np
        );
    err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    return C;
}

// Times only wmmaMatmul, like benchmarkMatMul times only its kernel.
// The fp32 -> fp16 conversion (preProcess) runs once before timing and
// the copy back to an unpadded C (postProcess) is not run. FLOPs are
// counted on the unpadded M x N x K, so padding shows up as lost speed.
template<int WARP_ROWS, int WARP_COLS>
BenchmarkResult benchmarkWmmaMatMul(
    const DeviceTensor& A,
    const DeviceTensor& B,
    int warmupIters = 3,
    int timedIters = 10
) {
    ConstDeviceTensorView a = A.view();
    ConstDeviceTensorView b = B.view();

    if (a.ndim != 2 || b.ndim != 2) {
        throw std::invalid_argument(
            "Both A and B must be 2D tensors."
        );
    }

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    if (K != b.shape[0]) {
        throw std::invalid_argument(
            "Inner dimensions of A and B must match."
        );
    }

    if (timedIters <= 0) {
        throw std::invalid_argument(
            "benchmark requires at least one timed iteration"
        );
    }

    size_t Mp = ((M + 15) / 16) * 16;
    size_t Kp = ((K + 15) / 16) * 16;
    size_t Np = ((N + 15) / 16) * 16;

    DeviceBuffer<half> Ap(Mp * Kp);
    DeviceBuffer<half> Bp(Kp * Np);
    DeviceBuffer<float> Cp(Mp * Np);

    constexpr int THREADS = 256;
    constexpr int BLOCKS = 256;

    preProcess <<<BLOCKS, THREADS>>>(
        a.data,
        b.data,
        Ap.data(), Bp.data(),
        M, N, K,
        Mp, Np, Kp
        );

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("wmma preProcess launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    constexpr int WARPS_PER_BLOCK =
        WARP_ROWS * WARP_COLS;

    dim3 block(32, WARPS_PER_BLOCK);

    dim3 grid(
        (Np + WARP_COLS * 16 - 1) / (WARP_COLS * 16),
        (Mp + WARP_ROWS * 16 - 1) / (WARP_ROWS * 16)
    );

    // Warm-up launches: absorb one-time costs (module load, caches).
    for (int i = 0; i < warmupIters; i++) {
        wmmaMatmul<WARP_ROWS, WARP_COLS><<<grid, block>>>(
            Ap.data(), Bp.data(), Cp.data(),
            Mp, Np, Kp
            );
    }

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("wmma warm-up launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("wmma warm-up execution failed: ") +
            cudaGetErrorString(err)
        );
    }

    CudaEvent start;
    CudaEvent stop;

    cudaEventRecord(start.get());

    for (int i = 0; i < timedIters; i++) {
        wmmaMatmul<WARP_ROWS, WARP_COLS><<<grid, block>>>(
            Ap.data(), Bp.data(), Cp.data(),
            Mp, Np, Kp
            );
    }

    cudaEventRecord(stop.get());

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("wmma benchmark launch failed: ") +
            cudaGetErrorString(err)
        );
    }

    err = cudaEventSynchronize(stop.get());

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("wmma benchmark execution failed: ") +
            cudaGetErrorString(err)
        );
    }

    float totalMs = 0.0f;
    cudaEventElapsedTime(&totalMs, start.get(), stop.get());

    BenchmarkResult result;
    result.avgMs = totalMs / timedIters;

    double flops = 2.0 * M * N * K;
    result.gflops = flops / (result.avgMs * 1e-3) / 1e9;

    return result;
}
