#pragma once

#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "buffer.cuh"
#include "tensor.cuh"
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
