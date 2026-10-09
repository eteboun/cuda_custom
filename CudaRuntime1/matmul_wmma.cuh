#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "tensor.cuh"

using namespace nvcuda;

constexpr int WARP_SIZE = 32;

// Defined in matmul_wmma.cu: they are not templates, so a definition
// here would be compiled into every .cu that includes this header.
__global__ void preProcess(
    const float* A,
    const float* B,
    half* Ap,
    half* Bp,
    const size_t M,
    const size_t N,
    const size_t K,
    size_t paddedM,
    size_t paddedN,
    size_t paddedK
);

__global__ void postProcess(
    const float* Cp,
    float* C,
    size_t M,
    size_t N,
    size_t Np
);

template<int WARP_ROWS, int WARP_COLS>
__global__ void wmmaMatmul(
    const half* Ap,
    const half* Bp,
    float* Cp,
    const size_t Mp,
    const size_t Np,
    const size_t Kp
) {
    int tId = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = tId / WARP_SIZE;

    int globalWarpRow = blockIdx.y * WARP_ROWS;
    int globalWarpCol = blockIdx.x * WARP_COLS;

    int localWarpRow = warpId / WARP_COLS;
    int localWarpCol = warpId % WARP_COLS;

    wmma::fragment<
        wmma::matrix_a,
        16, 16, 16,
        half,
        wmma::row_major
    > aFrag;

    wmma::fragment <
        wmma::matrix_b,
        16, 16, 16,
        half,
        wmma::row_major
    > bFrag;

    wmma::fragment<
        wmma::accumulator,
        16, 16, 16,
        float
    > cFrag;

    int tileRows = Mp / 16;
    int tileCols = Np / 16;

    int rowStride = gridDim.y * WARP_ROWS;
    int colStride = gridDim.x * WARP_COLS;

    int kIters = Kp / 16;
    for (int warpRow = globalWarpRow + localWarpRow;
        warpRow < tileRows;
        warpRow += rowStride)
    {
        for (int warpCol = globalWarpCol + localWarpCol;
            warpCol < tileCols;
            warpCol += colStride)
        {
            wmma::fill_fragment(cFrag, 0.0f);

            for (int k = 0; k < kIters; k++)
            {
                wmma::load_matrix_sync(
                    aFrag,
                    Ap + warpRow * 16 * Kp + k * 16,
                    Kp
                );

                wmma::load_matrix_sync(
                    bFrag,
                    Bp + warpCol * 16 + k * 16 * Np,
                    Np
                );

                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }

            wmma::store_matrix_sync(
                Cp + warpRow * 16 * Np + warpCol * 16,
                cFrag,
                Np,
                wmma::mem_row_major
            );
        }
    }
}
