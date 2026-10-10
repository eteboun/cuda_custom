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

template<
    int BLOCK_X,
    int BLOCK_Y,
    int WARP_ROWS,
    int WARP_COLS
>
__device__ void loadSharedMemory(
    half As[][16],
    half Bs[][16 * WARP_COLS],
    const half* Ap,
    const half* Bp,
    size_t Kp,
    size_t Np,
    const int globalWarpRow,
    const int globalWarpCol,
    int k
) {
    constexpr int stride = BLOCK_X * BLOCK_Y;

    // Fixed round counts, rounded up. When a tile does not split evenly
    // over the block, the last round is partial and the guard stops the
    // threads past the tile; otherwise the guard compiles away.
    constexpr int aElems = 256 * WARP_ROWS;
    constexpr int rowIters = (aElems + stride - 1) / stride;
    #pragma unroll
    for (int i = 0;
        i < rowIters;
        i++)
    {
        int tId = threadIdx.y * BLOCK_X + threadIdx.x + i * stride;

        if constexpr (aElems % stride != 0) {
            if (tId >= aElems) break;
        }

        int r = tId / 16;
        int c = tId % 16;

        As[r][c] =
            Ap[(globalWarpRow * 16 + r) * Kp
            + k * 16 + c];
    }

    constexpr int bElems = 256 * WARP_COLS;
    constexpr int colIters = (bElems + stride - 1) / stride;
    #pragma unroll
    for (int i = 0;
        i < colIters;
        i++)
    {
        int tId = threadIdx.y * BLOCK_X + threadIdx.x + i * stride;

        if constexpr (bElems % stride != 0) {
            if (tId >= bElems) break;
        }

        int r = tId / (16 * WARP_COLS);
        int c = tId % (16 * WARP_COLS);

        Bs[r][c] =
            Bp[(k * 16 + r) * Np
            + globalWarpCol * 16 + c];
    }
}

template<
    int BLOCK_X,
    int BLOCK_Y,
    int WARP_ROWS,
    int WARP_COLS
>
__device__ void processMatmul(
    const half* Ap,
    const half* Bp,
    float* Cp,
    const size_t Mp,
    const size_t Np,
    const size_t Kp,
    const int globalWarpRow,
    const int globalWarpCol
) {
    __shared__ half As[16 * WARP_ROWS][16];
    __shared__ half Bs[16][16 * WARP_COLS];

    int tId = threadIdx.y * BLOCK_X + threadIdx.x;
    int warpId = tId / WARP_SIZE;

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

    wmma::fill_fragment(cFrag, 0.0f);

    int kIters = Kp / 16;
    for (int k = 0; k < kIters; k++)
    {
        loadSharedMemory<
            BLOCK_X,
            BLOCK_Y,
            WARP_ROWS,
            WARP_COLS
        >(
            As, Bs,
            Ap, Bp,
            Kp, Np,
            globalWarpRow, globalWarpCol,
            k
        );

        __syncthreads();

        wmma::load_matrix_sync(
            aFrag,
            &As[localWarpRow * 16][0],
            16
        );

        wmma::load_matrix_sync(
            bFrag,
            &Bs[0][localWarpCol * 16],
            16 * WARP_COLS
        );

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncthreads();
    }

    int warpRow = globalWarpRow + localWarpRow;
    int warpCol = globalWarpCol + localWarpCol;

    wmma::store_matrix_sync(
        Cp + warpRow * 16 * Np + warpCol * 16,
        cFrag,
        Np,
        wmma::mem_row_major
    );
}

template<
    int BLOCK_X,
    int BLOCK_Y,
    int WARP_ROWS,
    int WARP_COLS
>
__global__ void wmmaMatmul(
    const half* Ap,
    const half* Bp,
    float* Cp,
    const size_t Mp,
    const size_t Np,
    const size_t Kp
) {
    // One warp per 16x16 output tile: the block must hold exactly
    // WARP_ROWS x WARP_COLS warps.
    static_assert(BLOCK_X * BLOCK_Y == WARP_SIZE * WARP_ROWS * WARP_COLS,
        "BLOCK_X * BLOCK_Y must equal 32 * WARP_ROWS * WARP_COLS");

    int rowStride = gridDim.y * WARP_ROWS;
    int colStride = gridDim.x * WARP_COLS;

    int rowIters = Mp / 16;
    int colIters = Np / 16;

    for (int globalWarpRow = blockIdx.y * WARP_ROWS;
        globalWarpRow < rowIters;
        globalWarpRow += rowStride)
    {
        for (int globalWarpCol = blockIdx.x * WARP_COLS;
            globalWarpCol < colIters;
            globalWarpCol += colStride)
        {
            processMatmul<
                BLOCK_X,
                BLOCK_Y,
                WARP_ROWS,
                WARP_COLS
            >(
                Ap, Bp, Cp,
                Mp, Np, Kp,
                globalWarpRow, globalWarpCol
            );
        }
    }
}

