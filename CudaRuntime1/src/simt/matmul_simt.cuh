#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "tensor.cuh"

template<
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_M,
    bool CHECK_N,
    bool CHECK_K
>
__device__ void loadSharedMemory(
    ConstDeviceTensorView A,
    ConstDeviceTensorView B,
    float (*As)[TILE_K],
    float (*Bs)[TILE_N],
    int tileX,
    int tileY,
    int loadCount
) {
    int globalY = tileY * TILE_M;
    int globalX = tileX * TILE_N;
    int globalStep = loadCount * TILE_K;

    int tId = threadIdx.y * BLOCK_X + threadIdx.x;
    constexpr int nThreads = BLOCK_X * BLOCK_Y;

    constexpr int bElems = TILE_K * TILE_N;
    constexpr int bIters = (bElems + nThreads - 1) / nThreads;
    constexpr int aElems = TILE_M * TILE_K;
    constexpr int aIters = (aElems + nThreads - 1) / nThreads;

    #pragma unroll
    for (int it = 0; it < bIters; ++it)
    {
        int idx = tId + it * nThreads;

        if constexpr (bElems % nThreads != 0) {
            if (idx >= bElems)
                break;
        }

        int localK = idx / TILE_N;
        int localN = idx % TILE_N;

        int kB = globalStep + localK;
        int col = globalX + localN;

        bool valid = true;

        if constexpr (CHECK_K)
            valid = kB < B.shape[0];

        if constexpr (CHECK_N)
            valid = valid && col < B.shape[1];

        if (valid)
            Bs[localK][localN] =
            B.data[kB * B.strides[0] + col];
        else
            Bs[localK][localN] = 0.0f;
    }

    #pragma unroll
    for (int it = 0; it < aIters; ++it)
    {
        int idx = tId + it * nThreads;

        if constexpr (aElems % nThreads != 0) {
            if (idx >= aElems)
                break;
        }

        int localM = idx / TILE_K;
        int localK = idx % TILE_K;

        int kA = globalStep + localK;
        int row = globalY + localM;

        bool valid = true;

        if constexpr (CHECK_K)
            valid = kA < A.shape[1];

        if constexpr (CHECK_M)
            valid = valid && row < A.shape[0];

        if (valid)
            As[localM][localK] =
            A.data[row * A.strides[0] + kA];
        else
            As[localM][localK] = 0.0f;
    }
}

template<
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS
>
__device__ void evaluateMatMul(
    float (*As)[TILE_K],
    float (*Bs)[TILE_N],
    float* threadResults
) {
    constexpr int xPieces =
        (TILE_N + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces =
        (TILE_M + BLOCK_Y - 1) / BLOCK_Y;

    float aRegs[yPieces];
    float bRegs[xPieces];

    for (int k = 0; k < TILE_K; k++)
    {
        for (int i = 0; i < yPieces; i++)
        {
            int localY =
                threadIdx.y + BLOCK_Y * i;

            if constexpr (CHECK_BOUNDS) {
                if (localY < TILE_M)
                    aRegs[i] = As[localY][k];
                else
                    aRegs[i] = 0.0f;
            }
            else {
                aRegs[i] = As[localY][k];
            }
        }

        for (int j = 0; j < xPieces; j++)
        {
            int localX =
                threadIdx.x + BLOCK_X * j;

            if constexpr (CHECK_BOUNDS) {
                if (localX < TILE_N)
                    bRegs[j] = Bs[k][localX];
                else
                    bRegs[j] = 0.0f;
            }
            else {
                bRegs[j] = Bs[k][localX];
            }
        }

        for (int i = 0; i < yPieces; i++)
        {
            for (int j = 0; j < xPieces; j++)
            {
                int idx =
                    i * xPieces + j;

                threadResults[idx] +=
                    aRegs[i] * bRegs[j];
            }
        }
    }
}

template<
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS,
    bool CHECK_M,
    bool CHECK_N
>
__device__ void writeRes(
    DeviceTensorView C,
    int tileX,
    int tileY,
    float* threadResults
) {
    constexpr int xPieces =
        (TILE_N + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces =
        (TILE_M + BLOCK_Y - 1) / BLOCK_Y;
    constexpr int maxResults =
        xPieces * yPieces;

    int globalY = tileY * TILE_M;
    int globalX = tileX * TILE_N;

    for (int idx = 0; idx < maxResults; idx++)
    {
        int i = idx / xPieces;
        int j = idx % xPieces;

        int localY = threadIdx.y + BLOCK_Y * i;
        int localX = threadIdx.x + BLOCK_X * j;

        if constexpr (CHECK_BOUNDS) {
            if (localX >= TILE_N || localY >= TILE_M) continue;
        }

        int row = globalY + localY;
        int col = globalX + localX;

        if constexpr (CHECK_M) {
            if (row >= C.shape[0])
                continue;
        }

        if constexpr (CHECK_N) {
            if (col >= C.shape[1])
                continue;
        }

        C.data[
            row * C.strides[0]
                + col * C.strides[1]
        ] = threadResults[idx];
    }
}

template<
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS,
    bool CHECK_M,
    bool CHECK_N,
    bool CHECK_K
>
__device__ void processK(
    ConstDeviceTensorView A,
    ConstDeviceTensorView B,
    float (*As)[TILE_K],
    float (*Bs)[TILE_N],
    int tileX,
    int tileY,
    int k,
    float* threadResults
) {
    loadSharedMemory<
        TILE_M, TILE_N, TILE_K,
        BLOCK_X, BLOCK_Y,
        CHECK_M, CHECK_N, CHECK_K
    >(
        A, B,
        As, Bs,
        tileX, tileY,
        k
    );

    __syncthreads();

    evaluateMatMul<
        TILE_M, TILE_N, TILE_K,
        BLOCK_X, BLOCK_Y,
        CHECK_BOUNDS
    >(
        As, Bs,
        threadResults
    );

    __syncthreads();
}

template<
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS,
    bool CHECK_M,
    bool CHECK_N
>
__device__ void processX(
    ConstDeviceTensorView A,
    ConstDeviceTensorView B,
    DeviceTensorView C,
    float (*As)[TILE_K],
    float (*Bs)[TILE_N],
    int tileX,
    int tileY,
    int fastK,
    bool remK
) {
    constexpr int xPieces =
        (TILE_N + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces =
        (TILE_M + BLOCK_Y - 1) / BLOCK_Y;
    constexpr int maxResults =
        xPieces * yPieces;

    float threadResults[maxResults] = {};

    for (int k = 0; k < fastK; ++k) {
        processK<
            TILE_M, TILE_N, TILE_K,
            BLOCK_X, BLOCK_Y,
            CHECK_BOUNDS,
            CHECK_M, CHECK_N,
            false
        >(
            A, B,
            As, Bs,
            tileX, tileY,
            k,
            threadResults
        );
    }
    if (remK) {
        processK<
            TILE_M, TILE_N, TILE_K,
            BLOCK_X, BLOCK_Y,
            CHECK_BOUNDS,
            CHECK_M, CHECK_N,
            true
        >(
            A, B,
            As, Bs,
            tileX, tileY,
            fastK,
            threadResults
        );
    }

    writeRes<
        TILE_M, TILE_N, TILE_K,
        BLOCK_X, BLOCK_Y,
        CHECK_BOUNDS, CHECK_M, CHECK_N
    >(
        C,
        tileX, tileY,
        threadResults
    );
}

template<
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS,
    bool CHECK_M
>
__device__ void processY(
    ConstDeviceTensorView A,
    ConstDeviceTensorView B,
    DeviceTensorView C,
    float (*As)[TILE_K],
    float (*Bs)[TILE_N],
    int tileY,
    int fastX,
    bool remX,
    int fastK,
    bool remK
) {
    int tileX = blockIdx.x;

    while (tileX < fastX) {
        processX<
            TILE_M, TILE_N, TILE_K,
            BLOCK_X, BLOCK_Y,
            CHECK_BOUNDS,
            CHECK_M, false
        >(
            A, B, C,
            As, Bs,
            tileX, tileY,
            fastK, remK
        );

        tileX += gridDim.x;
    }

    if (remX && tileX == fastX) {
        processX<
            TILE_M, TILE_N, TILE_K,
            BLOCK_X, BLOCK_Y,
            CHECK_BOUNDS,
            CHECK_M, true
        >(
            A, B, C,
            As, Bs,
            tileX, tileY,
            fastK, remK
        );
    }
}

template<
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int BLOCK_X,
    int BLOCK_Y
>
__global__ void deviceMatMul(
    ConstDeviceTensorView A,
    ConstDeviceTensorView B,
    DeviceTensorView C
) {

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    constexpr bool CHECK_BOUNDS =
        (TILE_M % BLOCK_Y != 0) || (TILE_N % BLOCK_X != 0);

    int fastY = C.shape[0] / TILE_M;
    bool remY = C.shape[0] % TILE_M > 0;

    int fastX = C.shape[1] / TILE_N;
    bool remX = C.shape[1] % TILE_N > 0;

    int fastK = A.shape[1] / TILE_K;
    bool remK = A.shape[1] % TILE_K > 0;

    int tileY = blockIdx.y;
    while (tileY < fastY) {
        processY<
            TILE_M, TILE_N, TILE_K,
            BLOCK_X, BLOCK_Y,
            CHECK_BOUNDS,
            false
        >(
            A, B, C,
            As, Bs,
            tileY,
            fastX, remX,
            fastK, remK
        );

        tileY += gridDim.y;
    }
    if (remY && tileY == fastY) {
        processY<
            TILE_M, TILE_N, TILE_K,
            BLOCK_X, BLOCK_Y,
            CHECK_BOUNDS,
            true
        >(
            A, B, C,
            As, Bs,
            tileY,
            fastX, remX,
            fastK, remK
        );
    }
}

