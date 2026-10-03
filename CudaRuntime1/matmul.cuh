#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "tensor.cuh"

template<int TILE, int BLOCK_X, int BLOCK_Y>
__device__ void loadSharedMemory(
    DeviceTensorView A,
    DeviceTensorView B,
    float (*As)[TILE],
    float (*Bs)[TILE],
    int tileX,
    int tileY,
    int loadCount
) {
    for (int localY = threadIdx.y;
        localY < TILE;
        localY += BLOCK_Y)
    {
        for (int localX = threadIdx.x;
            localX < TILE;
            localX += BLOCK_X) {

            int globalY = tileY * TILE;
            int globalX = tileX * TILE;

            int row = globalY + localY;
            int col = globalX + localX;

            int kA = localX + loadCount * TILE;
            int kB = localY + loadCount * TILE;

            if (kA < A.shape[1] && row < A.shape[0]) {
                As[localY][localX] = A.data[
                    row * A.strides[0]
                        + kA * A.strides[1]
                ];
            }
            else {
                As[localY][localX] = 0.0f;
            }

            if (kB < B.shape[0] && col < B.shape[1]) {
                Bs[localY][localX] = B.data[
                    kB * B.strides[0]
                        + col * B.strides[1]
                ];
            }
            else {
                Bs[localY][localX] = 0.0f;
            }
        }
    }
}

template<int TILE, int BLOCK_X, int BLOCK_Y>
__device__ void evaluateMatMul(
    float (*As)[TILE],
    float (*Bs)[TILE],
    float* threadResults
) {
    constexpr int xPieces =
        (TILE + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces =
        (TILE + BLOCK_Y - 1) / BLOCK_Y;

    float aRegs[yPieces];
    float bRegs[xPieces];

    for (int k = 0; k < TILE; k++)
    {
        for (int i = 0; i < yPieces; i++)
        {
            int localY =
                threadIdx.y + BLOCK_Y * i;

            if (localY < TILE)
                aRegs[i] = As[localY][k];
            else
                aRegs[i] = 0.0f;
        }

        for (int j = 0; j < xPieces; j++)
        {
            int localX =
                threadIdx.x + BLOCK_X * j;

            if (localX < TILE)
                bRegs[j] = Bs[k][localX];
            else
                bRegs[j] = 0.0f;
        }

        for (int i = 0; i < yPieces; i++)
        {
            for (int j = 0; j < xPieces; j++)
            {
                int localY =
                    threadIdx.y + BLOCK_Y * i;

                int localX =
                    threadIdx.x + BLOCK_X * j;

                if (localX >= TILE ||
                    localY >= TILE)
                    continue;

                int idx =
                    i * xPieces + j;

                threadResults[idx] +=
                    aRegs[i] * bRegs[j];
            }
        }
    }
}

template<int TILE, int BLOCK_X, int BLOCK_Y>
__device__ void writeRes(
    DeviceTensorView C,
    int tileX,
    int tileY,
    float* threadResults
) {
    constexpr int xPieces =
        (TILE + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces =
        (TILE + BLOCK_Y - 1) / BLOCK_Y;
    constexpr int maxResults =
        xPieces * yPieces;

    for (int idx = 0; idx < maxResults; idx++)
    {
        int i = idx / xPieces;
        int j = idx % xPieces;

        int localY = threadIdx.y + BLOCK_Y * i;
        int localX = threadIdx.x + BLOCK_X * j;

        if (localX >= TILE || localY >= TILE) continue;

        int globalY = tileY * TILE;
        int globalX = tileX * TILE;

        int row = globalY + localY;
        int col = globalX + localX;

        if (row >= C.shape[0] || col >= C.shape[1]) continue;

        C.data[
            row * C.strides[0]
                + col * C.strides[1]
        ] = threadResults[idx];

    }
}

template<int TILE, int BLOCK_X, int BLOCK_Y>
__global__ void deviceMatMul(
    DeviceTensorView A,
    DeviceTensorView B,
    DeviceTensorView C
) {

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int loadCounts = (A.shape[1] + TILE - 1) / TILE;

    constexpr int xPieces =
        (TILE + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces =
        (TILE + BLOCK_Y - 1) / BLOCK_Y;
    constexpr int maxResults =
        xPieces * yPieces;

    int xTiles =
        (C.shape[1] + TILE - 1) / TILE;
    int yTiles =
        (C.shape[0] + TILE - 1) / TILE;

    for (int tileY = blockIdx.y;
        tileY < yTiles;
        tileY += gridDim.y)
    {
        for (int tileX = blockIdx.x;
            tileX < xTiles;
            tileX += gridDim.x)
        {

            float threadResults[maxResults] = {};

            for (int loadCount = 0; loadCount < loadCounts; loadCount++)
            {
                loadSharedMemory<TILE, BLOCK_X, BLOCK_Y>(
                    A, B,
                    As, Bs,
                    tileX, tileY,
                    loadCount
                );

                __syncthreads();

                evaluateMatMul<TILE, BLOCK_X, BLOCK_Y>(
                    As, Bs,
                    threadResults
                );

                __syncthreads();
            }

            writeRes<TILE, BLOCK_X, BLOCK_Y>(
                C,
                tileX, tileY,
                threadResults
            );
        }
    }
}

template<int TILE, int BLOCK_X, int BLOCK_Y>
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
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    deviceMatMul<TILE, BLOCK_X, BLOCK_Y>
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
