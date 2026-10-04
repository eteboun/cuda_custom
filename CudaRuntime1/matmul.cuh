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
    int globalY = tileY * TILE;
    int globalX = tileX * TILE;
    int globalStep = loadCount * TILE;

    for (int localY = threadIdx.y;
        localY < TILE;
        localY += BLOCK_Y)
    {
        int row = globalY + localY;
        int kB = localY + globalStep;

        bool validARow = row < A.shape[0];
        bool validBRow = kB < B.shape[0];

        size_t aRow = validARow ? row * A.strides[0] : 0;
        size_t bRow = validBRow ? kB * B.strides[0] : 0;

        for (int localX = threadIdx.x;
            localX < TILE;
            localX += BLOCK_X)
        {
            int col = globalX + localX;
            int kA = localX + globalStep;

            if (validARow && kA < A.shape[1]) {
                As[localY][localX] =
                    A.data[aRow + kA * A.strides[1]];
            }
            else {
                As[localY][localX] = 0.0f;
            }

            if (validBRow && col < B.shape[1]) {
                Bs[localY][localX] =
                    B.data[bRow + col * B.strides[1]];
            }
            else {
                Bs[localY][localX] = 0.0f;
            }
        }
    }
}

template<
    int TILE,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS
>
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

            if constexpr (CHECK_BOUNDS) {
                if (localY < TILE)
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
                if (localX < TILE)
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
                if constexpr (CHECK_BOUNDS) {
                    int localY =
                        threadIdx.y + BLOCK_Y * i;

                    int localX =
                        threadIdx.x + BLOCK_X * j;

                    if (localX >= TILE || localY >= TILE)
                        continue;
                }

                int idx =
                    i * xPieces + j;

                threadResults[idx] +=
                    aRegs[i] * bRegs[j];
            }
        }
    }
}

template<
    int TILE,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS
>
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

    int globalY = tileY * TILE;
    int globalX = tileX * TILE;

    for (int idx = 0; idx < maxResults; idx++)
    {
        int i = idx / xPieces;
        int j = idx % xPieces;

        int localY = threadIdx.y + BLOCK_Y * i;
        int localX = threadIdx.x + BLOCK_X * j;

        if constexpr (CHECK_BOUNDS) {
            if (localX >= TILE || localY >= TILE) continue;
        }

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

    // Fast path: if the block tiles the TILE evenly, no thread's
    // localX/localY can reach TILE, so those checks can be skipped.
    constexpr bool CHECK_BOUNDS =
        (TILE % BLOCK_X != 0) || (TILE % BLOCK_Y != 0);

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

                evaluateMatMul<TILE, BLOCK_X, BLOCK_Y, CHECK_BOUNDS>(
                    As, Bs,
                    threadResults
                );

                __syncthreads();
            }

            writeRes<TILE, BLOCK_X, BLOCK_Y, CHECK_BOUNDS>(
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

template<int TILE, int BLOCK_X, int BLOCK_Y>
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
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    // Warm-up launches: absorb one-time costs (module load, caches).
    for (int i = 0; i < warmupIters; i++) {
        deviceMatMul<TILE, BLOCK_X, BLOCK_Y>
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
        deviceMatMul<TILE, BLOCK_X, BLOCK_Y>
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
