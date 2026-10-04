#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "tensor.cuh"

template<
    int TILE,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_M,
    bool CHECK_N,
    bool CHECK_K
>
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

        // Disabled checks stay `true` at compile time, so the
        // compiler drops them from the conditions below.
        bool validARow = true;
        bool validBRow = true;

        if constexpr (CHECK_M)
            validARow = row < A.shape[0];

        if constexpr (CHECK_K)
            validBRow = kB < B.shape[0];

        size_t aRow = validARow ? row * A.strides[0] : 0;
        size_t bRow = validBRow ? kB * B.strides[0] : 0;

        for (int localX = threadIdx.x;
            localX < TILE;
            localX += BLOCK_X)
        {
            int col = globalX + localX;
            int kA = localX + globalStep;

            // ==================== A -> As ====================

            bool validA = validARow;

            if constexpr (CHECK_K)
                validA = validA && kA < A.shape[1];

            if (validA) {
                As[localY][localX] =
                    A.data[aRow + kA * A.strides[1]];
            }
            else {
                As[localY][localX] = 0.0f;
            }

            // ==================== B -> Bs ====================

            bool validB = validBRow;

            if constexpr (CHECK_N)
                validB = validB && col < B.shape[1];

            if (validB) {
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

        if constexpr (CHECK_M && CHECK_N) {
            if (row >= C.shape[0] || col >= C.shape[1])
                continue;
        }
        else if constexpr (CHECK_M) {
            if (row >= C.shape[0])
                continue;
        }
        else if constexpr (CHECK_N) {
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
    int TILE,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS,
    bool CHECK_M,
    bool CHECK_N,
    bool CHECK_K
>
__device__ void processK(
    DeviceTensorView A,
    DeviceTensorView B,
    float (*As)[TILE],
    float (*Bs)[TILE],
    int tileX,
    int tileY,
    int k,
    float* threadResults
) {
    loadSharedMemory<
        TILE, BLOCK_X, BLOCK_Y,
        CHECK_M, CHECK_N, CHECK_K
    >(
        A, B,
        As, Bs,
        tileX, tileY,
        k
    );

    __syncthreads();

    evaluateMatMul<
        TILE, BLOCK_X, BLOCK_Y,
        CHECK_BOUNDS
    >(
        As, Bs,
        threadResults
    );

    __syncthreads();
}

template<
    int TILE,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS,
    bool CHECK_M,
    bool CHECK_N
>
__device__ void processX(
    DeviceTensorView A,
    DeviceTensorView B,
    DeviceTensorView C,
    float (*As)[TILE],
    float (*Bs)[TILE],
    int tileX,
    int tileY,
    int fastK,
    bool remK
) {
    constexpr int xPieces =
        (TILE + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces =
        (TILE + BLOCK_Y - 1) / BLOCK_Y;
    constexpr int maxResults =
        xPieces * yPieces;

    float threadResults[maxResults] = {};

    for (int k = 0; k < fastK; ++k) {
        processK<
            TILE, BLOCK_X,
            BLOCK_Y, CHECK_BOUNDS,
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
            TILE, BLOCK_X,
            BLOCK_Y, CHECK_BOUNDS,
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
        TILE, BLOCK_X,
        BLOCK_Y, CHECK_BOUNDS,
        CHECK_M, CHECK_N
    >(
        C,
        tileX, tileY,
        threadResults
    );
}

template<
    int TILE,
    int BLOCK_X,
    int BLOCK_Y,
    bool CHECK_BOUNDS,
    bool CHECK_M
>
__device__ void processY(
    DeviceTensorView A,
    DeviceTensorView B,
    DeviceTensorView C,
    float (*As)[TILE],
    float (*Bs)[TILE],
    int tileY,
    int fastX,
    bool remX,
    int fastK,
    bool remK
) {
    int tileX = blockIdx.x;

    while (tileX < fastX) {
        processX<
            TILE, BLOCK_X, BLOCK_Y,
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
            TILE, BLOCK_X, BLOCK_Y,
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

template<int TILE, int BLOCK_X, int BLOCK_Y>
__global__ void deviceMatMul(
    DeviceTensorView A,
    DeviceTensorView B,
    DeviceTensorView C
) {

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // Fast path: if the block tiles the TILE evenly, no thread's
    // localX/localY can reach TILE, so those checks can be skipped.
    constexpr bool CHECK_BOUNDS =
        (TILE % BLOCK_X != 0) || (TILE % BLOCK_Y != 0);

    int fastY = C.shape[0] / TILE;
    bool remY = C.shape[0] % TILE > 0;

    int fastX = C.shape[1] / TILE;
    bool remX = C.shape[1] % TILE > 0;

    int fastK = A.shape[1] / TILE;
    bool remK = A.shape[1] % TILE > 0;

    int tileY = blockIdx.y;
    while (tileY < fastY) {
        processY <
            TILE, BLOCK_X, BLOCK_Y,
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
            TILE, BLOCK_X, BLOCK_Y,
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
