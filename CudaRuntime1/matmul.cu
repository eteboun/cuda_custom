#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>

#include "matmul.cuh"

// Link cuBLAS without touching the project's linker settings.
#pragma comment(lib, "cublas.lib")

// Computes rows [rowBegin, rowEnd) of C.
// i-k-j loop order: the inner loop walks B and C contiguously,
// which is cache friendly and lets the compiler vectorize it.
static void cpuMatmulRows(
    const float* A,
    const float* B,
    float* C,
    int rowBegin,
    int rowEnd,
    int K,
    int N
) {
    for (int i = rowBegin; i < rowEnd; ++i) {
        float* cRow = C + static_cast<size_t>(i) * N;

        for (int j = 0; j < N; ++j) {
            cRow[j] = 0.0f;
        }

        for (int k = 0; k < K; ++k) {
            const float a = A[static_cast<size_t>(i) * K + k];
            const float* bRow = B + static_cast<size_t>(k) * N;

            for (int j = 0; j < N; ++j) {
                cRow[j] += a * bRow[j];
            }
        }
    }
}

std::vector<float> cpuMatmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int K,
    int N
) {
    if (A.size() != static_cast<size_t>(M) * K ||
        B.size() != static_cast<size_t>(K) * N) {
        throw std::invalid_argument(
            "cpuMatmul: data size does not match M, K, N"
        );
    }

    std::vector<float> C(static_cast<size_t>(M) * N);

    // Split the rows of C across all hardware threads.
    int numThreads =
        static_cast<int>(std::thread::hardware_concurrency());
    numThreads = std::max(1, std::min(numThreads, M));

    int rowsPerThread = (M + numThreads - 1) / numThreads;

    std::vector<std::thread> threads;

    for (int t = 0; t < numThreads; ++t) {
        int rowBegin = t * rowsPerThread;
        int rowEnd = std::min(M, rowBegin + rowsPerThread);

        if (rowBegin >= rowEnd) {
            break;
        }

        threads.emplace_back(
            cpuMatmulRows,
            A.data(), B.data(), C.data(),
            rowBegin, rowEnd,
            K, N
        );
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    return C;
}

BenchmarkResult benchmarkCpuMatMul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int K,
    int N,
    std::vector<float>& C
) {
    // A single run: the CPU is slow enough that one run is already
    // long compared to timer resolution.
    auto start = std::chrono::high_resolution_clock::now();

    C = cpuMatmul(A, B, M, K, N);

    auto stop = std::chrono::high_resolution_clock::now();

    BenchmarkResult result;
    result.avgMs =
        std::chrono::duration<float, std::milli>(stop - start).count();

    double flops = 2.0 * M * N * K;
    result.gflops = flops / (result.avgMs * 1e-3) / 1e9;

    return result;
}

static void checkCublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + " failed: " +
            cublasGetStatusString(status)
        );
    }
}

// Owns a cublasHandle_t so it is destroyed even if an exception is thrown.
class CublasHandle {
private:
    cublasHandle_t handle;

public:
    CublasHandle() {
        checkCublas(cublasCreate(&handle), "cublasCreate");
    }

    ~CublasHandle() {
        cublasDestroy(handle);
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle;
    }
};

// cuBLAS is column-major, our tensors are row-major.
// A row-major matrix read as column-major is its transpose, so
// C = A * B (row-major) is the same memory as C^T = B^T * A^T (column-major).
// Passing B first and A second computes exactly that, with no copies.
static void cublasRowMajorSgemm(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, N,
            A, K,
            &beta,
            C, N
        ),
        "cublasSgemm"
    );
}

BenchmarkResult benchmarkCublasMatMul(
    const DeviceTensor& A,
    const DeviceTensor& B,
    std::vector<float>& C,
    int warmupIters,
    int timedIters
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

    // cublasSgemm takes int dimensions.
    int M = static_cast<int>(a.shape[0]);
    int K = static_cast<int>(a.shape[1]);
    int N = static_cast<int>(b.shape[1]);

    DeviceTensor deviceC({ a.shape[0], b.shape[1] });

    float* c = deviceC.view().data;

    // Handle creation is slow and done once, so it stays out of the timing.
    CublasHandle handle;

    // Warm-up also absorbs cuBLAS's one-time kernel selection and loading.
    for (int i = 0; i < warmupIters; i++) {
        cublasRowMajorSgemm(handle.get(), a.data, b.data, c, M, K, N);
    }

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cuBLAS warm-up execution failed: ") +
            cudaGetErrorString(err)
        );
    }

    CudaEvent start;
    CudaEvent stop;

    cudaEventRecord(start.get());

    for (int i = 0; i < timedIters; i++) {
        cublasRowMajorSgemm(handle.get(), a.data, b.data, c, M, K, N);
    }

    cudaEventRecord(stop.get());

    err = cudaEventSynchronize(stop.get());

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cuBLAS benchmark execution failed: ") +
            cudaGetErrorString(err)
        );
    }

    float totalMs = 0.0f;
    cudaEventElapsedTime(&totalMs, start.get(), stop.get());

    BenchmarkResult result;
    result.avgMs = totalMs / timedIters;

    double flops = 2.0 * M * N * K;
    result.gflops = flops / (result.avgMs * 1e-3) / 1e9;

    C = deviceC.toHost();

    return result;
}
