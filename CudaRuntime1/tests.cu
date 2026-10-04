#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tensor.cuh"
#include "matmul.cuh"
#include "tests.cuh"

// Returns true if every element matches within a relative tolerance.
// NaN/inf compare false against any tolerance, so reject them explicitly.
bool compare(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float relTol = 1e-4f
) {
    if (actual.size() != expected.size()) {
        std::cout << "    size mismatch: expected " << expected.size()
            << ", got " << actual.size() << '\n';
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        float tol = relTol * std::max(1.0f, std::fabs(expected[i]));

        if (!std::isfinite(actual[i]) ||
            !std::isfinite(expected[i]) ||
            std::fabs(actual[i] - expected[i]) > tol) {
            std::cout << "    mismatch at index " << i
                << ": expected " << expected[i]
                << ", got " << actual[i] << '\n';
            return false;
        }
    }

    return true;
}

template<int TILE, int BLOCK_X, int BLOCK_Y>
bool runBenchmark(
    const char* kind,
    const DeviceTensor& A,
    const DeviceTensor& B,
    const std::vector<float>& expected,
    const BenchmarkResult& cpu,
    const BenchmarkResult& cublas
) {
    static_assert(BLOCK_X * BLOCK_Y <= 1024,
        "a block cannot have more than 1024 threads");
    static_assert(2 * TILE * TILE * sizeof(float) <= 48 * 1024,
        "static shared memory is limited to 48 KB per block");

    constexpr int xPieces = (TILE + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces = (TILE + BLOCK_Y - 1) / BLOCK_Y;
    constexpr int resultsPerThread = xPieces * yPieces;

    // Correctness: one separate launch, checked against the CPU result.
    std::vector<float> actual =
        hostMatmul<TILE, BLOCK_X, BLOCK_Y>(A, B).toHost();
    bool passed = compare(actual, expected);

    BenchmarkResult result = benchmarkMatMul<TILE, BLOCK_X, BLOCK_Y>(A, B);

    std::cout
        << "  " << (passed ? "[PASS]" : "[FAIL]")
        << " " << std::left << std::setw(7) << kind << std::right
        << " TILE=" << std::setw(2) << TILE
        << " BLOCK=" << std::setw(2) << BLOCK_X
        << "x" << std::left << std::setw(2) << BLOCK_Y << std::right
        << " threads=" << std::setw(4) << BLOCK_X * BLOCK_Y
        << " results/thread=" << std::setw(2) << resultsPerThread
        << " | " << std::fixed << std::setprecision(4)
        << std::setw(10) << result.avgMs << " ms"
        << " | " << std::setprecision(1)
        << std::setw(8) << result.gflops << " GFLOP/s"
        << " | " << std::setw(7) << cpu.avgMs / result.avgMs << "x CPU"
        << " | " << std::setw(5) << 100.0 * result.gflops / cublas.gflops
        << "% cuBLAS\n"
        << std::defaultfloat;

    return passed;
}

int runSize(
    int M,
    int K,
    int N
) {
    std::vector<float> hostA(static_cast<size_t>(M) * K);
    std::vector<float> hostB(static_cast<size_t>(K) * N);

    for (size_t i = 0; i < hostA.size(); ++i) {
        hostA[i] =
            static_cast<float>(static_cast<int>(i % 7) - 3);
    }

    for (size_t i = 0; i < hostB.size(); ++i) {
        hostB[i] =
            static_cast<float>(static_cast<int>(i % 5) - 2);
    }

    std::cout << "\n[BENCH] "
        << M << "x" << K << " @ " << K << "x" << N << '\n';

    std::vector<float> expected;
    BenchmarkResult cpu =
        benchmarkCpuMatMul(hostA, hostB, M, K, N, expected);

    std::cout
        << "  [CPU ] " << std::fixed << std::setprecision(4)
        << std::setw(10) << cpu.avgMs << " ms"
        << " | " << std::setprecision(1)
        << std::setw(8) << cpu.gflops << " GFLOP/s\n"
        << std::defaultfloat;

    DeviceTensor A(hostA, { M, K });
    DeviceTensor B(hostB, { K, N });

    int failures = 0;

    std::vector<float> cublasResult;
    BenchmarkResult cublas =
        benchmarkCublasMatMul(A, B, cublasResult);
    bool cublasPassed = compare(cublasResult, expected);
    failures += !cublasPassed;

    std::cout
        << "  " << (cublasPassed ? "[PASS]" : "[FAIL]")
        << " cuBLAS  " << std::fixed << std::setprecision(4)
        << std::setw(10) << cublas.avgMs << " ms"
        << " | " << std::setprecision(1)
        << std::setw(8) << cublas.gflops << " GFLOP/s"
        << " | " << std::setw(7) << cpu.avgMs / cublas.avgMs << "x CPU\n"
        << std::defaultfloat;

    // Perfect fit: one thread per output element (TILE == BLOCK_X == BLOCK_Y).
    failures += !runBenchmark< 8,  8,  8>("perfect", A, B, expected, cpu, cublas);
    failures += !runBenchmark<16, 16, 16>("perfect", A, B, expected, cpu, cublas);
    failures += !runBenchmark<32, 32, 32>("perfect", A, B, expected, cpu, cublas);

    // Multiple results per thread (register tiling).
    failures += !runBenchmark<16,  8,  8>("multi", A, B, expected, cpu, cublas);
    failures += !runBenchmark<32, 16, 16>("multi", A, B, expected, cpu, cublas);
    failures += !runBenchmark<32,  8,  8>("multi", A, B, expected, cpu, cublas);
    failures += !runBenchmark<32, 32,  8>("multi", A, B, expected, cpu, cublas);
    failures += !runBenchmark<64, 16, 16>("multi", A, B, expected, cpu, cublas);
    failures += !runBenchmark<64, 32,  8>("multi", A, B, expected, cpu, cublas);
    failures += !runBenchmark<64,  8,  8>("multi", A, B, expected, cpu, cublas);

    return failures;
}

int runBenchmarks() {
    int failures = 0;

    failures += runSize(512, 512, 512);
    failures += runSize(1024, 1024, 1024);
    failures += runSize(2048, 2048, 2048);
    failures += runSize(4096, 4096, 4096);

    // Non-square, non-tile-aligned shape.
    failures += runSize(1000, 3000, 2000);

    std::cout << '\n'
        << (failures == 0 ? "All configurations passed.\n"
                          : "Some configurations FAILED.\n");

    return failures;
}
