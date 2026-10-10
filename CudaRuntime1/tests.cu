#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#include "tensor.cuh"
#include "matmul.cuh"
#include "matmul_simt_host.cuh"
#include "matmul_wmma_host.cuh"
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

struct BestConfig {
    double gflops = 0.0;
    float avgMs = 0.0f;
    int tileM = 0;
    int tileN = 0;
    int tileK = 0;
    int blockX = 0;
    int blockY = 0;
};

struct BestWmmaConfig {
    double gflops = 0.0;
    float avgMs = 0.0f;
    int warpRows = 0;
    int warpCols = 0;
};

template<int TILE_M, int TILE_N, int TILE_K, int BLOCK_X, int BLOCK_Y>
bool runBenchmark(
    const DeviceTensor& A,
    const DeviceTensor& B,
    const std::vector<float>& expected,
    const BenchmarkResult& cpu,
    const BenchmarkResult& cublas,
    BestConfig& best,
    int warmupIters,
    int timedIters
) {
    static_assert(BLOCK_X * BLOCK_Y <= 1024,
        "a block cannot have more than 1024 threads");
    static_assert(
        (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float) <= 48 * 1024,
        "static shared memory is limited to 48 KB per block");

    constexpr int xPieces = (TILE_N + BLOCK_X - 1) / BLOCK_X;
    constexpr int yPieces = (TILE_M + BLOCK_Y - 1) / BLOCK_Y;
    constexpr int resultsPerThread = xPieces * yPieces;

    // perfect: one output per thread; square/wide/tall: shape of the block.
    const char* kind =
        (resultsPerThread == 1) ? "perfect" :
        (BLOCK_X == BLOCK_Y) ? "square" :
        (BLOCK_X > BLOCK_Y) ? "wide" : "tall";

    // Correctness: one separate launch, checked against the CPU result.
    std::vector<float> actual =
        hostSimtMatmul<TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y>(A, B).toHost();
    bool passed = compare(actual, expected);

    BenchmarkResult result =
        benchmarkMatMul<TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y>(
            A, B, warmupIters, timedIters
        );

    if (passed && result.gflops > best.gflops) {
        best.gflops = result.gflops;
        best.avgMs = result.avgMs;
        best.tileM = TILE_M;
        best.tileN = TILE_N;
        best.tileK = TILE_K;
        best.blockX = BLOCK_X;
        best.blockY = BLOCK_Y;
    }

    std::cout
        << "  " << (passed ? "[PASS]" : "[FAIL]")
        << " simt " << std::left << std::setw(7) << kind << std::right
        << " TILE=" << std::setw(3) << TILE_M
        << "x" << std::setw(3) << TILE_N
        << "x" << std::setw(2) << TILE_K
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

template<int WARP_ROWS, int WARP_COLS>
bool runWmmaBenchmark(
    const DeviceTensor& A,
    const DeviceTensor& B,
    const std::vector<float>& expected,
    const BenchmarkResult& cpu,
    const BenchmarkResult& cublas,
    BestWmmaConfig& best,
    int warmupIters,
    int timedIters
) {
    constexpr int threads = 32 * WARP_ROWS * WARP_COLS;

    static_assert(threads <= 1024,
        "a block cannot have more than 1024 threads");

    // Correctness: one separate run, checked against the CPU result.
    // The test data are small integers, which fp16 holds exactly, so the
    // usual tolerance still applies.
    std::vector<float> actual =
        hostWmmaMatmul<WARP_ROWS, WARP_COLS>(A, B).toHost();
    bool passed = compare(actual, expected);

    BenchmarkResult result =
        benchmarkWmmaMatMul<WARP_ROWS, WARP_COLS>(
            A, B, warmupIters, timedIters
        );

    if (passed && result.gflops > best.gflops) {
        best.gflops = result.gflops;
        best.avgMs = result.avgMs;
        best.warpRows = WARP_ROWS;
        best.warpCols = WARP_COLS;
    }

    std::cout
        << "  " << (passed ? "[PASS]" : "[FAIL]")
        << " wmma    WARPS=" << WARP_ROWS << "x" << WARP_COLS
        << " threads=" << std::setw(4) << threads
        << " block tile=" << std::setw(2) << WARP_ROWS * 16
        << "x" << std::left << std::setw(2) << WARP_COLS * 16 << std::right
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

// One row of the per-size comparison table.
void printCompareRow(
    const std::string& label,
    float avgMs,
    double gflops,
    const BenchmarkResult& cpu,
    const BenchmarkResult& cublas
) {
    std::cout
        << "    " << std::left << std::setw(36) << label << std::right
        << std::fixed << std::setprecision(4)
        << std::setw(11) << avgMs << " ms"
        << " | " << std::setprecision(1)
        << std::setw(8) << gflops << " GFLOP/s"
        << " | " << std::setw(8) << cpu.avgMs / avgMs << "x CPU"
        << " | " << std::setw(6) << 100.0 * gflops / cublas.gflops
        << "% cuBLAS\n"
        << std::defaultfloat;
}

int runSize(
    int M,
    int K,
    int N,
    int warmupIters,
    int timedIters
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
        << M << "x" << K << " @ " << K << "x" << N
        << "  (warmup=" << warmupIters
        << ", avg over " << timedIters << " runs)\n";

    std::vector<float> expected;
    BenchmarkResult cpu =
        benchmarkCpuMatMul(hostA, hostB, M, K, N, expected);

    std::cout
        << "  [CPU ] " << std::fixed << std::setprecision(4)
        << std::setw(10) << cpu.avgMs << " ms"
        << " | " << std::setprecision(1)
        << std::setw(8) << cpu.gflops << " GFLOP/s\n"
        << std::defaultfloat;

    DeviceTensor A(hostA, { static_cast<size_t>(M), static_cast<size_t>(K) });
    DeviceTensor B(hostB, { static_cast<size_t>(K), static_cast<size_t>(N) });

    int failures = 0;

    std::vector<float> cublasResult;
    BenchmarkResult cublas = benchmarkCublasMatMul(
        A, B, cublasResult, warmupIters, timedIters
    );
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

    BestConfig best;

    // Reference: the best square tiles from the previous sweep.
    failures += !runBenchmark< 32,  32,  32,   8,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 32,  32,  32,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 48,  48,  48,  24,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 48,  48,  48,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  64,  16,  16>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  64,  32,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  64,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  64,   8,  16>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);

    // TILE_K sweep on the 64x64 output tile: less shared memory per block
    // (more blocks per SM) against more __syncthreads per output tile.
    // TILE_K = 24 does not divide any test size, so the K edge is exercised.
    failures += !runBenchmark< 64,  64,   8,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  16,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  24,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  32,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  16,  16,  16>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  64,  32,  16,  16>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);

    // Rectangular output tiles (TILE_M != TILE_N).
    failures += !runBenchmark< 32,  64,  32,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  32,  32,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark<128,  32,  32,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 32, 128,  32,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark<128,  64,  16,  16,  16>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64, 128,  16,  32,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 96,  64,  16,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64,  96,  16,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);

    // Large output tiles: 64 results per thread, small TILE_K so shared
    // memory stays under 48 KB.
    failures += !runBenchmark<128,  64,  16,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark< 64, 128,  16,  16,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark<128, 128,   8,  16,  16>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark<128, 128,  16,  16,  16>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);
    failures += !runBenchmark<128, 128,   8,  32,   8>(A, B, expected, cpu, cublas, best, warmupIters, timedIters);

    // Tensor cores (WMMA, fp16 inputs, fp32 accumulate). Each warp computes
    // one 16x16 output tile; a block holds WARP_ROWS x WARP_COLS warps.
    BestWmmaConfig bestWmma;

    std::cout << '\n';
    failures += !runWmmaBenchmark<1, 1>(A, B, expected, cpu, cublas, bestWmma, warmupIters, timedIters);
    failures += !runWmmaBenchmark<2, 1>(A, B, expected, cpu, cublas, bestWmma, warmupIters, timedIters);
    failures += !runWmmaBenchmark<1, 2>(A, B, expected, cpu, cublas, bestWmma, warmupIters, timedIters);
    failures += !runWmmaBenchmark<2, 2>(A, B, expected, cpu, cublas, bestWmma, warmupIters, timedIters);
    failures += !runWmmaBenchmark<4, 2>(A, B, expected, cpu, cublas, bestWmma, warmupIters, timedIters);
    failures += !runWmmaBenchmark<2, 4>(A, B, expected, cpu, cublas, bestWmma, warmupIters, timedIters);
    failures += !runWmmaBenchmark<4, 4>(A, B, expected, cpu, cublas, bestWmma, warmupIters, timedIters);

    // CPU, cuBLAS, and the best SIMT and WMMA configurations side by side.
    std::cout << "\n  [COMPARE] "
        << M << "x" << K << " @ " << K << "x" << N << '\n';

    printCompareRow("CPU (fp32, all threads)",
        cpu.avgMs, cpu.gflops, cpu, cublas);
    printCompareRow("cuBLAS gemmEx (fp16 in, fp32 out)",
        cublas.avgMs, cublas.gflops, cpu, cublas);

    if (best.gflops > 0.0) {
        printCompareRow(
            "SIMT best TILE=" + std::to_string(best.tileM) +
            "x" + std::to_string(best.tileN) +
            "x" + std::to_string(best.tileK) +
            " BLOCK=" + std::to_string(best.blockX) +
            "x" + std::to_string(best.blockY),
            best.avgMs, best.gflops, cpu, cublas);
    }

    if (bestWmma.gflops > 0.0) {
        printCompareRow(
            "WMMA best WARPS=" + std::to_string(bestWmma.warpRows) +
            "x" + std::to_string(bestWmma.warpCols) + " (fp16)",
            bestWmma.avgMs, bestWmma.gflops, cpu, cublas);
    }

    return failures;
}

int runBenchmarks() {
    int failures = 0;

    // Iteration counts are chosen so each timed loop lasts long enough
    // (hundreds of ms) to average out launch jitter and clock changes:
    // a 512x512 run takes ~0.1 ms, a 4096x4096 run tens of ms.
    // Warm-up also lets the laptop GPU reach a steady boost clock.
    failures += runSize(512, 512, 512, 50, 1000);
    failures += runSize(2048, 2048, 2048, 20, 200);
    failures += runSize(4096, 4096, 4096, 10, 50);

    std::cout << '\n'
        << (failures == 0 ? "All configurations passed.\n"
                          : "Some configurations FAILED.\n");

    return failures;
}
