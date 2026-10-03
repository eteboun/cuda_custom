#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

#include "tensor.cuh"
#include "matmul.cuh"

std::vector<float> referenceMatmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int K,
    int N
) {
    std::vector<float> C(M * N, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * N + j] +=
                    A[i * K + k] *
                    B[k * N + j];
            }
        }
    }

    return C;
}

bool compare(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float epsilon = 1e-4f
) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        // NaN/inf compare false against epsilon, so reject them explicitly.
        if (!std::isfinite(actual[i]) ||
            !std::isfinite(expected[i]) ||
            std::fabs(actual[i] - expected[i]) > epsilon) {
            std::cout
                << "Mismatch at index " << i
                << ": expected " << expected[i]
                << ", got " << actual[i]
                << '\n';

            return false;
        }
    }

    return true;
}

void runTest(
    const char* name,
    int M,
    int K,
    int N
) {
    std::vector<float> hostA(M * K);
    std::vector<float> hostB(K * N);

    // Non-trivial deterministic values.
    // Cast to int before subtracting: size_t would wrap around to ~2^64.
    for (size_t i = 0; i < hostA.size(); ++i) {
        hostA[i] =
            static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6);
    }

    for (size_t i = 0; i < hostB.size(); ++i) {
        hostB[i] =
            static_cast<float>(static_cast<int>((i * 5 + 1) % 11) - 5);
    }

    DeviceTensor A(hostA, { M, K });
    DeviceTensor B(hostB, { K, N });

    DeviceTensor C = hostMatmul<16, 8, 8>(A, B);

    std::vector<float> expected =
        referenceMatmul(hostA, hostB, M, K, N);

    std::vector<float> actual = C.toHost();

    if (compare(actual, expected)) {
        std::cout << "[PASS] " << name
            << " (" << M << "x" << K
            << " @ " << K << "x" << N
            << ")\n";
    }
    else {
        std::cout << "[FAIL] " << name
            << " (" << M << "x" << K
            << " @ " << K << "x" << N
            << ")\n";
    }
}

int main() {
    // 1. Everything smaller than TILE.
    runTest(
        "small",
        3, 5, 4
    );

    // 2. Perfect TILE boundaries.
    runTest(
        "perfect tiles",
        32, 32, 32
    );

    // 3. Partial M tile only.
    runTest(
        "partial M",
        37, 32, 32
    );

    // 4. Partial N tile only.
    runTest(
        "partial N",
        32, 32, 43
    );

    // 5. Partial K tile only.
    // Very important for shared-memory zero padding.
    runTest(
        "partial K",
        32, 37, 32
    );

    // 6. Everything partial.
    runTest(
        "all partial",
        37, 29, 43
    );

    // 7. Very asymmetric matrix.
    runTest(
        "wide output",
        7, 23, 101
    );

    // 8. Opposite asymmetry.
    runTest(
        "tall output",
        103, 19, 5
    );

    // 9. K spans many tiles.
    runTest(
        "many K tiles",
        31, 131, 29
    );

    // 10. Large enough to exercise many C tiles.
    runTest(
        "many output tiles",
        137, 67, 149
    );

    return 0;
}