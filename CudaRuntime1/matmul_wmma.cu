#include "matmul_wmma.cuh"

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
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;

    size_t sizeA = paddedM * paddedK;
    for (size_t i = idx;
        i < sizeA;
        i += stride) {

        size_t r = i / paddedK;
        size_t c = i % paddedK;

        if (r < M && c < K)
            Ap[i] = __float2half(A[r * K + c]);

        else
            Ap[i] = __float2half(0.0f);
    }

    size_t sizeB = paddedK * paddedN;
    for (size_t i = idx;
        i < sizeB;
        i += stride) {

        size_t r = i / paddedN;
        size_t c = i % paddedN;

        if (r < K && c < N)
            Bp[i] = __float2half(B[r * N + c]);

        else
            Bp[i] = __float2half(0.0f);
    }
}

__global__ void postProcess(
    const float* Cp,
    float* C,
    size_t M,
    size_t N,
    size_t Np
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;

    for (size_t i = idx; i < M * N; i += stride)
    {
        size_t row = i / N;
        size_t col = i % N;

        C[i] = Cp[row * Np + col];
    }
}
