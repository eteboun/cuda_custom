#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "buffer.cuh"

// Evicts the GPU's L2 cache by writing a buffer twice its size, so the
// next kernel starts with a cold cache and reads its inputs from DRAM.
// Used by benchmarks whose working set would otherwise stay in L2
// between timed runs (e.g. gemv on a 4096x4096 fp16 matrix is 32 MB).
class L2Flusher {
private:
    DeviceBuffer<char> buffer;

    static size_t flushBytes() {
        int device = 0;
        int l2Bytes = 0;

        cudaError_t err = cudaGetDevice(&device);

        if (err == cudaSuccess) {
            err = cudaDeviceGetAttribute(
                &l2Bytes, cudaDevAttrL2CacheSize, device
            );
        }

        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("L2 size query failed: ") +
                cudaGetErrorString(err)
            );
        }

        // Twice the L2 size, so no line of the previous working set
        // survives; at least 1 MB in case the query returns 0.
        size_t bytes = 2 * static_cast<size_t>(l2Bytes);
        return bytes < (1 << 20) ? (1 << 20) : bytes;
    }

public:
    L2Flusher()
        : buffer(flushBytes())
    {
    }

    // Queued on the default stream, so it finishes before the next
    // kernel launched there starts.
    void flush() {
        cudaError_t err = cudaMemsetAsync(buffer.data(), 0, buffer.size());

        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("L2 flush failed: ") +
                cudaGetErrorString(err)
            );
        }
    }
};
