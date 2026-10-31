# cuda_custom

A learning project for speeding up matrix multiplication (GEMM) with CUDA step by step. The same problem is solved with different approaches, and each one is measured against a CPU baseline and cuBLAS.

Target hardware: RTX 4070 Laptop GPU (sm_89).

## Layout

```
CudaRuntime1/
├── src/
│   ├── core/        DeviceBuffer, DeviceTensor, L2Flusher
│   ├── reference/   CPU matmul and cuBLAS baseline
│   ├── simt/        Tiled matmul on CUDA cores
│   └── wmma/        Tensor core (WMMA) matmul
├── tests/           Correctness checks + benchmark sweep
└── scripts/         test.bat, profile.bat
```

## Architecture

**Core.** `DeviceBuffer<T>` owns GPU memory with RAII (movable, not copyable). `DeviceTensor` adds shape/stride information on top and is passed to kernels as a `DeviceTensorView`. CUDA errors are turned into exceptions.

**References.** On the CPU, a multithreaded matmul with i-k-j loop order is the source of truth for correctness. cuBLAS `cublasGemmEx` (fp16 inputs, fp32 accumulate/output) is the speed target; row-major layout is handled without copies by swapping A and B.

**SIMT kernel.** Each block computes a `TILE_M x TILE_N` output tile. Slices of A and B that are `TILE_K` wide are staged in shared memory, and each thread accumulates several outputs in registers (register tiling). Tile and block sizes are template parameters; edge checks (`CHECK_M/N/K`) are compiled in with `if constexpr` only when needed.

**WMMA kernel.** Inputs are first converted to fp16 and padded to a whole block tile. Each warp computes one 16x16 output tile on the tensor cores with `wmma::mma_sync`; a block holds `WARP_ROWS x WARP_COLS` warps, and tiles are loaded through shared memory. Accumulation is in fp32.

**Tests.** `tests.cu` sweeps many SIMT and WMMA configurations on 512, 2048 and 4096 sizes. It checks every result against the CPU output, prints the time, GFLOP/s and ratio to cuBLAS, and summarizes the best configuration per size. Timing uses warm-up runs and CUDA events, and only kernel time is counted.

## Build and run

```bat
scripts\test.bat                 :: build + run all tests
scripts\profile.bat [simt|wmma]  :: Nsight Compute profile (needs admin rights)
```

Requirements: CUDA Toolkit (nvcc, cuBLAS) and MSVC. Adjust the `-ccbin` path in the scripts to your own Visual Studio installation.
