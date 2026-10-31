# cuda_custom

A learning project for speeding up matrix multiplication (GEMM) with CUDA step by step. The same problem is solved with different approaches, and each one is measured against a CPU baseline and cuBLAS.

Target hardware: RTX 4070 Laptop GPU (sm_89).

## Layout

```
CudaRuntime1/
├── src/
│   ├── core/        DeviceBuffer, DeviceTensor
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

## Results

WMMA kernel vs. cuBLAS `GemmEx`, both with fp16 inputs and fp32 accumulation/output. One run of `scripts\test.bat` on an RTX 4070 Laptop GPU (CUDA 13.4, driver 616.92, `-O3 -arch=sm_89`); square matrices, average kernel time, all configurations passed the correctness check. "WMMA best" is the fastest warp layout in the sweep for that size.

| Size | WMMA best | cuBLAS GemmEx | WMMA / cuBLAS |
|---|---|---|---|
| 512³  | 0.029 ms · 9.3 TFLOP/s (`WARPS=4x4`)  | 0.020 ms · 13.3 TFLOP/s | 70% |
| 2048³ | 1.29 ms · 13.3 TFLOP/s (`WARPS=4x2`)  | 0.44 ms · 38.9 TFLOP/s  | 34% |
| 4096³ | 11.99 ms · 11.5 TFLOP/s (`WARPS=4x2`) | 5.59 ms · 24.6 TFLOP/s  | 47% |

## Benchmark limitations

This is a learning project and the benchmark is deliberately simple. Treat the numbers as rough, relative indicators, not rigorous measurements:

- **Single run, no variance.** Each number is one average with no repeats or standard deviation. Laptop GPU clocks are not locked, so boost and thermal/power throttling move results between runs. Example: cuBLAS drops from 38.9 to 24.6 TFLOP/s going from 2048³ to 4096³.
- **Optimistic "best".** Picking the fastest of seven noisy warp layouts biases the WMMA result upward.
- **Kernel time only.** Host↔device copies, the fp32→fp16 conversion and the WMMA padding are not timed.
- **Warm caches, back-to-back launches.** The same inputs run repeatedly between one pair of CUDA events. At 512³ the whole working set fits in L2 and launch overhead is a noticeable part of a ~30 µs kernel.
- **Narrow coverage.** Only square power-of-two sizes. The test data are small integers that fp16 represents exactly, so the correctness check does not stress fp16 rounding error.
