#pragma once

// Sweeps many TILE / BLOCK_X / BLOCK_Y configurations (and cuBLAS) on
// 512x512, 2048x2048 and 4096x4096 matmuls, checks each against the CPU
// result, prints the timings and the fastest configuration per size.
// Returns the number of configurations that failed.
int runBenchmarks();
