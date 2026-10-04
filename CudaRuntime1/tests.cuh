#pragma once

// Runs benchmarkMatMul for several TILE / BLOCK_X / BLOCK_Y configurations
// and cuBLAS over a set of matrix sizes, checks each against the CPU result
// and prints the timings. Returns the number of configurations that failed.
int runBenchmarks();
