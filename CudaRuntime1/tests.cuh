#pragma once

// Benchmarks cuBLAS and the TILE=64, BLOCK=16x16 kernel on 2048x2048,
// checks both against the CPU result and prints the timings.
// Returns the number of configurations that failed.
int runBenchmarks();
