# CudaRuntime1 – CUDA matmul learning project

## Build and run
- scripts\test.bat (build + test), scripts\profile.bat [simt|wmma|both]
- Manually (from the CudaRuntime1/ folder): nvcc -O3 -arch=sm_89 -lineinfo -Isrc/core -Isrc/reference -Isrc/simt -Isrc/wmma -Itests tests/main.cu tests/tests.cu src/reference/matmul.cu src/wmma/matmul_wmma.cu src/core/utils.cpp -o test.exe
- GPU: RTX 4070 Laptop (sm_89). Use Release for benchmarks, not Debug (-G).
- Profiling: scripts\profile.bat (ncu needs admin rights), reports go to profiles/.

## Code style
- 4-space indentation, no tabs.

## Workflow
- This is a learning repo: don't change code unless I ask; check first and tell me.
- Only commit source files (no exe or .ncu-repz).
