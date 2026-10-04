@echo off

echo === Compiling ===
nvcc -O3 -arch=sm_89 -lineinfo main.cu matmul.cu tests.cu utils.cpp -o test.exe
if errorlevel 1 exit /b 1

echo === Profiling deviceMatMul ===
call ncu --set full -k regex:deviceMatMul -f -c 1 -o mykernel .\test.exe
if errorlevel 1 exit /b 1

echo === Opening report ===
start "" ncu-ui mykernel.ncu-repz

echo === Done ===