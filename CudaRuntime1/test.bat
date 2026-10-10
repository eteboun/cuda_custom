@echo off

echo === Compiling ===
nvcc -O3 -arch=sm_89 -lineinfo -ccbin "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64" main.cu matmul.cu matmul_wmma.cu tests.cu utils.cpp -o test.exe
if errorlevel 1 exit /b 1

echo === Running tests ===
.\test.exe
