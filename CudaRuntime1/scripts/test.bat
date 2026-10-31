@echo off
setlocal

rem Builds from the project folder and runs the tests.
cd /d "%~dp0.."

echo === Compiling ===
nvcc -O3 -arch=sm_89 -lineinfo -ccbin "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64" -Isrc/core -Isrc/reference -Isrc/simt -Isrc/wmma -Itests tests/main.cu tests/tests.cu src/reference/matmul.cu src/wmma/matmul_wmma.cu src/core/utils.cpp -o test.exe
if errorlevel 1 exit /b 1

echo === Running tests ===
.\test.exe
