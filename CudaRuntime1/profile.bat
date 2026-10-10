@echo off
setlocal

rem Usage: profile.bat [simt ^| wmma ^| both]   (default: both)
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=both"

if /i not "%TARGET%"=="simt" if /i not "%TARGET%"=="wmma" if /i not "%TARGET%"=="both" (
    echo Unknown kernel "%TARGET%". Use simt, wmma or both.
    exit /b 1
)

echo === Compiling ===
nvcc -O3 -arch=sm_89 -lineinfo -ccbin "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64" main.cu matmul.cu matmul_wmma.cu tests.cu utils.cpp -o test.exe
if errorlevel 1 exit /b 1

if /i "%TARGET%"=="wmma" goto wmma

echo === Profiling deviceMatMul (SIMT) ===
call ncu --set full -k regex:deviceMatMul -f -c 1 -o simt .\test.exe
if errorlevel 1 exit /b 1
start "" ncu-ui simt.ncu-repz
if /i "%TARGET%"=="simt" goto done

:wmma
echo === Profiling wmmaMatmul (WMMA) ===
call ncu --set full -k regex:wmmaMatmul -f -c 1 -o wmma .\test.exe
if errorlevel 1 exit /b 1
start "" ncu-ui wmma.ncu-repz

:done
echo === Done ===
