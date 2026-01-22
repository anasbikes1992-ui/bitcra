@echo off
setlocal enabledelayedexpansion
echo Building BitCrack Replica (Hybrid CPU+CUDA)...

:: 1. Try to find Visual Studio environment
echo Looking for Visual Studio...

set "VSPATH="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" set "VSPATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not defined VSPATH if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" set "VSPATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

if defined VSPATH (
    echo [INFO] Found Visual Studio at: "!VSPATH!"
    call "!VSPATH!" x64 >nul
)

:: 2. Check for NVCC
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] nvcc CUDA Compiler not found in PATH.
    pause
    exit /b 1
)

:: 3. Compile Hybrid Application
echo Compiling...
nvcc -O3 -std=c++17 -allow-unsupported-compiler ^
    -I"./include" -I"./cuda" -I"../cudaMath" ^
    "src/main.cpp" "src/CpuEngine.cpp" "cuda/CudaEngine.cu" ^
    -lcuda -lcudart -o "Replica.exe"

if %errorlevel% equ 0 (
    echo [SUCCESS] Replica.exe created!
    echo Run it with: .\Replica.exe --cpu [hash] OR .\Replica.exe --cuda [hash]
) else (
    echo [ERROR] Compilation failed.
)

pause
