cmake_minimum_required(VERSION 3.18)
project(BitCrackReplica LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture (e.g., 7.5 for T4/RTX 20, 8.6 for RTX 30)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

include_directories(include cuda)

add_executable(Replica 
    src/main.cpp 
    cuda/CudaEngine.cu
)

set_target_properties(Replica PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(Replica PRIVATE CUDA::cudart)
