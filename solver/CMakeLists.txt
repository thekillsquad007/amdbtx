cmake_minimum_required(VERSION 3.22)
project(btx-gbt-solve-hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROCm
find_package(hip REQUIRED)
find_package(hipblas QUIET)

# Source files
set(SOURCES src/main.cpp)
set(HIP_SOURCES src/matmul_kernel.hip)

# Main executable
add_executable(btx-gbt-solve ${SOURCES} ${HIP_SOURCES})

target_include_directories(btx-gbt-solve PRIVATE src)

# ROCm libraries
target_link_libraries(btx-gbt-solve PRIVATE hip::hiprt)
if(hipblas_FOUND)
    target_link_libraries(btx-gbt-solve PRIVATE hip::hipblas)
endif()

# Set GPU architectures - support all AMD GPUs from gfx803 to gfx1102
set(CMAKE_HIP_ARCHITECTURES "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1010;gfx1030;gfx1100;gfx1101;gfx1102")

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(btx-gbt-solve PRIVATE -O3 -DNDEBUG)
endif()
