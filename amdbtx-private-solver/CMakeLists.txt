cmake_minimum_required(VERSION 3.22)
project(btx-gbt-solve-hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set GPU architectures
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx803;gfx900;gfx906;gfx90a;gfx1010;gfx1030;gfx1100;gfx1102")
endif()

# Tell cmake where ROCm is
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install path")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# Use clang++ for both C++ and HIP — clang knows the __host__/__device__
# builtins that field.h uses. The HIP compiler already wraps clang++.
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/clang++")

# Source files — main.cpp is now treated as a HIP file so it gets
# the HIP platform defines and builtins
add_executable(btx-gbt-solve
    src/main.cpp
    src/matmul_kernel.hip
)

# Tell cmake main.cpp should be compiled by the HIP compiler
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(btx-gbt-solve PRIVATE
    src
    ${ROCM_PATH}/include
)

# Link HIP runtime library (amdhip64 is the shared lib in ROCm 6.x)
target_link_directories(btx-gbt-solve PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(btx-gbt-solve PRIVATE amdhip64)

# hipblas is optional
find_library(HIPBLAS_LIB NAMES hipblas PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
if(HIPBLAS_LIB)
    message(STATUS "Found hipblas: ${HIPBLAS_LIB}")
    target_link_libraries(btx-gbt-solve PRIVATE ${HIPBLAS_LIB})
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(btx-gbt-solve PRIVATE -O3 -DNDEBUG)
endif()
