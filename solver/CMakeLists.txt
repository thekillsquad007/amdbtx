cmake_minimum_required(VERSION 3.22)
project(btx-gbt-solve-hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(HIPCC hipcc PATHS /opt/rocm/bin /opt/rocm-*/bin)
if(NOT HIPCC)
    message(FATAL_ERROR "hipcc not found. Install rocm-dev or hip-devel.")
endif()

set(SOURCES
    src/main.cpp
    src/sha256.cpp
    src/field.cpp
    src/matrix.cpp
    src/noise.cpp
    src/transcript.cpp
    src/solve.cpp
    src/solve_gpu.hip
    src/gpu_sha256.hip
    src/matmul_kernel.hip
)

set(HIP_ARCHS
    gfx900;gfx906;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201
    CACHE STRING "Semicolon-separated list of HIP offload architectures"
)

set(HIP_ARCH_FLAGS "")
foreach(ARCH IN LISTS HIP_ARCHS)
    list(APPEND HIP_ARCH_FLAGS "--offload-arch=${ARCH}")
endforeach()

if(WIN32)
    set(HIP_LINK_FLAGS "")
else()
    set(HIP_LINK_FLAGS "-lpthread" "-Wl,-rpath,/opt/rocm/lib:/opt/rocm-7.2.0/lib")
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/btx-gbt-solve-hip
    COMMAND ${HIPCC}
        -x hip -O3 -std=c++17
        ${HIP_ARCH_FLAGS}
        -D__HIP_PLATFORM_AMD__
        -I${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SOURCES}
        ${HIP_LINK_FLAGS}
        -o ${CMAKE_CURRENT_BINARY_DIR}/btx-gbt-solve-hip
    DEPENDS ${SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building btx-gbt-solve-hip with ${HIPCC}"
)

add_custom_target(btx-gbt-solve-hip ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/btx-gbt-solve-hip
)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/btx-gbt-solve-hip DESTINATION bin)
