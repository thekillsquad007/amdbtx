#!/usr/bin/env bash
# Clean HIP solver build for AMD GPUs (run inside distrobox with ROCm).
set -euo pipefail

SOLVER_SRC="${SOLVER_SRC:-/var/home/bazzite/amdbtx-private-solver}"
BUILD_DIR="${SOLVER_SRC}/build"
OUTPUT="${BUILD_DIR}/btx-gbt-solve"

echo "[build] Cleaning old build..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

echo "[build] Configuring with CMake..."
cmake -S "${SOLVER_SRC}" -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES="gfx803;gfx900;gfx906;gfx90a;gfx1010;gfx1030;gfx1100;gfx1102"

echo "[build] Building..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "[build] Verifying..."
file "${OUTPUT}"
ldd "${OUTPUT}" 2>&1 | head -5

echo "[build] Done. Binary at: ${OUTPUT}"
echo "[build] Size: $(du -h "${OUTPUT}" | cut -f1)"
