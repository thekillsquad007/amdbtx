#!/usr/bin/env bash
# Clean HIP solver build for AMD GPUs (run inside distrobox with ROCm).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOLVER_SRC="${SOLVER_SRC:-${SCRIPT_DIR}/solver}"
BUILD_DIR="${SOLVER_SRC}/build"
OUTPUT="${BUILD_DIR}/btx-gbt-solve-hip"

echo "[build] Building HIP solver..."
bash "${SOLVER_SRC}/build.sh"
BUILD_DIR="${SOLVER_SRC}/build"

echo "[build] Verifying..."
file "${OUTPUT}"
ldd "${OUTPUT}" 2>&1 | head -5

echo "[build] Done. Binary at: ${OUTPUT}"
echo "[build] Size: $(du -h "${OUTPUT}" | cut -f1)"
