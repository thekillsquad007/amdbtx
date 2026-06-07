#!/usr/bin/env bash
# One-command setup: build solver, install wheel, configure, and mine.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="${HOME}/.amdbtx-miner"
CONFIG_PATH="${INSTALL_DIR}/config.yaml"
SOLVER_DIR="${HOME}/amdbtx-private-solver"
SOLVER_PATH="${INSTALL_DIR}/bin/btx-gbt-solve-hip"
PAYOUT_ADDRESS="${PAYOUT_ADDRESS:-}"

echo "=== 1. Build HIP solver ==="
if [ -d "${SOLVER_DIR}/src" ]; then
    BUILD_DIR="${SOLVER_DIR}/build"
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cmake -S "${SOLVER_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_ARCHITECTURES="gfx803;gfx900;gfx906;gfx90a;gfx1010;gfx1030;gfx1100;gfx1101;gfx1102"
    cmake --build "${BUILD_DIR}" -j"$(nproc)"
    mkdir -p "${INSTALL_DIR}/bin"
    cp "${BUILD_DIR}/btx-gbt-solve-hip" "${SOLVER_PATH}"
    echo "[ok] solver built from source → ${SOLVER_PATH}"
else
    SOLVER_URL="${PREBUILDS_BASE:-https://github.com/thekillsquad007/amdbtx/releases/download/amdbtx-prebuilds-v1.0}/btx-gbt-solve-hip"
    mkdir -p "${INSTALL_DIR}/bin"
    echo "downloading pre-built solver..."
    curl -fsSL -o "${SOLVER_PATH}" "${SOLVER_URL}"
    chmod +x "${SOLVER_PATH}"
    echo "[ok] solver downloaded → ${SOLVER_PATH}"
fi

echo ""
echo "=== 2. Install Python wheel ==="
WHEEL_URL="https://github.com/thekillsquad007/amdbtx/releases/download/amdbtx-prebuilds-v1.0/amdbtx_miner-1.0.0-py3-none-any.whl"
python3 -m pip install --user --force-reinstall "${WHEEL_URL}" 2>&1 | tail -2

echo ""
echo "=== 3. Write config ==="
mkdir -p "${INSTALL_DIR}"
cat > "${CONFIG_PATH}" << CFG
pool_host: "stratum.minebtx.com"
pool_port: 3333
pool_tls: false
payout_address: "${PAYOUT_ADDRESS}"
worker_name: "${WORKER_NAME:-default}"
gbt_solve_path: "SOLVER_PATH_PLACEHOLDER"
solver_backend: "rocm"
solver_threads: 8
solver_batch_size: 128
solver_prefetch_depth: 8
solver_prepare_workers: 16
solver_pipeline_async: 1
gpu_inputs: 0
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
CFG
sed -i "s|SOLVER_PATH_PLACEHOLDER|${SOLVER_PATH}|" "${CONFIG_PATH}"

echo ""
echo "=== 4. Done ==="
echo "Config: ${CONFIG_PATH}"
echo "Solver: ${SOLVER_PATH}"
echo ""
if [ -z "${PAYOUT_ADDRESS}" ]; then
    echo "Set your payout address in ${CONFIG_PATH}, then run:"
else
    echo "Ready to mine! Run:"
fi
echo "  ~/.local/bin/amdbtx-miner"
