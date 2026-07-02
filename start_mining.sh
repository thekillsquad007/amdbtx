#!/usr/bin/env bash
# Build and run from the checked-out repository.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="${HOME}/.amdbtx-miner"
CONFIG_PATH="${INSTALL_DIR}/config.yaml"
SOLVER_PATH="${INSTALL_DIR}/bin/btx-gbt-solve-hip"
PAYOUT_ADDRESS="${PAYOUT_ADDRESS:-}"

echo "=== 1. Build HIP solver ==="
bash "${REPO_DIR}/solver/build.sh"
mkdir -p "${INSTALL_DIR}/bin"
install -m 0755 "${REPO_DIR}/solver/build/btx-gbt-solve-hip" "${SOLVER_PATH}"
echo "[ok] current solver installed: ${SOLVER_PATH}"

echo
echo "=== 2. Install Python wrapper ==="
python3 -m pip install --user --force-reinstall "${REPO_DIR}" 2>&1 | tail -2

echo
echo "=== 3. Write config ==="
mkdir -p "${INSTALL_DIR}"
cat > "${CONFIG_PATH}" <<CFG
pool_host: "stratum.bitminerpool.xyz"
pool_port: 3333
pool_tls: false
payout_address: "${PAYOUT_ADDRESS}"
worker_name: "${WORKER_NAME:-default}"
gbt_solve_path: "${SOLVER_PATH}"
solver_backend: "rocm"
solver_threads: 8
solver_batch_size: 65536
solver_prefetch_depth: 8
solver_prepare_workers: 16
solver_pipeline_async: 1
gpu_inputs: 1
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
pool_max_shares_per_slice: 0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
CFG

echo
echo "=== 4. Done ==="
echo "Config: ${CONFIG_PATH}"
echo "Solver: ${SOLVER_PATH}"
echo
if [[ -z "${PAYOUT_ADDRESS}" ]]; then
    echo "Set payout_address in ${CONFIG_PATH}, then run:"
fi
echo "amdbtx-miner --config ${CONFIG_PATH}"
