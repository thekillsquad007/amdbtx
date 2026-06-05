#!/usr/bin/env bash
# AMD BTX Miner - WSL setup script
# Run inside WSL Ubuntu 22.04+
# Usage: bash install_wsl.sh [--address btx1...] [--worker rig1]

set -euo pipefail

ADDRESS="${ADDRESS:-}"
WORKER="${WORKER:-}"
POOL="${POOL:-stratum.minebtx.com:3333}"
INSTALL_DIR="${HOME}/.amdbtx-miner"
SOLVER_PATH="${INSTALL_DIR}/bin/btx-gbt-solve"
CONFIG_PATH="${INSTALL_DIR}/config.yaml"

# GPU tuning defaults
GPU_THREADS=8
GPU_WORKERS=16
GPU_BATCH=128

log() { echo -e "\033[1;34m[amdbtx]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address) ADDRESS="$2"; shift 2 ;;
        --worker) WORKER="$2"; shift 2 ;;
        --pool) POOL="$2"; shift 2 ;;
        --help|-h) echo "Usage: $0 [--address btx1...] [--worker name] [--pool host:port]"; exit 0 ;;
        *) shift ;;
    esac
done

log "AMD BTX Miner - WSL setup"

# Check for ROCm installation (might be in /opt/rocm but not in PATH)
ROCM_PATH="/opt/rocm"
if [[ -d "$ROCM_PATH/bin" ]]; then
    export PATH="$ROCM_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
fi

# Ensure python3-venv is available
if ! python3 -m venv --help >/dev/null 2>&1 || [[ ! -f "$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null)/../bin/activate" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3")
        log "Installing python${PYTHON_VERSION}-venv..."
        sudo apt-get install -y -qq "python${PYTHON_VERSION}-venv" 2>/dev/null || sudo apt-get install -y -qq python3-venv 2>/dev/null || true
    fi
fi

# ROCm detection: rocm-smi may not exist in ROCm 7.x, check alternative tools
HAS_ROCM=false
ROCM_VERSION_DETECTED=""
if command -v rocm-smi >/dev/null 2>&1; then
    HAS_ROCM=true
    ROCM_VERSION_DETECTED=$(rocm-smi --version 2>/dev/null | head -1 || true)
    log "ROCm detected via rocm-smi: $ROCM_VERSION_DETECTED"
elif command -v rocminfo >/dev/null 2>&1; then
    HAS_ROCM=true
    ROCM_VERSION_DETECTED=$(rocminfo --version 2>/dev/null | head -1 || true)
    log "ROCm detected via rocminfo: $ROCM_VERSION_DETECTED"
elif command -v hipcc >/dev/null 2>&1; then
    HAS_ROCM=true
    ROCM_VERSION_DETECTED="$(hipcc --version 2>&1 | head -1 || true)"
    log "ROCm detected via hipcc: $ROCM_VERSION_DETECTED"
elif [[ -f /opt/rocm/share/doc/rocm-core/version ]]; then
    HAS_ROCM=true
    ROCM_VERSION_DETECTED=$(cat /opt/rocm/share/doc/rocm-core/version)
    log "ROCm detected at /opt/rocm: $ROCM_VERSION_DETECTED"
fi

# Check for AMD GPU
GPU_QUERY_CMD=""
if command -v rocm-smi >/dev/null 2>&1; then
    GPU_QUERY_CMD="rocm-smi"
elif command -v rocminfo >/dev/null 2>&1; then
    GPU_QUERY_CMD="rocminfo"
fi

GPU_NAME=""
GPU_ARCH=""
if [[ -n "$GPU_QUERY_CMD" ]]; then
    if [[ "$GPU_QUERY_CMD" == "rocm-smi" ]]; then
        GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | head -1 || true)
        GPU_ARCH=$(rocm-smi --showid 2>/dev/null | head -1 | grep -oP 'gfx[0-9a-f]+' || true)
    else
        # rocminfo: only use if it succeeded (exited 0)
        if ROCMINFO_OUT=$(rocminfo 2>/dev/null); then
            GPU_ARCH=$(echo "$ROCMINFO_OUT" | grep -oP 'gfx[0-9a-f]+' | head -1 || true)
            if [[ -n "$GPU_ARCH" ]]; then
                GPU_NAME=$(echo "$ROCMINFO_OUT" | grep -B5 "$GPU_ARCH" | grep "Name:" | head -1 | sed 's/.*Name:[ \t]*//; s/ (TM)//; s/ (R)//' || true)
            fi
        fi
    fi
    if [[ -n "$GPU_NAME" && "$GPU_NAME" != *"None"* ]]; then
        log "GPU detected: $GPU_NAME (arch: ${GPU_ARCH:-unknown})"
    else
        warn "$GPU_QUERY_CMD found but no AMD GPU detected"
    fi
fi

# Install ROCm only if not present at all
if ! $HAS_ROCM; then
    log "ROCm not detected. Installing..."
    UBUNTU_CODENAME=$(lsb_release -cs)
    # Ubuntu 24.04 (noble) needs ROCm 7.x
    if [[ "$UBUNTU_CODENAME" == "noble" ]]; then
        ROCM_VERSION="7.2"
    else
        ROCM_VERSION="6.0"
    fi
    sudo apt-get update -qq
    curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - 2>/dev/null || true
    echo "deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/$ROCM_VERSION $UBUNTU_CODENAME main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt-get update -qq 2>/dev/null || true
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rocm-hip-runtime hipblas hipsolver 2>/dev/null || \
        { warn "ROCm install may have issues - continuing"; }
    echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    # Enable AMD GPU detection on WSL2
    echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
    export PATH="/opt/rocm/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
fi

# Create venv and install Python package
log "Setting up Python virtual environment..."
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip wheel 2>/dev/null || true
"${INSTALL_DIR}/venv/bin/pip" install pyyaml 2>/dev/null || true

# Download assets
mkdir -p "${INSTALL_DIR}/bin"
PREBUILDS="https://github.com/thekillsquad007/amdbtx/releases/download/amdbtx-prebuilds-v1.0"

log "Downloading solver binary..."
sudo curl -fsSL "${PREBUILDS}/btx-gbt-solve" -o "$SOLVER_PATH"
sudo chmod +x "$SOLVER_PATH"

log "Downloading Python wheel..."
WHEEL="${INSTALL_DIR}/amdbtx_miner-1.0.0-py3-none-any.whl"
sudo curl -fsSL "${PREBUILDS}/amdbtx_miner-1.0.0-py3-none-any.whl" -o "$WHEEL"
"${INSTALL_DIR}/venv/bin/pip" install --force-reinstall "$WHEEL"

# Add venv to PATH in bashrc
grep -q "amdbtx-miner/venv/bin" ~/.bashrc || echo 'export PATH="$HOME/.amdbtx-miner/venv/bin:$PATH"' >> ~/.bashrc

# Get GPU arch for tuning and worker name
GPU_ARCH=""
GPU_WORKERS=16
GPU_THREADS=8
GPU_BATCH=128
WORKER_NAME="${WORKER}"
if [[ -n "$GPU_QUERY_CMD" ]]; then
    if [[ "$GPU_QUERY_CMD" == "rocm-smi" ]]; then
        GPU_ARCH=$(rocm-smi --showid 2>/dev/null | head -1 | grep -oP 'gfx[0-9a-f]+' || true)
        GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | head -1 | sed 's/.*: //; s/ (TM)//; s/ (R)//; s/ /-/g' || true)
    else
        # rocminfo: only use if it succeeded (exited 0)
        if ROCMINFO_OUT=$(rocminfo 2>/dev/null); then
            GPU_ARCH=$(echo "$ROCMINFO_OUT" | grep -oP 'gfx[0-9a-f]+' | head -1 || true)
            if [[ -n "$GPU_ARCH" ]]; then
                GPU_NAME=$(echo "$ROCMINFO_OUT" | grep -B5 "$GPU_ARCH" | grep "Name:" | head -1 | sed 's/.*Name:[ \t]*//; s/ (TM)//; s/ (R)//; s/ /-/g' || true)
            fi
        fi
    fi
    # Older GCN cards
    if [[ "$GPU_ARCH" == "gfx803" ]]; then
        GPU_WORKERS=8; GPU_THREADS=4; GPU_BATCH=64
    fi
    # If no worker provided, use GPU-based name
    if [[ -z "${WORKER:-}" ]]; then
        WORKER_NAME="${GPU_NAME:-amdgpu}-1"
    fi
fi

# Write config
log "Writing config..."
cat > "$CONFIG_PATH" <<EOF
pool_host: "${POOL%:*}"
pool_port: ${POOL##*:}
pool_tls: false
payout_address: "${ADDRESS:-btx1z...YOUR_ADDRESS_HERE...}"
worker_name: "${WORKER_NAME}"
gbt_solve_path: "${SOLVER_PATH}"
solver_backend: "rocm"
solver_threads: ${GPU_THREADS}
solver_prepare_workers: ${GPU_WORKERS}
solver_batch_size: ${GPU_BATCH}
solver_prefetch_depth: 8
solver_pipeline_async: 1
gpu_inputs: 0
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
venv_path: "${INSTALL_DIR}/venv"
EOF

log "Done!"
echo
echo "Config: $CONFIG_PATH"
echo "Solver: $SOLVER_PATH"
echo "Worker: $WORKER_NAME"
echo
echo "Launch miner:"
echo "  amdbtx-miner --config $CONFIG_PATH"
echo
echo "If amdbtx-miner not found, add to ~/.bashrc:"
echo "  export PATH=\"\$HOME/.amdbtx-miner/venv/bin:\$PATH\""