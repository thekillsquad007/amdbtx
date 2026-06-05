#!/usr/bin/env bash
# AMD BTX Miner - WSL setup script
# Works on any Ubuntu 22.04+ WSL2 with AMD GPU
# Usage: bash install_wsl.sh [--address btx1...] [--worker rig1] [--pool host:port]

set -euo pipefail

ADDRESS="${ADDRESS:-}"
WORKER="${WORKER:-}"
POOL="${POOL:-stratum.minebtx.com:3333}"
INSTALL_DIR="${HOME}/.amdbtx-miner"
SOLVER_PATH="${INSTALL_DIR}/bin/btx-gbt-solve"
CONFIG_PATH="${INSTALL_DIR}/config.yaml"
RUNTIME_DIR="${INSTALL_DIR}/runtime"

GPU_THREADS=8
GPU_WORKERS=16
GPU_BATCH=128

log() { echo -e "\033[1;34m[amdbtx]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
err() { echo -e "\033[1;31m[error]\033[0m $*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address) ADDRESS="$2"; shift 2 ;;
        --worker)  WORKER="$2"; shift 2 ;;
        --pool)    POOL="$2"; shift 2 ;;
        --help|-h) echo "Usage: $0 [--address btx1...] [--worker name] [--pool host:port]"; exit 0 ;;
        *) shift ;;
    esac
done

if [[ -z "$ADDRESS" ]]; then
    err "BTX payout address required. Usage: $0 --address btx1z..."
    exit 1
fi

log "AMD BTX Miner - WSL setup"
log "========================="

# ──────────────────────────────────────────────────────
# 1. WSL GPU passthrough check
# ──────────────────────────────────────────────────────
if grep -qi microsoft /proc/version 2>/dev/null; then
    IN_WSL=true
    if [[ -z "${HSA_ENABLE_DXG_DETECTION:-}" ]]; then
        warn "HSA_ENABLE_DXG_DETECTION=1 is not set in this session"
        warn "AMD GPU may not be visible to the solver"
        warn ""
        warn "To fix permanently, run this in WINDOWS PowerShell (Admin):"
        warn '  [Environment]::SetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "1", "User")'
        warn "  Then: wsl --shutdown; and relaunch WSL"
        warn ""
        warn "Setting for this session only..."
        export HSA_ENABLE_DXG_DETECTION=1
    fi
else
    IN_WSL=false
fi

# ──────────────────────────────────────────────────────
# 2. ROCm detection
# ──────────────────────────────────────────────────────
ROCM_LIB_DIRS=()
if [[ -d /opt/rocm/lib ]]; then
    ROCM_LIB_DIRS+=(/opt/rocm/lib)
fi
for d in /opt/rocm-*/lib; do
    [[ -d "$d" ]] && ROCM_LIB_DIRS+=("$d")
done

if [[ ${#ROCM_LIB_DIRS[@]} -eq 0 ]]; then
    err "No ROCm installation found in /opt/rocm*"
    err "Install ROCm first: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

PRIMARY_ROCM_LIB="/opt/rocm/lib"
if [[ -d /opt/rocm/bin ]]; then
    export PATH="/opt/rocm/bin:$PATH"
fi
export LD_LIBRARY_PATH="${ROCM_LIB_DIRS[0]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for d in "${ROCM_LIB_DIRS[@]:1}"; do
    export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done

HAS_ROCM=true
ROCM_VERSION=""
if [[ -f /opt/rocm/share/doc/rocm-core/version ]]; then
    ROCM_VERSION=$(cat /opt/rocm/share/doc/rocm-core/version)
elif command -v rocminfo >/dev/null 2>&1; then
    ROCM_VERSION=$(rocminfo --version 2>/dev/null | head -1 || true)
fi
log "ROCm: ${ROCM_VERSION:-detected} (libs: ${ROCM_LIB_DIRS[*]})"

# ──────────────────────────────────────────────────────
# 3. GPU detection
# ──────────────────────────────────────────────────────
GPU_ARCH=""
GPU_NAME=""
GPU_QUERY_CMD=""

if command -v rocm-smi >/dev/null 2>&1; then
    GPU_QUERY_CMD="rocm-smi"
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP '\S+.*' | head -1 | sed 's/ *$//; s/ (TM)//; s/ (R)//; s/ /-/g' || true)
    GPU_ARCH=$(rocm-smi --showid 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | head -1 || true)
elif command -v rocminfo >/dev/null 2>&1; then
    if ROCMINFO_OUT=$(rocminfo 2>/dev/null); then
        GPU_QUERY_CMD="rocminfo"
        GPU_ARCH=$(echo "$ROCMINFO_OUT" | grep -oP 'gfx[0-9a-f]+' | head -1 || true)
        if [[ -n "$GPU_ARCH" ]]; then
            GPU_NAME=$(echo "$ROCMINFO_OUT" | grep -B5 "$GPU_ARCH" | grep "Name:" | head -1 | sed 's/.*Name:[ \t]*//; s/ (TM)//; s/ (R)//; s/ /-/g' || true)
        fi
    fi
fi

if [[ -n "$GPU_ARCH" ]]; then
    log "GPU: ${GPU_NAME:-unknown} (arch: $GPU_ARCH)"
else
    warn "No AMD GPU detected"
    if [[ "$IN_WSL" == "true" ]]; then
        warn "WSL2 requires HSA_ENABLE_DXG_DETECTION=1 in Windows environment"
        warn "See step 1 above"
    fi
fi

# GPU tuning by arch
case "$GPU_ARCH" in
    gfx803|gfx900)  GPU_WORKERS=8;  GPU_THREADS=4; GPU_BATCH=64  ;;
    gfx906|gfx90a)  GPU_WORKERS=12; GPU_THREADS=8; GPU_BATCH=128 ;;
    gfx1010|gfx1030) GPU_WORKERS=12; GPU_THREADS=8; GPU_BATCH=128 ;;
    gfx1100|gfx1101|gfx1102|gfx1150|gfx1151|gfx1200|gfx1201)
                    GPU_WORKERS=16; GPU_THREADS=8; GPU_BATCH=128 ;;
    *)              GPU_WORKERS=16; GPU_THREADS=8; GPU_BATCH=128 ;;
esac

# ──────────────────────────────────────────────────────
# 4. Build solver runtime (library resolver)
# ──────────────────────────────────────────────────────
# The pre-built solver binary links against specific sonames from ROCm 6.0:
#   libamdhip64.so.6, libhipblas.so.2
# ROCm 7.x ships different sonames. We create a local runtime dir
# with symlinks that maps what the solver needs to what's installed.
log "Building solver runtime..."

mkdir -p "$RUNTIME_DIR"

# Solver expected sonames -> search pattern in installed ROCm
declare -A SOLVER_LIBS=(
    ["libamdhip64.so.6"]="libamdhip64.so"
    ["libhipblas.so.2"]="libhipblas.so"
)

# Also find transitive deps of hipblas (rocblas, rocsolver)
# These are loaded at runtime by the hipblas library
declare -A TRANSITIVE_LIBS=()

resolve_lib() {
    local soname="$1"
    local base="$2"
    local target=""

    # 1. Check if soname already exists in any ROCm lib dir
    for d in "${ROCM_LIB_DIRS[@]}"; do
        if [[ -f "$d/$soname" ]]; then
            target="$d/$soname"
            break
        fi
    done

    # 2. Find latest versioned .so in any ROCm lib dir and symlink
    if [[ -z "$target" ]]; then
        for d in "${ROCM_LIB_DIRS[@]}"; do
            # Find the real .so file (not a symlink) with the highest version
            latest=$(find "$d" -maxdepth 1 -name "${base}.so.*" ! -type l 2>/dev/null | sort -V | tail -1 || true)
            if [[ -n "$latest" ]]; then
                ln -sfn "$latest" "$RUNTIME_DIR/$soname"
                log "  $soname -> $(basename $latest) (from $d)"
                # Check transitive deps of this library
                check_transitive_deps "$latest"
                return 0
            fi
            # Also check for unversioned symlink pointing to a versioned file
            if [[ -L "$d/${base}.so" ]]; then
                real_target=$(readlink -f "$d/${base}.so")
                if [[ -f "$real_target" ]]; then
                    ln -sfn "$real_target" "$RUNTIME_DIR/$soname"
                    log "  $soname -> $(basename $real_target) (from $d)"
                    check_transitive_deps "$real_target"
                    return 0
                fi
            fi
        done
    fi

    if [[ -n "$target" ]]; then
        ln -sfn "$target" "$RUNTIME_DIR/$soname"
        log "  $soname -> $target (direct)"
        check_transitive_deps "$target"
        return 0
    fi

    warn "  $soname: not found in any ROCm installation"
    return 1
}

check_transitive_deps() {
    local lib_path="$1"
    # Parse ldd output to find ROCm deps that also need resolving
    for dep in $(LD_LIBRARY_PATH="${ROCM_LIB_DIRS[*]}:/opt/rocm/lib" ldd "$lib_path" 2>/dev/null | grep 'not found' | awk '{print $1}'); do
        # Skip if already resolved
        if [[ -L "$RUNTIME_DIR/$dep" || -f "$RUNTIME_DIR/$dep" ]]; then
            continue
        fi
        # Get base name (e.g., librocblas.so from librocblas.so.5)
        base="${dep%%.so*}.so"
        TRANSITIVE_LIBS["$dep"]="$base"
    done
}

LIBS_OK=true
for soname in "${!SOLVER_LIBS[@]}"; do
    base="${SOLVER_LIBS[$soname]}"
    if ! resolve_lib "$soname" "$base"; then
        LIBS_OK=false
    fi
done

# Resolve transitive deps (may take multiple passes for deep deps)
for _pass in 1 2 3; do
    if [[ ${#TRANSITIVE_LIBS[@]} -eq 0 ]]; then
        break
    fi
    for soname in "${!TRANSITIVE_LIBS[@]}"; do
        base="${TRANSITIVE_LIBS[$soname]}"
        resolve_lib "$soname" "$base" || true
    done
    TRANSITIVE_LIBS=()
done

# Also symlink rocblas kernel library directory if it exists
for d in "${ROCM_LIB_DIRS[@]}"; do
    if [[ -d "$d/rocblas/library" ]]; then
        ln -sfn "$d/rocblas" "$RUNTIME_DIR/rocblas" 2>/dev/null || true
        log "  rocblas kernels -> $d/rocblas/"
        break
    fi
done

# Set runtime LD_LIBRARY_PATH: runtime dir first, then all ROCm dirs
RUNTIME_LD_PATH="$RUNTIME_DIR"
for d in "${ROCM_LIB_DIRS[@]}"; do
    RUNTIME_LD_PATH="$RUNTIME_LD_PATH:$d"
done

# ──────────────────────────────────────────────────────
# 5. Verify solver can load all libraries
# ──────────────────────────────────────────────────────
# Download solver binary first so we can verify
mkdir -p "${INSTALL_DIR}/bin"
PREBUILDS="https://github.com/thekillsquad007/amdbtx/releases/download/amdbtx-prebuilds-v1.0"

log "Downloading solver binary..."
curl -fsSL "${PREBUILDS}/btx-gbt-solve" -o "$SOLVER_PATH"
chmod +x "$SOLVER_PATH"

log "Verifying solver libraries..."
MISSING=$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>&1 | grep 'not found' || true)
if [[ -n "$MISSING" ]]; then
    warn "Solver has unresolved libraries:"
    echo "$MISSING" | while read -r line; do warn "  $line"; done

    # Try installing missing libs via apt as fallback
    warn "Attempting to install missing libraries via apt..."
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
    if [[ "$UBUNTU_CODENAME" == "noble" ]]; then
        ROCM_REPO_VER="7.2"
    else
        ROCM_REPO_VER="6.0"
    fi

    # Write repo config (skip if already correct)
    REPO_FILE="/etc/apt/sources.list.d/rocm.list"
    EXPECTED_REPO="deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/$ROCM_REPO_VER $UBUNTU_CODENAME main"
    CURRENT_REPO=$(cat "$REPO_FILE" 2>/dev/null || echo "")
    if [[ "$CURRENT_REPO" != "$EXPECTED_REPO" ]]; then
        log "Updating ROCm apt repo to $ROCM_REPO_VER..."
        curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - 2>/dev/null || true
        echo "$EXPECTED_REPO" | sudo tee "$REPO_FILE" >/dev/null
    fi

    # Try apt install with timeout (don't hang the installer)
    log "Installing hipblas, rocblas, rocsolver (this may take a minute)..."
    if timeout 120 sudo apt-get update -qq 2>/dev/null && \
       timeout 120 sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq hipblas rocblas rocsolver 2>/dev/null; then
        log "apt install succeeded, re-scanning libraries..."
        # Re-add any new ROCm dirs
        for d in /opt/rocm-*/lib; do
            [[ -d "$d" ]] && ROCM_LIB_DIRS+=("$d")
        done
        # Re-resolve libs
        for soname in "${!SOLVER_LIBS[@]}"; do
            base="${SOLVER_LIBS[$soname]}"
            resolve_lib "$soname" "$base" || true
        done
        # Update runtime LD path
        RUNTIME_LD_PATH="$RUNTIME_DIR"
        for d in "${ROCM_LIB_DIRS[@]}"; do
            RUNTIME_LD_PATH="$RUNTIME_LD_PATH:$d"
        done
        # Re-check
        MISSING=$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>&1 | grep 'not found' || true)
    else
        warn "apt install timed out or failed"
    fi

    if [[ -n "$MISSING" ]]; then
        err "Solver still has unresolved libraries after all attempts"
        err "The miner may not work. Missing:"
        echo "$MISSING" | while read -r line; do err "  $line"; done
    fi
else
    log "All solver libraries resolved"
fi

# ──────────────────────────────────────────────────────
# 6. Python setup
# ──────────────────────────────────────────────────────
if ! python3 -m venv --help >/dev/null 2>&1; then
    log "Installing python3-venv..."
    PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3")
    if command -v apt-get >/dev/null 2>&1; then
        timeout 60 sudo apt-get install -y -qq "python${PYTHON_VER}-venv" 2>/dev/null || \
        timeout 60 sudo apt-get install -y -qq python3-venv 2>/dev/null || \
        { err "Cannot install python3-venv. Install it manually: sudo apt install python3-venv"; exit 1; }
    fi
fi

log "Setting up Python virtual environment..."
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip wheel 2>/dev/null || true
"${INSTALL_DIR}/venv/bin/pip" install pyyaml 2>/dev/null || true

log "Downloading Python wheel..."
WHEEL="${INSTALL_DIR}/amdbtx_miner-1.0.0-py3-none-any.whl"
curl -fsSL "${PREBUILDS}/amdbtx_miner-1.0.0-py3-none-any.whl" -o "$WHEEL"
"${INSTALL_DIR}/venv/bin/pip" install --force-reinstall "$WHEEL"

# ──────────────────────────────────────────────────────
# 7. Environment persistence
# ──────────────────────────────────────────────────────
log "Configuring environment..."
grep -q 'amdbtx-miner' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'ENVEOF'

# AMD BTX Miner
export PATH="/opt/rocm/bin:$HOME/.amdbtx-miner/venv/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.amdbtx-miner/runtime:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_ENABLE_DXG_DETECTION=1
ENVEOF

# ──────────────────────────────────────────────────────
# 8. Worker name and config
# ──────────────────────────────────────────────────────
WORKER_NAME="${WORKER:-}"
if [[ -z "$WORKER_NAME" ]]; then
    if [[ -n "$GPU_NAME" && "$GPU_NAME" != *"None"* ]]; then
        WORKER_NAME="${GPU_NAME}-1"
    elif [[ -n "$GPU_ARCH" ]]; then
        WORKER_NAME="${GPU_ARCH}-1"
    else
        WORKER_NAME="amdgpu-1"
    fi
fi

log "Writing config..."
cat > "$CONFIG_PATH" <<EOF
pool_host: "${POOL%:*}"
pool_port: ${POOL##*:}
pool_tls: false
payout_address: "${ADDRESS}"
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
runtime_ld_path: "${RUNTIME_LD_PATH}"
EOF

# ──────────────────────────────────────────────────────
# 9. Auto-benchmark
# ──────────────────────────────────────────────────────
log "Running auto-benchmark (~2 minutes, finds optimal solver config)..."
LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    "${INSTALL_DIR}/venv/bin/amdbtx-miner" --config "$CONFIG_PATH" --benchmark 2>&1 || \
    warn "Benchmark failed - using default tuning for $GPU_ARCH"

# ──────────────────────────────────────────────────────
# 10. Summary
# ──────────────────────────────────────────────────────
echo
log "Setup complete!"
echo
echo "  Config:  $CONFIG_PATH"
echo "  Solver:  $SOLVER_PATH"
echo "  Runtime: $RUNTIME_DIR"
echo "  Worker:  $WORKER_NAME"
echo
if [[ "$IN_WSL" == "true" && -z "${HSA_ENABLE_DXG_DETECTION:-}" ]]; then
    echo -e "\033[1;33m  IMPORTANT: GPU not detected in WSL\033[0m"
    echo "  Run this in Windows PowerShell (Admin) to enable GPU passthrough:"
    echo '    [Environment]::SetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "1", "User")'
    echo "    wsl --shutdown"
    echo "    Then reopen WSL and run: amdbtx-miner --config $CONFIG_PATH"
    echo
fi
echo "  Launch miner:"
echo "    amdbtx-miner --config $CONFIG_PATH"
