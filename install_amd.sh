#!/usr/bin/env bash
# AMDBTX miner — AMD GPU installer for the MineBtx pool.
#
# Usage:
#   curl -fsSL <url> | bash
#   bash install_amd.sh --address btx1z...
#
# Installs:
#   1. ROCm 6.x (if not present)
#   2. amdbtx-miner Python wrapper
#   3. btx-gbt-solve-hip solver binary (prebuilt or from source)
#   4. Writes tuned config for detected AMD GPU
#   5. Runs GPU acceleration smoke test
#
# Dev fee: 2% time-sliced (transparent, logged).
# Dev wallet: btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4

set -euo pipefail

# Preserve real user's HOME even if run with sudo
if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    export HOME="$REAL_HOME"
fi

# ─── Configurables ──────────────────────────────────────────────────────────
PREBUILDS_TAG="${PREBUILDS_TAG:-amdbtx-prebuilds-v1.0}"
PREBUILDS_BASE="${PREBUILDS_BASE:-https://github.com/thekillsquad007/amdbtx/releases/download/${PREBUILDS_TAG}}"
SOLVER_NAME="${SOLVER_NAME:-btx-gbt-solve-hip}"
SOLVER_URL="${PREBUILDS_BASE}/${SOLVER_NAME}"
DEFAULT_POOL="${DEXBTX_POOL:-stratum.minebtx.com:3333}"

INSTALL_DIR="${HOME}/.amdbtx-miner"
SOLVER_PATH="${INSTALL_DIR}/bin/btx-gbt-solve-hip"
CONFIG_PATH="${INSTALL_DIR}/config.yaml"
VENV_DIR="${INSTALL_DIR}/venv"
LAUNCHER_PATH="${HOME}/.local/bin/amdbtx-miner"

DEV_WALLET="btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4"

# ─── Parse CLI ──────────────────────────────────────────────────────────────
ADDRESS=""
WORKER=""
POOL="${DEFAULT_POOL}"
ASSUME_YES=0
SKIP_PROMPT=0
LOCAL_SOLVER=""
SKIP_PIP=0
SKIP_ROCM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address) ADDRESS="$2"; shift 2 ;;
        --worker)  WORKER="$2";  shift 2 ;;
        --pool)    POOL="$2";    shift 2 ;;
        --yes|-y)  ASSUME_YES=1; SKIP_PROMPT=1; shift ;;
        --skip-prompt) SKIP_PROMPT=1; shift ;;
        --local-solver) LOCAL_SOLVER="$2"; shift 2 ;;
        --skip-pip)    SKIP_PIP=1; shift ;;
        --skip-rocm)   SKIP_ROCM=1; shift ;;
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Helpers ────────────────────────────────────────────────────────────────
log()  { echo -e "\033[1;34m[amdbtx]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
err()  { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || err "missing required tool: $1"
}

confirm() {
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    read -rp "$1 [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

# ─── OS + GPU Detection ────────────────────────────────────────────────────
log "AMDBTX miner installer — AMD GPU edition"

OS="$(uname -s)"
if [[ "$OS" != "Linux" ]]; then
    err "unsupported OS: $OS (AMD ROCm requires Linux)"
fi

# Detect AMD GPU
HAS_AMD=0
GPU_NAME=""
GPU_ARCH=""
if command -v rocm-smi >/dev/null 2>&1; then
    GPU_NAME="$(rocm-smi --showproductname 2>/dev/null | head -1 || true)"
    if [[ -n "$GPU_NAME" && "$GPU_NAME" != *"None"* ]]; then
        HAS_AMD=1
        log "detected AMD GPU: ${GPU_NAME}"
        GPU_ARCH="$(rocm-smi --showid 2>/dev/null | head -1 | grep -oP 'gfx[0-9a-f]+' || true)"
        if [[ -n "$GPU_ARCH" ]]; then
            log "GPU arch: ${GPU_ARCH}"
        fi
    fi
fi
if [[ "$HAS_AMD" -eq 0 ]] && command -v rocminfo >/dev/null 2>&1; then
    if ROCMINFO_OUT=$(rocminfo 2>/dev/null); then
        GPU_NAME=$(echo "$ROCMINFO_OUT" | awk '
            /Agent [0-9]/ { mktname=""; devtype=""; }
            /Marketing Name:/ { sub(/^.*Marketing Name: */, ""); mktname=$0; sub(/ *$/, "", mktname); }
            /Device Type.*GPU/ { devtype="GPU"; }
            devtype=="GPU" && mktname!="" { print mktname; devtype=""; exit; }
        ' || true)
        GPU_ARCH=$(echo "$ROCMINFO_OUT" | awk '
            /Agent [0-9]/ { devtype=""; arch=""; }
            /Device Type.*GPU/ { devtype="GPU"; }
            /gfx[0-9a-f]+/ && devtype=="GPU" && arch=="" { match($0, /gfx[0-9a-f]+/); arch=substr($0, RSTART, RLENGTH); }
            END { print arch; }
        ' || true)
        if [[ -n "$GPU_ARCH" ]]; then
            HAS_AMD=1
            if [[ -z "$GPU_NAME" || "$GPU_NAME" == *"None"* ]]; then
                GPU_NAME="AMD-$GPU_ARCH"
            fi
            log "detected AMD GPU: ${GPU_NAME} (arch: ${GPU_ARCH})"
        fi
    fi
fi
if [[ "$HAS_AMD" -eq 0 ]]; then
    warn "no AMD GPU detected via rocm-smi — solver will run on CPU only (much slower)"
    warn "ensure /dev/kfd and /dev/dri are accessible in your container"
fi

# Container environment detection
if [[ -f /.dockerenv ]] || grep -qE "docker|containerd|kubepods|lxc" /proc/self/cgroup 2>/dev/null; then
    log "container environment detected"
    # Check GPU device access
    if [[ ! -e /dev/kfd ]]; then
        warn "/dev/kfd not found — GPU acceleration may not work"
        warn "Container may need --privileged or --device=/dev/kfd --device=/dev/dri"
    fi
    if [[ ! -d /dev/dri ]]; then
        warn "/dev/dri not found — GPU acceleration may not work"
        warn "Container may need --device=/dev/dri"
    fi
fi

# ─── Python ─────────────────────────────────────────────────────────────────
need curl
need sha256sum

PYTHON=""
for cand in python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON="$cand"
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
            break
        fi
        PYTHON=""
    fi
done

if [[ -z "$PYTHON" ]]; then
    log "installing python3.10 via apt..."
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10 python3.10-venv python3-pip
    PYTHON=python3.10
fi
log "using Python: $($PYTHON --version 2>&1)"

# ─── Install ROCm (if missing) ─────────────────────────────────────────────
ROCM_LIB_PRESENT=0
if find /opt/rocm /opt/rocm-* -maxdepth 2 -name 'libamdhip64.so*' -print -quit 2>/dev/null | grep -q .; then
    ROCM_LIB_PRESENT=1
fi

if [[ "$SKIP_ROCM" -eq 1 ]]; then
    log "skipping ROCm installation (--skip-rocm)"
elif [[ "$ROCM_LIB_PRESENT" -eq 1 ]]; then
    log "ROCm runtime libraries detected under /opt; skipping package installation"
elif ! command -v rocm-smi >/dev/null 2>&1; then
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
    ROCM_REPO_VER=$([[ "$UBUNTU_CODENAME" == "noble" ]] && echo "7.2" || echo "6.0")
    log "ROCm not detected. Installing ROCm ${ROCM_REPO_VER} packages..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq || true
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wget gnupg2 curl || true

        log "adding ROCm repo..."
        echo "deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/${ROCM_REPO_VER} ${UBUNTU_CODENAME} main" | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null

        # Remove stale ROCm 5.x sources if any
        for old in /etc/apt/sources.list.d/rocm*.list; do
            [[ -f "$old" ]] && grep -q "rocm/apt/5\." "$old" 2>/dev/null && sudo rm -f "$old" && log "removed stale ROCm 5.x source: $old"
        done

        log "updating package lists..."
        sudo apt-get update -qq 2>&1 | grep -v "Warning:" || true

        # Install only the runtime needed by the pre-built miner. Full rocm-dev
        # pulls compiler/debugger packages that conflict on WSL/Noble.
        log "installing ROCm runtime packages (may take several minutes)..."
        if [[ "$ROCM_REPO_VER" == "7.2" ]]; then
            RUNTIME_PACKAGES=(hip-runtime-amd rocminfo)
        else
            RUNTIME_PACKAGES=(rocm-hip-runtime rocminfo)
        fi
        if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${RUNTIME_PACKAGES[@]}" 2>&1 | tail -10; then
            warn "ROCm runtime install failed — diagnosing..."
            sudo apt-get install -y "${RUNTIME_PACKAGES[@]}" 2>&1 | grep -E "Depends:|not going|broken|held|not installable" | head -10 >&2 || true
            warn "install ROCm runtime manually, then rerun with --skip-rocm"
        fi

        # Add to PATH
        if ! grep -q '/opt/rocm/bin' ~/.bashrc 2>/dev/null; then
            echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
        fi
        export PATH=/opt/rocm/bin:$PATH

        if command -v rocm-smi >/dev/null 2>&1; then
            log "ROCm installed successfully"
        else
            warn "ROCm install completed but rocm-smi not in PATH"
            warn "Add to your shell: export PATH=/opt/rocm/bin:\$PATH"
        fi
    else
        err "apt-get not available; install ROCm 6.x manually then re-run"
    fi
fi

# ─── Install pip + runtime deps ──────────────────────────────────────────────
if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
    log "python venv support not present; installing via apt..."
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip
fi

log "creating private Python environment..."
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip wheel pyyaml

# ─── Fetch pre-built Python package + solver binary from releases ───────────
mkdir -p "${INSTALL_DIR}/bin"
WHEEL_FILENAME="amdbtx_miner-1.0.0-py3-none-any.whl"
TMP_WHEEL="$(mktemp -d)/${WHEEL_FILENAME}"
TMP_SOLVER="$(mktemp)"
trap 'rm -f "$TMP_SOLVER"; rm -rf "$(dirname "$TMP_WHEEL")"' EXIT

# Download and install pre-built Python wheel
if [[ "$SKIP_PIP" -eq 1 ]]; then
    log "skipping amdbtx-miner pip install (--skip-pip)"
else
    log "downloading amdbtx-miner wheel from ${PREBUILDS_BASE}..."
    WHEEL_URL="${PREBUILDS_BASE}/${WHEEL_FILENAME}"
    curl -fsSL "$WHEEL_URL" -o "$TMP_WHEEL" 2>/dev/null || err "failed to download Python wheel from GitHub releases"
    "$VENV_DIR/bin/python" -m pip install --quiet --upgrade "$TMP_WHEEL"
    mkdir -p "$(dirname "$LAUNCHER_PATH")"
    cat > "$LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/amdbtx-miner" "\$@"
EOF
    chmod +x "$LAUNCHER_PATH"
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) : ;;
        *) warn "add to your shell rc: export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
    esac
fi

# Download pre-built solver binary
if [[ -n "$LOCAL_SOLVER" ]]; then
    log "using local solver at ${LOCAL_SOLVER}"
    cp "$LOCAL_SOLVER" "$TMP_SOLVER"
else
    log "downloading solver binary from ${SOLVER_URL}..."
    curl -fsSL "$SOLVER_URL" -o "$TMP_SOLVER" 2>/dev/null || {
        err "failed to download solver binary from GitHub releases
    URL: ${SOLVER_URL}
    Ensure you have a compatible AMD GPU and internet access.
    You can also provide a local binary: --local-solver /path/to/btx-gbt-solve-hip"
    }
fi

install -m 0755 "$TMP_SOLVER" "$SOLVER_PATH"
log "solver installed → $SOLVER_PATH"

# ─── Build solver runtime (library resolver) ──────────────────────────────
# The pre-built solver binary links against specific sonames (e.g. libamdhip64.so.6, libhipblas.so.2)
# ROCm 7.x ships different sonames. We create a local runtime dir with symlinks.
RUNTIME_DIR="${INSTALL_DIR}/runtime"
mkdir -p "$RUNTIME_DIR"

ROCM_LIB_DIRS=()
if [[ -d /opt/rocm/lib ]]; then ROCM_LIB_DIRS+=(/opt/rocm/lib); fi
for d in /opt/rocm-*/lib; do [[ -d "$d" ]] && ROCM_LIB_DIRS+=("$d"); done

resolve_lib() {
    local soname="$1" base="$2" target=""
    for d in "${ROCM_LIB_DIRS[@]}"; do
        [[ -f "$d/$soname" ]] && { ln -sfn "$d/$soname" "$RUNTIME_DIR/$soname"; log " $soname -> $d/$soname"; return 0; }
    done
    for d in "${ROCM_LIB_DIRS[@]}"; do
        latest=$(find "$d" -maxdepth 1 -name "${base}.so.*" 2>/dev/null | sort -V | tail -1 || true)
        if [[ -n "$latest" ]]; then
            real_file=$(readlink -f "$latest")
            if [[ -f "$real_file" ]]; then
                ln -sfn "$real_file" "$RUNTIME_DIR/$soname"; log " $soname -> $(basename $real_file) ($d)"; return 0
            fi
        fi
    done
    warn " $soname: not found in any ROCm installation"
    return 1
}

log "building solver runtime..."
declare -A SOLVER_LIBS=( ["libamdhip64.so.6"]="libamdhip64" ["libhipblas.so.2"]="libhipblas" )
for soname in "${!SOLVER_LIBS[@]}"; do
    resolve_lib "$soname" "${SOLVER_LIBS[$soname]}" || true
done

# Symlink rocblas kernels if available
for d in "${ROCM_LIB_DIRS[@]}"; do
    if [[ -d "$d/rocblas/library" ]]; then
        ln -sfn "$d/rocblas" "$RUNTIME_DIR/rocblas" 2>/dev/null || true
        log "  rocblas kernels -> $d/rocblas/"
        break
    fi
done

# Build LD path: runtime dir first, then all ROCm dirs
RUNTIME_LD_PATH="$RUNTIME_DIR"
for d in "${ROCM_LIB_DIRS[@]}"; do RUNTIME_LD_PATH="$RUNTIME_LD_PATH:$d"; done

# Verify solver can load all libraries
MISSING=$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>&1 | grep 'not found' || true)
if [[ -n "$MISSING" ]]; then
    warn "solver has unresolved libraries:"
    echo "$MISSING" | while read -r line; do warn "  $line"; done
    # Fallback: try apt install
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
    ROCM_REPO_VER=$([[ "$UBUNTU_CODENAME" == "noble" ]] && echo "7.2" || echo "6.0")
    log "attempting to install missing libraries via apt..."
    curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - 2>/dev/null || true
    echo "deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/$ROCM_REPO_VER $UBUNTU_CODENAME main" | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null
    if timeout 120 sudo apt-get update -qq 2>/dev/null && \
       timeout 120 sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq hipblas rocblas rocsolver 2>/dev/null; then
        log "apt install succeeded"
        for d in /opt/rocm-*/lib; do [[ -d "$d" ]] && ROCM_LIB_DIRS+=("$d"); done
        for soname in "${!SOLVER_LIBS[@]}"; do resolve_lib "$soname" "${SOLVER_LIBS[$soname]}" || true; done
        RUNTIME_LD_PATH="$RUNTIME_DIR"
        for d in "${ROCM_LIB_DIRS[@]}"; do RUNTIME_LD_PATH="$RUNTIME_LD_PATH:$d"; done
    else
        warn "apt install timed out or failed - solver may not work"
    fi
else
    log "all solver libraries resolved"
fi

# If rocm-smi/rocminfo was unavailable before installation, use the solver's
# own HIP startup line as a final GPU detection fallback.
if [[ "$HAS_AMD" -eq 0 ]]; then
    SOLVER_PROBE="$(printf '%s\n' '{"version":536870912,"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000","merkle_root":"0000000000000000000000000000000000000000000000000000000000000000","time":1779672814,"bits":"1d17c609","seed_a":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","seed_b":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","block_height":1,"nonce_start":0,"max_tries":1,"max_seconds":1,"share_target":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}' | LD_LIBRARY_PATH="$RUNTIME_LD_PATH" HSA_ENABLE_DXG_DETECTION=1 "$SOLVER_PATH" --daemon --backend hip --epsilon-bits 0 --batch-size 1 2>&1 | head -3 || true)"
    if echo "$SOLVER_PROBE" | grep -q 'HIP GPU detected:'; then
        HAS_AMD=1
        GPU_NAME="$(echo "$SOLVER_PROBE" | sed -nE 's/^HIP GPU detected: (.*) arch=(gfx[0-9a-f]+) memory=.*/\1/p' | head -1)"
        GPU_ARCH="$(echo "$SOLVER_PROBE" | grep -oE 'gfx[0-9a-f]+' | head -1)"
        log "detected AMD GPU via solver: ${GPU_NAME:-AMD GPU} ${GPU_ARCH:+(arch: $GPU_ARCH)}"
    fi
fi

# ─── Config ─────────────────────────────────────────────────────────────────
if [[ -z "$ADDRESS" && "$SKIP_PROMPT" -eq 0 ]]; then
    echo
    echo "Enter your BTX payout address (format: btx1z...):"
    read -rp "  address: " ADDRESS
fi

if [[ -n "$ADDRESS" ]]; then
    if [[ ! "$ADDRESS" =~ ^btx1z[0-9a-zA-Z]{50,}$ ]]; then
        warn "address does not match expected btx1z... format — proceeding anyway, but double-check"
    fi
fi

if [[ -z "$WORKER" ]]; then
    WORKER="$(hostname -s 2>/dev/null || echo default)"
fi

# GPU-specific tuning
GPU_WORKERS=16
GPU_THREADS=8
GPU_BATCH=128
GPU_PREFETCH=8

# Older GCN cards (gfx803) benefit from lower settings
if [[ "$GPU_ARCH" == "gfx803" || "$GPU_NAME" == *"RX 4"* || "$GPU_NAME" == *"RX 5"* ]]; then
    GPU_WORKERS=8
    GPU_THREADS=4
    GPU_BATCH=64
    GPU_PREFETCH=4
fi

if [[ ! -f "$CONFIG_PATH" || "$ASSUME_YES" -eq 1 ]]; then
    cat > "$CONFIG_PATH" <<YAML
# AMDBTX miner config — generated by install_amd.sh
# Pool connection
pool_host: "${POOL%:*}"
pool_port: ${POOL##*:}
pool_tls: false

# Worker identity
payout_address: "${ADDRESS}"
worker_name: "${WORKER}"

# Solver binary
gbt_solve_path: "${SOLVER_PATH}"

# Solver tuning (AMD GPU: ${GPU_NAME:-CPU only})
solver_backend: "rocm"
solver_threads: ${GPU_THREADS}
solver_batch_size: ${GPU_BATCH}
solver_prefetch_depth: ${GPU_PREFETCH}
solver_prepare_workers: ${GPU_WORKERS}
solver_pipeline_async: 1
gpu_inputs: 0

# Slice sizing
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0

# Reconnect
reconnect_initial_s: 1.0
reconnect_max_s: 60.0

# Dev fee (2% time-sliced, transparent)
# Dev wallet: ${DEV_WALLET}
# Mines with your address 98% of the time, dev wallet 2%.
# All switches logged at INFO level.

log_level: "INFO"

runtime_ld_path: "${RUNTIME_LD_PATH}"
venv_path: "${VENV_DIR}"
YAML
    log "config written → $CONFIG_PATH"
fi

# ─── GPU smoke test ─────────────────────────────────────────────────────────
if [[ "$HAS_AMD" -eq 1 ]]; then
    log "running GPU acceleration smoke test..."
    
    # First, check if the binary is even executable and has HIP support
    if ! "$SOLVER_PATH" --help >/dev/null 2>&1 && ! "$SOLVER_PATH" --version >/dev/null 2>&1; then
        warn "solver binary not responding — may need ROCm libraries"
        warn "Try: export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH"
    fi
    
    SMOKE_JOB='{"version":536870912,"prev_hash":"0ab38fdff2ef667dcddac7f50c3696080c26697615f7b6b9af5c3a1ba0a5fb7e","merkle_root":"d906f02ed11d8936770423263b56c5ffe1ea1b15c8a2867afb161adb6fd76eb7","time":1779672814,"bits":"1d17c609","seed_a":"8460daf3ff446cc55a7115de88ee24c8a2bf182eedde43abb9cf4cc94cc209bf","seed_b":"7f2e377616feb92d2e9857cab390595b7d6b8d24373a2da394f8d97197b5f437","block_height":110806,"nonce_start":1,"max_tries":200000,"max_seconds":30,"share_target":"00ffffff00000000000000000000000000000000000000000000000000000000"}'
    SMOKE_OUT="$(printf '%s\n' "$SMOKE_JOB" | LD_LIBRARY_PATH="$RUNTIME_LD_PATH" "$SOLVER_PATH" \
        --daemon --backend hip --epsilon-bits 18 --batch-size ${GPU_BATCH} 2>&1 || true)"
    SMOKE_LAST_LINE="$(echo "$SMOKE_OUT" | grep -E '^\{.*\}$' | tail -1)"
    if [[ -z "$SMOKE_LAST_LINE" ]]; then
        warn "GPU smoke test: solver produced no JSON output."
        warn "Possible causes:"
        warn "  1. /dev/kfd or /dev/dri not accessible (check: ls -la /dev/kfd /dev/dri)"
        warn "  2. ROCm libraries not in LD_LIBRARY_PATH"
        warn "  3. GPU not supported by installed ROCm version"
        warn "Try: export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH"
        warn "Miner will start anyway — GPU acceleration may not work"
    elif echo "$SMOKE_LAST_LINE" | grep -q '"found":true'; then
        ELAPSED="$(echo "$SMOKE_LAST_LINE" | sed -E 's/.*"elapsed_s":([0-9.e+-]+).*/\1/')"
        log "GPU smoke test: PASS (found a share in ${ELAPSED}s)"
    else
        warn "GPU smoke test: solver ran but didn't find a share — could be hard luck"
        warn "Check 'rocm-smi' to verify GPU is detected"
    fi
fi

# ─── Summary ────────────────────────────────────────────────────────────────
echo
log "AMDBTX miner installed."
echo
echo "  Pool:     ${POOL}"
echo "  Address:  ${ADDRESS:-<edit ${CONFIG_PATH} and set payout_address>}"
echo "  Worker:   ${WORKER}"
echo "  GPU:      ${GPU_NAME:-CPU only}"
echo "  Dev fee:  2% (time-sliced, transparent)"
echo "  Dev fee wallet: ${DEV_WALLET}"
echo
echo "Setup PATH (add to ~/.bashrc if not already):"
echo " export PATH=/opt/rocm/bin:\$PATH"
echo " export LD_LIBRARY_PATH=$RUNTIME_DIR:/opt/rocm/lib:\$LD_LIBRARY_PATH"
echo
echo "Launch the miner:"
echo "  amdbtx-miner --config ${CONFIG_PATH}"
echo
echo "Or, for a long-running daemon:"
echo "  tmux new -d -s amdbtx 'amdbtx-miner --config ${CONFIG_PATH} 2>&1 | tee -a ${INSTALL_DIR}/miner.log'"
echo "  tmux attach -t amdbtx"
echo
echo "Stats + payouts via Telegram: @btxdexbot   /stats /mybalance /help"
echo
echo "Troubleshooting:"
echo "  - GPU not detected:  rocm-smi   (should list your AMD GPU)"
echo "  - Permission denied: Ensure /dev/kfd and /dev/dri are accessible"
echo "  - ROCm not in PATH:  export PATH=/opt/rocm/bin:\$PATH"
echo "  - Check miner logs:  tail -f ${INSTALL_DIR}/miner.log"
echo
