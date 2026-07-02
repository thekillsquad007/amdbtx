#!/usr/bin/env bash
# AMDBTX miner — AMD GPU installer for the MineBtx pool.
#
# Usage:
#   curl -fsSL <url> | bash
#   bash install_amd.sh --address btx1z...
#
# Installs:
#   1. ROCm runtime + HIP compiler (if not present)
#   2. amdbtx-miner Python wrapper (from source by default)
#   3. btx-gbt-solve-hip solver (compiled for your GPU arch by default)
#   4. Writes tuned config for detected AMD GPU
#   5. Runs GPU acceleration smoke test
#
# Default: compile solver from source (broader AMD arch support than prebuilds).
# Fast path: pass --use-prebuilt to download release wheel + multi-arch binary.
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
SOURCE_REF="${AMDBTX_SOURCE_REF:-main}"
SOURCE_REPO="${AMDBTX_SOURCE_REPO:-thekillsquad007/amdbtx}"
RELEASE_REPO="${AMDBTX_RELEASE_REPO:-thekillsquad007/amdbtx-releases}"
PREBUILDS_TAG="${PREBUILDS_TAG:-amdbtx-prebuilds-v1.1.9}"
PREBUILDS_BASE="${PREBUILDS_BASE:-https://github.com/${RELEASE_REPO}/releases/download/${PREBUILDS_TAG}}"
WHEEL_FILENAME="${AMDBTX_WHEEL_FILENAME:-amdbtx_miner-1.1.9-py3-none-any.whl}"
EXPECTED_MINER_VERSION="1.1.9"
EXPECTED_WHEEL_SHA256="336201199bc02eddf3cb4c2a41804d9fa7d65622d049bd9eaf0c8230918ad7b8"
EXPECTED_SOLVER_VERSION="2.2.0"
EXPECTED_SOLVER_SHA256="b4dc4194e348edbc691dbec9dad1b75bbf5fd626d099784d9a757ab6f172b21d"
DEFAULT_POOL="${DEXBTX_POOL:-stratum.bitminerpool.xyz:3333}"
# Used when --compile-all-archs is set (or AMDBTX_COMPILE_ALL_ARCHS=1).
ALL_HIP_ARCHS="${AMDBTX_HIP_ARCHS:-gfx803 gfx900 gfx906 gfx908 gfx90a gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151}"

INSTALL_DIR="${HOME}/.amdbtx-miner"
SOLVER_PATH="${INSTALL_DIR}/bin/btx-gbt-solve-hip"
CONFIG_PATH="${INSTALL_DIR}/config.yaml"
VENV_DIR="${AMDBTX_VENV_DIR:-${HOME}/.local/share/amdbtx-miner/venv}"
LAUNCHER_PATH="${HOME}/.local/bin/amdbtx-miner"

DEV_WALLET="btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4"

# ─── Parse CLI ──────────────────────────────────────────────────────────────
ADDRESS=""
WORKER=""
POOL="${DEFAULT_POOL}"
ASSUME_YES=0
SKIP_PROMPT=0
LOCAL_SOLVER=""
USE_PREBUILT=0
COMPILE_ALL_ARCHS=0
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
        --use-prebuilt) USE_PREBUILT=1; shift ;;
        --source-ref) SOURCE_REF="$2"; shift 2 ;;
        --compile-all-archs) COMPILE_ALL_ARCHS=1; shift ;;
        --skip-pip)    SKIP_PIP=1; shift ;;
        --skip-rocm)   SKIP_ROCM=1; shift ;;
        --help|-h)
            sed -n '2,22p' "$0"
            echo "  --use-prebuilt       download release wheel+solver instead of compiling"
            echo "  --source-ref REF     git ref for source compile (default: main)"
            echo "  --compile-all-archs  compile solver for all common gfx targets"
            echo "  --local-solver PATH  install an existing solver binary"
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "${AMDBTX_COMPILE_ALL_ARCHS:-0}" == "1" ]]; then
    COMPILE_ALL_ARCHS=1
fi

# ─── Helpers ────────────────────────────────────────────────────────────────
log()  { echo -e "\033[1;34m[amdbtx]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
err()  { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || err "missing required tool: $1"
}

have_apt() {
    command -v apt-get >/dev/null 2>&1
}

sudo_cmd() {
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        err "sudo is required to install missing system packages. Install sudo or run as root."
    fi
}

apt_install() {
    have_apt || return 1
    sudo_cmd apt-get update -qq
    sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@"
}

ensure_hip_compiler() {
    if command -v hipcc >/dev/null 2>&1; then
        return 0
    fi
    for cand in /opt/rocm/bin/hipcc /opt/rocm-*/bin/hipcc; do
        if [[ -x "$cand" ]]; then
            export PATH="$(dirname "$cand"):$PATH"
            return 0
        fi
    done
    warn "HIP compiler (hipcc) not found; installing dev packages..."
    if have_apt; then
        for pkg_set in "hip-dev rocminfo" "rocm-dev rocminfo" "hip-dev"; do
            # shellcheck disable=SC2086
            if apt_install $pkg_set 2>/dev/null; then
                break
            fi
        done
    fi
    export PATH=/opt/rocm/bin:${PATH:-}
    command -v hipcc >/dev/null 2>&1 || err "hipcc still missing after dev package install; install ROCm HIP dev tools manually"
}

rocm_repo_for_codename() {
    local codename="$1"
    local version_id="${2:-}"
    if [[ -n "${ROCM_REPO_VER:-}" ]]; then
        echo "$ROCM_REPO_VER"
        return 0
    fi
    case "$codename" in
        noble|oracular|plucky) echo "7.2" ;;
        jammy) echo "6.4" ;;
        *) echo "" ;;
    esac
}

python_apt_packages_for_codename() {
    local codename="$1"
    local version_id="$2"
    case "$codename" in
        noble|oracular|plucky) echo "python3 python3-venv python3-pip" ;;
        jammy) echo "python3.10 python3.10-venv python3-pip" ;;
        *)
            # Fallback for unknown codenames
            if [[ -n "$version_id" ]]; then
                local major="${version_id%%.*}"
                if [[ "$major" -ge 24 ]]; then
                    echo "python3 python3-venv python3-pip"
                elif [[ "$major" -ge 22 ]]; then
                    echo "python3.10 python3.10-venv python3-pip"
                else
                    echo ""
                fi
            else
                echo ""
            fi
            ;;
    esac
}

confirm() {
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    read -rp "$1 [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

# ─── OS + GPU Detection ────────────────────────────────────────────────────
log "AMDBTX miner installer — AMD GPU edition"

# HSA_ENABLE_DXG_DETECTION is required for AMD GPU detection in WSL2.
# Setting it early ensures rocminfo/rocm-smi can see the GPU later.
export HSA_ENABLE_DXG_DETECTION=1

OS="$(uname -s)"
if [[ "$OS" != "Linux" ]]; then
    err "unsupported OS: $OS (AMD ROCm requires Linux)"
fi

OS_ID=""
OS_VERSION_ID=""
UBUNTU_CODENAME=""
IS_HIVEOS=0
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS_ID="${ID:-}"
    OS_VERSION_ID="${VERSION_ID:-}"
    UBUNTU_CODENAME="${VERSION_CODENAME:-}"
fi
if [[ -e /hive || -e /etc/hive-release ]] || grep -qi 'hive\s*os\|hiveos' /etc/os-release 2>/dev/null; then
    IS_HIVEOS=1
fi
if [[ -z "$UBUNTU_CODENAME" ]]; then
    UBUNTU_CODENAME="$(lsb_release -cs 2>/dev/null || true)"
fi
if [[ -z "$UBUNTU_CODENAME" ]]; then
    UBUNTU_CODENAME="jammy"
fi

if [[ "$OS_ID" == "ubuntu" ]]; then
    log "detected Ubuntu ${OS_VERSION_ID:-unknown} (${UBUNTU_CODENAME})"
elif [[ -n "$OS_ID" ]]; then
    warn "detected ${OS_ID} ${OS_VERSION_ID:-}; installer is best tested on Ubuntu 22.04/24.04"
fi
if [[ "$IS_HIVEOS" -eq 1 ]]; then
    log "detected HiveOS; preserving HiveOS-managed AMD/ROCm driver stack"
    SKIP_ROCM=1
fi

if have_apt; then
    log "installing base system dependencies..."
    apt_install \
        ca-certificates curl wget gnupg2 lsb-release coreutils findutils grep gawk sed \
        python3 python3-venv python3-pip \
        libstdc++6 libgcc-s1 libc6 libelf1 libdrm2 libdrm-amdgpu1 libnuma1 zlib1g libzstd1 \
        pciutils procps >/dev/null || warn "some base packages failed to install; continuing"
else
    warn "apt-get not found; installer can still run, but you must provide Python 3.10+, curl, and ROCm runtime libraries"
fi

# Detect AMD GPU (first pass: rocm-smi/rocminfo if ROCm is already installed)
HAS_AMD=0
GPU_NAME=""
GPU_ARCH=""
detect_amd_gpu() {
    local has=0 name="" arch=""
    if command -v rocm-smi >/dev/null 2>&1; then
        name="$(rocm-smi --showproductname 2>/dev/null | head -1 || true)"
        if [[ -n "$name" && "$name" != *"None"* ]]; then
            has=1
            arch="$(rocm-smi --showid 2>/dev/null | head -1 | grep -oP 'gfx[0-9a-f]{3,}' || true)"
        fi
    fi
    if [[ "$has" -eq 0 ]] && command -v rocminfo >/dev/null 2>&1; then
        local out
        out=$(rocminfo 2>/dev/null) || true
        if [[ -n "$out" ]]; then
            name=$(echo "$out" | awk '
                /Agent [0-9]/ { mktname=""; devtype=""; }
                /Marketing Name:/ { sub(/^.*Marketing Name: */, ""); mktname=$0; sub(/ *$/, "", mktname); }
                /Device Type.*GPU/ { devtype="GPU"; }
                devtype=="GPU" && mktname!="" { print mktname; devtype=""; exit; }
            ' || true)
            arch=$(echo "$out" | awk '
                /Agent [0-9]/ { devtype=""; arch=""; }
                /Device Type.*GPU/ { devtype="GPU"; }
                /gfx[0-9a-f]{3,}/ && devtype=="GPU" && arch=="" { match($0, /gfx[0-9a-f]{3,}/); arch=substr($0, RSTART, RLENGTH); }
                END { print arch; }
            ' || true)
            if [[ -n "$arch" ]]; then
                has=1
                [[ -z "$name" || "$name" == *"None"* ]] && name="AMD-$arch"
            fi
        fi
    fi
    if [[ "$has" -eq 1 ]]; then
        HAS_AMD=1
        GPU_NAME="$name"
        GPU_ARCH="$arch"
        if [[ -n "$GPU_ARCH" ]]; then
            log "detected AMD GPU: ${GPU_NAME} (arch: ${GPU_ARCH})"
        else
            log "detected AMD GPU: ${GPU_NAME}"
        fi
    fi
}
detect_amd_gpu
if [[ "$HAS_AMD" -eq 0 ]]; then
    warn "no AMD GPU detected yet (will retry after ROCm install)"
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
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON="$cand"
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
            break
        fi
        PYTHON=""
    fi
done

if [[ -z "$PYTHON" ]]; then
    PY_PKGS="$(python_apt_packages_for_codename "$UBUNTU_CODENAME" "$OS_VERSION_ID")"
    if [[ -z "$PY_PKGS" ]]; then
        err "Python 3.10+ is required. Please use Ubuntu 22.04+ or install Python 3.10+ manually."
    fi
    log "installing Python runtime (${PY_PKGS}) via apt..."
    # shellcheck disable=SC2086
    apt_install $PY_PKGS || err "failed to install Python 3.10+"
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
            PYTHON="$cand"
            break
        fi
    done
    [[ -n "$PYTHON" ]] || err "Python install completed but no Python 3.10+ executable was found"
fi
log "using Python: $($PYTHON --version 2>&1)"

# ─── Install ROCm (if missing) ─────────────────────────────────────────────
ROCM_LIB_PRESENT=0
if find /opt/rocm /opt/rocm-* -maxdepth 2 -name 'libamdhip64.so*' -print -quit 2>/dev/null | grep -q .; then
    ROCM_LIB_PRESENT=1
elif command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libamdhip64\.so'; then
    ROCM_LIB_PRESENT=1
fi

if [[ "$SKIP_ROCM" -eq 1 ]]; then
    log "skipping ROCm installation (--skip-rocm)"
elif [[ "$ROCM_LIB_PRESENT" -eq 1 ]]; then
    log "ROCm runtime libraries detected; skipping package installation"
else
    ROCM_REPO_VER="$(rocm_repo_for_codename "$UBUNTU_CODENAME" "$OS_VERSION_ID")"
    log "ROCm not detected. Installing ROCm packages..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo_cmd apt-get update -qq || true
        sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wget gnupg2 curl ca-certificates || true

        # First try system packages (Ubuntu 26.04+ has ROCm in main repos).
        # Package names differ between Ubuntu and Radeon repositories.
        ROCM_INSTALL_OK=0
        for pkg_set in "rocm-hip-runtime rocminfo" "hip-runtime-amd rocminfo" "rocm-hip-runtime" "hip-runtime-amd"; do
            if sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $pkg_set 2>/dev/null; then
                ROCM_INSTALL_OK=1
                log "ROCm installed from system repositories"
                break
            fi
        done

        if [[ "$ROCM_INSTALL_OK" -eq 0 ]]; then
            if [[ -z "$ROCM_REPO_VER" ]]; then
                err "ROCm packages were not available from system apt, and no supported Radeon repo mapping exists for Ubuntu ${OS_VERSION_ID:-unknown} (${UBUNTU_CODENAME}). Install ROCm manually, then rerun with --skip-rocm."
            fi
            log "system packages not available, trying external ROCm ${ROCM_REPO_VER} repo..."
            echo "deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/${ROCM_REPO_VER} ${UBUNTU_CODENAME} main" | sudo_cmd tee /etc/apt/sources.list.d/rocm.list > /dev/null

            # Remove stale ROCm 5.x sources if any
            for old in /etc/apt/sources.list.d/rocm*.list; do
                [[ -f "$old" ]] && grep -q "rocm/apt/5\." "$old" 2>/dev/null && sudo_cmd rm -f "$old" && log "removed stale ROCm 5.x source: $old"
            done

            log "updating package lists..."
            sudo_cmd apt-get update -qq 2>&1 | grep -v "Warning:" || true

            # Install only the runtime needed by the pre-built miner. Full rocm-dev
            # pulls compiler/debugger packages that conflict on WSL/Noble.
            log "installing ROCm runtime packages (may take several minutes)..."
            if [[ "$ROCM_REPO_VER" == "7."* ]]; then
                RUNTIME_PACKAGES=(hip-runtime-amd rocminfo)
            else
                RUNTIME_PACKAGES=(rocm-hip-runtime rocminfo)
            fi
            if ! sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${RUNTIME_PACKAGES[@]}" 2>&1 | tail -10; then
                warn "ROCm runtime install failed — diagnosing..."
                sudo_cmd apt-get install -y "${RUNTIME_PACKAGES[@]}" 2>&1 | grep -E "Depends:|not going|broken|held|not installable" | head -10 >&2 || true
                warn "install ROCm runtime manually, then rerun with --skip-rocm"
            fi
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

        # Re-detect GPU now that ROCm is installed
        detect_amd_gpu
        if [[ "$HAS_AMD" -eq 0 ]]; then
            warn "GPU still not detected via rocm-smi/rocminfo — will try solver probe later"
        fi
    else
        err "apt-get not available; install ROCm 6.x manually then re-run"
    fi
fi

# ─── Install pip + runtime deps ──────────────────────────────────────────────
if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
    log "python venv support not present; installing via apt..."
    apt_install python3-venv python3-pip || err "failed to install python3-venv"
fi

# ─── Clean previous installation ────────────────────────────────────────────
if [[ -d "$INSTALL_DIR" ]]; then
    log "removing previous installation at ${INSTALL_DIR}..."
    rm -rf "$INSTALL_DIR"
fi
if [[ -d "$VENV_DIR" ]]; then
    log "removing previous virtual environment at ${VENV_DIR}..."
    rm -rf "$VENV_DIR"
fi
if [[ -f "$LAUNCHER_PATH" ]]; then
    rm -f "$LAUNCHER_PATH"
fi

log "creating private Python environment..."
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip wheel pyyaml

# ─── Fetch source snapshot (default path) ────────────────────────────────
mkdir -p "${INSTALL_DIR}/bin"
TMP_BUILD_DIR="$(mktemp -d)"
REPO_SRC_DIR="${TMP_BUILD_DIR}/repo"
cleanup_temp() {
    rm -rf "$TMP_BUILD_DIR" 2>/dev/null || true
}
trap cleanup_temp EXIT

RESOLVED_SOURCE_REF="$SOURCE_REF"
SOURCE_ARCHIVE_SHA256=""
SOURCE_COMMIT=""
if [[ "$USE_PREBUILT" -eq 0 ]]; then
    SOURCE_API_URL="https://api.github.com/repos/${SOURCE_REPO}/commits/${SOURCE_REF}"
    SOURCE_API_JSON="$(curl -fsSL "$SOURCE_API_URL" 2>/dev/null || true)"
    SOURCE_COMMIT="$(printf '%s' "$SOURCE_API_JSON" | sed -nE 's/^[[:space:]]*"sha":[[:space:]]*"([0-9a-f]{40})",?$/\1/p' | head -1)"
    if [[ -n "$SOURCE_COMMIT" ]]; then
        RESOLVED_SOURCE_REF="$SOURCE_COMMIT"
    fi
    SOURCE_ARCHIVE_URL="https://github.com/${SOURCE_REPO}/archive/${RESOLVED_SOURCE_REF}.tar.gz"
    log "downloading AMDBTX source (${RESOLVED_SOURCE_REF})..."
    mkdir -p "$REPO_SRC_DIR"
    curl -fsSL "$SOURCE_ARCHIVE_URL" -o "$TMP_BUILD_DIR/repo.tar.gz" 2>/dev/null || \
        err "failed to download source from ${SOURCE_ARCHIVE_URL}"
    SOURCE_ARCHIVE_SHA256="$(sha256sum "$TMP_BUILD_DIR/repo.tar.gz" | awk '{print $1}')"
    tar xzf "$TMP_BUILD_DIR/repo.tar.gz" -C "$REPO_SRC_DIR" --strip-components=1
fi

if [[ "$SKIP_PIP" -eq 1 ]]; then
    log "skipping amdbtx-miner pip install (--skip-pip)"
elif [[ "$USE_PREBUILT" -eq 1 ]]; then
    WHEEL_PATH="${TMP_BUILD_DIR}/${WHEEL_FILENAME}"
    WHEEL_URL="${PREBUILDS_BASE}/${WHEEL_FILENAME}"
    log "downloading amdbtx-miner ${EXPECTED_MINER_VERSION} wheel..."
    curl -fsSL "$WHEEL_URL" -o "$WHEEL_PATH" 2>/dev/null || \
        err "failed to download Python wheel from ${WHEEL_URL}"
    WHEEL_SHA256="$(sha256sum "$WHEEL_PATH" | awk '{print $1}')"
    [[ "$WHEEL_SHA256" == "$EXPECTED_WHEEL_SHA256" ]] || \
        err "wheel checksum mismatch: got ${WHEEL_SHA256}"
    "$VENV_DIR/bin/python" -m pip install --quiet --upgrade "$WHEEL_PATH"
    INSTALLED_MINER_VERSION="$(
        "$VENV_DIR/bin/python" -c 'import amdbtx_miner; print(amdbtx_miner.__version__)'
    )"
else
    log "installing amdbtx-miner from source..."
    "$VENV_DIR/bin/python" -m pip install --quiet --upgrade "$REPO_SRC_DIR"
    INSTALLED_MINER_VERSION="$(
        "$VENV_DIR/bin/python" -c 'import amdbtx_miner; print(amdbtx_miner.__version__)'
    )"
    WHEEL_SHA256="source"
fi
if [[ "$SKIP_PIP" -eq 0 ]]; then
    [[ "$INSTALLED_MINER_VERSION" == "$EXPECTED_MINER_VERSION" ]] || \
        warn "installed miner version ${INSTALLED_MINER_VERSION}, expected ${EXPECTED_MINER_VERSION}"
    "$VENV_DIR/bin/python" -c '
import amdbtx_miner
assert "matmul_parent_mtp_seed_v3" in amdbtx_miner.PROTOCOL_CAPABILITIES
' || err "installed miner does not advertise BTX V3 parent-MTP support"
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

# ─── Install solver binary ───────────────────────────────────────────────
SOLVER_BUILT_FROM_SOURCE=0
if [[ -n "$LOCAL_SOLVER" ]]; then
    log "using local solver at ${LOCAL_SOLVER}"
    install -m 0755 "$LOCAL_SOLVER" "$SOLVER_PATH"
elif [[ "$USE_PREBUILT" -eq 1 ]]; then
    SOLVER_ASSET_PATH="${TMP_BUILD_DIR}/btx-gbt-solve-hip"
    SOLVER_URL="${PREBUILDS_BASE}/btx-gbt-solve-hip"
    log "downloading btx-gbt-solve-hip ${EXPECTED_SOLVER_VERSION} (prebuilt)..."
    curl -fsSL "$SOLVER_URL" -o "$SOLVER_ASSET_PATH" 2>/dev/null || \
        err "failed to download solver binary from ${SOLVER_URL}"
    SOLVER_ASSET_SHA256="$(sha256sum "$SOLVER_ASSET_PATH" | awk '{print $1}')"
    [[ "$SOLVER_ASSET_SHA256" == "$EXPECTED_SOLVER_SHA256" ]] || \
        err "solver checksum mismatch: got ${SOLVER_ASSET_SHA256}"
    install -m 0755 "$SOLVER_ASSET_PATH" "$SOLVER_PATH"
else
    SOLVER_SRC_DIR="$REPO_SRC_DIR/solver"
    [[ -f "$SOLVER_SRC_DIR/build.sh" ]] || err "solver sources missing at ${SOLVER_SRC_DIR}/build.sh"

    ensure_hip_compiler

    # Align HIP dev packages with the runtime ROCm version when possible.
    ROCM_DEV_NEEDED=0
    set +e
    ROCM_LIB_VER=$(dpkg -l libamdhip64-dev 2>/dev/null | awk '/libamdhip64-dev/ {print $3}' | head -1)
    ROCM_VER=$(echo "${ROCM_LIB_VER:-}" | grep -oP '^\d+\.\d+' | head -1) || true
    HIPCC_VER=$(hipcc --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1) || true
    set -e
    HIPCC_VER="${HIPCC_VER:-0}"
    if [[ -n "$ROCM_VER" && -n "$HIPCC_VER" && "$ROCM_VER" != "$HIPCC_VER" ]]; then
        ROCM_DEV_NEEDED=1
    fi
    if [[ "$ROCM_DEV_NEEDED" -eq 1 ]]; then
        log "installing matching HIP dev packages..."
        sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq hip-dev 2>/dev/null || true
    fi

    BUILD_LOG="$TMP_BUILD_DIR/build.log"
    if [[ "$COMPILE_ALL_ARCHS" -eq 1 ]]; then
        log "compiling solver for all common AMD GPU architectures..."
        export AMDBTX_HIP_ARCHS="$ALL_HIP_ARCHS"
    elif [[ -n "$GPU_ARCH" ]]; then
        log "compiling solver for detected GPU arch ${GPU_ARCH}..."
        export AMDBTX_HIP_ARCHS="$GPU_ARCH"
    else
        log "compiling solver (auto-detect arch, fallback to common AMD targets)..."
        unset AMDBTX_HIP_ARCHS || true
    fi
    if bash "$SOLVER_SRC_DIR/build.sh" > "$BUILD_LOG" 2>&1; then
        log "solver compilation successful"
        install -m 0755 "$SOLVER_SRC_DIR/build/btx-gbt-solve-hip" "$SOLVER_PATH"
        SOLVER_BUILT_FROM_SOURCE=1
    else
        warn "solver compilation failed (see log below). Try: sudo apt install hip-dev"
        cat "$BUILD_LOG" >&2
        err "solver compilation failed. Use --use-prebuilt for release binaries, or install hip-dev/rocm-dev."
    fi
fi

log "solver installed → $SOLVER_PATH"
SOLVER_VERSION_OUTPUT="$("$SOLVER_PATH" --version 2>&1 || true)"
[[ "$SOLVER_VERSION_OUTPUT" == *"$EXPECTED_SOLVER_VERSION"* ]] || \
    err "solver is not fork-ready: ${SOLVER_VERSION_OUTPUT:-version unavailable}"
SOLVER_SHA256="$(sha256sum "$SOLVER_PATH" | awk '{print $1}')"
cat > "${INSTALL_DIR}/install-source.txt" <<EOF
build_mode=$([[ "$SOLVER_BUILT_FROM_SOURCE" -eq 1 ]] && echo source || echo prebuilt)
source_repo=${SOURCE_REPO}
source_ref=${SOURCE_REF}
source_commit=${RESOLVED_SOURCE_REF:-unknown}
source_archive_sha256=${SOURCE_ARCHIVE_SHA256:-skipped}
release_repo=${RELEASE_REPO}
prebuilds_tag=${PREBUILDS_TAG}
wheel_filename=${WHEEL_FILENAME}
wheel_sha256=${WHEEL_SHA256:-skipped}
solver_sha256=${SOLVER_SHA256}
solver_version=${SOLVER_VERSION_OUTPUT}
EOF
if [[ "$SOLVER_BUILT_FROM_SOURCE" -eq 1 ]]; then
    log "solver built from source (${RESOLVED_SOURCE_REF})"
else
    log "release repo: ${RELEASE_REPO}"
    log "release tag: ${PREBUILDS_TAG}"
fi
log "miner version: ${INSTALLED_MINER_VERSION:-skipped}"
log "solver sha256: ${SOLVER_SHA256}"

# ─── Build solver runtime (library resolver) ──────────────────────────────
# The pre-built solver binary links against specific sonames (e.g. libamdhip64.so.6, libhipblas.so.2)
# ROCm 7.x ships different sonames. We create a local runtime dir with symlinks.
RUNTIME_DIR="${INSTALL_DIR}/runtime"
mkdir -p "$RUNTIME_DIR"

ROCM_LIB_DIRS=()
add_rocm_lib_dir() {
    local d="$1"
    [[ -n "$d" && -d "$d" ]] || return 0
    local existing
    for existing in "${ROCM_LIB_DIRS[@]}"; do
        [[ "$existing" == "$d" ]] && return 0
    done
    ROCM_LIB_DIRS+=("$d")
}

collect_rocm_lib_dirs() {
    local d path
    ROCM_LIB_DIRS=()
    add_rocm_lib_dir /opt/rocm/lib
    for d in /opt/rocm-*/lib; do [[ -d "$d" ]] && add_rocm_lib_dir "$d"; done
    if command -v ldconfig >/dev/null 2>&1; then
        while read -r path; do
            [[ -n "$path" ]] && add_rocm_lib_dir "$(dirname "$path")"
        done < <(ldconfig -p 2>/dev/null | awk '/libamdhip64|libhsa-runtime64|librocprofiler-register|libhip|libroc/ {print $NF}')
    fi
}
collect_rocm_lib_dirs

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

resolve_ldd_missing_libs() {
    local missing soname base
    missing="$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>/dev/null | awk '/not found/ {print $1}' || true)"
    [[ -n "$missing" ]] || return 0
    while read -r soname; do
        [[ -n "$soname" ]] || continue
        case "$soname" in
            libamdhip64.so.*|libhsa-runtime64.so.*|librocprofiler-register.so.*|libhip*.so.*|libroc*.so.*|libamd*.so.*)
                base="${soname%%.so*}"
                resolve_lib "$soname" "$base" || true
                ;;
        esac
    done <<< "$missing"
}

link_resolved_rocm_libs() {
    local line path soname
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>/dev/null | while read -r line; do
        path="$(echo "$line" | awk '/=> \// {print $3}')"
        [[ -n "$path" && -f "$path" ]] || continue
        soname="$(basename "$path")"
        case "$soname" in
            libamd*|libhsa*|libroc*|libhip*) ln -sfn "$(readlink -f "$path")" "$RUNTIME_DIR/$soname" 2>/dev/null || true ;;
        esac
    done
}

log "building solver runtime..."
if [[ "$SOLVER_BUILT_FROM_SOURCE" -eq 1 ]]; then
    # Source builds should run against the same libraries selected by the linker.
    # Do not force /opt/rocm symlinks here; distrobox hosts can expose a newer
    # /opt/rocm while Ubuntu hipcc links against system ROCm libraries.
    RUNTIME_LD_PATH=""
    while read -r libdir; do
        [[ -n "$libdir" ]] || continue
        case ":$RUNTIME_LD_PATH:" in
            *":$libdir:"*) ;;
            *) RUNTIME_LD_PATH="${RUNTIME_LD_PATH:+$RUNTIME_LD_PATH:}$libdir" ;;
        esac
    done < <(ldd "$SOLVER_PATH" 2>/dev/null | awk '/=> \// {print $3}' | while read -r p; do
        case "$(basename "$p")" in
            libamd*|libhsa*|libroc*|libhip*) dirname "$p" ;;
        esac
    done)
    [[ -n "$RUNTIME_LD_PATH" ]] || RUNTIME_LD_PATH="$RUNTIME_DIR"
else
declare -A SOLVER_LIBS=(
    ["libamdhip64.so.7"]="libamdhip64"
    ["libamdhip64.so.6"]="libamdhip64"
    ["libhsa-runtime64.so.1"]="libhsa-runtime64"
    ["librocprofiler-register.so.0"]="librocprofiler-register"
)
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
resolve_ldd_missing_libs
link_resolved_rocm_libs
fi

# Verify solver can load all libraries
MISSING=$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>&1 | grep 'not found' || true)
if [[ -n "$MISSING" ]]; then
    warn "solver has unresolved libraries:"
    echo "$MISSING" | while read -r line; do warn "  $line"; done

    if [[ "$SKIP_ROCM" -eq 1 ]]; then
        warn "not installing ROCm packages because ROCm installation is disabled"
    else
    # Fallback: try apt install. System packages are preferred because Ubuntu
    # 26.04+ ships ROCm directly; the Radeon repo is only a fallback for known
    # codenames.
    ROCM_REPO_VER="$(rocm_repo_for_codename "$UBUNTU_CODENAME" "$OS_VERSION_ID")"
    ROCM_APT_OK=0
    for pkg_set in "rocm-hip-runtime rocminfo" "hip-runtime-amd rocminfo" "rocm-hip-runtime" "hip-runtime-amd"; do
        if sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $pkg_set libdrm-amdgpu1 libnuma1 libelf1 libzstd1 2>/dev/null; then
            ROCM_APT_OK=1
            log "apt install succeeded (system packages)"
            break
        fi
    done

    if [[ "$ROCM_APT_OK" -eq 0 && -n "$ROCM_REPO_VER" ]]; then
        log "system packages not available, trying external ROCm ${ROCM_REPO_VER} repo..."
        curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | sudo_cmd apt-key add - 2>/dev/null || true
        echo "deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/$ROCM_REPO_VER $UBUNTU_CODENAME main" | sudo_cmd tee /etc/apt/sources.list.d/rocm.list >/dev/null
        if sudo_cmd apt-get update -qq 2>/dev/null; then
            if [[ "$ROCM_REPO_VER" == "7."* ]]; then
                EXTRA_ROCM_PACKAGES=(hip-runtime-amd rocminfo)
            else
                EXTRA_ROCM_PACKAGES=(rocm-hip-runtime rocminfo)
            fi
            if sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${EXTRA_ROCM_PACKAGES[@]}" libdrm-amdgpu1 libnuma1 libelf1 libzstd1 2>/dev/null; then
                ROCM_APT_OK=1
                log "apt install succeeded (external ROCm repo)"
                # Install matching dev packages for solver compilation
                sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq hip-dev 2>/dev/null || true
            fi
        fi
    elif [[ "$ROCM_APT_OK" -eq 0 ]]; then
        warn "no external ROCm repo mapping for ${UBUNTU_CODENAME}; relying on existing libraries"
    fi

    if [[ "$ROCM_APT_OK" -eq 1 ]]; then
        collect_rocm_lib_dirs
        for soname in "${!SOLVER_LIBS[@]}"; do resolve_lib "$soname" "${SOLVER_LIBS[$soname]}" || true; done
        RUNTIME_LD_PATH="$RUNTIME_DIR"
        for d in "${ROCM_LIB_DIRS[@]}"; do RUNTIME_LD_PATH="$RUNTIME_LD_PATH:$d"; done
        resolve_ldd_missing_libs
        link_resolved_rocm_libs
        MISSING=$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>&1 | grep 'not found' || true)
    else
        warn "apt install failed or was not applicable - solver may not work"
    fi
    fi
else
    log "all solver libraries resolved"
fi
FINAL_MISSING=$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" ldd "$SOLVER_PATH" 2>&1 | grep 'not found' || true)
if [[ -n "$FINAL_MISSING" ]]; then
    warn "solver still has unresolved libraries after automatic setup:"
    echo "$FINAL_MISSING" | while read -r line; do warn "  $line"; done
    warn "GPU mining may not start until ROCm runtime packages are fixed."
fi

# Solver probe: launch the solver binary briefly and parse its HIP startup line.
# This is the most reliable GPU detection — it works even when rocm-smi/rocminfo
# are missing or misconfigured, and provides the exact GPU model + arch + VRAM.
log "probing AMD GPU via solver binary..."
PROBE_TMP="$(mktemp -t amdbtx-probe.XXXXXX 2>/dev/null || mktemp)"
printf '%s\n' '{"version":536870912,"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000","merkle_root":"0000000000000000000000000000000000000000000000000000000000000000","time":1779672814,"bits":"1d17c609","seed_a":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","seed_b":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","block_height":1,"nonce_start":0,"max_tries":1,"max_seconds":1,"share_target":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}' > "$PROBE_TMP" || true
SOLVER_PROBE=$(LD_LIBRARY_PATH="$RUNTIME_LD_PATH" HSA_ENABLE_DXG_DETECTION=1 "$SOLVER_PATH" --daemon --backend hip --epsilon-bits 0 --batch-size 1 < "$PROBE_TMP" 2>&1) || true
rm -f "$PROBE_TMP"
PROBE_GPU_NAME=""
PROBE_GPU_ARCH=""
if [[ -n "$SOLVER_PROBE" ]]; then
    PROBE_GPU_NAME=$(echo "$SOLVER_PROBE" | sed -nE 's/^HIP GPU detected: (.*) arch=(gfx[0-9a-f]+) memory=.*/\1/p') || true
    PROBE_GPU_ARCH=$(echo "$SOLVER_PROBE" | grep -oE 'gfx[0-9a-f]{3,}' | head -1) || true
fi
if [[ -n "$PROBE_GPU_ARCH" ]]; then
    HAS_AMD=1
    GPU_NAME="${PROBE_GPU_NAME:-AMD GPU}"
    GPU_ARCH="$PROBE_GPU_ARCH"
    log "detected AMD GPU via solver: ${GPU_NAME} (arch: ${GPU_ARCH})"
else
    log "solver probe: no GPU arch in output (using earlier detection)"
    if [[ "$HAS_AMD" -eq 1 ]]; then
        if [[ -n "$GPU_ARCH" ]]; then
            log "using GPU detected earlier via rocm-smi/rocminfo: ${GPU_NAME} (arch: ${GPU_ARCH})"
        else
            log "using GPU detected earlier via rocm-smi/rocminfo: ${GPU_NAME}"
        fi
    else
        warn "no AMD GPU detected by solver probe either — solver will run on CPU only (much slower)"
        warn "ensure /dev/kfd and /dev/dri are accessible"
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
GPU_BATCH=4096
GPU_PREFETCH=8

# Older GCN cards (gfx803) benefit from lower settings
if [[ "$GPU_ARCH" == "gfx803" || "$GPU_NAME" == *"RX 4"* || "$GPU_NAME" == *"RX 5"* ]]; then
    GPU_WORKERS=8
    GPU_THREADS=4
    GPU_BATCH=64
    GPU_PREFETCH=4
fi

# RDNA 2/3/4: larger scan batches keep the GPU fed (see --benchmark)
if [[ "$GPU_ARCH" == gfx103* ]]; then
    GPU_BATCH=1048576
fi
if [[ "$GPU_ARCH" == gfx110* || "$GPU_ARCH" == gfx115* ]]; then
    GPU_BATCH=4194304
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
gpu_inputs: 1

# Slice sizing
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
pool_max_shares_per_slice: 0

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
    SMOKE_LAST_LINE="$(echo "$SMOKE_OUT" | grep -E '^\{.*\}$' | tail -1 || true)"
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
