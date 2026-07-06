#!/usr/bin/env bash
set -e
set -o pipefail

MINER_VERSION="1.2.2"
MINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CUSTOM_CONFIG_FILENAME:-${MINER_DIR}/amdbtx.yaml}"
ARGS_FILE="${CONFIG_FILE}.args"
LOG_BASENAME="${CUSTOM_LOG_BASENAME:-/var/log/miner/amdbtx/lastrun}"
LOG_FILE="${LOG_BASENAME}.log"
START_FILE="/var/run/amdbtx.started"

export HOME="${HOME:-/root}"
INSTALL_DIR="${HOME}/.amdbtx-miner"
VENV_DIR="${AMDBTX_VENV_DIR:-${HOME}/.local/share/amdbtx-miner/venv}"
MINER_BIN="${VENV_DIR}/bin/amdbtx-miner"
SOLVER_BIN="${INSTALL_DIR}/bin/btx-gbt-solve-hip"
INSTALLER_URL="${AMDBTX_INSTALLER_URL:-https://raw.githubusercontent.com/thekillsquad007/amdbtx/v${MINER_VERSION}/install_amd.sh}"

add_existing_path() {
    local path
    for path in "$@"; do
        [[ -d "$path" ]] || continue
        case ":${PATH:-}:" in
            *":$path:"*) ;;
            *) export PATH="$path:${PATH:-}" ;;
        esac
    done
}

add_existing_ld_path() {
    local path
    for path in "$@"; do
        [[ -d "$path" ]] || continue
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$path:"*) ;;
            *) export LD_LIBRARY_PATH="$path:${LD_LIBRARY_PATH:-}" ;;
        esac
    done
}

setup_hive_rocm_env() {
    local rocm_dir
    for rocm_dir in /opt/rocm /opt/rocm-*; do
        [[ -d "$rocm_dir" ]] || continue
        add_existing_path "$rocm_dir/bin"
        add_existing_ld_path "$rocm_dir/lib" "$rocm_dir/lib64" "$rocm_dir/hip/lib"
    done
    add_existing_ld_path /usr/lib/x86_64-linux-gnu /usr/local/lib
}

installed_version() {
    "${VENV_DIR}/bin/python" -c 'import amdbtx_miner; print(amdbtx_miner.__version__)' 2>/dev/null || true
}

os_distro() {
    # Detect distro: 22 (Ubuntu 22) or 20 (Ubuntu 20) or unknown
    local id version
    if [[ -f /etc/os-release ]]; then
        id="$(grep -oP '(?<=^ID=).*' /etc/os-release 2>/dev/null | tr -d '"')"
        version="$(grep -oP '(?<=^VERSION_ID=).*' /etc/os-release 2>/dev/null | tr -d '"')"
        case "$id" in
            ubuntu)
                case "$version" in
                    22*) echo "22" ;;
                    20*) echo "20" ;;
                    *) echo "unknown" ;;
                esac
                ;;
            *) echo "unknown" ;;
        esac
    else
        echo "unknown"
    fi
}

use_bundled_binary() {
    local binary="${MINER_DIR}/amdbtx-miner-linux-ubuntu${1}"
    if [[ -x "$binary" ]]; then
        echo "AMDBTX: using bundled portable binary (ubuntu${1})"
        mkdir -p "$(dirname "$MINER_BIN")"
        mkdir -p "$(dirname "$SOLVER_BIN")"
        cp -a "$binary" "$MINER_BIN"
        # The portable binary includes the solver internally; mark SOLVER_BIN as present
        touch "$SOLVER_BIN"
        chmod +x "$MINER_BIN" "$SOLVER_BIN"
        return 0
    fi
    return 1
}

ensure_installed() {
    local version installer install_mode distro

    # Try bundled binaries first
    distro="$(os_distro)"
    case "$distro" in
        22|20)
            if use_bundled_binary "$distro"; then
                version="$(installed_version)"
                [[ "$version" == "$MINER_VERSION" ]] || {
                    echo "AMDBTX: bundled version '$version', expected '$MINER_VERSION'" >&2
                }
                if [[ -x "$MINER_BIN" ]]; then
                    return 0
                fi
            fi
            ;;
    esac

    version="$(installed_version)"
    if [[ "$version" == "$MINER_VERSION" && -x "$MINER_BIN" && -x "$SOLVER_BIN" ]]; then
        return 0
    fi

    command -v curl >/dev/null 2>&1 || {
        echo "AMDBTX: curl is required for first-run installation" >&2
        return 1
    }

    installer="/tmp/amdbtx-install-${MINER_VERSION}.sh"
    echo "AMDBTX: installing upstream v${MINER_VERSION} (ROCm packages are skipped for HiveOS)"
    curl -fsSL "$INSTALLER_URL" -o "$installer"
    chmod +x "$installer"

    install_mode="${AMDBTX_HIVE_INSTALL_MODE:-prebuilt}"
    if [[ "$install_mode" == "source" ]]; then
        AMDBTX_SOURCE_REF="v${MINER_VERSION}" bash "$installer" \
            --yes \
            --skip-rocm \
            --pool "btx-sg.lproute.com:8660"
    else
        AMDBTX_SOURCE_REF="v${MINER_VERSION}" bash "$installer" \
            --yes \
            --skip-rocm \
            --use-prebuilt \
            --pool "btx-sg.lproute.com:8660"
    fi

    version="$(installed_version)"
    [[ "$version" == "$MINER_VERSION" ]] || {
        echo "AMDBTX: installed version '$version', expected '$MINER_VERSION'" >&2
        return 1
    }
    [[ -x "$MINER_BIN" && -x "$SOLVER_BIN" ]] || {
        echo "AMDBTX: miner or solver binary is missing after installation" >&2
        return 1
    }
}

read_extra_args() {
    EXTRA_ARGS=()
    [[ -s "$ARGS_FILE" ]] || return 0
    # HiveOS extra CLI arguments are intentionally simple whitespace-separated flags.
    # Use Extra config YAML lines when values need spaces or complex quoting.
    read -r -a EXTRA_ARGS < "$ARGS_FILE"
}

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$START_FILE")"
date +%s > "$START_FILE"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

setup_hive_rocm_env

if [[ ! -s "$CONFIG_FILE" ]]; then
    echo "AMDBTX: config file not found: $CONFIG_FILE"
    echo "AMDBTX: run h-config.sh through HiveOS or set CUSTOM_CONFIG_FILENAME"
    exit 1
fi

ensure_installed
read_extra_args

echo "AMDBTX: starting miner v${MINER_VERSION}"
echo "AMDBTX: config=$CONFIG_FILE"
exec "$MINER_BIN" --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
