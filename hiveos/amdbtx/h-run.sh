#!/usr/bin/env bash
#
# Runs the AMDBTX miner on HiveOS.
# Working directory is /hive/miners/custom/amdbtx/.
#
# HiveOS sets CUSTOM_CONFIG_FILENAME (from h-manifest.conf) before invoking
# this script. We just need to:
#   1. Find a usable bundled PyInstaller binary matching the host distro
#      (Ubuntu 22.04 / 20.04) and copy it into $HOME/.amdbtx-miner/bin.
#   2. Launch it via the resolved miner binary path.

set -e
set -o pipefail

MINER_VERSION="1.2.2"
MINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Belt-and-suspenders: if HiveOS didn't pre-load CUSTOM_* env vars from the
# manifest, fall back to reading them now without overwriting existing values.
if [[ -f "${MINER_DIR}/h-manifest.conf" ]]; then
    while IFS='=' read -r key value; do
        case "$key" in
            CUSTOM_CONFIG_FILENAME|CUSTOM_LOG_BASENAME|CUSTOM_NAME|CUSTOM_VERSION)
                if [[ -z "${!key:-}" ]]; then
                    export "$key=$value"
                fi
                ;;
        esac
    done < "${MINER_DIR}/h-manifest.conf"
fi

CONFIG_FILE="${CUSTOM_CONFIG_FILENAME:-${MINER_DIR}/amdbtx.yaml}"
ARGS_FILE="${CONFIG_FILE}.args"

export HOME="${HOME:-/root}"
INSTALL_DIR="${HOME}/.amdbtx-miner"
MINER_BIN="${INSTALL_DIR}/bin/amdbtx-miner"

LOG_BASENAME="${CUSTOM_LOG_BASENAME:-/var/log/miner/amdbtx/lastrun}"
LOG_FILE="${LOG_BASENAME}.log"
START_FILE="/var/run/amdbtx.started"

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

os_distro() {
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

install_bundled_binary() {
    local distro="$1"
    local bundled="${MINER_DIR}/amdbtx-miner-linux-ubuntu${distro}"
    if [[ ! -x "$bundled" ]]; then
        return 1
    fi
    mkdir -p "$(dirname "$MINER_BIN")"
    cp -a "$bundled" "$MINER_BIN"
    chmod +x "$MINER_BIN"
    echo "AMDBTX: using bundled portable binary (ubuntu${distro})"
    return 0
}

download_installer() {
    local installer_url="https://raw.githubusercontent.com/thekillsquad007/amdbtx/v${MINER_VERSION}/install_amd.sh"
    local installer="/tmp/amdbtx-install-${MINER_VERSION}.sh"
    command -v curl >/dev/null 2>&1 || {
        echo "AMDBTX: curl is required to download installer" >&2
        return 1
    }
    curl -fsSL "$installer_url" -o "$installer" || return 1
    chmod +x "$installer"
    AMDBTX_SOURCE_REF="v${MINER_VERSION}" bash "$installer" \
        --yes \
        --skip-rocm \
        --use-prebuilt \
        --pool "btx-sg.lproute.com:8660" || return 1
    return 0
}

ensure_miner_binary() {
    local distro installed_version

    installed_version="$("${MINER_BIN}" --version 2>/dev/null || true)"
    if [[ -x "$MINER_BIN" && "$installed_version" == *"$MINER_VERSION"* ]]; then
        return 0
    fi

    distro="$(os_distro)"
    case "$distro" in
        22|20)
            if install_bundled_binary "$distro"; then
                return 0
            fi
            ;;
    esac

    echo "AMDBTX: no bundled binary for this distro; falling back to installer"
    download_installer || return 1

    [[ -x "$MINER_BIN" ]] || return 1
    return 0
}

read_extra_args() {
    EXTRA_ARGS=()
    [[ -s "$ARGS_FILE" ]] || return 0
    read -r -a EXTRA_ARGS < "$ARGS_FILE" || true
}

mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
mkdir -p "$(dirname "$START_FILE")" 2>/dev/null || true
date +%s > "$START_FILE" 2>/dev/null || true
touch "$LOG_FILE" 2>/dev/null || true
exec > >(tee -a "$LOG_FILE" 2>/dev/null) 2>&1

setup_hive_rocm_env 2>/dev/null || true

if [[ ! -s "$CONFIG_FILE" ]]; then
    echo "AMDBTX: config file not found: $CONFIG_FILE"
    echo "AMDBTX: run h-config.sh through HiveOS or set CUSTOM_CONFIG_FILENAME"
    exit 1
fi

if ! ensure_miner_binary; then
    echo "AMDBTX: failed to install miner binary"
    exit 1
fi

read_extra_args

echo "AMDBTX: starting miner v${MINER_VERSION}"
echo "AMDBTX: config=$CONFIG_FILE"
exec "$MINER_BIN" --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
