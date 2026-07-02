#!/usr/bin/env bash
# AMD BTX Miner - WSL wrapper around the unified Linux installer.

set -euo pipefail

if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    export HOME="$REAL_HOME"
fi

ADDRESS="${ADDRESS:-}"
WORKER="${WORKER:-}"
POOL="${POOL:-btx-sg.lproute.com:8660}"

log() { echo -e "\033[1;34m[amdbtx]\033[0m $*"; }
err() { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address) ADDRESS="$2"; shift 2 ;;
        --worker) WORKER="$2"; shift 2 ;;
        --pool) POOL="$2"; shift 2 ;;
        --help|-h) echo "Usage: $0 --address btx1... [--worker name] [--pool host:port]"; exit 0 ;;
        *) err "unknown arg: $1" ;;
    esac
done

[[ -n "$ADDRESS" ]] || err "BTX payout address required. Usage: $0 --address btx1z..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_AMD_SH="$SCRIPT_DIR/install_amd.sh"
[[ -f "$INSTALL_AMD_SH" ]] || err "install_amd.sh not found next to install_wsl.sh"

export HSA_ENABLE_DXG_DETECTION=1

args=(--address "$ADDRESS" --pool "$POOL" --yes)
if [[ -n "$WORKER" ]]; then
    args+=(--worker "$WORKER")
fi

log "Using unified Linux installer inside WSL"
exec bash "$INSTALL_AMD_SH" "${args[@]}"
