#!/usr/bin/env bash
# AMDBTX source installer — compiles the HIP solver for your GPU architecture.
#
# This is the recommended installer when you need broad AMD arch support beyond
# the prebuilt multi-arch release binary (gfx900/906/1030/1100/1101).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd_source.sh | bash -s -- --address btx1z... --yes
#   bash install_amd_source.sh --address btx1z... [--worker name] [--pool host:port]
#
# Options forwarded to install_amd.sh:
#   --compile-all-archs   build for every common gfx target (slower, universal binary)
#   --source-ref REF      git branch/tag/commit (default: main)
#   --skip-rocm           skip ROCm package installation
#
# For prebuilt release assets instead, use install_amd.sh --use-prebuilt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_AMD_SH="${SCRIPT_DIR}/install_amd.sh"
[[ -f "$INSTALL_AMD_SH" ]] || {
    echo "install_amd.sh not found next to ${BASH_SOURCE[0]}" >&2
    exit 1
}

exec bash "$INSTALL_AMD_SH" "$@"