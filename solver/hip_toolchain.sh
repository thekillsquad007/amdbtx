#!/usr/bin/env bash
# Shared ROCm/HIP compiler discovery for build.sh and install_amd.sh.
# Prefer a working /opt/rocm*/hipcc over broken /usr/bin/hipcc wrappers.

set -euo pipefail

HIPCC=""
ROCM_INCLUDE=""
ROCM_LIB=""
ROCM_ROOT=""
HIP_TOOLCHAIN_OK=0

_hipcc_works() {
    local cc="$1"
    [[ -n "$cc" && -x "$cc" ]] || return 1
    local out=""
    out="$("$cc" --version 2>&1)" || return 1
    [[ "$out" != *"not found"* && "$out" != *"No such file"* ]] || return 1
    return 0
}

_set_rocm_root() {
    local root="$1"
    ROCM_ROOT="$root"
    export ROCM_PATH="$root"
    export HIP_PATH="$root"
    if [[ -d "$root/lib/llvm/bin" ]]; then
        export HIP_CLANG_PATH="$root/lib/llvm/bin"
    elif [[ -d "$root/llvm/bin" ]]; then
        export HIP_CLANG_PATH="$root/llvm/bin"
    fi
    if [[ -d "$root/include/hip" ]]; then
        ROCM_INCLUDE="$root/include"
        ROCM_LIB="$root/lib"
    fi
}

_try_hipcc_at() {
    local cc="$1"
    if _hipcc_works "$cc"; then
        HIPCC="$cc"
        HIP_TOOLCHAIN_OK=1
        return 0
    fi
    return 1
}

_resolve_from_hipconfig() {
    command -v hipconfig >/dev/null 2>&1 || return 1

    local hip_path clang_path
    for hip_path in \
        "$(hipconfig --hip-path 2>/dev/null | head -n1 | tr -d '\r')" \
        "$(hipconfig --path 2>/dev/null | head -n1 | tr -d '\r')"; do
        [[ -n "$hip_path" && "$hip_path" == /* && -d "$hip_path" ]] || continue
        _set_rocm_root "$hip_path"
        if [[ -x "$hip_path/bin/hipcc" ]] && _try_hipcc_at "$hip_path/bin/hipcc"; then
            return 0
        fi
        if [[ -x "$hip_path/lib/llvm/bin/clang++" ]] && _try_hipcc_at "$hip_path/lib/llvm/bin/clang++"; then
            return 0
        fi
    done

    clang_path="$(hipconfig 2>/dev/null | awk -F: '/HIP_CLANG_PATH/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
    if [[ -n "$clang_path" && -x "$clang_path/clang++" ]]; then
        _try_hipcc_at "$clang_path/clang++" && return 0
    fi
    return 1
}

_resolve_from_opt_rocm() {
    local rocm
    for rocm in /opt/rocm /opt/rocm-*/; do
        [[ -d "$rocm" ]] || continue
        _set_rocm_root "$rocm"
        if [[ -x "$rocm/bin/hipcc" ]] && _try_hipcc_at "$rocm/bin/hipcc"; then
            return 0
        fi
        if [[ -x "$rocm/lib/llvm/bin/clang++" ]] && _try_hipcc_at "$rocm/lib/llvm/bin/clang++"; then
            return 0
        fi
    done
    return 1
}

_resolve_from_system_paths() {
    if [[ -d /usr/include/hip ]]; then
        ROCM_INCLUDE="/usr/include"
        if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
            ROCM_LIB="/usr/lib/x86_64-linux-gnu"
        else
            ROCM_LIB="/usr/lib"
        fi
    fi

    local rocm
    for rocm in /opt/rocm /opt/rocm-*/; do
        [[ -d "$rocm/include/hip" ]] || continue
        _set_rocm_root "$rocm"
        break
    done

    local cc
    for cc in \
        /opt/rocm/bin/hipcc \
        /opt/rocm-*/bin/hipcc \
        /opt/rocm/lib/llvm/bin/clang++ \
        /opt/rocm-*/lib/llvm/bin/clang++; do
        [[ -e "$cc" ]] || continue
        if _try_hipcc_at "$cc"; then
            return 0
        fi
    done

    if command -v hipcc >/dev/null 2>&1; then
        _try_hipcc_at "$(command -v hipcc)" && return 0
    fi
    return 1
}

# Populates HIPCC, ROCM_INCLUDE, ROCM_LIB, HIP_TOOLCHAIN_OK.
resolve_hip_toolchain() {
    HIPCC=""
    ROCM_INCLUDE=""
    ROCM_LIB=""
    ROCM_ROOT=""
    HIP_TOOLCHAIN_OK=0

    _resolve_from_opt_rocm && return 0
    _resolve_from_hipconfig && return 0
    _resolve_from_system_paths && return 0
    return 1
}

hip_toolchain_ready() {
    resolve_hip_toolchain
}