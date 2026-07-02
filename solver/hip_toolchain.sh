#!/usr/bin/env bash
# Shared ROCm/HIP compiler discovery for build.sh and install_amd.sh.
# Prefer /opt/rocm*/bin/hipcc over raw clang++ (mixing /opt/rocm clang with
# /usr/include/hip breaks with undeclared __AMDGCN_WAVEFRONT_SIZE on Ubuntu).

set -euo pipefail

HIPCC=""
ROCM_INCLUDE=""
ROCM_LIB=""
ROCM_ROOT=""
HIP_TOOLCHAIN_OK=0
HIP_TOOLCHAIN_NOTE=""

_hipcc_works() {
    local cc="$1"
    [[ -n "$cc" && -x "$cc" ]] || return 1
    local out=""
    out="$("$cc" --version 2>&1)" || return 1
    [[ "$out" != *"not found"* && "$out" != *"No such file"* ]] || return 1
    return 0
}

_realpath() {
    readlink -f "$1" 2>/dev/null || echo "$1"
}

_set_rocm_root() {
    local root="$1"
    ROCM_ROOT="$root"
    export ROCM_PATH="$root"
    export HIP_PATH="$root"
    export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
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

_set_system_hip_paths() {
    if [[ -d /usr/include/hip ]]; then
        ROCM_INCLUDE="/usr/include"
        if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
            ROCM_LIB="/usr/lib/x86_64-linux-gnu"
        else
            ROCM_LIB="/usr/lib"
        fi
    fi
}

_compiler_is_raw_clang() {
    local cc="$1"
    local base
    base="$(basename "$(_realpath "$cc")")"
    [[ "$base" == "clang++" || "$base" == "clang" ]]
}

_toolchain_pair_is_mixed() {
    # /opt/rocm clang++ + Ubuntu /usr/include/hip headers is a common Bazzite failure mode.
    [[ -d /usr/include/hip ]] || return 1
    [[ -n "$ROCM_ROOT" && -d "$ROCM_ROOT/include/hip" ]] || return 1
    _compiler_is_raw_clang "$HIPCC"
}

_export_hip_compile_env() {
    if [[ -n "$ROCM_ROOT" ]]; then
        export ROCM_PATH="$ROCM_ROOT"
        export HIP_PATH="$ROCM_ROOT"
    fi
    export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
    if [[ -n "${HIP_CLANG_PATH:-}" ]]; then
        export HIP_CLANG_PATH
    fi
    if [[ -n "$ROCM_INCLUDE" ]]; then
        export HIP_INCLUDE_PATH="$ROCM_INCLUDE"
        export CPLUS_INCLUDE_PATH="${ROCM_INCLUDE}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
        export C_INCLUDE_PATH="${ROCM_INCLUDE}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
    fi
    if [[ -n "$ROCM_LIB" ]]; then
        export HIP_LIB_PATH="$ROCM_LIB"
        export LD_LIBRARY_PATH="${ROCM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

_hip_toolchain_smoke() {
    local arch="${AMDBTX_HIP_ARCHS:-gfx1030}"
    arch="${arch%% *}"
    [[ -n "$arch" ]] || arch="gfx1030"

    local tmpdir src err rc
    tmpdir="$(mktemp -d)"
    src="${tmpdir}/smoke.hip"
    err="${tmpdir}/err.log"
    cat > "$src" <<'EOF'
#include <hip/hip_runtime.h>
__global__ void btx_smoke_kernel() {}
int main() {
    int count = 0;
    return hipGetDeviceCount(&count);
}
EOF

    _export_hip_compile_env
    set +e
    if _compiler_is_raw_clang "$HIPCC"; then
        "$HIPCC" -x hip -O0 -std=c++17 \
            -D__HIP_PLATFORM_AMD__ \
            -I"$ROCM_INCLUDE" \
            --offload-arch="$arch" \
            "$src" -L"$ROCM_LIB" -lamdhip64 -o "${tmpdir}/smoke" 2>"$err"
    else
        "$HIPCC" -O0 -std=c++17 \
            --offload-arch="$arch" \
            "$src" -L"$ROCM_LIB" -lamdhip64 -o "${tmpdir}/smoke" 2>"$err"
    fi
    rc=$?
    set -e
    rm -rf "$tmpdir"
    [[ "$rc" -eq 0 ]]
}

_try_toolchain() {
    local cc="$1"
    local note="${2:-}"
    HIPCC=""
    HIP_TOOLCHAIN_OK=0
    HIP_TOOLCHAIN_NOTE=""
    _hipcc_works "$cc" || return 1
    HIPCC="$cc"
    if _toolchain_pair_is_mixed; then
        return 1
    fi
    _hip_toolchain_smoke || return 1
    HIP_TOOLCHAIN_OK=1
    HIP_TOOLCHAIN_NOTE="$note"
    return 0
}

_try_opt_rocm_hipcc() {
    local rocm cc
    for rocm in /opt/rocm /opt/rocm-*/; do
        [[ -d "$rocm" ]] || continue
        _set_rocm_root "$rocm"
        cc="$rocm/bin/hipcc"
        [[ -x "$cc" ]] || continue
        if _try_toolchain "$cc" "opt/rocm hipcc ($rocm)"; then
            return 0
        fi
    done
    return 1
}

_try_system_hipcc() {
    local cc
    _set_system_hip_paths
    for cc in /usr/bin/hipcc /usr/local/bin/hipcc; do
        [[ -x "$cc" ]] || continue
        ROCM_ROOT=""
        unset HIP_CLANG_PATH || true
        if _try_toolchain "$cc" "system hipcc ($cc)"; then
            return 0
        fi
    done
    if command -v hipcc >/dev/null 2>&1; then
        cc="$(command -v hipcc)"
        if _try_toolchain "$cc" "PATH hipcc ($cc)"; then
            return 0
        fi
    fi
    return 1
}

_try_hipconfig() {
    command -v hipconfig >/dev/null 2>&1 || return 1

    local hip_path cc
    for hip_path in \
        "$(hipconfig --hip-path 2>/dev/null | head -n1 | tr -d '\r')" \
        "$(hipconfig --path 2>/dev/null | head -n1 | tr -d '\r')"; do
        [[ -n "$hip_path" && "$hip_path" == /* && -d "$hip_path" ]] || continue
        _set_rocm_root "$hip_path"
        cc="$hip_path/bin/hipcc"
        [[ -x "$cc" ]] || continue
        if _try_toolchain "$cc" "hipconfig hipcc ($cc)"; then
            return 0
        fi
    done
    return 1
}

# Populates HIPCC, ROCM_INCLUDE, ROCM_LIB, HIP_TOOLCHAIN_OK.
resolve_hip_toolchain() {
    HIPCC=""
    ROCM_INCLUDE=""
    ROCM_LIB=""
    ROCM_ROOT=""
    HIP_TOOLCHAIN_OK=0
    HIP_TOOLCHAIN_NOTE=""

    _try_opt_rocm_hipcc && return 0
    _try_hipconfig && return 0
    _try_system_hipcc && return 0
    return 1
}

hip_toolchain_ready() {
    resolve_hip_toolchain
}