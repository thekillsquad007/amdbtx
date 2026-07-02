#!/usr/bin/env bash
# Build the BTX GPU solver on the target system.
# Auto-detects the installed ROCm version and GPU architecture.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "$BUILD_DIR"

# --- Find HIP compiler (prefer /opt/rocm over system hipcc) ---
HIPCC=""
for cand in /opt/rocm /opt/rocm-*; do
    if [[ -f "$cand/bin/hipcc" ]]; then
        HIPCC="$cand/bin/hipcc"
        break
    fi
done
if [[ -z "$HIPCC" ]] && command -v hipcc >/dev/null 2>&1; then
    HIPCC="hipcc"
fi
if [[ -z "$HIPCC" ]]; then
    for cand in /opt/rocm /opt/rocm-*; do
        if [[ -f "$cand/lib/llvm/bin/clang++" ]]; then
            HIPCC="$cand/lib/llvm/bin/clang++"
            break
        fi
    done
fi
if [[ -z "$HIPCC" ]]; then
    echo "Error: HIP compiler (hipcc) not found."
    echo "Install rocm-dev, hip-devel, or hip-dev first."
    exit 1
fi
echo "Using HIP compiler: $HIPCC"

# --- Find ROCm include/lib paths ---
# Use hipconfig by default (guarantees headers match the HIP compiler version).
# Only prefer /opt/rocm when no hipconfig is available.
ROCM_INCLUDE=""
ROCM_LIB=""
if command -v hipconfig >/dev/null 2>&1; then
    for hipconfig_flag in --hip-path --path; do
        HIP_PATH="$(hipconfig "$hipconfig_flag" 2>/dev/null | head -n1 | tr -d '\r' || true)"
        # ROCm 7.x sometimes prints "HIP version: ..." instead of a directory.
        if [[ -n "$HIP_PATH" && "$HIP_PATH" == /* && -d "$HIP_PATH/include/hip" ]]; then
            ROCM_INCLUDE="$HIP_PATH/include"
            ROCM_LIB="$HIP_PATH/lib"
            break
        fi
    done
fi
if [[ -z "$ROCM_INCLUDE" ]]; then
    for cand in /opt/rocm /opt/rocm-*; do
        if [[ -d "$cand/include/hip" ]]; then
            ROCM_INCLUDE="$cand/include"
            ROCM_LIB="$cand/lib"
            break
        fi
    done
fi
if [[ -z "$ROCM_INCLUDE" ]]; then
    # System install (Ubuntu 26.04+)
    if [[ -d "/usr/include/hip" ]]; then
        ROCM_INCLUDE="/usr/include"
        ROCM_LIB="/usr/lib/x86_64-linux-gnu"
    fi
fi
if [[ -z "$ROCM_INCLUDE" ]]; then
    echo "Error: HIP headers not found. Install rocm-dev or hip-dev."
    exit 1
fi
echo "ROCm include: $ROCM_INCLUDE"
echo "ROCm lib: $ROCM_LIB"

# --- Determine GPU architecture (discrete GPUs only; skip iGPU gfx90c) ---
IGPU_ARCH_RE='^gfx90c$'

pick_discrete_archs_from_rocminfo() {
    rocminfo 2>/dev/null | awk '
        function flush() {
            if (dev == "GPU" && arch != "" && arch != "gfx90c") {
                gpus[arch] = 1
            }
        }
        /^(\*\*\*)?[[:space:]]*Agent[[:space:]]+[0-9]+/ { flush(); dev = ""; arch = "" }
        /Device Type:.*GPU/ { dev = "GPU" }
        /Name:[[:space:]]*gfx/ { match($0, /gfx[0-9a-f]+/); arch = substr($0, RSTART, RLENGTH) }
        END { flush(); for (a in gpus) print a }
    ' | sort -u | tr '\n' ' '
}

ARCHS="${AMDBTX_HIP_ARCHS:-}"
if [[ -n "$ARCHS" ]]; then
    echo "Using requested GPU architectures: $ARCHS"
fi
if [[ -z "$ARCHS" ]] && command -v rocminfo >/dev/null 2>&1; then
    ARCHS="$(pick_discrete_archs_from_rocminfo)"
fi
if [[ -z "$ARCHS" ]] && command -v rocm-smi >/dev/null 2>&1; then
    ARCHS=$(rocm-smi --showid 2>/dev/null | grep -oP 'gfx[0-9a-f]{3,}' | grep -v '^gfx90c$' | sort -u | tr '\n' ' ')
fi
if [[ -z "$ARCHS" ]]; then
    ARCHS="gfx803 gfx900 gfx906 gfx908 gfx90a gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201"
    echo "Warning: no discrete GPU arch detected; compiling for common discrete targets"
fi

ARCHS=$(echo "$ARCHS" | tr ' ' '\n' | grep -Ev "$IGPU_ARCH_RE" | sort -u | tr '\n' ' ')
if [[ -z "$ARCHS" ]]; then
    ARCHS="$(pick_discrete_archs_from_rocminfo)"
fi
if [[ -z "$ARCHS" ]]; then
    echo "Error: no discrete GPU architecture found (only iGPU gfx90c present?)"
    exit 1
fi

ARCH_FLAGS=""
for arch in $ARCHS; do
    ARCH_FLAGS="$ARCH_FLAGS --offload-arch=$arch"
done
echo "Target architectures: $ARCHS"

# --- Compile ---
SOURCES=(
    "$SRC_DIR/main.cpp"
    "$SRC_DIR/sha256.cpp"
    "$SRC_DIR/field.cpp"
    "$SRC_DIR/matrix.cpp"
    "$SRC_DIR/noise.cpp"
    "$SRC_DIR/transcript.cpp"
    "$SRC_DIR/solve.cpp"
    "$SRC_DIR/solve_gpu.hip"
    "$SRC_DIR/gpu_sha256.hip"
    "$SRC_DIR/matmul_kernel.hip"
)

echo "Building btx-gbt-solve-hip..."
$HIPCC \
    -x hip \
    -O3 \
    -std=c++17 \
    -mllvm -amdgpu-early-inline-all=true \
    $ARCH_FLAGS \
    -D__HIP_PLATFORM_AMD__ \
    -I"$ROCM_INCLUDE" \
    -I"$SRC_DIR" \
    "${SOURCES[@]}" \
    -L"$ROCM_LIB" \
    -lamdhip64 \
    -lpthread \
    -Wl,-rpath,"$ROCM_LIB" \
    -o "${BUILD_DIR}/btx-gbt-solve-hip"

echo "Build successful: ${BUILD_DIR}/btx-gbt-solve-hip"
