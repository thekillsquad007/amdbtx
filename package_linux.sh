#!/bin/bash
# Package amdbtx-miner for Linux portable (single-file executable with bundled ROCm runtime)
# Builds the solver with ORIGIN-based RUNPATH so it finds bundled .so files at runtime.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$REPO_ROOT/dist/amdbtx-miner-linux"
ENTRY_POINT="$REPO_ROOT/src/run_miner.py"
SOLVER_BUILD="$REPO_ROOT/solver/build-portable"

# === Find ROCm ===
# Look in /opt/rocm-* first (newer distros), then /opt/rocm, then anywhere under /opt
ROCM_LIB=""
for cand in /opt/rocm-*/lib /opt/rocm/lib; do
    if [[ -d "$cand" ]] && [[ -f "$cand/libamdhip64.so" ]]; then
        ROCM_LIB="$cand"
        break
    fi
done
if [[ -z "$ROCM_LIB" ]]; then
    ROCM_LIB="$(find /opt -maxdepth 3 -name 'libamdhip64.so' -path '*/lib/*' 2>/dev/null | head -1 | xargs -r dirname)"
fi
if [[ -z "$ROCM_LIB" ]] || [[ ! -d "$ROCM_LIB" ]]; then
    echo "ERROR: ROCm not found. Install ROCm (amdgpu + rocm-runtime) or set ROCM_LIB_PATH."
    exit 1
fi
echo "ROCm lib: $ROCM_LIB"

echo "=== Building solver (with ORIGIN RUNPATH) ==="
cd "$REPO_ROOT/solver"
rm -rf "$SOLVER_BUILD"
cmake -S . -B "$SOLVER_BUILD" \
    -DHIP_ARCHS="gfx900;gfx906;gfx1030;gfx1031;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
cmake --build "$SOLVER_BUILD" -j"$(nproc)"

SOLVER_BIN="$SOLVER_BUILD/btx-gbt-solve-hip"
if [[ ! -x "$SOLVER_BIN" ]]; then
    echo "ERROR: solver binary not found at $SOLVER_BIN"
    exit 1
fi

# Verify RUNPATH was applied correctly (works on both DT_RUNPATH and DT_RPATH)
if ! readelf -d "$SOLVER_BIN" | grep -qE '(RUNPATH|RPATH)' ; then
    echo "ERROR: solver has no RUNPATH/RPATH set"
    readelf -d "$SOLVER_BIN"
    exit 1
fi
if ! readelf -d "$SOLVER_BIN" | grep -Eq '(RUNPATH|RPATH).*\$ORIGIN' ; then
    echo "WARNING: solver RUNPATH does not contain \$ORIGIN, bundled .so may not be found"
fi
echo "RUNPATH: $(readelf -d "$SOLVER_BIN" | grep -E '(RUNPATH|RPATH)')"

echo "=== Staging ROCm runtime libraries ==="
RUNTIME_DIR="$SOLVER_BUILD/runtime"
mkdir -p "$RUNTIME_DIR"

# Copy the canonical SONAMEs (dlopen via bare 'libamdhip64.so' needs these)
# and the versioned .so files (dynamic linker uses these)
declare -a SONAMES=(
    "libamdhip64.so"
    "libamdhip64.so.7"
    "libhsa-runtime64.so"
    "libhsa-runtime64.so.1"
    "librocprofiler-register.so"
    "librocprofiler-register.so.0"
)
for lib in "${SONAMES[@]}"; do
    src="$ROCM_LIB/$lib"
    [[ -f "$src" ]] || continue
    cp -a "$src" "$RUNTIME_DIR/" 2>/dev/null || cp "$src" "$RUNTIME_DIR/" 2>/dev/null || true
done

# Copy the rest of the matching unversioned artifacts (libamdhip64.so.7.2.x and similar)
for lib in "$ROCM_LIB"/*.so*; do
    [[ -f "$lib" ]] || continue
    base="$(basename "$lib")"
    case "$base" in
        libamdhip64*|libhsa-runtime64*|librocprofiler-register*)
            cp -a "$lib" "$RUNTIME_DIR/"
            ;;
    esac
done

if ls "$RUNTIME_DIR"/*.so* 1>/dev/null 2>&1; then
    echo "Bundled runtime ($(ls "$RUNTIME_DIR"/*.so* | wc -l) files):"
    ls "$RUNTIME_DIR/" | sed 's/^/  /'
else
    echo "WARNING: no .so files copied — runtime dir empty"
    exit 1
fi

# Bundle amdgpu.ids if available (needed by libdrm_amdgpu on HiveOS)
AMDGPU_IDS=""
for cand in /usr/share/libdrm/amdgpu.ids /usr/share/amdgpu.ids /opt/rocm*/share/libdrm/amdgpu.ids; do
    if [[ -f "$cand" ]]; then
        AMDGPU_IDS="$cand"
        break
    fi
done
if [[ -n "$AMDGPU_IDS" ]]; then
    cp -a "$AMDGPU_IDS" "$RUNTIME_DIR/"
    echo "Bundled amdgpu.ids from $AMDGPU_IDS"
fi

echo "=== Installing Python deps ==="
pip3 install --user --break-system-packages pyyaml pyinstaller 2>/dev/null \
    || pip3 install --break-system-packages pyyaml pyinstaller \
    || python3 -m pip install --user pyyaml pyinstaller 2>/dev/null \
    || python3 -m pip install pyyaml pyinstaller
if ! command -v pyinstaller >/dev/null && ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "ERROR: pyinstaller not available; cannot package"
    exit 1
fi

echo "=== Packaging with PyInstaller ==="
mkdir -p "$(dirname "$OUT_DIR")"
rm -rf "$REPO_ROOT/build" "$REPO_ROOT"/*.spec 2>/dev/null || true

python3 -m PyInstaller --clean --onefile --console \
    --name "amdbtx-miner" \
    --distpath "$OUT_DIR" \
    --add-binary "$SOLVER_BIN:." \
    --add-binary "$RUNTIME_DIR:runtime" \
    --hidden-import yaml \
    --collect-submodules amdbtx_miner \
    "$ENTRY_POINT"

echo "=== Done ==="
EXE="$OUT_DIR/amdbtx-miner"
if [[ ! -x "$EXE" ]]; then
    echo "ERROR: $EXE not produced"
    exit 1
fi
ls -lh "$EXE"
echo ""
echo "Smoke test: $EXE --version (or --help)"
echo ""
echo "Usage: $EXE --payout-address YOUR_BTX_ADDRESS"
echo "       $EXE --auto-tune   (force batch-size sweep and exit)"
