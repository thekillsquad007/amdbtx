#!/bin/bash
# Package amdbtx-miner for Linux portable (single-file executable with bundled ROCm runtime)
# Builds the solver with ORIGIN-based RUNPATH so it finds bundled .so files at runtime.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$REPO_ROOT/dist/amdbtx-miner-linux"
ENTRY_POINT="$REPO_ROOT/src/run_miner.py"
SOLVER_BUILD="$REPO_ROOT/solver/build-portable"
ROCM_LIB="/opt/rocm/lib"

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

# Verify RUNPATH was applied correctly
if ! readelf -d "$SOLVER_BIN" | grep -q "ORIGIN"; then
    echo "ERROR: solver RUNPATH does not contain ORIGIN. Mobile reload the source."
    readelf -d "$SOLVER_BIN" | grep -i "RPATH\|RUNPATH"
    exit 1
fi

echo "=== Staging ROCm runtime libs ==="
RUNTIME_DIR="$SOLVER_BUILD/runtime"
mkdir -p "$RUNTIME_DIR"
# Symlinks (used by dlruntime) + the real versioned .so files
for lib in libamdhip64.so libamdhip64.so.7 libhsa-runtime64.so libhsa-runtime64.so.1 librocprofiler-register.so librocprofiler-register.so.0; do
    src="$ROCM_LIB/$lib"
    if [[ -f "$src" ]]; then
        cp -a "$src" "$RUNTIME_DIR/" 2>/dev/null || cp "$src" "$RUNTIME_DIR/" 2>/dev/null || true
    fi
done
# Also copy the unversioned sonames — needed by dlextract
for lib in libamdhip64.so.7.2.70200 libhsa-runtime64.so.1.18.70200 librocprofiler-register.so.0.6.0; do
    src="$ROCM_LIB/$lib"
    if [[ -f "$src" ]]; then
        cp -a "$src" "$RUNTIME_DIR/" 2>/dev/null || true
    fi
done

if ls "$RUNTIME_DIR"/*.so* 1>/dev/null 2>&1; then
    echo "Bundled runtime libs:"
    ls -la "$RUNTIME_DIR/"*.so* 2>/dev/null
else
    echo "WARNING: no .so files copied — runtime dir empty"
fi

echo "=== Installing Python deps ==="
pip3 install --user --break-system-packages pyyaml pyinstaller 2>/dev/null || pip3 install --break-system-packages pyyaml pyinstaller

echo "=== Packaging with PyInstaller ==="
mkdir -p "$OUT_DIR"
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
ls -lh "$EXE"
echo ""
echo "Usage: $EXE --payout-address YOUR_BTX_ADDRESS --pool-host btx-sg.lproute.com --pool-port 8660"
echo "Usage: $EXE --auto-tune   (force batch-size sweep and exit)"
