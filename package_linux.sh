#!/bin/bash
# Package amdbtx-miner for Linux (single-file executable)
# Builds solver, then bundles everything with PyInstaller

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$REPO_ROOT/dist/amdbtx-miner-linux"
ENTRY_POINT="$REPO_ROOT/src/run_miner.py"

echo "=== Building solver ==="
cd "$REPO_ROOT/solver"
rm -rf build-linux
cmake -S . -B build-linux -DHIP_ARCHS=gfx1101
cmake --build build-linux -j"$(nproc)"
SOLVER_BIN="$REPO_ROOT/solver/build-linux/btx-gbt-solve-hip"
if [[ ! -x "$SOLVER_BIN" ]]; then
    echo "ERROR: solver binary not found or not executable at $SOLVER_BIN"
    exit 1
fi

echo "=== Installing Python deps ==="
pip3 install --user pyyaml pyinstaller 2>/dev/null || pip3 install pyyaml pyinstaller

echo "=== Packaging with PyInstaller ==="
mkdir -p "$OUT_DIR"
rm -rf "$REPO_ROOT/build" "$REPO_ROOT/*.spec" 2>/dev/null || true

# Use --add-binary so the solver keeps +x
python3 -m PyInstaller --clean --onefile --console \
    --name "amdbtx-miner" \
    --distpath "$OUT_DIR" \
    --add-binary "$SOLVER_BIN:." \
    --hidden-import yaml \
    --collect-submodules amdbtx_miner \
    "$ENTRY_POINT"

echo "=== Done ==="
EXE="$OUT_DIR/amdbtx-miner"
ls -lh "$EXE"
echo ""
echo "Usage: $EXE --payout-address YOUR_BTX_ADDRESS"
echo "Usage: $EXE --payout-address YOUR_BTX_ADDRESS --auto-tune"