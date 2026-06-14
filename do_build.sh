#!/usr/bin/env bash
# Clean build and release upload for amdbtx.
# Run this OUTSIDE distrobox. It will:
#   1. Rebuild the Python wheel (local)
#   2. Enter distrobox to rebuild the solver
#   3. Upload both assets to GitHub release
#   4. Revoke old PAT and create new one (prompts you)
set -euo pipefail

RELEASE_TAG="${RELEASE_TAG:-amdbtx-prebuilds-v1.1.1}"
REPO="thekillsquad007/amdbtx"
SRC_DIR="/var/home/bazzite/amdbtx-private-src"
SOLVER_SRC="/var/home/bazzite/amdbtx-private-solver"
DISTROBOX="${DISTROBOX:-ubuntu-rocm}"

# ── 1. Python wheel ────────────────────────────────────────────────
echo "=== Step 1: Clean build Python wheel ==="
rm -rf "${SRC_DIR}/build" "${SRC_DIR}/dist" "${SRC_DIR}"/*.egg-info
python3 -m build --wheel "${SRC_DIR}"
WHEEL_PATH=$(ls "${SRC_DIR}"/dist/amdbtx_miner-*.whl)
echo "Wheel: ${WHEEL_PATH}"

# ── 2. Solver binary (via distrobox) ───────────────────────────────
echo "=== Step 2: Build HIP solver in distrobox ==="
AMDBTX_HIP_ARCHS="${AMDBTX_HIP_ARCHS:-gfx900 gfx906 gfx1030 gfx1100 gfx1101}"
distrobox enter "${DISTROBOX}" -- env AMDBTX_HIP_ARCHS="${AMDBTX_HIP_ARCHS}" \
    bash /var/home/bazzite/amdbtx/build_solver.sh
SOLVER_BINARY="${SOLVER_SRC}/build/btx-gbt-solve-hip"
if [ ! -f "${SOLVER_BINARY}" ]; then
    echo "ERROR: solver binary not found at ${SOLVER_BINARY}"
    exit 1
fi
echo "Solver: ${SOLVER_BINARY}"

# ── 3. Upload to GitHub release ────────────────────────────────────
echo "=== Step 3: Upload assets to GitHub release ==="
# Prompt for PAT
read -rsp "Enter GitHub PAT (classic, with public_repo scope): " GH_TOKEN
echo

# Delete old assets
echo "Fetching existing assets..."
RELEASE_ID=$(curl -fsSL \
    -H "Authorization: token ${GH_TOKEN}" \
    "https://api.github.com/repos/${REPO}/releases/tags/${RELEASE_TAG}" |
    jq -r '.id')
[[ "$RELEASE_ID" =~ ^[0-9]+$ ]] || {
    echo "ERROR: release ${RELEASE_TAG} was not found"
    exit 1
}
ASSETS=$(curl -s -H "Authorization: token ${GH_TOKEN}" "https://api.github.com/repos/${REPO}/releases/${RELEASE_ID}/assets")
echo "${ASSETS}" | jq -r '
    .[]
    | select(.name == "btx-gbt-solve-hip" or (.name | test("^amdbtx_miner-.*\\.whl$")))
    | "\(.id)\t\(.name)"
' | while IFS=$'\t' read -r asset_id name; do
    echo "Deleting old ${name} (asset ${asset_id})..."
    curl -s -X DELETE -H "Authorization: token ${GH_TOKEN}" \
        "https://api.github.com/repos/${REPO}/releases/assets/${asset_id}"
done

# Upload wheel
echo "Uploading wheel..."
curl -s -X POST \
    -H "Authorization: token ${GH_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    "https://uploads.github.com/repos/${REPO}/releases/${RELEASE_ID}/assets?name=$(basename ${WHEEL_PATH})" \
    --data-binary @"${WHEEL_PATH}" | jq '{state: .state, name: .name, size: .size}'

# Upload solver
echo "Uploading solver..."
curl -s -X POST \
    -H "Authorization: token ${GH_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    "https://uploads.github.com/repos/${REPO}/releases/${RELEASE_ID}/assets?name=btx-gbt-solve-hip" \
    --data-binary @"${SOLVER_BINARY}" | jq '{state: .state, name: .name, size: .size}'

echo ""
echo "=== DONE ==="
echo "Revoke the old PAT at: https://github.com/settings/tokens"
echo "New PAT used above is still visible in shell history — clear it."
