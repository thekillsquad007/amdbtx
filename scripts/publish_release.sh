#!/usr/bin/env bash
# Build and publish AMDBTX prebuilds to GitHub Releases.
# Requires: GH_TOKEN (classic PAT with repo scope) or `gh auth login`.
#
# Usage:
#   GH_TOKEN=ghp_... ./scripts/publish_release.sh
#   ./scripts/publish_release.sh --tag amdbtx-prebuilds-v1.1.9
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${RELEASE_TAG:-amdbtx-prebuilds-v1.1.9}"
REPO="${AMDBTX_RELEASE_REPO:-thekillsquad007/amdbtx-releases}"
ARCHS="${AMDBTX_HIP_ARCHS:-gfx900 gfx906 gfx1030 gfx1100 gfx1101 gfx1200 gfx1201}"
MINER_VERSION="$(python3 -c "import tomllib; print(tomllib.load(open('$ROOT/pyproject.toml','rb'))['project']['version'])")"
WHEEL_NAME="amdbtx_miner-${MINER_VERSION}-py3-none-any.whl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

GH="${GH_BIN:-gh}"
if ! command -v "$GH" >/dev/null 2>&1; then
  GH="/tmp/gh-install/gh_2.65.0_linux_amd64/bin/gh"
fi

if [[ -z "${GH_TOKEN:-}" ]] && ! "$GH" auth status >/dev/null 2>&1; then
  echo "ERROR: set GH_TOKEN or run: gh auth login"
  exit 1
fi
api() {
  if [[ -n "${GH_TOKEN:-}" ]]; then
    curl -fsSL -H "Authorization: token ${GH_TOKEN}" "$@"
  else
    "$GH" api "$@"
  fi
}

echo "=== Build wheel ==="
rm -rf "$ROOT/dist" "$ROOT/build" "$ROOT/src"/*.egg-info
python3 -m pip install -q build
python3 -m build --wheel "$ROOT"
WHEEL_PATH="$ROOT/dist/$WHEEL_NAME"
[[ -f "$WHEEL_PATH" ]] || { echo "missing $WHEEL_PATH"; exit 1; }

echo "=== Build HIP solver (archs: $ARCHS) ==="
AMDBTX_HIP_ARCHS="$ARCHS" bash "$ROOT/build_solver.sh"
SOLVER_PATH="$ROOT/solver/build/btx-gbt-solve-hip"
[[ -f "$SOLVER_PATH" ]] || { echo "missing $SOLVER_PATH"; exit 1; }

WHEEL_SHA="$(sha256sum "$WHEEL_PATH" | awk '{print $1}')"
SOLVER_SHA="$(sha256sum "$SOLVER_PATH" | awk '{print $1}')"
echo "wheel sha256: $WHEEL_SHA"
echo "solver sha256: $SOLVER_SHA"

RELEASE_BODY="$(cat <<EOF
## AMDBTX v${MINER_VERSION} — HIP scan-batch performance

### Highlights
- Fix HIP scan batch clamp: honor \`BTX_MATMUL_MAX_SCAN_BATCH\` (was hard-capped at 131072).
- Default \`solver_batch_size: 4194304\` for RX 7800 XT class GPUs (~2.7× scan throughput vs 131k clamp).
- \`amdbtx-miner --benchmark\` sweeps GPU-sized batches (131k–16M).
- Experimental \`BTX_MATMUL_FAST_V3_SCAN=1\` (opt-in; ~2% on RDNA3 in profiling).

### RX 7800 XT sweep (bits=1c4916ad)
| Scan batch | MN/s |
|---:|---:|
| 131072 | ~97 |
| 524288 | ~173 |
| 4194304 | ~260 |
| 8388608 | ~254 |

### Checksums
- \`${WHEEL_NAME}\`: \`${WHEEL_SHA}\`
- \`btx-gbt-solve-hip\`: \`${SOLVER_SHA}\`
EOF
)"

echo "=== Ensure GitHub release $TAG on $REPO ==="
if [[ -n "${GH_TOKEN:-}" ]]; then
  if ! api "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" >/dev/null 2>&1; then
    api -X POST "https://api.github.com/repos/${REPO}/releases" \
      -H "Content-Type: application/json" \
      -d "$(jq -n --arg tag "$TAG" --arg name "AMDBTX v${MINER_VERSION} HIP Scan-Batch Perf" --arg body "$RELEASE_BODY" \
        '{tag_name:$tag,name:$name,body:$body,draft:false,prerelease:false}')"
  else
    RELEASE_ID="$(api "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" | jq -r '.id')"
    api -X PATCH "https://api.github.com/repos/${REPO}/releases/${RELEASE_ID}" \
      -H "Content-Type: application/json" \
      -d "$(jq -n --arg name "AMDBTX v${MINER_VERSION} HIP Scan-Batch Perf" --arg body "$RELEASE_BODY" \
        '{name:$name,body:$body,draft:false,prerelease:false}')"
  fi
  RELEASE_ID="$(api "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" | jq -r '.id')"
else
  if ! "$GH" release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
    "$GH" release create "$TAG" -R "$REPO" --title "AMDBTX v${MINER_VERSION} HIP Scan-Batch Perf" --notes "$RELEASE_BODY"
  else
    "$GH" release edit "$TAG" -R "$REPO" --title "AMDBTX v${MINER_VERSION} HIP Scan-Batch Perf" --notes "$RELEASE_BODY"
  fi
  RELEASE_ID="$("$GH" release view "$TAG" -R "$REPO" --json id -q .id)"
fi

echo "=== Upload / replace release assets ==="
delete_asset() {
  local name="$1"
  if [[ -n "${GH_TOKEN:-}" ]]; then
    api "https://api.github.com/repos/${REPO}/releases/${RELEASE_ID}/assets" \
      | jq -r --arg n "$name" '.[] | select(.name==$n) | .id' \
      | while read -r aid; do
          [[ -n "$aid" ]] || continue
          echo "Deleting old $name (asset $aid)"
          api -X DELETE "https://api.github.com/repos/${REPO}/releases/assets/${aid}" >/dev/null
        done
  else
    "$GH" release view "$TAG" -R "$REPO" --json assets -q '.assets[] | select(.name=="'"$name"'") | .id' \
      | while read -r aid; do
          [[ -n "$aid" ]] || continue
          echo "Deleting old $name (asset $aid)"
          "$GH" api -X DELETE "repos/${REPO}/releases/assets/${aid}" >/dev/null
        done
  fi
}

upload_asset() {
  local file="$1"
  local name="$2"
  delete_asset "$name"
  echo "Uploading $name ..."
  if [[ -n "${GH_TOKEN:-}" ]]; then
    curl -fsSL -X POST \
      -H "Authorization: token ${GH_TOKEN}" \
      -H "Content-Type: application/octet-stream" \
      "https://uploads.github.com/repos/${REPO}/releases/${RELEASE_ID}/assets?name=${name}" \
      --data-binary @"$file" | jq '{state,name,size}'
  else
    "$GH" release upload "$TAG" -R "$REPO" "$file#${name}"
  fi
}

upload_asset "$WHEEL_PATH" "$WHEEL_NAME"
upload_asset "$SOLVER_PATH" "btx-gbt-solve-hip"

cat <<EOF

=== DONE ===
Release: https://github.com/${REPO}/releases/tag/${TAG}
Update install_amd.sh pins:
  PREBUILDS_TAG=${TAG}
  WHEEL_FILENAME=${WHEEL_NAME}
  EXPECTED_MINER_VERSION=${MINER_VERSION}
  EXPECTED_WHEEL_SHA256=${WHEEL_SHA}
  EXPECTED_SOLVER_VERSION=2.2.0
  EXPECTED_SOLVER_SHA256=${SOLVER_SHA}
EOF