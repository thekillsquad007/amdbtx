#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="${ROOT_DIR}/hiveos/amdbtx"
VERSION="1.2.2_hiveos"
OUT_DIR="${ROOT_DIR}/dist"
OUT_FILE="${OUT_DIR}/amdbtx-${VERSION}.tar.gz"

for file in h-config.sh h-run.sh h-stats.sh; do
    [[ -f "${PACKAGE_DIR}/${file}" ]] || {
        echo "missing ${PACKAGE_DIR}/${file}" >&2
        exit 1
    }
done

# Copy portable binaries into the package
for distro in ubuntu22 ubuntu20; do
    src="${ROOT_DIR}/dist/amdbtx-miner-linux-${distro}/amdbtx-miner"
    dst="${PACKAGE_DIR}/amdbtx-miner-linux-${distro}"
    if [[ -f "$src" ]]; then
        cp -a "$src" "$dst"
        chmod 0755 "$dst"
        echo "bundled $dst ($(du -h "$dst" | cut -f1))"
    else
        echo "WARNING: $src not found, skipping" >&2
    fi
done

mkdir -p "$OUT_DIR"
chmod 0755 "${PACKAGE_DIR}/h-config.sh" "${PACKAGE_DIR}/h-run.sh" "${PACKAGE_DIR}/h-stats.sh"
tar -czf "$OUT_FILE" -C "${ROOT_DIR}/hiveos" amdbtx

echo "$OUT_FILE"
