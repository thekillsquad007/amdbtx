#!/bin/bash
#
# build_linux_portable.sh
#
# Run this on your Bazzite (or other Linux) machine that has ROCm working.
#
# It uses distrobox to create Ubuntu 22.04 and Ubuntu 20.04 build environments
# so the resulting amdbtx-miner binary is compatible with those older glibc versions.
#
# Requirements on the host:
#   - distrobox (usually preinstalled on Bazzite)
#   - podman or docker
#   - Working ROCm on the host (we pass the GPU through)
#
# Usage:
#   ./build_linux_portable.sh
#
# This will produce:
#   dist/amdbtx-miner-linux-ubuntu22/amdbtx-miner
#   dist/amdbtx-miner-linux-ubuntu20/amdbtx-miner
#
# Then you can tar them and upload to the release.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$REPO_ROOT/dist"

# Architectures to build for (add/remove as needed)
HIP_ARCHS="gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"

echo "=== amdbtx Linux portable builder (via distrobox) ==="
echo "Host ROCm will be passed through to the containers."
echo "Target glibc versions:"
echo "  - Ubuntu 22.04 (glibc 2.35)"
echo "  - Ubuntu 20.04 (glibc 2.31)"
echo

command -v distrobox >/dev/null 2>&1 || {
    echo "ERROR: distrobox not found. On Bazzite it should be available."
    echo "Install with: ujust install distrobox   or   flatpak install distrobox"
    exit 1
}

build_in_distrobox() {
    local name=$1
    local image=$2
    local out_name=$3

    echo
    echo "=== Setting up $name ($image) ==="

    if ! distrobox list | grep -q "$name"; then
        echo "Creating distrobox '$name'..."
        distrobox create --image "$image" --name "$name" --yes
    else
        echo "Using existing distrobox '$name'"
    fi

    echo "Entering $name to install build dependencies and build..."

    # We run a big heredoc inside the container
    distrobox enter "$name" -- bash -c '
        set -e
        echo "Inside container: $(cat /etc/os-release | grep PRETTY_NAME)"

        # Update and install basic build tools
        sudo apt-get update
        sudo apt-get install -y \
            build-essential cmake git python3 python3-pip \
            wget curl ca-certificates

        # Install ROCm for this Ubuntu version
        # Using ROCm 6.2.4 as a reasonably recent version that supports RDNA4 (gfx1200)
        # You can change the version if needed.
        echo "Installing ROCm..."
        wget -qO - https://repo.radeon.com/rocm/apt/6.2.4/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2.4/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rocm.list
        sudo apt-get update
        sudo apt-get install -y rocm-hip-runtime-dev rocm-hip-runtime

        # Make sure hipcc is in PATH
        export PATH="/opt/rocm/bin:$PATH"

        # Go to the mounted repo
        cd /mnt/e/Business/amdbtx   # This path may need adjustment on your machine!
                                    # On Bazzite the Windows drive may appear differently.
                                    # Common locations: /run/media or just use the full path you have.

        # If the path above does not exist, the script will fail here.
        # In that case, change the cd line or run with the repo copied inside the container.

        echo "Building solver..."
        cd solver
        rm -rf build-db
        cmake -S . -B build-db -DHIP_ARCHS="'"$HIP_ARCHS"'"
        cmake --build build-db -j"$(nproc)"

        SOLVER_BIN="build-db/btx-gbt-solve-hip"
        if [ ! -x "$SOLVER_BIN" ]; then
            echo "ERROR: solver build failed"
            exit 1
        fi

        echo "Installing PyInstaller..."
        pip3 install --break-system-packages --user pyyaml pyinstaller

        export PATH="$HOME/.local/bin:$PATH"

        echo "Running PyInstaller..."
        cd ..
        rm -rf build dist/amdbtx-miner-linux-"'"$out_name"'"
        mkdir -p dist/amdbtx-miner-linux-"'"$out_name"'"

        python3 -m PyInstaller --clean --onefile --console \
            --name amdbtx-miner \
            --distpath dist/amdbtx-miner-linux-"'"$out_name"'" \
            --add-binary "$SOLVER_BIN:." \
            --hidden-import yaml \
            --collect-submodules amdbtx_miner \
            src/run_miner.py

        echo "Build finished inside container."
        ls -lh dist/amdbtx-miner-linux-"'"$out_name"'"/amdbtx-miner
    ' 2>&1 || echo "Build for $name finished (or failed - check output above)"
}

# On Bazzite the repo is probably at a different location.
# We will try to auto-detect a reasonable path or let the user override.
REPO_MOUNT="/mnt/e/Business/amdbtx"
if [ ! -d "$REPO_MOUNT" ]; then
    echo "WARNING: $REPO_MOUNT does not exist on this machine."
    echo "You will need to either:"
    echo "  1. Copy the repo into the container, or"
    echo "  2. Edit this script and change REPO_MOUNT"
    echo
    echo "For now we will continue. The build will fail inside the container until you fix the path."
fi

# Build for Ubuntu 22.04
build_in_distrobox "amdbtx-ubuntu22" "ubuntu:22.04" "ubuntu22"

# Build for Ubuntu 20.04
build_in_distrobox "amdbtx-ubuntu20" "ubuntu:20.04" "ubuntu20"

echo
echo "=== All builds attempted ==="
echo "Check the following locations:"
ls -lh "$REPO_ROOT/dist"/amdbtx-miner-linux-*/amdbtx-miner 2>/dev/null || echo "No binaries found yet."

echo
echo "When successful, you can package them like this:"
echo "  cd dist"
echo "  tar -czf amdbtx-miner-linux-ubuntu22-v1.2.0.tar.gz -C amdbtx-miner-linux-ubuntu22 amdbtx-miner"
echo "  tar -czf amdbtx-miner-linux-ubuntu20-v1.2.0.tar.gz -C amdbtx-miner-linux-ubuntu20 amdbtx-miner"
echo
echo "Then upload both tarballs to the GitHub release."