#!/bin/bash
#
# build_linux_portable_bazzite.sh
#
# This script is meant to be run on your Bazzite laptop (or any distrobox-friendly machine)
# that has a working AMD GPU + ROCm.
#
# It will use distrobox to create Ubuntu 22.04 and Ubuntu 20.04 build environments
# so the final amdbtx-miner binary has older glibc (2.35 for 22.04, 2.31 for 20.04).
#
# Usage:
#   1. Copy or git clone this repo to your Bazzite machine.
#   2. cd into the repo
#   3. chmod +x build_linux_portable_bazzite.sh
#   4. ./build_linux_portable_bazzite.sh
#
# Requirements on host:
#   - distrobox (pre-installed on Bazzite)
#   - podman (usually the backend on Bazzite)
#   - Working ROCm on the host (so the containers can see the GPU if needed)
#
# The script will try to build two versions:
#   - Ubuntu 22.04 target (recommended for most people)
#   - Ubuntu 20.04 target (more compatibility, harder with recent ROCm)
#
# Output will be in:
#   dist/amdbtx-miner-linux-ubuntu22/
#   dist/amdbtx-miner-linux-ubuntu20/
#
# After it finishes, you can tar the binaries and upload them to the GitHub release.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$REPO_ROOT/dist"

# Change these if you want different ROCm versions inside the containers.
# For gfx1200 (RX 9070 XT) you generally want ROCm 6.2+ or 7.x.
ROCM_UBUNTU22_VERSION="6.2.4"   # Good balance for Ubuntu 22.04 + RDNA4
ROCM_UBUNTU20_VERSION="6.1.2"   # Last-ish version with decent Ubuntu 20.04 support

# Architectures - include your 9070 XT
HIP_ARCHS="gfx1030;gfx1031;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"

echo "=== amdbtx Linux Portable Builder (Bazzite / distrobox) ==="
echo "Repo: $REPO_ROOT"
echo "Building for glibc targets:"
echo "  - Ubuntu 22.04 (glibc 2.35)"
echo "  - Ubuntu 20.04 (glibc 2.31)"
echo

if ! command -v distrobox >/dev/null 2>&1; then
    echo "ERROR: distrobox not found."
    echo "On Bazzite you can usually install it with:"
    echo "  ujust install distrobox"
    echo "or"
    echo "  flatpak install distrobox"
    exit 1
fi

build_for() {
    local distro_name=$1
    local ubuntu_image=$2
    local rocm_version=$3
    local out_dir_name=$4
    local box_name="amdbtx-build-$distro_name"

    echo
    echo "=========================================="
    echo "Building for $distro_name (Ubuntu base: $ubuntu_image)"
    echo "ROCm version inside container: $rocm_version"
    echo "Output dir: dist/$out_dir_name"
    echo "=========================================="

    # Create the distrobox if it doesn't exist
    if ! distrobox list | grep -q "$box_name"; then
        echo "Creating distrobox container: $box_name"
        distrobox create \
            --image "$ubuntu_image" \
            --name "$box_name" \
            --yes \
            --pull
    else
        echo "Container $box_name already exists, reusing it."
    fi

    # Enter the container and do the build
    # We pass the current directory as /repo inside the container
    distrobox enter "$box_name" -- bash -c "
        set -e
        echo '=== Inside $box_name ==='
        cat /etc/os-release | grep PRETTY_NAME

        # Install basic tools
        sudo apt-get update
        sudo apt-get install -y \
            build-essential cmake git python3 python3-pip \
            wget gnupg lsb-release ca-certificates

        # Add ROCm repo for the chosen version
        echo 'Adding ROCm repository...'
        wget -qO - https://repo.radeon.com/rocm/apt/$rocm_version/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg
        echo \"deb [arch=amd64] https://repo.radeon.com/rocm/apt/$rocm_version/ \$(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/rocm.list

        sudo apt-get update
        sudo apt-get install -y rocm-hip-runtime-dev rocm-hip-runtime

        # Make hipcc available
        export PATH=\"/opt/rocm/bin:\$PATH\"
        export LD_LIBRARY_PATH=\"/opt/rocm/lib:\$LD_LIBRARY_PATH\"

        # Mount point for the source
        cd /repo

        echo '=== Building HIP solver ==='
        cd solver
        rm -rf build-distrobox
        cmake -S . -B build-distrobox -DHIP_ARCHS=\"$HIP_ARCHS\"
        cmake --build build-distrobox -j\$(nproc)

        SOLVER=\"build-distrobox/btx-gbt-solve-hip\"
        if [ ! -x \"\$SOLVER\" ]; then
            echo 'ERROR: Solver build failed!'
            exit 1
        fi
        echo \"Solver built: \$(ls -lh \$SOLVER)\"

        echo '=== Installing PyInstaller ==='
        pip3 install --break-system-packages --user pyyaml pyinstaller

        export PATH=\"\$HOME/.local/bin:\$PATH\"

        echo '=== Running PyInstaller ==='
        cd ..
        rm -rf build dist/$out_dir_name
        mkdir -p dist/$out_dir_name

        python3 -m PyInstaller --clean --onefile --console \
            --name amdbtx-miner \
            --distpath dist/$out_dir_name \
            --add-binary \"\$SOLVER:.\" \
            --hidden-import yaml \
            --collect-submodules amdbtx_miner \
            src/run_miner.py

        echo '=== Build complete inside container ==='
        ls -lh dist/$out_dir_name/amdbtx-miner
    " 2>&1 || echo "Build for $distro_name finished with some output (check above for errors)"
}

# Make sure we are in the repo root
cd "$REPO_ROOT"

# Build Ubuntu 22.04 target
build_for "ubuntu22" "ubuntu:22.04" "$ROCM_UBUNTU22_VERSION" "amdbtx-miner-linux-ubuntu22"

# Build Ubuntu 20.04 target
build_for "ubuntu20" "ubuntu:20.04" "$ROCM_UBUNTU20_VERSION" "amdbtx-miner-linux-ubuntu20"

echo
echo "=== All builds finished ==="
echo "Binaries should be in:"
ls -lh "$DIST_DIR"/amdbtx-miner-linux-*/amdbtx-miner 2>/dev/null || echo "(none found yet - check the output above for errors)"

echo
echo "To package for release:"
echo "  cd dist"
echo "  tar -czf amdbtx-miner-linux-ubuntu22-v1.2.0.tar.gz -C amdbtx-miner-linux-ubuntu22 amdbtx-miner"
echo "  tar -czf amdbtx-miner-linux-ubuntu20-v1.2.0.tar.gz -C amdbtx-miner-linux-ubuntu20 amdbtx-miner"
echo
echo "Then upload the two .tar.gz files to the GitHub release."