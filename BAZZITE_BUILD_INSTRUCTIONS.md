# Instructions for building on your Bazzite laptop (with ROCm + 9070 XT)

## 1. Copy the repo to your Bazzite machine

On your Bazzite laptop, clone or copy this repository:

```bash
git clone https://github.com/thekillsquad007/amdbtx.git
cd amdbtx
```

## 2. Make the script executable

```bash
chmod +x build_linux_portable_bazzite.sh
```

## 3. Run the build

```bash
./build_linux_portable_bazzite.sh
```

This script will:

- Use `distrobox` (which is usually already on Bazzite) to create two containers:
  - One based on Ubuntu 22.04 (for glibc 2.35 compatibility)
  - One based on Ubuntu 20.04 (for glibc 2.31 compatibility)
- Inside each container it installs the appropriate ROCm version + build tools.
- It builds the HIP solver (with gfx1200 support for your 9070 XT).
- It runs PyInstaller to create a single-file `amdbtx-miner` binary.
- Output goes to:
  - `dist/amdbtx-miner-linux-ubuntu22/amdbtx-miner`
  - `dist/amdbtx-miner-linux-ubuntu20/amdbtx-miner`

## 4. Package the results

After the script finishes successfully:

```bash
cd dist

# Ubuntu 22.04 version (recommended for most users)
tar -czf amdbtx-miner-linux-ubuntu22-v1.2.0.tar.gz \
    -C amdbtx-miner-linux-ubuntu22 amdbtx-miner

# Ubuntu 20.04 version (maximum compatibility)
tar -czf amdbtx-miner-linux-ubuntu20-v1.2.0.tar.gz \
    -C amdbtx-miner-linux-ubuntu20 amdbtx-miner

ls -lh *.tar.gz
```

## 5. Upload to GitHub release

You can use the GitHub CLI:

```bash
# Make sure you are logged in
gh auth login

# Upload both files to the v1.2.0 release
gh release upload v1.2.0 \
    amdbtx-miner-linux-ubuntu22-v1.2.0.tar.gz \
    amdbtx-miner-linux-ubuntu20-v1.2.0.tar.gz
```

## Important notes

- Building for Ubuntu 20.04 is harder because newer ROCm versions dropped official packages for it. The script uses ROCm 6.1.2 inside the 20.04 container. If it fails, we can drop 20.04 support or try an even older ROCm.
- Your 9070 XT (gfx1200) is included in the architectures.
- The resulting binaries should run on the target Ubuntu versions (and usually newer) **without** the user needing to install ROCm system packages (because we bundle the solver + Python runtime).
- The binaries will still need the ROCm runtime libraries at runtime on the target machine (libamdhip64.so.7 etc.). We can discuss full bundling later if you want zero dependencies.

## If something goes wrong

The script prints a lot of output. Common issues:
- distrobox / podman not installed → install via `ujust install distrobox`
- ROCm repo key / package issues inside the container (especially on 20.04)
- Path problems (the script mounts the repo at `/repo` inside the container)

Let me know the output if it fails and we can adjust.

Once the two tarballs are uploaded, the release will have proper Linux support for both Ubuntu 22.04 and 20.04.