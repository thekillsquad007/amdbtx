# AMD BTX Miner - Windows installer with WSL support
# Usage: .\install_amd.ps1 -Address "btx1..." -Worker "rig1"
#
# Requirements: WSL2 with Ubuntu 22.04+, AMD GPU with WSL support

param(
    [Parameter(Mandatory=$true)]
    [string]$Address,
    [string]$Worker = "",
    [string]$Pool = "stratum.minebtx.com:3333",
    [switch]$SkipWslInstall
)

$ErrorActionPreference = "Stop"

function Log { Write-Host "[amdbtx] $args" -ForegroundColor Cyan }
function Warn { Write-Host "[warn] $args" -ForegroundColor Yellow }
function Err { Write-Host "[error] $args" -ForegroundColor Red; exit 1 }

Log "AMD BTX Miner - Windows/WSL Installer"

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Warn "Not running as admin - WSL install may require elevation" }

# Install WSL if needed
if (-not $SkipWslInstall) {
    $wslList = wsl -l 2>$null
    if ($LASTEXITCODE -ne 0) {
        Log "Installing WSL2..."
        wsl --install -d Ubuntu-22.04
        Log "WSL2 installed. Please restart Windows and re-run this script."
        exit 0
    }
}

# Find WSL distro
$wslDistro = $null
$distros = wsl -l --quiet 2>$null
foreach ($d in $distros) { if ($d -match "Ubuntu") { $wslDistro = $d; break } }
if (-not $wslDistro) { Err "No Ubuntu WSL distro found. Install: wsl --install -d Ubuntu-22.04" }
Log "Using WSL distro: $wslDistro"

# Parse pool
$poolParts = $Pool -split ':'
$poolHost = $poolParts[0]
$poolPort = $poolParts[1]

Log "Setting up WSL environment..."
$installScript = @"
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq curl ca-certificates gnupg 2>/dev/null || true

# Detect Ubuntu version for correct ROCm repo
UBUNTU_CODENAME=\$(lsb_release -cs)
if [[ "\$UBUNTU_CODENAME" == "noble" ]]; then
    ROCM_VERSION="7.2"
else
    ROCM_VERSION="6.0"
fi

# ROCm install (only if not present)
# Check multiple ROCm indicators (rocm-smi may not exist in ROCm 7.x)
HAS_ROCM=false
if command -v rocm-smi >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1 || [[ -f /opt/rocm/share/doc/rocm-core/version ]]; then
    HAS_ROCM=true
fi

if ! \$HAS_ROCM; then
    echo "deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/\$ROCM_VERSION \$UBUNTU_CODENAME main" > /etc/apt/sources.list.d/rocm.list
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq rocm-hip-runtime hipblas hipsolver 2>/dev/null || echo 'ROCm may need manual install'
fi

# Ensure ROCm in PATH
export PATH="/opt/rocm/bin:\$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:\$LD_LIBRARY_PATH"

# Create venv and install Python package
python3 -m venv ~/.amdbtx-miner/venv
~/.amdbtx-miner/venv/bin/pip install --upgrade pip wheel 2>/dev/null || true
~/.amdbtx-miner/venv/bin/pip install pyyaml 2>/dev/null || true

# Enable AMD GPU detection on WSL2
echo 'export PATH=/opt/rocm/bin:\$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
echo 'export PATH="\$HOME/.amdbtx-miner/venv/bin:\$PATH"' >> ~/.bashrc
mkdir -p ~/.amdbtx-miner/bin

PREBUILDS='https://github.com/thekillsquad007/amdbtx/releases/download/amdbtx-prebuilds-v1.0'
curl -fsSL "\$PREBUILDS/btx-gbt-solve" -o ~/.amdbtx-miner/bin/btx-gbt-solve || true
chmod +x ~/.amdbtx-miner/bin/btx-gbt-solve || true
curl -fsSL "\$PREBUILDS/amdbtx_miner-1.0.0-py3-none-any.whl" -o ~/.amdbtx-miner/amdbtx_miner.whl || true
~/.amdbtx-miner/venv/bin/pip install --force-reinstall ~/.amdbtx-miner/amdbtx_miner.whl || true

# Detect GPU for worker name and tuning
GPU_THREADS=8
GPU_WORKERS=16
GPU_BATCH=128
WORKER_NAME=""
GPU_QUERY=""
if command -v rocm-smi >/dev/null 2>&1; then
    GPU_QUERY="rocm-smi"
elif command -v rocminfo >/dev/null 2>&1; then
    GPU_QUERY="rocminfo"
fi
if [[ -n "\$GPU_QUERY" ]]; then
    if [[ "\$GPU_QUERY" == "rocm-smi" ]]; then
        GPU_ARCH=\$(rocm-smi --showid 2>/dev/null | head -1 | grep -oP 'gfx[0-9a-f]+' || true)
        GPU_NAME=\$(rocm-smi --showproductname 2>/dev/null | head -1 | sed 's/.*: //; s/ (TM)//; s/ (R)//; s/ /-/g' || true)
    else
        GPU_ARCH=\$(rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | head -1 || true)
        if [[ -n "\$GPU_ARCH" ]]; then
            GPU_NAME=\$(rocminfo 2>/dev/null | grep -B5 "\$GPU_ARCH" | grep "Name:" | head -1 | sed 's/.*Name:[ \t]*//; s/ (TM)//; s/ (R)//; s/ /-/g' || true)
        fi
    fi
    if [[ "\$GPU_ARCH" == "gfx803" ]]; then
        GPU_THREADS=4; GPU_WORKERS=8; GPU_BATCH=64
    fi
    if [[ -z "\$WORKER_NAME" ]]; then
        WORKER_NAME="\${GPU_NAME:-amdgpu}-1"
    fi
fi

# Write config file
cat > ~/.amdbtx-miner/config.yaml << CFGEOF
pool_host: "$poolHost"
pool_port: $poolPort
pool_tls: false
payout_address: "$Address"
worker_name: "\$WORKER_NAME"
gbt_solve_path: "\$HOME/.amdbtx-miner/bin/btx-gbt-solve"
solver_backend: "rocm"
solver_threads: \$GPU_THREADS
solver_prepare_workers: \$GPU_WORKERS
solver_batch_size: \$GPU_BATCH
solver_prefetch_depth: 8
solver_pipeline_async: 1
gpu_inputs: 0
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
venv_path: "\$HOME/.amdbtx-miner/venv"
CFGEOF

# Output tuning for reading back
echo "GPU_THREADS=\$GPU_THREADS"
echo "GPU_WORKERS=\$GPU_WORKERS"
echo "GPU_BATCH=\$GPU_BATCH"
echo "WORKER_NAME=\$WORKER_NAME"
"@

$output = wsl -d $wslDistro bash -c $installScript
Write-Host $output -ForegroundColor Gray

# Parse GPU tuning from output
$gpuThreads = 8; $gpuWorkers = 16; $gpuBatch = 128; $detectedWorker = ""
foreach ($line in $output -split "`n") {
    if ($line -match '^GPU_THREADS=(.*)') { $gpuThreads = $matches[1] }
    elseif ($line -match '^GPU_WORKERS=(.*)') { $gpuWorkers = $matches[1] }
    elseif ($line -match '^GPU_BATCH=(.*)') { $gpuBatch = $matches[1] }
    elseif ($line -match '^WORKER_NAME=(.*)') { $detectedWorker = $matches[1] }
}

$finalWorker = if ($Worker) { $Worker } elseif ($detectedWorker) { $detectedWorker } else { "amdgpu-1" }

Log ""
Log "Install complete!"
echo ""
echo "Launch miner:"
echo "  wsl -d $wslDistro -e amdbtx-miner"
echo ""
echo "Config: ~/.amdbtx-miner/config.yaml"
echo "Address: $Address"
echo "Worker: $finalWorker"
echo "GPU: threads=$gpuThreads workers=$gpuWorkers batch=$gpuBatch"
echo ""
echo "Troubleshoot:"
echo "  - Check GPU: wsl -d $wslDistro -e rocm-smi"
echo "  - Check logs: wsl -d $wslDistro -e tail -f ~/.amdbtx-miner/miner.log"