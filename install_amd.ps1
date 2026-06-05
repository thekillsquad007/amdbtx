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
Log "========================================"

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Set HSA_ENABLE_DXG_DETECTION for WSL GPU passthrough
$hsaEnv = [Environment]::GetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "User")
if ($hsaEnv -ne "1") {
    Log "Setting HSA_ENABLE_DXG_DETECTION=1 for WSL GPU passthrough..."
    if ($isAdmin) {
        [Environment]::SetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "1", "Machine")
        Log "Set system-wide (Machine level)"
    } else {
        [Environment]::SetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "1", "User")
        Log "Set for current user (run as admin to set system-wide)"
    }
    $env:HSA_ENABLE_DXG_DETECTION = "1"
    Log "Environment variable set. WSL must be restarted for it to take effect."
}

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
$distros = wsl -l --quiet 2>$null | Where-Object { $_ -match "Ubuntu" }
if ($distros) {
    # Prefer noble (24.04) over jammy (22.04)
    foreach ($d in $distros) {
        $d = $d.Trim()
        if ($d -and $d -ne "") {
            if ($d -match "24\.04|noble") { $wslDistro = $d; break }
            if (-not $wslDistro) { $wslDistro = $d }
        }
    }
}
if (-not $wslDistro) { Err "No Ubuntu WSL distro found. Install: wsl --install -d Ubuntu-22.04" }
$wslDistro = $wslDistro.Trim()
Log "Using WSL distro: $wslDistro"

# Find the install script path (should be next to this .ps1 file)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$installWslSh = Join-Path $scriptDir "install_wsl.sh"

# Build install command
$installArgs = "--address `"$Address`""
if ($Worker) { $installArgs += " --worker `"$Worker`"" }
$installArgs += " --pool `"$Pool`""

if (Test-Path $installWslSh) {
    # Use the WSL install script directly
    Log "Running install_wsl.sh inside WSL..."
    $repoPath = $scriptDir -replace '\\', '/' -replace '^([A-Za-z]):', '/mnt/$1L'
    $repoPath = $repoPath -replace '([A-Z]):', { "/mnt/$($_.Groups[1].Value.ToLower())" }
    $repoPath = $repoPath -replace '\\', '/'

    $wslCmd = "cd '$repoPath' && bash install_wsl.sh $installArgs"
    wsl -d $wslDistro bash -c $wslCmd
} else {
    # Fallback: run inline install
    Log "install_wsl.sh not found, running inline setup..."
    $inlineCmd = @"
set -e
export DEBIAN_FRONTEND=noninteractive
export HSA_ENABLE_DXG_DETECTION=1

# Ensure python3-venv
python3 -m venv --help >/dev/null 2>&1 || sudo apt-get install -y -qq python3-venv 2>/dev/null || true

# Create venv
python3 -m venv ~/.amdbtx-miner/venv
~/.amdbtx-miner/venv/bin/pip install --upgrade pip wheel pyyaml 2>/dev/null || true

# Download assets
mkdir -p ~/.amdbtx-miner/bin
PREBUILDS='https://github.com/thekillsquad007/amdbtx/releases/download/amdbtx-prebuilds-v1.0'
curl -fsSL "`$PREBUILDS/btx-gbt-solve" -o ~/.amdbtx-miner/bin/btx-gbt-solve
chmod +x ~/.amdbtx-miner/bin/btx-gbt-solve
curl -fsSL "`$PREBUILDS/amdbtx_miner-1.0.0-py3-none-any.whl" -o /tmp/amdbtx_miner.whl
~/.amdbtx-miner/venv/bin/pip install --force-reinstall /tmp/amdbtx_miner.whl

# Build runtime
mkdir -p ~/.amdbtx-miner/runtime
ROCM_LIBS=(/opt/rocm/lib)
for d in /opt/rocm-*/lib; do [[ -d "`$d" ]] && ROCM_LIBS+=("`$d"); done
RUNTIME_LD=""
for d in "`${ROCM_LIBS[@]}"; do
    for f in libamdhip64.so libhipblas.so; do
        latest=`$(find "`$d" -maxdepth 1 -name "`$f.*" ! -type l 2>/dev/null | sort -V | tail -1)
        if [[ -n "`$latest" ]]; then
            soname=`$(echo "`$f" | sed 's/\.so$//').so.`$(echo "`$latest" | grep -oP '\.so\.\K[0-9]+' | head -1)
            ln -sfn "`$latest" ~/.amdbtx-miner/runtime/"`$soname"
        fi
    done
    RUNTIME_LD="`$RUNTIME_LD:`$d"
done
RUNTIME_LD="$HOME/.amdbtx-miner/runtime`$RUNTIME_LD"

# GPU detection
GPU_THREADS=8; GPU_WORKERS=16; GPU_BATCH=128; WORKER_NAME="amdgpu-1"
if command -v rocminfo >/dev/null 2>&1; then
    if ROCMINFO_OUT=`$(rocminfo 2>/dev/null); then
        GPU_ARCH=`$(echo "`$ROCMINFO_OUT" | grep -oP 'gfx[0-9a-f]+' | head -1)
        if [[ -n "`$GPU_ARCH" ]]; then
            GPU_NAME=`$(echo "`$ROCMINFO_OUT" | grep -B5 "`$GPU_ARCH" | grep "Name:" | head -1 | sed 's/.*Name:[ \t]*//; s/ (TM)//; s/ (R)//; s/ /-/g')
            WORKER_NAME="`${GPU_NAME:-amdgpu}-1"
        fi
    fi
fi

# Write config
cat > ~/.amdbtx-miner/config.yaml << EOF
pool_host: "$($Pool -split ':')[0]"
pool_port: $($Pool -split ':')[1]
pool_tls: false
payout_address: "$Address"
worker_name: "`$WORKER_NAME"
gbt_solve_path: "`$HOME/.amdbtx-miner/bin/btx-gbt-solve"
solver_backend: "rocm"
solver_threads: `$GPU_THREADS
solver_prepare_workers: `$GPU_WORKERS
solver_batch_size: `$GPU_BATCH
solver_prefetch_depth: 8
solver_pipeline_async: 1
gpu_inputs: 0
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
venv_path: "`$HOME/.amdbtx-miner/venv"
runtime_ld_path: "`$RUNTIME_LD"
EOF

# Environment
grep -q 'amdbtx-miner' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'BASHRC'
export PATH="/opt/rocm/bin:$HOME/.amdbtx-miner/venv/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.amdbtx-miner/runtime:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_ENABLE_DXG_DETECTION=1
BASHRC

echo "WORKER_NAME=`$WORKER_NAME"
echo "RUNTIME_LD=`$RUNTIME_LD"
"@

    $output = wsl -d $wslDistro bash -c $inlineCmd
    Write-Host $output -ForegroundColor Gray
}

# Restart WSL to pick up new environment variables
Log "Restarting WSL to apply GPU passthrough environment..."
wsl --shutdown 2>$null
Start-Sleep -Seconds 3

Log ""
Log "Setup complete!"
echo ""
echo "Launch miner:"
echo "  wsl -d $wslDistro -e amdbtx-miner --config ~/.amdbtx-miner/config.yaml"
echo ""
echo "Or enter WSL and run:"
echo "  wsl -d $wslDistro"
echo "  amdbtx-miner --config ~/.amdbtx-miner/config.yaml"
echo ""
echo "Troubleshoot:"
echo "  Check GPU:   wsl -d $wslDistro -e rocminfo"
echo "  Check logs:  wsl -d $wslDistro -e tail -f ~/.amdbtx-miner/miner.log"
echo "  Check libs:  wsl -d $wslDistro -e ldd ~/.amdbtx-miner/bin/btx-gbt-solve"
