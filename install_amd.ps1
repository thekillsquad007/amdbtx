# AMD BTX Miner - Windows installer with WSL support
# Usage: .\install_amd.ps1 -Address "btx1..." -Worker "rig1"

param(
    [Parameter(Mandatory=$true)]
    [string]$Address,
    [string]$Worker = "",
    [string]$Pool = "btx-sg.lproute.com:8660",
    [switch]$SkipWslInstall
)

$ErrorActionPreference = "Stop"

function Log { param([string]$Message) Write-Host "[amdbtx] $Message" -ForegroundColor Cyan }
function Warn { param([string]$Message) Write-Host "[warn] $Message" -ForegroundColor Yellow }
function Err { param([string]$Message) Write-Host "[error] $Message" -ForegroundColor Red; exit 1 }

function ShellQuoteForWsl {
    param([string]$Value)
    return "'" + ($Value -replace "'", "'\''") + "'"
}

Log "AMD BTX Miner - Windows/WSL Installer"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$hsaEnv = [Environment]::GetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "User")
if ($hsaEnv -ne "1") {
    Log "Setting HSA_ENABLE_DXG_DETECTION=1 for WSL GPU passthrough"
    if ($isAdmin) {
        [Environment]::SetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "1", "Machine")
        Log "Set HSA_ENABLE_DXG_DETECTION at Machine level"
    } else {
        [Environment]::SetEnvironmentVariable("HSA_ENABLE_DXG_DETECTION", "1", "User")
        Log "Set HSA_ENABLE_DXG_DETECTION at User level"
    }
    $env:HSA_ENABLE_DXG_DETECTION = "1"
}

if (-not $SkipWslInstall) {
    $null = wsl -l 2>$null
    if ($LASTEXITCODE -ne 0) {
        Log "Installing WSL2 with Ubuntu"
        wsl --install -d Ubuntu
        Log "WSL2 install started. Restart Windows if prompted, finish Ubuntu setup, then rerun this installer."
        exit 0
    }
}

$distros = @(
    wsl -l --quiet 2>$null |
        ForEach-Object { ($_ -replace "`0", "").Trim() } |
        Where-Object { $_ -match "Ubuntu" }
)

$wslDistro = $null
foreach ($pattern in @("26\.04", "24\.04|noble", "22\.04|jammy", "^Ubuntu$", "Ubuntu")) {
    $match = $distros | Where-Object { $_ -match $pattern } | Select-Object -First 1
    if ($match) { $wslDistro = $match; break }
}

if (-not $wslDistro) {
    if ($SkipWslInstall) { Err "No Ubuntu WSL distro found." }
    Log "No Ubuntu WSL distro found. Installing Ubuntu."
    wsl --install -d Ubuntu
    Log "Ubuntu install started. Finish Ubuntu setup, then rerun this installer."
    exit 0
}

Log "Using WSL distro: $wslDistro"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$installWslSh = Join-Path $scriptDir "install_wsl.sh"
if (-not (Test-Path $installWslSh)) {
    Err "install_wsl.sh not found next to install_amd.ps1. Run install_amd.cmd from the downloaded repo folder."
}

$repoPath = (wsl -d $wslDistro wslpath -a $scriptDir 2>$null | Select-Object -First 1)
if ($repoPath) { $repoPath = ($repoPath -replace "`0", "").Trim() }
if (-not $repoPath) { Err "Could not map Windows path to WSL path: $scriptDir" }

$installArgs = @("--address", (ShellQuoteForWsl $Address), "--pool", (ShellQuoteForWsl $Pool))
if ($Worker) { $installArgs += @("--worker", (ShellQuoteForWsl $Worker)) }

$cmd = "cd $(ShellQuoteForWsl $repoPath) && bash ./install_wsl.sh $($installArgs -join ' ')"
Log "Running installer inside WSL"
wsl -d $wslDistro bash -lc $cmd
if ($LASTEXITCODE -ne 0) { Err "WSL installer failed with exit code $LASTEXITCODE" }

Log "Restarting WSL to apply GPU passthrough environment"
wsl --shutdown 2>$null
Start-Sleep -Seconds 3

Log "Setup complete"
Write-Host ""
Write-Host "Launch miner:"
Write-Host "  wsl -d $wslDistro -e amdbtx-miner --config ~/.amdbtx-miner/config.yaml"
Write-Host ""
Write-Host "Troubleshoot:"
Write-Host "  Check GPU:   wsl -d $wslDistro -e rocminfo"
Write-Host "  Check logs:  wsl -d $wslDistro -e tail -f ~/.amdbtx-miner/miner.log"
Write-Host "  Check libs:  wsl -d $wslDistro -e ldd ~/.amdbtx-miner/bin/btx-gbt-solve-hip"
