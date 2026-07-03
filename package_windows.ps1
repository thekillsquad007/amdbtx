# Package amdbtx-miner for Windows portable use
# Builds solver, then bundles everything with PyInstaller into a single .exe

$ErrorActionPreference = "Stop"
$repoRoot = "E:\Business\amdbtx"
$outDir = "$repoRoot\dist\amdbtx-miner"
$entryPoint = "$repoRoot\src\run_miner.py"

# --- Build solver (multi-arch) ---
Write-Output "=== Building solver ==="
& "$repoRoot\solver\build_windows.ps1"
if ($LASTEXITCODE -ne 0) { throw "Solver build failed" }

# --- Install Python deps ---
Write-Output "=== Installing Python deps ==="
pip install pyyaml 2>&1 | Out-Null

# --- Run PyInstaller ---
Write-Output "=== Packaging with PyInstaller ==="
$solverBin = "$repoRoot\solver\build-win\btx-gbt-solve-hip"
$solverName = "btx-gbt-solve-hip.exe"

# Copy solver with .exe extension for PyInstaller data bundling
Copy-Item $solverBin "$env:TEMP\$solverName" -Force

python -m PyInstaller --clean --onefile --console `
    --name "amdbtx-miner" `
    --distpath "$outDir" `
    --add-data "$env:TEMP\$solverName;." `
    --hidden-import yaml `
    --collect-submodules amdbtx_miner `
    "$entryPoint"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# --- Clean up intermediate PyInstaller files ---
Remove-Item -Recurse -Force "$repoRoot\build" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$repoRoot\__pycache__" -ErrorAction SilentlyContinue
Remove-Item "$repoRoot\*.spec" -ErrorAction SilentlyContinue

Write-Output "=== Done ==="
Write-Output "Package: $outDir\amdbtx-miner.exe"
Write-Output "Size:    $([math]::Round((Get-Item "$outDir\amdbtx-miner.exe").Length / 1MB, 1)) MB"
Write-Output ""
Write-Output "Usage: amdbtx-miner.exe --payout-address YOUR_BTX_ADDRESS"
Write-Output "Usage: amdbtx-miner.exe --payout-address YOUR_BTX_ADDRESS --pool-host POOL --pool-port PORT"
