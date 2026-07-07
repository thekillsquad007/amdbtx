# Build solver + bundle ROCm DLLs + package with PyInstaller
$ErrorActionPreference = "Stop"
$repoRoot = "E:\Business\amdbtx"
$entryPoint = "$repoRoot\src\run_miner.py"
$outDir = "$repoRoot\dist\amdbtx-miner-windows"

# === Find ROCm SDK ===
# Try: ROCm SDK installed in Python site-packages, or set ROCM_PATH env var, or known install dirs
$rocmCore = $env:ROCM_PATH
if (-not $rocmCore) {
    # Check common locations
    $candidates = @(
        "$env:APPDATA\Python\Python313\site-packages\_rocm_sdk_core",
        "$env:APPDATA\Python\Python312\site-packages\_rocm_sdk_core",
        "$env:APPDATA\Python\Python311\site-packages\_rocm_sdk_core",
        "$env:APPDATA\Python\Python310\site-packages\_rocm_sdk_core",
        "C:\Program Files\AMD\ROCm\*\bin"
    )
    foreach ($c in $candidates) {
        $resolved = Resolve-Path $c -ErrorAction SilentlyContinue
        if ($resolved) { $rocmCore = $resolved[0].Path; break }
    }
}
if (-not $rocmCore -or -not (Test-Path "$rocmCore\bin\hipcc.exe")) {
    Write-Error "ROCm SDK not found. Set ROCM_PATH or install ROCm SDK for Python."
    exit 1
}
Write-Output "ROCm SDK: $rocmCore"

# Find hipcc.exe
$hipcc = (Get-ChildItem -Recurse -Filter hipcc.exe -Path $rocmCore | Select-Object -First 1).FullName
if (-not $hipcc) { $hipcc = "$rocmCore\bin\hipcc.exe" }
Write-Output "HIPCC: $hipcc"

# Find Python
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = "python" }
$pyVer = & $python --version 2>&1
Write-Output "Python: $pyVer"

# Find PyInstaller
$pyinstaller = (Get-Command pyinstaller -ErrorAction SilentlyContinue).Source
if (-not $pyinstaller) { 
    & $python -m PyInstaller --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Installing PyInstaller..."
        & $python -m pip install pyinstaller
    }
}

Write-Output "=== 1. Build solver ==="
$buildDir = "$repoRoot\solver\build-win"
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }

cmake -S "$repoRoot\solver" -B $buildDir "-DHIPCC=$hipcc" "-DHIP_ARCHS=gfx1030;gfx1031;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
cmake --build $buildDir -j8
if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }
$solverBin = "$buildDir\btx-gbt-solve-hip"
if (-not (Test-Path $solverBin)) { throw "Solver binary not found at $solverBin" }
Write-Output "Solver: $solverBin"

Write-Output "=== 2. Stage ROCm runtime DLLs ==="
$runtimeDir = "$buildDir\runtime"
if (Test-Path $runtimeDir) { Remove-Item -Recurse -Force $runtimeDir }
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

$rocmDlls = @("amdhip64_7.dll", "amd_comgr0701.dll", "hiprtc-builtins0701.dll", "hiprtc0701.dll")
$found = 0
foreach ($dll in $rocmDlls) {
    $src = "$rocmCore\bin\$dll"
    if (Test-Path $src) {
        Copy-Item $src "$runtimeDir\$dll" -Force
        Write-Output "  Bundled $dll"
        $found++
    } else {
        Write-Output "  WARNING: $dll not found at $src"
    }
}
if ($found -eq 0) { throw "No ROCm DLLs found! Cannot package." }

Write-Output "=== 3. Package with PyInstaller ==="
$solverName = "btx-gbt-solve-hip.exe"
Copy-Item $solverBin "$env:TEMP\$solverName" -Force

if (Test-Path "$outDir") { Remove-Item -Recurse -Force $outDir }

& $python -m PyInstaller --clean --onefile --console `
    --name "amdbtx-miner" `
    --distpath "$outDir" `
    --add-data "$env:TEMP\$solverName;." `
    --add-data "$runtimeDir;runtime" `
    --hidden-import yaml `
    --collect-submodules amdbtx_miner `
    "$entryPoint"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Write-Output "=== 4. Clean up ==="
Remove-Item "$env:TEMP\$solverName" -Force -ErrorAction SilentlyContinue

Write-Output "=== Done ==="
$exe = "$outDir\amdbtx-miner.exe"
if (Test-Path $exe) {
    Write-Output "Package: $exe"
    Write-Output "Size:    $([math]::Round((Get-Item $exe).Length / 1MB, 1)) MB"
} else {
    Write-Error "Expected executable not found at $exe"
    exit 1
}
