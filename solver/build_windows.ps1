# Build btx-gbt-solve-hip on Windows with AMD ROCm HIP SDK
# Prerequisites:
#   1. AMD ROCm HIP SDK via: pip install rocm_sdk_core
#   2. Visual Studio 2022 Build Tools with C++ workload (provides MSVC headers/linker)
#   3. CMake (install via winget: winget install Kitware.CMake)

$ErrorActionPreference = "Stop"

# Paths
$rocmCore = "$env:APPDATA\Python\Python312\site-packages\_rocm_sdk_core"
$scripts  = "$env:APPDATA\Python\Python312\Scripts"
$msvcVer  = "14.44.35207"
$msvcRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\$msvcVer"
$sdkRoot  = "C:\Program Files (x86)\Windows Kits\10"
$sdkVer   = "10.0.26100.0"

# Verify dependencies
if (-not (Test-Path $rocmCore))   { throw "ROCm SDK not found at $rocmCore. Install: pip install rocm_sdk_core" }
if (-not (Test-Path $msvcRoot))   { throw "MSVC not found at $msvcRoot. Install Visual Studio Build Tools with C++ workload" }
if (-not (Test-Path "$msvcRoot\bin\Hostx64\x64\dumpbin.exe")) { throw "MSVC linker not found at $msvcRoot" }
$cmakePath = Get-Command cmake.exe -ErrorAction SilentlyContinue
if (-not $cmakePath)              { throw "CMake not found. Install: winget install Kitware.CMake" }

# Set build environment
$env:PATH    = "$scripts;$rocmCore\bin;$rocmCore\lib\llvm\bin;C:\Program Files\CMake\bin;$env:PATH"
$env:INCLUDE = "$msvcRoot\include;$msvcRoot\include\winrt;$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\um;$sdkRoot\Include\$sdkVer\shared"
$env:LIB     = "$msvcRoot\lib\x64;$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64"

# Clean old build
$buildDir = Join-Path $PSScriptRoot "build-win"
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
Write-Output "Building btx-gbt-solve-hip for Windows HIP..."

cmake -S $PSScriptRoot -B $buildDir "-DHIPCC=$scripts\hipcc.exe" "-DHIP_ARCHS=gfx1030;gfx1031;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" 2>&1
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }

cmake --build $buildDir -j8 2>&1
if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }

$binary = Join-Path $buildDir "btx-gbt-solve-hip"
Write-Output "Build successful!"
Write-Output "Binary: $binary"
Write-Output "Size:   $((Get-Item $binary).Length / 1KB) KB"

# Verify
$proc = Start-Process -FilePath $binary -ArgumentList "--version" -NoNewWindow -RedirectStandardOutput "$env:TEMP\btx_ver_out.txt" -RedirectStandardError "$env:TEMP\btx_ver_err.txt" -PassThru -Wait
Write-Output "Version: $(Get-Content "$env:TEMP\btx_ver_err.txt" -Raw)"
Write-Output "---"
Write-Output "DLL dependencies:"
& "$msvcRoot\bin\Hostx64\x64\dumpbin.exe" /dependents $binary 2>&1 | Select-String "\.dll"
