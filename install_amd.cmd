@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "ADDRESS=%~1"
set "WORKER=%~2"

if "%ADDRESS%"=="" (
  echo AMD BTX Miner Windows/WSL installer
  echo.
  set /p "ADDRESS=Enter your BTX payout address ^(btx1z...^): "
)

if "%ADDRESS%"=="" (
  echo [error] BTX payout address is required.
  exit /b 1
)

echo [amdbtx] Launching installer with PowerShell execution policy bypass...
if "%WORKER%"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_amd.ps1" -Address "%ADDRESS%"
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_amd.ps1" -Address "%ADDRESS%" -Worker "%WORKER%"
)
exit /b %ERRORLEVEL%
