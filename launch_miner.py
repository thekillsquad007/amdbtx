#!/usr/bin/env python3
"""Launcher for AMD BTX Miner on Windows/WSL."""
import subprocess
import sys
import os
from pathlib import Path

def main():
    # Try to run in WSL if available
    if os.name == "nt":
        # Check if WSL is installed and Ubuntu exists
        try:
            result = subprocess.run(
                ["wsl", "-l", "--quiet"],
                capture_output=True,
                text=True
            )
            distros = result.stdout.strip().split("\n")
            if any("ubuntu" in d.lower() for d in distros):
                print("[INFO] Launching in WSL...")
                # Run the miner in WSL
                wsl_cmd = [
                    "wsl", "-d", "Ubuntu-22.04",
                    "bash", "-c",
                    "export HSA_ENABLE_DXG_DETECTION=1; source ~/.bashrc 2>/dev/null; amdbtx-miner"
                ]
                # Find config path
                config = Path.home() / ".amdbtx-miner" / "config.yaml"
                if config.exists():
                    wsl_cmd[4] = f"source ~/.bashrc 2>/dev/null; amdbtx-miner --config ~/.amdbtx-miner/config.yaml"
                subprocess.run(" ".join(wsl_cmd))
                return
        except FileNotFoundError:
            pass

    # Fallback: try direct execution
    print("[INFO] WSL not found, cannot mine (ROCm requires Linux)")
    print("Install WSL2 with Ubuntu and run the installer again.")
    sys.exit(1)

if __name__ == "__main__":
    main()