import subprocess
import shutil


def detect_gpu_info() -> dict:
    info = {"gpu_detected": False, "gpu_name": "", "gpu_arch": ""}

    if shutil.which("rocm-smi"):
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--showid"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.split("\n"):
                if "gfx" in line:
                    arch = line.split("gfx")[-1].split()[0]
                    if arch:
                        info["gpu_arch"] = f"gfx{arch}"
                if "Card" in line or "GPU" in line:
                    info["gpu_name"] = line.strip(": ")
            info["gpu_detected"] = bool(info["gpu_arch"])
        except subprocess.SubprocessError:
            pass

    if not info["gpu_detected"] and shutil.which("rocminfo"):
        try:
            out = subprocess.check_output(
                ["rocminfo"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.split("\n"):
                if "gfx" in line.lower() and "Agent" in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith("gfx") and len(p) > 3:
                            info["gpu_arch"] = p
            info["gpu_detected"] = bool(info["gpu_arch"])
        except subprocess.SubprocessError:
            pass

    return info
