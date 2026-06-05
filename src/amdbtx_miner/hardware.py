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
                    info["gpu_arch"] = line.split("gfx")[-1].split()[0]
                if "Card" in line or "GPU" in line:
                    info["gpu_name"] = line
            info["gpu_detected"] = bool(info["gpu_arch"])
        except subprocess.SubprocessError:
            pass

    return info