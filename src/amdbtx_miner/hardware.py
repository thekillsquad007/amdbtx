import subprocess
import shutil
import os
import re
import platform
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SUBPROCESS_TIMEOUT_SEC = 5.0


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=SUBPROCESS_TIMEOUT_SEC
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _cpu_threads_total() -> int | None:
    n = os.cpu_count()
    return int(n) if n else None


def _ram_gb_total() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
    except OSError:
        pass
    return None


def _os_string() -> str:
    sys = platform.system()
    rel = platform.release()
    if sys == "Linux":
        try:
            with open("/etc/os-release") as f:
                fields = {}
                for line in f:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        fields[k.strip()] = v.strip().strip('"')
                name = fields.get("PRETTY_NAME") or fields.get("NAME", "Linux")
                return f"{name} / {rel}"
        except OSError:
            return f"Linux / {rel}"
    return f"{sys} / {rel}"


def _enumerate_amd_gpus() -> list[dict[str, Any]]:
    gpus = []
    env = {**os.environ, "HSA_ENABLE_DXG_DETECTION": "1"}

    if shutil.which("rocm-smi"):
        out = _run_env(["rocm-smi", "--showproductname", "--showid"], env)
        if out:
            gpu_name = ""
            gpu_arch = ""
            vram_mb = None
            for line in out.split("\n"):
                m = re.search(r"gfx[0-9a-f]+", line)
                if m and not gpu_arch:
                    gpu_arch = m.group()
                if re.search(r"Card|GPU|product", line, re.IGNORECASE):
                    name = re.sub(r"^.*?:\s*", "", line).strip()
                    name = re.sub(r"\s*\(TM\)\s*|\s*\(R\)\s*", "", name)
                    if name and name != "None":
                        gpu_name = name
            vram_out = _run_env(["rocm-smi", "--showmeminfo", "vram"], env)
            if vram_out:
                vm = re.search(r"(\d+)\s*(?:MiB|MB)", vram_out)
                if vm:
                    try:
                        vram_mb = int(vm.group(1))
                    except ValueError:
                        pass
            if gpu_arch:
                gpus.append({
                    "model": gpu_name or "AMD GPU",
                    "vram_gb": round(vram_mb / 1024, 2) if vram_mb else None,
                    "compute_capability": gpu_arch,
                    "gpu_uuid": f"amdgpu-{gpu_arch}",
                    "pcie_link": None,
                })

    if not gpus:
        rocminfo_path = shutil.which("rocminfo")
        if not rocminfo_path:
            for d in sorted(Path("/opt").glob("rocm*/bin")):
                candidate = d / "rocminfo"
                if candidate.is_file():
                    rocminfo_path = str(candidate)
                    break
        if rocminfo_path:
            try:
                result = subprocess.run(
                    [rocminfo_path], capture_output=True, text=True, timeout=10, env=env
                )
                if result.returncode == 0:
                    current_agent = {}
                    gpus_list = []
                    for line in result.stdout.splitlines():
                        stripped = line.strip()
                        if re.match(r"^Agent\s+\d+", stripped):
                            if current_agent.get("gpu_arch") and current_agent.get("is_gpu"):
                                gpus_list.append(current_agent)
                            current_agent = {"gpu_arch": "", "marketing_name": "", "is_gpu": False, "vram_mb": None}
                        m = re.search(r"gfx[0-9a-f]+", stripped)
                        if m and not current_agent.get("gpu_arch"):
                            current_agent["gpu_arch"] = m.group()
                        if "Device Type:" in stripped and "GPU" in stripped:
                            current_agent["is_gpu"] = True
                        if "Marketing Name:" in stripped:
                            name = stripped.split(":", 1)[1].strip()
                            name = re.sub(r"\s*\(TM\)\s*|\s*\(R\)\s*", "", name)
                            if name and name != "None":
                                current_agent["marketing_name"] = name
                        mem_match = re.match(r"Size:\s*(\d+)\s*(KB|MB|GB)", stripped)
                        if mem_match and not current_agent.get("vram_mb"):
                            val = int(mem_match.group(1))
                            unit = mem_match.group(2)
                            if unit == "KB":
                                current_agent["vram_mb"] = val / 1024
                            elif unit == "MB":
                                current_agent["vram_mb"] = val
                            elif unit == "GB":
                                current_agent["vram_mb"] = val * 1024
                    if current_agent.get("gpu_arch") and current_agent.get("is_gpu"):
                        gpus_list.append(current_agent)
                    for g in gpus_list:
                        gpus.append({
                            "model": g.get("marketing_name") or "AMD GPU",
                            "vram_gb": round(g["vram_mb"] / 1024, 2) if g.get("vram_mb") else None,
                            "compute_capability": g["gpu_arch"],
                            "gpu_uuid": f"amdgpu-{g['gpu_arch']}",
                            "pcie_link": None,
                        })
            except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                pass

    return gpus


def _run_env(cmd: list[str], env: dict) -> str | None:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=SUBPROCESS_TIMEOUT_SEC, env=env
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _hostname() -> str | None:
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return None


def _detect_environment() -> dict[str, Any]:
    info = {
        "is_containerized": False,
        "cpu_threads_effective": None,
        "rental_provider": None,
        "power_cap_writable": False,
    }
    try:
        if os.path.exists("/.dockerenv"):
            info["is_containerized"] = True
    except Exception:
        pass
    try:
        with open("/proc/self/cgroup", "r") as f:
            cg = f.read()
            if any(m in cg for m in ("docker", "containerd", "kubepods", "lxc")):
                info["is_containerized"] = True
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            parts = f.read().strip().split()
            if len(parts) == 2 and parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                if quota > 0 and period > 0:
                    info["cpu_threads_effective"] = round(quota / period, 2)
    except Exception:
        pass
    hn = (_hostname() or "").lower()
    if "vast.ai" in hn or "vast-" in hn:
        info["rental_provider"] = "vast.ai"
    elif "runpod" in hn:
        info["rental_provider"] = "runpod"
    elif "autodl" in hn:
        info["rental_provider"] = "autodl"
    return info


def _probe_active_backend(solver_path: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"active": None, "cuda_archs": None}
    if not solver_path or not os.path.isfile(solver_path):
        return result
    try:
        out = subprocess.run(
            ["strings", solver_path], capture_output=True, text=True, timeout=15
        )
        if out.returncode == 0:
            gfx = sorted(set(re.findall(r"\b(gfx\d+[a-z]?)\b", out.stdout)))
            if gfx:
                result["cuda_archs"] = ",".join(gfx)
            if any("hip" in s.lower() or "rocm" in s.lower() for s in out.stdout.split("\n")):
                result["active"] = "rocm"
            elif any("cuda" in s.lower() for s in out.stdout.split("\n")):
                result["active"] = "cuda"
    except Exception:
        pass
    return result


def _probe_solver_gpu(solver_path: str | None) -> dict[str, Any] | None:
    if not solver_path or not os.path.isfile(solver_path):
        return None
    env = {**os.environ, "HSA_ENABLE_DXG_DETECTION": "1"}
    # WSL needs librocdxg from rocm-7.2.3+ before /opt/rocm/lib (often lacks DXG).
    ld_parts = [str(Path.home() / ".amdbtx-miner" / "runtime"), "/usr/lib/wsl/lib"]
    ld_parts.extend(str(d) for d in sorted(Path("/opt").glob("rocm-*/lib"), reverse=True))
    ld_parts.append("/opt/rocm/lib")
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_parts.extend(p for p in existing_ld.split(":") if p)
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in ld_parts if os.path.isdir(p)))
    try:
        proc = subprocess.Popen(
            [solver_path, "--daemon", "--backend", "hip", "--batch-size", "1", "--epsilon-bits", "0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            line = proc.stderr.readline() if proc.stderr else ""
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None

    m = re.search(r"HIP GPU detected:\s*(.*?)\s+arch=(gfx[0-9a-f]+)\s+memory=(\d+)MB", line)
    if not m:
        return None
    return {
        "model": m.group(1).strip() or "AMD GPU",
        "vram_gb": round(int(m.group(3)) / 1024, 2),
        "compute_capability": m.group(2),
        "gpu_uuid": f"amdgpu-{m.group(2)}",
        "pcie_link": None,
    }


def collect_static_hardware(
    miner_version: str,
    cpu_threads_allocated: int | None = None,
    solver_env: dict[str, str | int | None] | None = None,
    solver_path: str | None = None,
) -> dict[str, Any]:
    gpus = _enumerate_amd_gpus()
    if not gpus:
        solver_gpu = _probe_solver_gpu(solver_path)
        if solver_gpu:
            gpus = [solver_gpu]
    env = _detect_environment()
    backend = _probe_active_backend(solver_path)
    out = {
        "cpu_model": _cpu_model(),
        "cpu_threads_total": _cpu_threads_total(),
        "cpu_threads_allocated": cpu_threads_allocated,
        "ram_gb_total": _ram_gb_total(),
        "os": _os_string(),
        "miner_version": miner_version,
        "driver_version": None,
        "cuda_version": None,
        "gpus": gpus,
        "host_hostname": _hostname(),
        "is_containerized": env["is_containerized"],
        "cpu_threads_effective": env["cpu_threads_effective"],
        "rental_provider": env["rental_provider"],
        "power_cap_writable": env["power_cap_writable"],
        "numa": None,
        "active_backend": backend["active"],
        "cuda_arch_supported": backend["cuda_archs"],
    }
    if solver_env:
        out["solver_env"] = {k: (str(v) if v is not None else "") for k, v in solver_env.items()}
    return out


def detect_gpu_info(solver_path: str | None = None) -> dict:
    info = {"gpu_detected": False, "gpu_name": "", "gpu_arch": "", "gpus": []}
    gpus = _enumerate_amd_gpus()
    if not gpus:
        solver_gpu = _probe_solver_gpu(solver_path)
        if solver_gpu:
            gpus = [solver_gpu]
    info["gpus"] = gpus
    if gpus:
        best = pick_best_gpu(gpus)
        info["gpu_detected"] = True
        info["gpu_name"] = best.get("model", "")
        info["gpu_arch"] = best.get("compute_capability", "")
    return info


def pick_best_gpu(gpus: list[dict]) -> dict | None:
    if not gpus:
        return None
    if len(gpus) == 1:
        return gpus[0]
    sorted_gpus = sorted(
        gpus,
        key=lambda g: (
            1 if (g.get("compute_capability") or "").startswith("gfx9") else 0,
            -(g.get("vram_gb") or 0),
        ),
    )
    return sorted_gpus[0]


def pick_best_gpu_index(gpus: list[dict]) -> int:
    if not gpus:
        return -1
    best = pick_best_gpu(gpus)
    for i, g in enumerate(gpus):
        if g is best:
            return i
    return 0


def parse_gpu_devices_spec(value: Any) -> list[int] | str | None:
    """Normalize gpu_devices config/CLI value. Returns 'all', list of ints, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() == "all":
            return "all"
        return [int(part.strip()) for part in stripped.split(",") if part.strip()]
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(x) for x in value]
    raise ValueError(f"invalid gpu_devices value: {value!r}")


def resolve_gpu_devices(cfg: dict, gpu_info: dict) -> list[int]:
    """Pick which GPU indices to mine on.

    Priority:
    1. gpu_devices: \"all\", [0, 1], or \"0,1\"
    2. gpu_device >= 0 (single forced GPU)
    3. auto (-1): best dGPU when multiple are present, else GPU 0
    """
    gpus = gpu_info.get("gpus") or []
    num_detected = len(gpus)
    if num_detected == 0:
        num_detected = 1

    spec = parse_gpu_devices_spec(cfg.get("gpu_devices"))
    if spec == "all":
        if not gpus:
            log.warning("gpu_devices=all but no GPUs detected; using GPU 0")
            return [0]
        return list(range(len(gpus)))
    if isinstance(spec, list) and spec:
        for idx in spec:
            if idx < 0 or (gpus and idx >= len(gpus)):
                log.warning("gpu_devices includes GPU %d (detected %d); using anyway", idx, len(gpus))
        return spec

    forced = int(cfg.get("gpu_device", -1))
    if forced >= 0:
        return [forced]

    if len(gpus) > 1:
        best = pick_best_gpu_index(gpus)
        return [best if best >= 0 else 0]
    return [0]


def hardware_summary_string(hw: dict[str, Any]) -> str:
    bits = []
    if hw.get("cpu_model"):
        bits.append(f"CPU={hw['cpu_model']}")
    if hw.get("cpu_threads_total"):
        bits.append(f"threads={hw['cpu_threads_total']}")
    if hw.get("ram_gb_total"):
        bits.append(f"RAM={hw['ram_gb_total']}GB")
    gpus = hw.get("gpus") or []
    if gpus:
        models = ", ".join(f"{g.get('model', '?')}" for g in gpus)
        bits.append(f"GPUs=[{models}]")
    return " ".join(bits)
