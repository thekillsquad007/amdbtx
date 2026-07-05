import hashlib
import json
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


def _run_env(cmd: list[str], env: dict) -> str | None:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=SUBPROCESS_TIMEOUT_SEC, env=env
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _rocm_miner_env() -> dict[str, str]:
    """ROC/HIP env for WSL DXG passthrough and distro ROCm installs."""
    env = {**os.environ, "HSA_ENABLE_DXG_DETECTION": "1"}
    ld_parts = [str(Path.home() / ".amdbtx-miner" / "runtime"), "/usr/lib/wsl/lib"]
    ld_parts.extend(str(d) for d in sorted(Path("/opt").glob("rocm-*/lib"), reverse=True))
    ld_parts.append("/opt/rocm/lib")
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_parts.extend(p for p in existing_ld.split(":") if p)
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in ld_parts if os.path.isdir(p)))
    return env


def _resolve_rocm_smi() -> str | None:
    rocm = shutil.which("rocm-smi")
    if rocm:
        return rocm
    for d in sorted(Path("/opt").glob("rocm*/bin"), reverse=True):
        candidate = d / "rocm-smi"
        if candidate.is_file():
            return str(candidate)
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


def _hostname() -> str | None:
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return None


_MODEL_TO_GFX: dict[str, str] = {
    "7900 xtx": "gfx1100", "7900 xt": "gfx1100", "7900 gre": "gfx1100",
    "7800 xt": "gfx1101", "7700 xt": "gfx1102", "7600": "gfx1102",
    "6900 xt": "gfx1030", "6800 xt": "gfx1030", "6800": "gfx1031",
    "6700 xt": "gfx1031", "6600 xt": "gfx1032", "6600": "gfx1032",
    "rx 6400": "gfx1032", "rx 6500": "gfx1032",
    "9070 xt": "gfx1200", "9070": "gfx1200",
    "9060 xt": "gfx1201", "9060": "gfx1201",
    "w7900": "gfx1100", "w7800": "gfx1101", "w7700": "gfx1102",
    "mi250x": "gfx90a", "mi250": "gfx90a", "mi100": "gfx908",
    "mi50": "gfx906", "mi60": "gfx906",
    "w7900": "gfx1100", "w7800": "gfx1101", "w7700": "gfx1102",
    "pro w7900": "gfx1100", "pro w7800": "gfx1101",
}


def _gfx_from_model_name(name: str) -> str | None:
    lower = name.lower()
    for key, gfx in _MODEL_TO_GFX.items():
        if key in lower:
            return gfx
    return None


def _amd_gfx_target(env: dict[str, str] | None = None,
                    model_fallback: str | None = None) -> str | None:
    rocminfo_path = shutil.which("rocminfo")
    if not rocminfo_path:
        for d in sorted(Path("/opt").glob("rocm*/bin"), reverse=True):
            candidate = d / "rocminfo"
            if candidate.is_file():
                rocminfo_path = str(candidate)
                break
    if rocminfo_path:
        try:
            result = subprocess.run(
                [rocminfo_path],
                capture_output=True,
                text=True,
                timeout=10,
                env=env or _rocm_miner_env(),
            )
            if result.returncode == 0:
                m = re.search(r"\bgfx\d{3,4}[a-z]?\b", result.stdout)
                if m:
                    return m.group(0)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass
    if model_fallback:
        mapped = _gfx_from_model_name(model_fallback)
        if mapped:
            return mapped
    # Fallback: try hipconfig
    hipconfig = shutil.which("hipconfig")
    if not hipconfig:
        for d in sorted(Path("/opt").glob("rocm*/bin"), reverse=True):
            candidate = d / "hipconfig"
            if candidate.is_file():
                hipconfig = str(candidate)
                break
    if hipconfig:
        try:
            out = subprocess.check_output(
                [hipconfig, "--arch"], stderr=subprocess.DEVNULL, timeout=5,
            )
            arch = out.decode("utf-8", errors="replace").strip()
            if re.match(r"gfx\d{3,4}[a-z]?$", arch):
                return arch
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass
    return None


def _amd_pci_bdfs() -> list[str]:
    """PCI bus IDs for AMD display/3D controllers (vendor 0x1002).

    Excludes integrated GPUs (typically on bus 0 root-complex slot) so the
    default enumeration prefers discrete GPUs. Set AMDBTX_INCLUDE_IGPU=1
    to opt back into including them.
    """
    bdfs: list[str] = []
    pci_root = "/sys/bus/pci/devices"
    if not os.path.isdir(pci_root):
        return bdfs
    include_igpu = os.environ.get("AMDBTX_INCLUDE_IGPU", "").lower() in ("1", "true", "yes")
    try:
        for entry in sorted(os.listdir(pci_root)):
            base = os.path.join(pci_root, entry)
            try:
                with open(os.path.join(base, "vendor")) as f:
                    if f.read().strip() != "0x1002":
                        continue
                with open(os.path.join(base, "class")) as f:
                    cls = f.read().strip()
                if not (cls.startswith("0x0300") or cls.startswith("0x0302")):
                    continue
                # Integrated GPUs sit on PCI bus 0 with secondary function 0
                # (e.g. "00:01.0"). Exclude by default unless AMDBTX_INCLUDE_IGPU
                bus = entry.split(":")[0]
                if not include_igpu and bus == "0000":
                    continue
                bdfs.append(entry)
            except OSError:
                continue
    except OSError:
        pass
    return bdfs


def _make_amd_gpu_uuid(gfx: str | None, ident: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", (gfx or "amdgpu").lower()) or "amdgpu"
    clean = re.sub(r"[^a-z0-9]", "", ident.lower()) or "unknown"
    return f"amd-{slug}-{clean}"


def _rocm_smi_json(args: list[str]) -> dict[str, Any] | None:
    rocm = _resolve_rocm_smi()
    if not rocm:
        return None
    out = _run_env([rocm, *args, "--json"], _rocm_miner_env())
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _amd_gpus_from_rocm_smi() -> list[dict[str, Any]]:
    """Per-GPU AMD info via rocm-smi JSON — matches dexbtx pool handshake shape."""
    if platform.system() != "Linux":
        return []
    j = _rocm_smi_json(["--showproductname", "--showuniqueid", "--showmeminfo", "vram"])
    if not j:
        return []
    first_model: str | None = None
    gpus: list[dict[str, Any]] = []
    card_idx = 0
    for card, info in j.items():
        if not str(card).lower().startswith("card"):
            continue
        low = {str(k).lower(): v for k, v in info.items()}
        model = (
            low.get("card series")
            or low.get("card model")
            or low.get("device name")
            or low.get("gpu id")
            or "AMD GPU"
        )
        model = str(model).strip() or "AMD GPU"
        if first_model is None:
            first_model = model
    gfx = _amd_gfx_target(model_fallback=first_model)
    pci_bdfs = _amd_pci_bdfs()
    card_idx = 0
    for card, info in j.items():
        if not str(card).lower().startswith("card"):
            continue
        low = {str(k).lower(): v for k, v in info.items()}
        model = (
            low.get("card series")
            or low.get("card model")
            or low.get("device name")
            or low.get("gpu id")
            or "AMD GPU"
        )
        model = str(model).strip() or "AMD GPU"
        if not re.search(r"amd|radeon|instinct", model, re.I):
            model = f"AMD {model}"
        uniq = str(low.get("unique id", "")).strip()
        ident = re.sub(r"[^a-z0-9]", "", uniq.lower()) if uniq else ""
        if not ident:
            if card_idx < len(pci_bdfs):
                ident = re.sub(r"[^a-z0-9]", "", pci_bdfs[card_idx].lower())
            else:
                ident = (
                    re.sub(r"[^a-z0-9]", "", str(card).lower())
                    + "-"
                    + re.sub(r"[^a-z0-9]", "", (_hostname() or "unknown").lower())
                )
        vram_gb: float | None = None
        for k, v in low.items():
            if "vram total memory" in k:
                try:
                    vram_gb = round(int(str(v)) / (1024 ** 3), 2)
                except ValueError:
                    pass
        gpus.append({
            "model": model,
            "vram_gb": vram_gb if vram_gb is not None else 0,
            "compute_capability": gfx or "rocm",
            "gpu_uuid": _make_amd_gpu_uuid(gfx, ident),
            "pcie_link": pci_bdfs[card_idx] if card_idx < len(pci_bdfs) else None,
        })
        card_idx += 1
    return gpus


def _amd_gpus_from_rocminfo() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    env = _rocm_miner_env()
    rocminfo_path = shutil.which("rocminfo")
    if not rocminfo_path:
        for d in sorted(Path("/opt").glob("rocm*/bin"), reverse=True):
            candidate = d / "rocminfo"
            if candidate.is_file():
                rocminfo_path = str(candidate)
                break
    if not rocminfo_path:
        return gpus
    pci_bdfs = _amd_pci_bdfs()
    try:
        result = subprocess.run(
            [rocminfo_path], capture_output=True, text=True, timeout=10, env=env
        )
        if result.returncode != 0:
            return gpus
        current_agent: dict[str, Any] = {}
        gpus_list: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if re.match(r"^Agent\s+\d+", stripped):
                if current_agent.get("gpu_arch") and current_agent.get("is_gpu"):
                    gpus_list.append(current_agent)
                current_agent = {
                    "gpu_arch": "",
                    "marketing_name": "",
                    "is_gpu": False,
                    "vram_mb": None,
                    "agent_id": stripped,
                }
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
        for idx, g in enumerate(gpus_list):
            gfx = g["gpu_arch"]
            if idx < len(pci_bdfs):
                ident = pci_bdfs[idx]
            else:
                ident = re.sub(r"[^a-z0-9]", "", g.get("agent_id", f"gpu{idx}").lower())
                ident += "-" + re.sub(r"[^a-z0-9]", "", (_hostname() or "unknown").lower())
            gpus.append({
                "model": g.get("marketing_name") or "AMD GPU",
                "vram_gb": round(g["vram_mb"] / 1024, 2) if g.get("vram_mb") else None,
                "compute_capability": gfx,
                "gpu_uuid": _make_amd_gpu_uuid(gfx, ident),
                "pcie_link": pci_bdfs[idx] if idx < len(pci_bdfs) else None,
            })
    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass
    return gpus


def _enumerate_amd_gpus() -> list[dict[str, Any]]:
    """Enumerate AMD GPUs for the pool hardware handshake."""
    gpus = _amd_gpus_from_rocm_smi()
    if gpus:
        return gpus
    return _amd_gpus_from_rocminfo()


def _max_gpu_index_from_pci() -> int:
    """Highest PCI slot index visible. Used to bound per-GPU probing."""
    try:
        entries = [e for e in os.listdir("/sys/bus/pci/devices")
                   if os.path.isfile(os.path.join("/sys/bus/pci/devices", e, "vendor"))]
    except OSError:
        return 0
    indices = []
    for e in entries:
        try:
            with open(os.path.join("/sys/bus/pci/devices", e, "vendor")) as f:
                if f.read().strip() != "0x1002":
                    continue
            with open(os.path.join("/sys/bus/pci/devices", e, "class")) as f:
                cls = f.read().strip()
            if not (cls.startswith("0x0300") or cls.startswith("0x0302")):
                continue
            indices.append(int(e.split(":")[1], 16))
        except (OSError, ValueError):
            continue
    return max(indices) if indices else 0


def _count_amd_pci_devices() -> int:
    """Count PCI AMD display/3D devices (matching _amd_pci_bdfs filter)."""
    pci_root = "/sys/bus/pci/devices"
    if not os.path.isdir(pci_root):
        return 0
    include_igpu = os.environ.get("AMDBTX_INCLUDE_IGPU", "").lower() in ("1", "true", "yes")
    count = 0
    try:
        for entry in sorted(os.listdir(pci_root)):
            base = os.path.join(pci_root, entry)
            try:
                with open(os.path.join(base, "vendor")) as f:
                    if f.read().strip() != "0x1002":
                        continue
                with open(os.path.join(base, "class")) as f:
                    cls = f.read().strip()
                if not (cls.startswith("0x0300") or cls.startswith("0x0302")):
                    continue
                bus = entry.split(":")[0]
                if not include_igpu and bus == "0000":
                    continue
                count += 1
            except OSError:
                continue
    except OSError:
        pass
    return count


def _probe_alive_gpu_indices(solver_path: str | None, total: int | None = None) -> list[int]:
    """Probe each candidate GPU index individually; return list of indices that
    initialize HIP successfully.

    Solves the 4xGPU+1broken-card failure mode where rocm-smi enumerates cards
    but HIP refuses one of them, so the solver reports No HIP GPU found and
    falls back to CPU.

    Only probes indices that have a corresponding PCI AMD display device — does
    not try every PCI slot up to 16.

    Does NOT exclude GPUs whose rocm-smi sensors report degraded values: HIP
    init success is the only criterion, so a working GPU with bad sensors still
    counts.
    """
    if not solver_path or not os.path.isfile(solver_path):
        return []
    if total is None or total <= 0:
        total = max(1, _count_amd_pci_devices())
    total = min(total, 16)
    alive: list[int] = []
    for idx in range(total):
        probe = _probe_solver_gpu_at_index(solver_path, idx)
        if probe is not None:
            alive.append(idx)
    return alive


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
                result["active"] = "hip"
            elif any("cuda" in s.lower() for s in out.stdout.split("\n")):
                result["active"] = "cuda"
    except Exception:
        pass
    return result


def _probe_solver_gpu(solver_path: str | None) -> dict[str, Any] | None:
    if not solver_path or not os.path.isfile(solver_path):
        return None
    return _probe_solver_gpu_at_index(solver_path, -1)


def _probe_solver_gpu_at_index(solver_path: str | None, device_index: int) -> dict[str, Any] | None:
    """Probe one GPU index. device_index=-1 means probe default device.

    Returns None if HIP fails to detect any GPU (e.g. broken hardware card).
    """
    if not solver_path or not os.path.isfile(solver_path):
        return None
    env = _rocm_miner_env()
    if device_index >= 0:
        env = {**env, "HIP_VISIBLE_DEVICES": str(device_index)}
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
    gfx = m.group(2)
    pci_bdfs = _amd_pci_bdfs()
    ident = pci_bdfs[device_index] if 0 <= device_index < len(pci_bdfs) else (
        pci_bdfs[0] if pci_bdfs else (_hostname() or "solver-probe")
    )
    return {
        "model": m.group(1).strip() or "AMD GPU",
        "vram_gb": round(int(m.group(3)) / 1024, 2),
        "compute_capability": gfx,
        "gpu_uuid": _make_amd_gpu_uuid(gfx, f"{device_index}-{ident}" if device_index >= 0 else ident),
        "pcie_link": pci_bdfs[device_index] if 0 <= device_index < len(pci_bdfs) else (
            pci_bdfs[0] if pci_bdfs else None
        ),
    }


def solver_sha256_hex(solver_path: str | None) -> str | None:
    if not solver_path:
        return None
    try:
        h = hashlib.sha256()
        with open(solver_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


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
    if hw.get("active_backend"):
        bits.append(f"backend={hw['active_backend']}")
    return " ".join(bits)


def _ram_gb_used() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            total_kb = None
            avail_kb = None
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
            if total_kb is not None and avail_kb is not None:
                return round((total_kb - avail_kb) / (1024 * 1024), 2)
    except OSError:
        pass
    return None


def _read_stat() -> tuple[int, ...] | None:
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        if not line.startswith("cpu "):
            return None
        parts = line.split()[1:]
        return tuple(int(x) for x in parts[:10])
    except (OSError, ValueError):
        return None


def _cpu_util_pct() -> float | None:
    try:
        import time as _time
        a = _read_stat()
        if a is None:
            return None
        _time.sleep(1.0)
        b = _read_stat()
        if b is None:
            return None
        idle_delta = b[3] - a[3]
        total_delta = sum(b) - sum(a)
        if total_delta <= 0:
            return None
        return round(100.0 * (1.0 - idle_delta / total_delta), 1)
    except Exception:
        return None


def _parse_rocm_metric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("n/a", "na", "none"):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _gpu_runtime() -> list[dict[str, Any]]:
    """Per-GPU runtime metrics for worker.report_metrics."""
    static_gpus = _enumerate_amd_gpus()
    if not static_gpus:
        probe = _probe_solver_gpu(None)
        if probe:
            static_gpus = [probe]

    j = _rocm_smi_json(["--showuse", "--showtemp", "--showpower", "--showuniqueid"])
    if not j:
        return [
            {
                "gpu_uuid": g.get("gpu_uuid"),
                "util_pct": None,
                "power_w": None,
                "temp_c": None,
            }
            for g in static_gpus
            if g.get("gpu_uuid")
        ]

    model_fallback = static_gpus[0].get("model") if static_gpus else None
    out: list[dict[str, Any]] = []
    card_idx = 0
    for card, info in j.items():
        if not str(card).lower().startswith("card"):
            continue
        low = {str(k).lower(): v for k, v in info.items()}
        uniq = str(low.get("unique id", "")).strip()
        ident = re.sub(r"[^a-z0-9]", "", uniq.lower()) if uniq else ""
        gfx = _amd_gfx_target(model_fallback=model_fallback)
        gpu_uuid = _make_amd_gpu_uuid(gfx, ident) if ident else None
        if not gpu_uuid and card_idx < len(static_gpus):
            gpu_uuid = static_gpus[card_idx].get("gpu_uuid")
        if not gpu_uuid:
            card_idx += 1
            continue

        util = None
        for k, v in low.items():
            if "gpu use" in k or "gpu activity" in k:
                parsed = _parse_rocm_metric(v)
                if parsed is not None:
                    util = int(parsed)
                break

        temp = None
        for k, v in low.items():
            if "temperature" in k and "edge" in k:
                parsed = _parse_rocm_metric(v)
                if parsed is not None:
                    temp = int(parsed)
                break
        if temp is None:
            for k, v in low.items():
                if "temperature" in k:
                    parsed = _parse_rocm_metric(v)
                    if parsed is not None:
                        temp = int(parsed)
                    break

        power = None
        for k, v in low.items():
            if "average graphics package power" in k or "current socket power" in k:
                parsed = _parse_rocm_metric(v)
                if parsed is not None:
                    power = round(parsed, 1)
                break

        out.append({
            "gpu_uuid": gpu_uuid,
            "util_pct": util,
            "power_w": power,
            "temp_c": temp,
        })
        card_idx += 1

    if not out and static_gpus:
        return [
            {
                "gpu_uuid": g.get("gpu_uuid"),
                "util_pct": None,
                "power_w": None,
                "temp_c": None,
            }
            for g in static_gpus
            if g.get("gpu_uuid")
        ]
    return out


def collect_runtime_metrics(
    session_id: str,
    solver_nps: float | None,
    shares_session_total: int,
    *,
    solver_sha256: str | None = None,
    solver_backend: str | None = None,
    wrapper_version: str | None = None,
) -> dict[str, Any]:
    """Build worker.report_metrics payload (pool uses this for vardiff + dashboard)."""
    payload: dict[str, Any] = {
        "session_id": session_id,
        "timestamp": int(__import__("time").time()),
        "cpu_util_pct": _cpu_util_pct(),
        "ram_gb_used": _ram_gb_used(),
        "gpus": _gpu_runtime(),
        "solver_nps": solver_nps,
        "shares_session_total": shares_session_total,
    }
    if wrapper_version:
        payload["wrapper_version"] = wrapper_version
    if solver_sha256:
        payload["solver_sha256"] = solver_sha256
    if solver_backend:
        payload["solver_backend"] = solver_backend
    return payload