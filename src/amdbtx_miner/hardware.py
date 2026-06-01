"""Hardware fingerprint collection for AMD GPUs via rocm-smi."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    """Run a command with timeout, return stdout or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        log.debug("command not found: %s", cmd[0])
    except subprocess.TimeoutExpired:
        log.debug("command timed out: %s", " ".join(cmd))
    except Exception as exc:
        log.debug("command failed: %s: %s", " ".join(cmd), exc)
    return None


def _rocm_smi_query(flags: List[str]) -> Optional[str]:
    """Run rocm-smi with given flags."""
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is None:
        return None
    return _run([rocm_smi] + flags)


def _parse_vram_total_kb(text: str) -> Optional[int]:
    """Parse 'Total Memory' from rocm-smi --showmeminfo vram output."""
    for line in text.splitlines():
        lower = line.lower().strip()
        if "total" in lower and ("memory" in lower or "vram" in lower):
            for token in lower.split():
                try:
                    return int(token)
                except ValueError:
                    continue
    return None


def amd_gpu_query() -> List[Dict[str, Any]]:
    """Query AMD GPUs: model, VRAM, compute units, gfx arch.

    Returns list of dicts with: model, vram_gb, compute_units, arch, gpu_id.
    """
    gpus: List[Dict[str, Any]] = []

    # Detect number of GPUs via rocm-smi device list
    show_id_out = _rocm_smi_query(["--showid"])
    gpu_ids: List[int] = []
    if show_id_out:
        for line in show_id_out.splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                gpu_ids.append(int(stripped))
    if not gpu_ids:
        # Fallback: assume at least one GPU if rocm-smi exists
        if shutil.which("rocm-smi") is not None:
            gpu_ids = [0]
        else:
            return []

    for gid in gpu_ids:
        info: Dict[str, Any] = {
            "gpu_id": gid,
            "model": None,
            "vram_gb": None,
            "compute_units": None,
            "arch": None,
        }

        # Product name
        prod_out = _rocm_smi_query(["--showproductname", "-d", str(gid)])
        if prod_out:
            for line in prod_out.splitlines():
                stripped = line.strip()
                if stripped and ":" not in stripped:
                    info["model"] = stripped
                    break
                elif ":" in stripped:
                    parts = stripped.split(":", 1)
                    if len(parts) == 2 and parts[1].strip():
                        info["model"] = parts[1].strip()
                        break

        # VRAM
        vram_out = _rocm_smi_query(["--showmeminfo", "vram", "-d", str(gid)])
        if vram_out:
            total_kb = _parse_vram_total_kb(vram_out)
            if total_kb is not None:
                info["vram_gb"] = round(total_kb / (1024 * 1024), 1)

        # Architecture from showclockinfo or showperfanalyzer
        clk_out = _rocm_smi_query(["--showclockinfo", "-d", str(gid)])
        if clk_out:
            for line in clk_out.splitlines():
                lower = line.lower()
                if "gfx" in lower:
                    # Extract e.g. "gfx1030" or "gfx906"
                    for token in lower.split():
                        if token.startswith("gfx"):
                            info["arch"] = token.strip()
                            break
                    break

        # Compute units — try showperf or showcu
        cu_out = _rocm_smi_query(["--showperfanalyzer", "-d", str(gid)])
        if cu_out:
            for line in cu_out.splitlines():
                lower = line.lower()
                if "compute unit" in lower or "cu" in lower:
                    for token in line.split():
                        try:
                            info["compute_units"] = int(token)
                            break
                        except ValueError:
                            continue

        gpus.append(info)

    return gpus


def _cpu_info() -> Dict[str, Any]:
    """Collect basic CPU info."""
    info: Dict[str, Any] = {
        "model": None,
        "cores_logical": os.cpu_count(),
        "arch": platform.machine(),
    }
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        info["model"] = platform.processor() or None
    return info


def _ram_info() -> Dict[str, Any]:
    """Collect RAM info from /proc/meminfo."""
    info: Dict[str, Any] = {"total_gb": None, "available_gb": None}
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                val_kb = int(parts[1])
                if key == "MemTotal":
                    info["total_gb"] = round(val_kb / (1024 * 1024), 1)
                elif key == "MemAvailable":
                    info["available_gb"] = round(val_kb / (1024 * 1024), 1)
    except (OSError, ValueError):
        pass
    return info


def _detect_container() -> Optional[str]:
    """Detect if running inside a container."""
    if os.path.exists("/.dockerenv"):
        return "docker"
    try:
        with open("/proc/1/cgroup", "r") as fh:
            content = fh.read()
            if "docker" in content:
                return "docker"
            if "lxc" in content:
                return "lxc"
            if "kubepods" in content:
                return "kubernetes"
    except OSError:
        pass
    # Check for podman
    if os.path.exists("/run/.containerenv"):
        return "podman"
    return None


def collect_static_hardware(
    miner_version: str,
    cpu_threads_allocated: Optional[int] = None,
    solver_env: Any = None,
    solver_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build hardware dict for mining.subscribe.

    Includes CPU info, RAM, AMD GPU details, OS, container detection.
    """
    cpu = _cpu_info()
    ram = _ram_info()
    gpus = amd_gpu_query()

    hw: Dict[str, Any] = {
        "miner_version": miner_version,
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "container": _detect_container(),
        "cpu_model": cpu["model"],
        "cpu_cores_logical": cpu["cores_logical"],
        "cpu_threads_allocated": cpu_threads_allocated or cpu["cores_logical"],
        "ram_total_gb": ram["total_gb"],
        "ram_available_gb": ram["available_gb"],
        "gpus": gpus,
        "gpu_count": len(gpus),
    }

    # Solver binary info
    if solver_path:
        expanded = os.path.expanduser(solver_path)
        hw["solver_path"] = expanded
        hw["solver_exists"] = os.path.isfile(expanded)
        hw["solver_executable"] = os.access(expanded, os.X_OK) if hw["solver_exists"] else False
    else:
        hw["solver_path"] = None
        hw["solver_exists"] = False
        hw["solver_executable"] = False

    if solver_env is not None:
        hw["solver_backend"] = getattr(solver_env, "backend", "rocm")
        hw["solver_threads"] = getattr(solver_env, "solver_threads", None)
        hw["solver_batch_size"] = getattr(solver_env, "batch_size", None)
    else:
        hw["solver_backend"] = "rocm"
        hw["solver_threads"] = None
        hw["solver_batch_size"] = None

    return hw


def collect_runtime_metrics(
    session_id: str,
    solver_nps: int,
    shares_session_total: int,
) -> Dict[str, Any]:
    """Build worker.report_metrics payload.

    Includes CPU util, RAM used, per-GPU util/power/temp via rocm-smi.
    """
    metrics: Dict[str, Any] = {
        "session_id": session_id,
        "shares_session_total": shares_session_total,
        "solver_nps": solver_nps,
    }

    # CPU utilization (1-second sample)
    try:
        load1, _load5, _load15 = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        metrics["cpu_load_1m"] = round(load1 / cpu_count * 100, 1)
    except OSError:
        metrics["cpu_load_1m"] = None

    # RAM usage
    ram = _ram_info()
    metrics["ram_total_gb"] = ram["total_gb"]
    metrics["ram_available_gb"] = ram["available_gb"]
    if ram["total_gb"] and ram["available_gb"] and ram["total_gb"] > 0:
        metrics["ram_used_gb"] = round(ram["total_gb"] - ram["available_gb"], 1)
        metrics["ram_pct"] = round(
            (1 - ram["available_gb"] / ram["total_gb"]) * 100, 1
        )
    else:
        metrics["ram_used_gb"] = None
        metrics["ram_pct"] = None

    # Per-GPU metrics
    gpu_metrics: List[Dict[str, Any]] = []
    gpus = amd_gpu_query()
    for gpu in gpus:
        gid = gpu["gpu_id"]
        gm: Dict[str, Any] = {"gpu_id": gid}

        # GPU utilization
        use_out = _rocm_smi_query(["--showuse", "-d", str(gid)])
        if use_out:
            for line in use_out.splitlines():
                lower = line.lower()
                if "gpu use" in lower or "use %" in lower or "utilization" in lower:
                    for token in line.replace("%", "").split():
                        try:
                            gm["gpu_util_pct"] = int(token)
                            break
                        except ValueError:
                            continue

        # Temperature
        temp_out = _rocm_smi_query(["--showtemp", "-d", str(gid)])
        if temp_out:
            for line in temp_out.splitlines():
                lower = line.lower()
                if "edge" in lower or "sensor" in lower or "temp" in lower:
                    for token in line.replace("c", "").replace("C", "").split():
                        try:
                            gm["temp_c"] = float(token)
                            break
                        except ValueError:
                            continue

        # Power
        power_out = _rocm_smi_query(["--showpower", "-d", str(gid)])
        if power_out:
            for line in power_out.splitlines():
                lower = line.lower()
                if "power" in lower or "watt" in lower:
                    for token in line.replace("W", "").replace("w", "").split():
                        try:
                            gm["power_w"] = float(token)
                            break
                        except ValueError:
                            continue

        # VRAM usage
        vram_out = _rocm_smi_query(["--showmeminfo", "vram", "-d", str(gid)])
        if vram_out:
            for line in vram_out.splitlines():
                lower = line.lower()
                if "used" in lower or "current" in lower:
                    parts = lower.split()
                    for i, token in enumerate(parts):
                        if token in ("used", "current") and i + 1 < len(parts):
                            try:
                                gm["vram_used_kb"] = int(parts[i + 1])
                                break
                            except ValueError:
                                continue

        gpu_metrics.append(gm)

    metrics["gpus"] = gpu_metrics
    metrics["timestamp"] = time.time()
    return metrics


def hardware_summary_string(hw: Dict[str, Any]) -> str:
    """One-line human summary for startup log."""
    parts: List[str] = []

    cpu_model = hw.get("cpu_model") or "unknown CPU"
    parts.append(cpu_model)

    ram_gb = hw.get("ram_total_gb")
    if ram_gb:
        parts.append(f"{ram_gb}GB RAM")

    gpu_count = hw.get("gpu_count", 0)
    if gpu_count > 0:
        gpus = hw.get("gpus", [])
        models = [g.get("model") or f"GPU#{g.get('gpu_id')}" for g in gpus]
        parts.append(f"{gpu_count}x AMD GPU: {', '.join(models)}")
    else:
        parts.append("no AMD GPUs detected")

    backend = hw.get("solver_backend", "rocm")
    parts.append(f"backend={backend}")

    return " | ".join(parts)
