from __future__ import annotations

import sys
import json
import time
import argparse
import logging
import itertools
from collections import deque
from pathlib import Path

try:
    from .config import load_config, validate_config
    from .hardware import detect_gpu_info, resolve_gpu_devices
    from .stratum_client import StratumClient, Job
    from .solo_client import SoloClient
    from .gbt_solve_wrapper import GBTSolveWrapper, MultiGPUSolver
    from . import __version__, USER_AGENT, DEV_WALLET
except ImportError:
    from config import load_config, validate_config
    from hardware import detect_gpu_info, resolve_gpu_devices
    from stratum_client import StratumClient, Job
    from solo_client import SoloClient
    from gbt_solve_wrapper import GBTSolveWrapper, MultiGPUSolver
    __version__ = "1.0.0"
    USER_AGENT = f"amdbtx-miner/{__version__}"
    DEV_WALLET = "btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4"

DEV_FEE_SLICE_S = 120
USER_SLICE_S = 58 * 60

log = logging.getLogger("amdbtx_miner")

# Pool dedupes (job_id, ntime, nonce); skip local resubmits after rotation/races.
_submitted_share_keys: set[tuple[str, int, int]] = set()


def _target_probability_from_hex(target_hex: str | None) -> float:
    """Return final-digest pass probability for a compact/share target hex."""
    if not target_hex:
        return 0.0
    target_str = str(target_hex).strip().lower()
    if target_str.startswith("0x"):
        target_str = target_str[2:]
    target_str = target_str[:64].ljust(64, "0")
    try:
        target = int(target_str, 16)
    except (TypeError, ValueError):
        return 0.0
    if target <= 0:
        return 0.0
    return min(target / float(1 << 256), 1.0)


def _expected_shares_from_gate(gate_passes: int, target_hex: str | None) -> float:
    """Expected accepted shares after the sigma gate for the current share target."""
    if gate_passes <= 0:
        return 0.0
    return float(gate_passes) * _target_probability_from_hex(target_hex)


def _format_share_eta(gate_per_s: float, target_hex: str | None) -> str:
    """Format expected wall-clock time to the next share at the observed gate rate."""
    probability = _target_probability_from_hex(target_hex)
    shares_per_s = max(float(gate_per_s), 0.0) * probability
    if shares_per_s <= 0:
        return "unknown"
    seconds = 1.0 / shares_per_s
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24.0:.1f}d"


def resolve_solver_path(configured_path: str = "") -> Path:
    if configured_path:
        return Path(configured_path).expanduser()

    # PyInstaller bundle: solver next to the executable
    if getattr(sys, "frozen", False):
        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(sys.executable).parent
        for name in ["btx-gbt-solve-hip.exe", "btx-gbt-solve-hip"]:
            candidate = bundle_dir / name
            if candidate.exists():
                return candidate

    solver_bin_dir = Path.home() / ".amdbtx-miner" / "bin"
    for name in ["btx-gbt-solve-hip.exe", "btx-gbt-solve-hip", "btx-gbt-solve"]:
        candidate = solver_bin_dir / name
        if candidate.exists():
            return candidate
    return solver_bin_dir / "btx-gbt-solve-hip"


def parse_args():
    p = argparse.ArgumentParser(description="AMD BTX Miner")
    p.add_argument("--config", default=str(Path.home() / ".amdbtx-miner" / "config.yaml"))
    p.add_argument("--payout-address", help="BTX payout address")
    p.add_argument("--worker-name", default=None)
    p.add_argument("--pool-host", default=None, help="pool hostname (default: config or btx-sg.lproute.com)")
    p.add_argument("--pool-port", type=int, default=None)
    p.add_argument("--solver-backend", default=None, choices=["rocm", "cpu"])
    p.add_argument("--solver-threads", type=int, default=None)
    p.add_argument("--solver-batch-size", type=int, default=None)
    p.add_argument("--gpu-device", type=int, default=-1, help="GPU device index (-1 = auto, single GPU)")
    p.add_argument(
        "--gpu-devices",
        default=None,
        help="Comma-separated GPU indices or 'all' for multi-GPU (e.g. 0,1). Overrides --gpu-device.",
    )
    p.add_argument("--benchmark", action="store_true", help="run benchmark to find optimal config")
    p.add_argument("--auto-tune", action="store_true",
                   help="force batch-size sweep and exit (saves to config)")
    p.add_argument("--no-auto-tune", action="store_true",
                   help="skip the automatic batch-size sweep at startup")
    p.add_argument("--auto-tune-seconds", type=int, default=3, metavar="N",
                   help="seconds per batch size in auto-tune (default: 3)")
    p.add_argument(
        "--experimental-rdna4",
        action="store_true",
        help="enable untested RDNA4 (gfx1200/gfx1201) WMMA fast path; requires solver built with gfx12",
    )
    p.add_argument("--solo", action="store_true", help="solo mine against a btxd node (local or remote)")
    p.add_argument("--rpc-url", default=None, help="btxd JSON-RPC URL (solo mode), e.g. http://192.168.1.15:19334")
    p.add_argument("--rpc-user", default=None, help="btxd RPC username (solo mode, required for remote nodes)")
    p.add_argument("--rpc-password", default=None, help="btxd RPC password (solo mode)")
    p.add_argument("--rpc-cookie-file", default=None, help="path to btxd .cookie file (solo mode, local node only)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _benchmark_job() -> dict:
    """Representative post-v3 mainnet job for the HIP fast path."""
    return {
        "version": 536870912,
        "prev_hash": "51619e6d8d37ab84bf7b9b8a6a8100d6fc1b92d2a6473b2bf153681a416215a1",
        "merkle_root": "f58785dbeb5a7033daa54958364388273cbf363cb50e3bcb0d2879e18e8bfeff",
        "time": int(time.time()),
        "bits": "1c4916ad",
        "seed_a": "43b5b748c3ad0928e56256e7c687c4907745220ba7053bc56905942c9a0fa1b2",
        "seed_b": "1190c8ed806ea11336f3ad6a20adb9da0beb7b05772afab86c38c67c919ae645",
        "block_height": 147000,
        "parent_mtp": 1782910000,
        "matmul_n": 512,
        "matmul_b": 16,
        "matmul_r": 8,
        "epsilon_bits": 18,
        "share_target": "00007331ec000000000000000000000000000000000000000000000000000000",
    }


def run_benchmark(cfg: dict, config_path: str = ""):
    import yaml

    def _parse_batch_sizes(value):
        if value is None:
            return [
                131072, 262144, 524288, 1048576,
                2097152, 4194304, 8388608, 16777216,
            ]
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
        else:
            parts = list(value)
        sizes = sorted({int(p) for p in parts if str(p).strip()})
        return [s for s in sizes if s > 0]

    solver_path = resolve_solver_path(cfg.get("gbt_solve_path", ""))
    if not solver_path.exists():
        print(f"[ERROR] Solver not found at {solver_path}")
        return

    runtime_ld = cfg.get("runtime_ld_path", "")
    backend = cfg.get("solver_backend", "rocm")
    bench_workers = cfg.get("solver_prepare_workers", 16)
    bench_threads = cfg.get("solver_threads", 8)
    bench_seconds = float(cfg.get("benchmark_seconds", 8.0))
    bench_tries = int(cfg.get("benchmark_tries_per_slice", 2_000_000))
    batch_opts = _parse_batch_sizes(cfg.get("benchmark_batch_sizes"))
    sweep_threads = bool(cfg.get("benchmark_sweep_threads", False))

    print("[BENCH] Starting benchmark to find optimal batch size...")
    print(f"[BENCH] workers={bench_workers} threads={bench_threads} "
          f"slice={bench_tries} nonces / {bench_seconds:.0f}s per test\n")

    configs = []
    thread_opts = sorted({bench_threads, 8, 12, 16, 20, 24}) if sweep_threads else [bench_threads]
    worker_opts = sorted({bench_workers, 8, 12, 16, 20, 24}) if sweep_threads else [bench_workers]
    for workers in worker_opts:
        for threads in thread_opts:
            for batch in batch_opts:
                configs.append({
                    "solver_prepare_workers": workers,
                    "solver_threads": threads,
                    "solver_batch_size": batch,
                    "solver_prefetch_depth": cfg.get("solver_prefetch_depth", 8),
                    "solver_pipeline_async": cfg.get("solver_pipeline_async", 1),
                })

    results = []
    bench_job = _benchmark_job()

    for params in configs:
        print(f"[BENCH] testing batch={params['solver_batch_size']}...", end=" ", flush=True)
        tries_for_batch = max(bench_tries, int(params["solver_batch_size"]))

        wrapper = GBTSolveWrapper(
            str(solver_path), backend, params["solver_threads"],
            prepare_workers=params["solver_prepare_workers"],
            batch_size=params["solver_batch_size"],
            runtime_ld_path=runtime_ld,
        )

        hip_ok = True
        for _ in range(2):
            warmup = wrapper.solve(bench_job, max_tries=tries_for_batch, max_seconds=bench_seconds)
            if warmup.get("backend") == "cpu":
                hip_ok = False
                break

        if hip_ok:
            start = time.time()
            nonces = 0
            while time.time() - start < bench_seconds:
                result = wrapper.solve(bench_job, max_tries=tries_for_batch, max_seconds=bench_seconds)
                if result.get("backend") == "cpu":
                    hip_ok = False
                    break
                nonces += result.get("tries_used", 0)

            elapsed = time.time() - start
            if hip_ok and elapsed > 0 and nonces > 0:
                khps = nonces / elapsed / 1000
                results.append((khps, params, nonces, elapsed))
                print(f"{khps:.0f} kH/s")
            else:
                print("CPU fallback (HIP failed) — skipped")

        else:
            print("CPU fallback (HIP failed) — skipped")

        wrapper.stop()

    if not results:
        print("\n[BENCH] No successful HIP runs — check GPU/ROCm and that block_height >= 125000")
        return None

    results.sort(reverse=True)
    best = results[0]
    print(f"\n[BENCH] Best config: workers={best[1]['solver_prepare_workers']} "
          f"threads={best[1]['solver_threads']} batch={best[1]['solver_batch_size']} "
          f"-> {best[0]:.0f} kH/s")
    print("[BENCH] All results:")
    for khps, params, n, e in results:
        print(f" {khps:>8.0f} kH/s w={params['solver_prepare_workers']:>2} "
              f"t={params['solver_threads']} b={params['solver_batch_size']:>3}")

    cp = Path(config_path) if config_path else Path.home() / ".amdbtx-miner" / "config.yaml"
    if cp.exists():
        with open(cp) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        yaml_cfg.update(best[1])
        with open(cp, "w") as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False)
        print(f"\n[BENCH] Updated {cp} with optimal settings")
    return best[1]


def auto_tune(cfg: dict, batch_sizes: list[int] | None = None,
              seconds_per_test: float = 3.0,
              config_path: str = "",
              force: bool = False) -> int:
    """Quick batch-size sweep; returns best solver_batch_size, updates cfg.

    When *force* is False and *cfg* already carries a tuned solver_batch_size
    (one of the test candidates), the sweep is skipped.
    """
    solver_path = resolve_solver_path(cfg.get("gbt_solve_path", ""))
    if not solver_path.exists():
        log.warning("auto-tune: solver not found at %s", solver_path)
        return int(cfg.get("solver_batch_size", 4194304))

    # Use config's benchmark_batch_sizes if caller didn't supply one.
    if batch_sizes is None:
        raw = cfg.get("benchmark_batch_sizes")
        if isinstance(raw, list) and all(isinstance(v, int) for v in raw):
            batch_sizes = raw
        else:
            batch_sizes = [131072, 262144, 524288, 1048576,
                           2097152, 4194304, 8388608, 16777216]

    # Short-circuit: already tuned (unless forced).
    current = int(cfg.get("solver_batch_size", 0) or 0)
    if not force and current in batch_sizes and cfg.get("auto_tuned", False):
        log.info("auto-tune: solver_batch_size=%d already tuned — skipping", current)
        return current

    backend = cfg.get("solver_backend", "rocm")
    workers = int(cfg.get("solver_prepare_workers", 16))
    threads = int(cfg.get("solver_threads", 8))
    runtime_ld = cfg.get("runtime_ld_path", "")

    log.info("auto-tune: sweeping batch sizes (%.0fs per test)...", seconds_per_test)
    results: list[tuple[float, int]] = []
    bench_job = _benchmark_job()
    best_so_far, best_batch = 0.0, batch_sizes[0]

    for batch in batch_sizes:
        tries = max(batch * 2, 500_000)
        wrapper = GBTSolveWrapper(
            str(solver_path), backend, threads,
            prepare_workers=workers, batch_size=batch,
            runtime_ld_path=runtime_ld,
        )
        hip_ok = True
        for _ in range(2):
            warmup = wrapper.solve(bench_job, max_tries=tries // 2, max_seconds=2.0)
            if warmup.get("backend") == "cpu":
                hip_ok = False
                break

        if not hip_ok:
            log.info("  batch=%d CPU fallback — skip", batch)
            wrapper.stop()
            continue

        start = time.time()
        nonces = 0
        while time.time() - start < seconds_per_test:
            r = wrapper.solve(bench_job, max_tries=tries, max_seconds=seconds_per_test)
            if r.get("backend") == "cpu":
                hip_ok = False
                break
            nonces += r.get("tries_used", 0)

        elapsed = time.time() - start
        wrapper.stop()

        if hip_ok and elapsed > 0 and nonces > 0:
            khps = nonces / elapsed / 1000
            results.append((khps, batch))
            marker = " <<" if khps > best_so_far else ""
            if khps > best_so_far:
                best_so_far, best_batch = khps, batch
            log.info("  batch=%d %9d  %8.0f kH/s%s", batch, nonces, khps, marker)

    if not results:
        log.warning("auto-tune: no batch size produced HIP results; keeping current value")
        return int(cfg.get("solver_batch_size", 4194304))

    log.info("auto-tune: best batch=%d (%s%.0f kH/s)", best_batch,
             "matches current " if best_batch == int(cfg.get("solver_batch_size", 0)) else "",
             best_so_far)

    # Only update if best is at least 3% faster than the fallback default
    fallback = int(cfg.get("solver_batch_size", 4194304))
    if best_batch != fallback and fallback in [b for _, b in results]:
        fallback_khps = next((kh for kh, b in results if b == fallback), 0)
        if fallback_khps > 0 and (best_so_far - fallback_khps) / fallback_khps < 0.03:
            log.info("auto-tune: best is within 3%% of current (%d); keeping current",
                     fallback)
            return fallback

    cfg["solver_batch_size"] = best_batch
    cfg["auto_tuned"] = True

    if config_path:
        cp = Path(config_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        try:
            yaml_cfg = yaml.safe_load(cp.read_text()) if cp.exists() else {}
            if not isinstance(yaml_cfg, dict):
                yaml_cfg = {}
            yaml_cfg["solver_batch_size"] = best_batch
            yaml_cfg["auto_tuned"] = True
            cp.write_text(yaml.dump(yaml_cfg, default_flow_style=False))
            log.info("auto-tune: saved solver_batch_size=%d to %s", best_batch, cp)
        except Exception as e:
            log.warning("auto-tune: failed to save config: %s", e)

    return best_batch


def _make_solver(cfg: dict, solver_path: Path, gpu_devices: list[int]) -> MultiGPUSolver:
    return MultiGPUSolver(
        solver_path=str(solver_path),
        backend=cfg.get("solver_backend", "rocm"),
        threads=cfg.get("solver_threads", 8),
        prepare_workers=cfg.get("solver_prepare_workers", 16),
        batch_size=cfg.get("solver_batch_size", 1024),
        prefetch_depth=cfg.get("solver_prefetch_depth", 8),
        pipeline_async=cfg.get("solver_pipeline_async", 1),
        gpu_inputs=cfg.get("gpu_inputs", 0),
        gpu_devices=gpu_devices,
        runtime_ld_path=cfg.get("runtime_ld_path", ""),
        experimental_rdna4_wmma=bool(cfg.get("experimental_rdna4_wmma")),
    )


def _format_khps_line(
    result: dict, nonce_khps: float, matmul_khps: float | None = None,
) -> str:
    matmul_part = ""
    if matmul_khps is not None and matmul_khps > 0:
        per_gpu_matmul = result.get("per_gpu_matmul_khps")
        if per_gpu_matmul and len(per_gpu_matmul) > 1:
            parts_m = "+".join(f"{v:.2f}" for v in per_gpu_matmul)
            matmul_part = f" matmul_khps={matmul_khps:.2f} total ({parts_m} per GPU)"
        else:
            matmul_part = f" matmul_khps={matmul_khps:.2f}"
    per_gpu = result.get("per_gpu_khps")
    if per_gpu and len(per_gpu) > 1:
        parts = "+".join(f"{v:.2f}" for v in per_gpu)
        return f"nonce_khps={nonce_khps:.2f} total ({parts} per GPU){matmul_part}"
    return f"nonce_khps={nonce_khps:.2f}{matmul_part}"


def _drain_pool_messages(client) -> None:
    if hasattr(client, "process_available_messages"):
        try:
            client.process_available_messages()
        except (BlockingIOError, TimeoutError):
            pass
        except ConnectionError:
            raise
        return
    if not getattr(client, "sock", None):
        return
    try:
        client.sock.setblocking(False)
        try:
            while True:
                msg = client._recv()
                if hasattr(client, "_dispatch_message"):
                    client._dispatch_message(msg)
                else:
                    client._handle_server_message(msg)
        except BlockingIOError:
            pass
        finally:
            client.sock.setblocking(True)
    except ConnectionError:
        raise
    except Exception:
        pass


def _share_submit_key(solve_job: Job, result: dict) -> tuple[str, int, int]:
    ntime = int(result.get("ntime") or solve_job.time)
    nonce = int(result["nonce64"])
    return (solve_job.job_id, ntime, nonce)


def _submit_pool_share(client, solve_job: Job, result: dict, *, solo: bool) -> bool:
    _drain_pool_messages(client)
    if not solo:
        key = _share_submit_key(solve_job, result)
        if key in _submitted_share_keys:
            log.info(
                "skip resubmit job=%s nonce=%016x (already submitted this session)",
                solve_job.job_id, key[2],
            )
            return False
    share_ntime = int(result.get("ntime") or solve_job.time)
    if (
        not solo
        and getattr(client, "_protocol", "stratum") == "luckypool"
        and share_ntime != int(solve_job.time)
    ):
        log.warning(
            "luckypool: skip submit — share ntime=%d differs from job ntime=%d; "
            "pool recomputes digest from the job header",
            share_ntime, solve_job.time,
        )
        return False
    if not solo and getattr(client, "_protocol", "stratum") == "luckypool":
        nonce_bits = int(getattr(solve_job, "luckypool_nonce_bits", 0) or 0)
        if nonce_bits > 0:
            nonce = int(result["nonce64"])
            lane_mask = ~((1 << nonce_bits) - 1) & ((1 << 64) - 1)
            if (nonce & lane_mask) != (int(solve_job.nonce64_start) & lane_mask):
                log.warning(
                    "luckypool: skip submit — nonce64=%d outside assigned lane start=%d bits=%d",
                    nonce, solve_job.nonce64_start, nonce_bits,
                )
                return False
    if not solo:
        _submitted_share_keys.add(_share_submit_key(solve_job, result))
    if not solo and client._current_job is not None:
        if client._current_job.job_id != solve_job.job_id:
            if (
                getattr(client, "_protocol", "stratum") == "luckypool"
                and (
                    client._current_job.block_height != solve_job.block_height
                    or client._current_job.prev_hash != solve_job.prev_hash
                )
            ):
                log.info(
                    "luckypool: drop stale share job=%s current=%s height=%d/%d",
                    solve_job.job_id, client._current_job.job_id,
                    solve_job.block_height, client._current_job.block_height,
                )
                return False
            log.info(
                "submitting found share for rotated job_id %s (current=%s); "
                "pool JobCache handles same-parent staleness",
                solve_job.job_id, client._current_job.job_id,
            )
    log.info(
        "FOUND! nonce=%d digest=%s target=%s is_block=%s job=%s ntime=%s",
        result.get("nonce64"), result.get("digest", ""),
        solve_job.target[:16] if solve_job.target else "none",
        result.get("is_block"), solve_job.job_id,
        result.get("ntime", solve_job.time),
    )
    wait_submit = bool(result.get("is_block")) if solo else (
        getattr(client, "_protocol", "stratum") == "luckypool"
    )
    client.submit_share(solve_job, result, wait=wait_submit)
    return True


def _accumulate_slice_stats(merged: dict, result: dict) -> None:
    merged["tries_used"] = merged.get("tries_used", 0) + int(result.get("tries_used", 0) or 0)
    merged["gate_passes"] = merged.get("gate_passes", 0) + int(result.get("gate_passes", 0) or 0)
    merged["words_hits"] = merged.get("words_hits", 0) + int(result.get("words_hits", 0) or 0)
    merged["cpu_verify_misses"] = (
        merged.get("cpu_verify_misses", 0) + int(result.get("cpu_verify_misses", 0) or 0)
    )
    if result.get("backend"):
        merged["backend"] = result["backend"]
    if result.get("per_gpu_khps"):
        merged["per_gpu_khps"] = result["per_gpu_khps"]
    if result.get("per_gpu_matmul_khps"):
        merged["per_gpu_matmul_khps"] = result["per_gpu_matmul_khps"]
    if result.get("num_gpus"):
        merged["num_gpus"] = result["num_gpus"]
    if result.get("error") and not merged.get("error"):
        merged["error"] = result["error"]
    elapsed = float(result.get("elapsed_s", 0) or 0)
    if elapsed > 0:
        merged["solver_elapsed_s"] = merged.get("solver_elapsed_s", 0.0) + elapsed


def _solve_slice_continuous(
    solver: MultiGPUSolver,
    client,
    solve_job: Job,
    *,
    solo: bool,
    nonce_start: int,
    nonces_per_slice: int,
    max_seconds_per_slice: float,
    max_shares_per_slice: int = 0,
) -> dict:
    """Scan a work slice; submit shares up to max_shares_per_slice (0 = unlimited)."""
    slice_t0 = time.time()
    slice_deadline = slice_t0 + max_seconds_per_slice
    cursor = nonce_start
    budget = nonces_per_slice
    shares_in_slice = 0
    merged: dict = {
        "found": False,
        "tries_used": 0,
        "gate_passes": 0,
        "words_hits": 0,
        "cpu_verify_misses": 0,
    }

    _drain_pool_messages(client)
    while budget > 0 and time.time() < slice_deadline:
        remaining_sec = max(0.05, slice_deadline - time.time())
        result = solver.solve(
            solve_job,
            nonce_start=cursor,
            max_tries=budget,
            max_seconds=remaining_sec,
        )
        _accumulate_slice_stats(merged, result)

        if result.get("found"):
            merged["found"] = True
            share_results = result.get("solutions") or [result]
            for share_result in share_results:
                if solo and not share_result.get("is_block"):
                    continue
                if max_shares_per_slice and shares_in_slice >= max_shares_per_slice:
                    break
                if _submit_pool_share(client, solve_job, share_result, solo=solo):
                    shares_in_slice += 1

            nonce_end = result.get("nonce64_end")
            if nonce_end is not None:
                cursor = int(nonce_end) + 1
            else:
                cursor = int(result.get("nonce64", cursor)) + 1
            used = int(result.get("tries_used", 0) or 0)
            budget = max(0, budget - (used if used > 0 else 1))
            if max_shares_per_slice and shares_in_slice >= max_shares_per_slice:
                break
            continue

        nonce_end = result.get("nonce64_end")
        if nonce_end is not None and int(nonce_end) >= cursor:
            cursor = int(nonce_end) + 1
            used = int(result.get("tries_used", 0) or 0)
            if used > 0:
                budget = max(0, budget - used)
        break

    merged["elapsed_s"] = time.time() - slice_t0
    merged["shares_in_slice"] = shares_in_slice
    merged["nonce64_end"] = cursor - 1 if cursor > nonce_start else nonce_start
    tries_work = merged.get("tries_used") or 0
    gate_work = merged.get("gate_passes") or 0
    solver_elapsed = merged.get("solver_elapsed_s") or merged["elapsed_s"]
    if solver_elapsed > 0:
        if tries_work > 0:
            merged["nonce_khps_total"] = tries_work / solver_elapsed / 1000
        if gate_work > 0:
            merged["matmul_khps_total"] = gate_work / solver_elapsed / 1000
    if shares_in_slice > 1:
        log.info(
            "slice submitted %d shares (continuous-feed; pool credits each at current vardiff)",
            shares_in_slice,
        )
    return merged


def run_mining_loop(client, solver: MultiGPUSolver, cfg: dict, *, solo: bool = False):
    fee_check = None
    if not solo:
        last_fee_switch = time.time()
        fee_active = False

        def fee_check():
            nonlocal last_fee_switch, fee_active
            elapsed = time.time() - last_fee_switch
            if fee_active and elapsed >= DEV_FEE_SLICE_S:
                client.send_authorize(cfg["payout_address"], cfg.get("worker_name", "default"))
                fee_active = False
                last_fee_switch = time.time()
                log.info("Dev fee: switched back to user address")
            elif not fee_active and elapsed >= USER_SLICE_S:
                client.send_authorize(DEV_WALLET, cfg.get("worker_name", "default"))
                fee_active = True
                last_fee_switch = time.time()
                log.info("Dev fee: switched to dev wallet (%s...)", DEV_WALLET[:12])

    nonces_per_slice = cfg.get("nonces_per_slice", 20_000_000)
    max_seconds_per_slice = cfg.get("solver_max_seconds_per_slice", 5.0)
    pool_max_shares = int(cfg.get("pool_max_shares_per_slice", 0)) if not solo else 0
    solver_batch_size = int(cfg.get("solver_batch_size", 0) or 0)
    if solver_batch_size > 1 and nonces_per_slice > solver_batch_size:
        aligned_nonces = (int(nonces_per_slice) // solver_batch_size) * solver_batch_size
        if aligned_nonces != nonces_per_slice:
            log.info(
                "aligning nonces_per_slice %d -> %d to avoid partial GPU batch "
                "(solver_batch_size=%d)",
                nonces_per_slice, aligned_nonces, solver_batch_size,
            )
            nonces_per_slice = aligned_nonces
    if not solo and pool_max_shares < 0:
        pool_max_shares = 0
    if not solo and pool_max_shares == 1:
        log.info(
            "pool mode: limiting to 1 submitted share per slice; "
            "set pool_max_shares_per_slice=0 to submit every valid solver result"
        )
    current_job = None
    log_interval = 30
    last_log = 0
    if not solo and hasattr(client, "start_metrics_reporter"):
        client.start_metrics_reporter(solver)

    while True:
        try:
            if current_job is None:
                current_job = client.get_job()
                log.info(
                    "new job=%s height=%d nonce_start=%d share_target=%s bits=%s eps=%d",
                    current_job.job_id, current_job.block_height, current_job.nonce64_start,
                    current_job.target[:16] if current_job.target else "none",
                    current_job.bits, current_job.epsilon_bits,
                )

            if fee_check:
                fee_check()

            solve_job = Job(
                job_id=current_job.job_id,
                version=current_job.version,
                prev_hash=current_job.prev_hash,
                merkle_root=current_job.merkle_root,
                time=current_job.time,
                bits=current_job.bits,
                target=current_job.target,
                seed_a=current_job.seed_a,
                seed_b=current_job.seed_b,
                block_height=current_job.block_height,
                matmul_n=current_job.matmul_n,
                matmul_b=current_job.matmul_b,
                matmul_r=current_job.matmul_r,
                epsilon_bits=current_job.epsilon_bits,
                parent_mtp=current_job.parent_mtp,
                nonce64_start=current_job.nonce64_start,
                clean_jobs=current_job.clean_jobs,
                received_at=current_job.received_at,
                luckypool_nonce_bits=current_job.luckypool_nonce_bits,
            )

            result = _solve_slice_continuous(
                solver,
                client,
                solve_job,
                solo=solo,
                nonce_start=solve_job.nonce64_start,
                nonces_per_slice=nonces_per_slice,
                max_seconds_per_slice=max_seconds_per_slice,
                max_shares_per_slice=pool_max_shares,
            )

            now = time.time()
            tries = result.get("tries_used", 0)
            gate_passes = result.get("gate_passes", 0)
            elapsed = result.get("elapsed_s", 0)
            # tries_used = sigma scans; gate_passes = full matmul work (post ε gate).
            if result.get("nonce_khps_total") is not None:
                nonce_khps = float(result["nonce_khps_total"])
            else:
                nonce_khps = tries / elapsed / 1000 if elapsed > 0 else 0
            matmul_khps = None
            if result.get("matmul_khps_total") is not None:
                matmul_khps = float(result["matmul_khps_total"])
            elif gate_passes and elapsed > 0:
                matmul_khps = gate_passes / elapsed / 1000
            gate_per_s = gate_passes / elapsed if elapsed > 0 else 0.0
            gate_ppm = (gate_passes / tries * 1_000_000) if tries else 0.0
            expected_shares = _expected_shares_from_gate(gate_passes, current_job.target)
            share_eta = _format_share_eta(gate_per_s, current_job.target)
            if result.get("error"):
                log.warning("solver error: %s", result["error"])

            words_hits = result.get("words_hits", 0)
            cpu_verify_misses = result.get("cpu_verify_misses", 0)
            if words_hits or cpu_verify_misses:
                log.info(
                    "digest path: words_hits=%d cpu_verify_misses=%d share_target=%s",
                    words_hits, cpu_verify_misses,
                    (current_job.target[:16] if current_job.target else "none"),
                )

            if tries > 0 and elapsed > 0:
                solver_elapsed = float(
                    result.get("solver_elapsed_s") or elapsed
                )
                slice_nps = tries / elapsed
                gpu_nps = tries / solver_elapsed if solver_elapsed > 0 else slice_nps
                observed_nps = gpu_nps if gpu_nps > 0 else slice_nps
                ema_prev = float(getattr(solver, "_ema_nonce_nps", 0) or 0)
                ema = observed_nps if ema_prev <= 0 else (ema_prev * 0.7 + observed_nps * 0.3)
                solver._ema_nonce_nps = ema
                peak = max(
                    float(getattr(solver, "_peak_nonce_nps", 0) or 0),
                    slice_nps,
                    gpu_nps,
                )
                solver._peak_nonce_nps = peak
                if hasattr(solver, "last_observed_nps"):
                    solver.last_observed_nps = ema

            if now - last_log >= log_interval or result.get("found") or result.get("error"):
                backend = result.get("backend", "?")
                a = getattr(client, "shares_accepted", 0)
                r = getattr(client, "shares_rejected", 0)
                khps_line = _format_khps_line(result, nonce_khps, matmul_khps)
                gpu_tag = f" gpus={solver.num_gpus}" if solver.num_gpus > 1 else ""
                pool_note = ""
                if not solo:
                    pool_diff = getattr(client, "_difficulty", None)
                    submit_worker = getattr(client, "_submit_worker", "") or getattr(client, "worker_name", "")
                    canonical = getattr(client, "_canonical_worker_name", "")
                    worker_tag = submit_worker or "?"
                    if canonical and canonical != submit_worker:
                        worker_tag = f"{submit_worker} ({canonical})"
                    if pool_diff is not None:
                        pool_note = (
                            f" pool_diff={pool_diff:g} worker={worker_tag}"
                            " (dashboard n/s = shares × per-share credit;"
                            " nonce_khps does not appear on pool; vardiff must ramp)"
                        )
                    if hasattr(client, "pool_credit_stats"):
                        stats = client.pool_credit_stats(60.0)
                        pool_note += (
                            f" pool_est_60s=a{int(stats['accepted'])}"
                            f" diff_avg={stats['avg_diff']:.6g}"
                            f" credit/min={stats['credit_per_min']:.6g}"
                        )
                slice_shares = result.get("shares_in_slice", 0)
                slice_tag = f" slice_shares={slice_shares}" if slice_shares else ""
                nonce_end = result.get("nonce64_end")
                if nonce_end is None:
                    nonce_end = current_job.nonce64_start + max(tries - 1, 0)
                log.info(
                    "solve: nonce=%d nonce_end=%d tries=%d gate=%d gate/s=%.0f "
                    "gate_ppm=%.2f exp_shares=%.6f share_eta=%s elapsed=%.2fs "
                    "%s backend=%s%s found=%s shares=%d/%d target=%s%s%s",
                    current_job.nonce64_start, nonce_end, tries, gate_passes, gate_per_s,
                    gate_ppm, expected_shares, share_eta, elapsed,
                    khps_line, backend, gpu_tag, result.get("found"), a, r,
                    (current_job.target[:12] if current_job.target else "none"),
                    pool_note, slice_tag,
                )
                last_log = now

            if client._current_job is not None:
                new_job = client._current_job
                client._current_job = None
                if getattr(client, "_protocol", "stratum") == "luckypool":
                    if new_job.job_id != current_job.job_id:
                        log.info(
                            "luckypool job replaced: %s -> %s (clean=%s height=%d)",
                            current_job.job_id, new_job.job_id,
                            new_job.clean_jobs, new_job.block_height,
                        )
                    _submitted_share_keys.clear()
                    current_job = new_job
                    continue
                if new_job.block_height != current_job.block_height:
                    log.info(
                        "job replaced: %s -> %s (clean=%s height=%d)",
                        current_job.job_id, new_job.job_id, new_job.clean_jobs,
                        new_job.block_height,
                    )
                    _submitted_share_keys.clear()
                    current_job = new_job
                    continue
                if new_job.clean_jobs and new_job.job_id != current_job.job_id:
                    log.info(
                        "job updated: %s -> %s (clean=%s height=%d nonce=%d)",
                        current_job.job_id, new_job.job_id, new_job.clean_jobs,
                        new_job.block_height, current_job.nonce64_start,
                    )
                current_job.merge_from(new_job)

            if solo:
                client.poll_template()
            elif getattr(client, "sock", None):
                _drain_pool_messages(client)

            if client._current_job is not None:
                new_job = client._current_job
                client._current_job = None
                if getattr(client, "_protocol", "stratum") == "luckypool":
                    if new_job.job_id != current_job.job_id:
                        log.info(
                            "luckypool job updated: %s -> %s (clean=%s height=%d)",
                            current_job.job_id, new_job.job_id,
                            new_job.clean_jobs, new_job.block_height,
                        )
                    _submitted_share_keys.clear()
                    current_job = new_job
                    continue
                if new_job.block_height != current_job.block_height:
                    log.info(
                        "new job: %s (clean=%s height=%d)",
                        new_job.job_id, new_job.clean_jobs, new_job.block_height,
                    )
                    _submitted_share_keys.clear()
                    current_job = new_job
                    continue
                if new_job.clean_jobs and new_job.job_id != current_job.job_id:
                    log.info(
                        "job updated: %s -> %s (clean=%s height=%d nonce=%d)",
                        current_job.job_id, new_job.job_id, new_job.clean_jobs,
                        new_job.block_height, current_job.nonce64_start,
                    )
                else:
                    log.debug(
                        "job updated: %s -> %s (nonce=%d)",
                        current_job.job_id, new_job.job_id, current_job.nonce64_start,
                    )
                current_job.merge_from(new_job)

            if solver.num_gpus > 1:
                next_nonce_start = solve_job.nonce64_start + nonces_per_slice * solver.num_gpus
            else:
                nonce_end = result.get("nonce64_end")
                if nonce_end is not None and nonce_end > solve_job.nonce64_start:
                    next_nonce_start = nonce_end + 1
                else:
                    next_nonce_start = solve_job.nonce64_start + nonces_per_slice

            if current_job.prev_hash == solve_job.prev_hash:
                current_job.nonce64_start = next_nonce_start

        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except ConnectionError as e:
            log.error("Connection lost: %s — reconnecting in 5s", e)
            time.sleep(5)
            try:
                if solo:
                    client = SoloClient(cfg)
                else:
                    client = StratumClient(
                        host=cfg["pool_host"],
                        port=cfg["pool_port"],
                        payout_address=cfg["payout_address"],
                        worker_name=cfg.get("worker_name", "default"),
                        cfg=cfg,
                    )
                    if hasattr(client, "start_metrics_reporter"):
                        client.start_metrics_reporter(solver)
                _submitted_share_keys.clear()
                current_job = None
            except Exception as e2:
                log.error("Reconnect failed: %s", e2)
                time.sleep(10)
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
            time.sleep(5)

    solver.stop()


def run_miner():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(Path(args.config)) if Path(args.config).exists() else {}

    if args.payout_address:
        cfg["payout_address"] = args.payout_address
    if args.worker_name:
        cfg["worker_name"] = args.worker_name
    if args.pool_host is not None:
        cfg["pool_host"] = args.pool_host
    if args.pool_port is not None:
        cfg["pool_port"] = args.pool_port
    if args.solver_backend is not None:
        cfg["solver_backend"] = args.solver_backend
    if args.solver_threads is not None:
        cfg["solver_threads"] = args.solver_threads
    if args.solver_batch_size is not None:
        cfg["solver_batch_size"] = args.solver_batch_size
    if args.gpu_device >= 0:
        cfg["gpu_device"] = args.gpu_device
    if args.gpu_devices is not None:
        cfg["gpu_devices"] = args.gpu_devices
    if args.log_level:
        cfg["log_level"] = args.log_level
    if args.solo:
        cfg["mining_mode"] = "solo"
    if args.rpc_url:
        cfg["rpc_url"] = args.rpc_url
    if args.rpc_user:
        cfg["rpc_user"] = args.rpc_user
    if args.rpc_password:
        cfg["rpc_password"] = args.rpc_password
    if args.rpc_cookie_file:
        cfg["rpc_cookie_file"] = args.rpc_cookie_file
    if args.experimental_rdna4:
        cfg["experimental_rdna4_wmma"] = True

    if args.benchmark:
        run_benchmark(cfg, config_path=str(args.config))
        return

    if args.auto_tune:
        auto_tune(cfg, seconds_per_test=max(2.0, float(args.auto_tune_seconds)),
                  config_path=str(args.config), force=True)
        return

    if not args.no_auto_tune:
        auto_tune(cfg, seconds_per_test=max(2.0, float(args.auto_tune_seconds)),
                  config_path=str(args.config))

    cfg = validate_config(cfg)

    if cfg.get("experimental_rdna4_wmma"):
        log.warning(
            "experimental RDNA4 WMMA enabled — untested on gfx1200/gfx1201; "
            "set experimental_rdna4_wmma: false in config if shares fail"
        )

    if int(cfg.get("solver_batch_size", 128) or 128) > 16777216:
        log.warning(
            "solver_batch_size=%s is very large; run --benchmark to confirm VRAM headroom",
            cfg["solver_batch_size"],
        )

    if not cfg.get("payout_address"):
        log.error("payout_address is required. Use --payout-address or set in config.yaml")
        sys.exit(1)

    solver_path = resolve_solver_path(cfg.get("gbt_solve_path", ""))

    gpu_info = detect_gpu_info(str(solver_path))
    if gpu_info["gpu_detected"]:
        log.info("GPU: %s (arch: %s)", gpu_info["gpu_name"], gpu_info["gpu_arch"])
        for i, g in enumerate(gpu_info.get("gpus", [])):
            log.info("  GPU %d: %s arch=%s vram=%sGB", i, g.get("model", "?"),
                     g.get("compute_capability", "?"), g.get("vram_gb", "?"))
    else:
        log.warning("No AMD GPU detected — solver will likely fail with backend=rocm")

    gpu_devices = resolve_gpu_devices(cfg, gpu_info)
    if len(gpu_devices) > 1 and cfg.get("solver_backend", "rocm") != "cpu":
        from .hardware import _probe_alive_gpu_indices
        alive = _probe_alive_gpu_indices(str(solver_path), total=len(gpu_devices))
        if alive:
            filtered = [i for i in gpu_devices if i in alive]
            skipped = [i for i in gpu_devices if i not in alive]
            if filtered and len(filtered) != len(gpu_devices):
                log.warning(
                    "GPU alive probe: skipped dead indices %s (alive %s); using %s",
                    skipped, alive, filtered,
                )
                gpu_devices = filtered
    if len(gpu_devices) == 1:
        log.info("mining on GPU %d", gpu_devices[0])
    else:
        log.info("multi-GPU mining on devices %s (hashrate stacks)", gpu_devices)

    solver = _make_solver(cfg, solver_path, gpu_devices)

    mining_mode = cfg.get("mining_mode", "pool")
    if mining_mode == "solo":
        log.info("solo mining mode: rpc=%s", cfg.get("rpc_url"))
        client = SoloClient(cfg)
        run_mining_loop(client, solver, cfg, solo=True)
        return

    client = StratumClient(
        host=cfg["pool_host"],
        port=cfg["pool_port"],
        payout_address=cfg["payout_address"],
        worker_name=cfg.get("worker_name", "default"),
        cfg=cfg,
    )
    run_mining_loop(client, solver, cfg, solo=False)


def main():
    run_miner()


if __name__ == "__main__":
    main()
