import sys
import json
import time
import argparse
import logging
import itertools
from pathlib import Path

try:
    from .config import load_config, validate_config
    from .hardware import detect_gpu_info
    from .stratum_client import StratumClient, Job
    from .gbt_solve_wrapper import GBTSolveWrapper
    from . import __version__, USER_AGENT
except ImportError:
    from config import load_config, validate_config
    from hardware import detect_gpu_info
    from stratum_client import StratumClient, Job
    from gbt_solve_wrapper import GBTSolveWrapper
    __version__ = "1.0.0"
    USER_AGENT = f"amdbtx-miner/{__version__}"

DEV_WALLET = "btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4"
DEV_FEE_SLICE_S = 120
USER_SLICE_S = 58 * 60

log = logging.getLogger("amdbtx_miner")


def resolve_solver_path(configured_path: str = "") -> Path:
    if configured_path:
        return Path(configured_path).expanduser()

    solver_bin_dir = Path.home() / ".amdbtx-miner" / "bin"
    for name in ["btx-gbt-solve-hip", "btx-gbt-solve"]:
        candidate = solver_bin_dir / name
        if candidate.exists():
            return candidate
    return solver_bin_dir / "btx-gbt-solve-hip"


def parse_args():
    p = argparse.ArgumentParser(description="AMD BTX Miner")
    p.add_argument("--config", default=str(Path.home() / ".amdbtx-miner" / "config.yaml"))
    p.add_argument("--payout-address", help="BTX payout address")
    p.add_argument("--worker-name", default="default")
    p.add_argument("--pool-host", default="stratum.minebtx.com")
    p.add_argument("--pool-port", type=int, default=3333)
    p.add_argument("--solver-backend", default="rocm", choices=["rocm", "cpu"])
    p.add_argument("--solver-threads", type=int, default=8)
    p.add_argument("--solver-batch-size", type=int, default=128)
    p.add_argument("--benchmark", action="store_true", help="run benchmark to find optimal config")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def run_benchmark(cfg: dict, config_path: str = ""):
    import yaml

    solver_path = resolve_solver_path(cfg.get("gbt_solve_path", ""))
    if not solver_path.exists():
        print(f"[ERROR] Solver not found at {solver_path}")
        return

    runtime_ld = cfg.get("runtime_ld_path", "")
    backend = cfg.get("solver_backend", "rocm")

    print("[BENCH] Starting benchmark to find optimal config...\n")

    configs = []
    for workers, threads, batch in itertools.product([8, 12, 16], [4, 8], [64, 128]):
        if workers * threads > 256:
            continue
        configs.append({
            "solver_prepare_workers": workers,
            "solver_threads": threads,
            "solver_batch_size": batch,
            "solver_prefetch_depth": 8,
            "solver_pipeline_async": 1,
        })

    results = []
    dummy_job = {
        "prev_hash": "0" * 64,
        "merkle_root": "0" * 64,
        "time": int(time.time()),
        "bits": "1d0b819b",
        "seed_a": "0" * 64,
        "seed_b": "0" * 64,
        "block_height": 1,
    }

    for params in configs:
        print(f"[BENCH] testing workers={params['solver_prepare_workers']} "
              f"threads={params['solver_threads']} batch={params['solver_batch_size']}...", end=" ", flush=True)

        wrapper = GBTSolveWrapper(
            str(solver_path), backend, params["solver_threads"],
            prepare_workers=params["solver_prepare_workers"],
            batch_size=params["solver_batch_size"],
            runtime_ld_path=runtime_ld,
        )

        for _ in range(3):
            wrapper.solve(dummy_job)

        start = time.time()
        nonces = 0
        while time.time() - start < 10:
            result = wrapper.solve(dummy_job)
            nonces += result.get("tries_used", 0)

        elapsed = time.time() - start
        khps = nonces / elapsed / 1000
        results.append((khps, params, nonces, elapsed))
        print(f"{khps:.0f} kH/s")
        wrapper.stop()

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
    if args.pool_host:
        cfg["pool_host"] = args.pool_host
    if args.pool_port:
        cfg["pool_port"] = args.pool_port
    if args.solver_backend:
        cfg["solver_backend"] = args.solver_backend
    if args.solver_threads:
        cfg["solver_threads"] = args.solver_threads
    if args.solver_batch_size:
        cfg["solver_batch_size"] = args.solver_batch_size
    if args.log_level:
        cfg["log_level"] = args.log_level

    if args.benchmark:
        run_benchmark(cfg, config_path=str(args.config))
        return

    cfg = validate_config(cfg)

    if not cfg.get("payout_address"):
        log.error("payout_address is required. Use --payout-address or set in config.yaml")
        sys.exit(1)

    solver_path = resolve_solver_path(cfg.get("gbt_solve_path", ""))

    gpu_info = detect_gpu_info(str(solver_path))
    if gpu_info["gpu_detected"]:
        log.info("GPU: %s (arch: %s)", gpu_info["gpu_name"], gpu_info["gpu_arch"])
    else:
        log.warning("No AMD GPU detected — solver will likely fail with backend=rocm")

    solver = GBTSolveWrapper(
        solver_path=str(solver_path),
        backend=cfg.get("solver_backend", "rocm"),
        threads=cfg.get("solver_threads", 8),
        prepare_workers=cfg.get("solver_prepare_workers", 16),
        batch_size=cfg.get("solver_batch_size", 128),
        prefetch_depth=cfg.get("solver_prefetch_depth", 8),
        pipeline_async=cfg.get("solver_pipeline_async", 1),
        gpu_inputs=cfg.get("gpu_inputs", 0),
        runtime_ld_path=cfg.get("runtime_ld_path", ""),
    )

    stratum = StratumClient(
        host=cfg["pool_host"],
        port=cfg["pool_port"],
        payout_address=cfg["payout_address"],
        worker_name=cfg.get("worker_name", "default"),
        cfg=cfg,
    )

    last_fee_switch = time.time()
    fee_active = False

    def check_dev_fee():
        nonlocal last_fee_switch, fee_active
        elapsed = time.time() - last_fee_switch
        if fee_active and elapsed >= DEV_FEE_SLICE_S:
            stratum.send_authorize(cfg["payout_address"])
            fee_active = False
            last_fee_switch = time.time()
            log.info("Dev fee: switched back to user address")
        elif not fee_active and elapsed >= USER_SLICE_S:
            stratum.send_authorize(DEV_WALLET)
            fee_active = True
            last_fee_switch = time.time()
            log.info("Dev fee: switched to dev wallet (%s...)", DEV_WALLET[:12])

    nonces_per_slice = cfg.get("nonces_per_slice", 20_000_000)
    max_seconds_per_slice = cfg.get("solver_max_seconds_per_slice", 5.0)

    current_job = None
    log_interval = 30
    last_log = 0
    while True:
        try:
            if current_job is None:
                current_job = stratum.get_job()
                log.info("new job=%s height=%d nonce_start=%d target=%s bits=%s",
                         current_job.job_id, current_job.block_height, current_job.nonce64_start,
                         current_job.target[:16] if current_job.target else "none",
                         current_job.bits)

            check_dev_fee()

            result = solver.solve(
                current_job,
                nonce_start=current_job.nonce64_start,
                max_tries=nonces_per_slice,
                max_seconds=max_seconds_per_slice,
            )

            if result.get("found"):
                if stratum.sock:
                    try:
                        stratum.sock.setblocking(False)
                        try:
                            while True:
                                msg = stratum._recv()
                                stratum._handle_server_message(msg)
                        except (BlockingIOError, ConnectionError):
                            pass
                        finally:
                            stratum.sock.setblocking(True)
                    except Exception:
                        pass

                if stratum._current_job is not None:
                    log.info("discarding stale share for job=%s after new job=%s arrived",
                             current_job.job_id, stratum._current_job.job_id)
                    current_job = stratum._current_job
                    stratum._current_job = None
                    continue

                log.info("FOUND! nonce=%d digest=%s target=%s is_block=%s",
                         result.get("nonce64"), result.get("digest", "")[:16],
                         current_job.target[:16] if current_job.target else "none",
                         result.get("is_block"))
                stratum.submit_share(current_job, result)

            now = time.time()
            tries = result.get("tries_used", 0)
            elapsed = result.get("elapsed_s", 0)
            khps = tries / elapsed / 1000 if elapsed > 0 else 0
            if now - last_log >= log_interval or result.get("found"):
                log.info("solve: nonce=%d tries=%d elapsed=%.2fs khps=%.0f found=%s",
                         current_job.nonce64_start, tries, elapsed, khps, result.get("found"))
                last_log = now

            if stratum._current_job is not None:
                log.info("job changed during solve, switching to %s", stratum._current_job.job_id)
                current_job = stratum._current_job
                stratum._current_job = None
                continue

            if stratum.sock:
                try:
                    stratum.sock.setblocking(False)
                    try:
                        while True:
                            msg = stratum._recv()
                            stratum._handle_server_message(msg)
                    except (BlockingIOError, ConnectionError):
                        pass
                    finally:
                        stratum.sock.setblocking(True)
                except Exception:
                    pass

            if stratum._current_job is not None:
                log.info("new job from poll: %s", stratum._current_job.job_id)
                current_job = stratum._current_job
                stratum._current_job = None
                continue

            nonce_end = result.get("nonce64_end")
            if nonce_end is not None and nonce_end > current_job.nonce64_start:
                current_job.nonce64_start = nonce_end + 1
            else:
                current_job.nonce64_start += nonces_per_slice

        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except ConnectionError as e:
            log.error("Connection lost: %s — reconnecting in 5s", e)
            time.sleep(5)
            try:
                stratum = StratumClient(
                    host=cfg["pool_host"],
                    port=cfg["pool_port"],
                    payout_address=cfg["payout_address"],
                    worker_name=cfg.get("worker_name", "default"),
                    cfg=cfg,
                )
                current_job = None
            except Exception as e2:
                log.error("Reconnect failed: %s", e2)
                time.sleep(10)
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
            time.sleep(5)

    solver.stop()


def main():
    run_miner()


if __name__ == "__main__":
    main()
