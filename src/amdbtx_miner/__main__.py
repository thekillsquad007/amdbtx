import sys
import json
import time
import argparse
import logging
import itertools
from pathlib import Path

try:
    from .config import load_config, validate_config
    from .hardware import detect_gpu_info, pick_best_gpu_index
    from .stratum_client import StratumClient, Job
    from .solo_client import SoloClient
    from .gbt_solve_wrapper import GBTSolveWrapper
    from . import __version__, USER_AGENT
except ImportError:
    from config import load_config, validate_config
    from hardware import detect_gpu_info, pick_best_gpu_index
    from stratum_client import StratumClient, Job
    from solo_client import SoloClient
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
    p.add_argument("--solver-backend", default=None, choices=["rocm", "cpu"])
    p.add_argument("--solver-threads", type=int, default=None)
    p.add_argument("--solver-batch-size", type=int, default=None)
    p.add_argument("--gpu-device", type=int, default=-1, help="GPU device index (-1 = auto)")
    p.add_argument("--benchmark", action="store_true", help="run benchmark to find optimal config")
    p.add_argument("--solo", action="store_true", help="solo mine against a btxd node (local or remote)")
    p.add_argument("--rpc-url", default=None, help="btxd JSON-RPC URL (solo mode), e.g. http://192.168.1.15:19334")
    p.add_argument("--rpc-user", default=None, help="btxd RPC username (solo mode, required for remote nodes)")
    p.add_argument("--rpc-password", default=None, help="btxd RPC password (solo mode)")
    p.add_argument("--rpc-cookie-file", default=None, help="path to btxd .cookie file (solo mode, local node only)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _benchmark_job() -> dict:
    """Representative mainnet job (height >= 125000 uses the V2 HIP path)."""
    return {
        "version": 536870912,
        "prev_hash": "51619e6d8d37ab84bf7b9b8a6a8100d6fc1b92d2a6473b2bf153681a416215a1",
        "merkle_root": "f58785dbeb5a7033daa54958364388273cbf363cb50e3bcb0d2879e18e8bfeff",
        "time": int(time.time()),
        "bits": "1d1ccc7b",
        "seed_a": "43b5b748c3ad0928e56256e7c687c4907745220ba7053bc56905942c9a0fa1b2",
        "seed_b": "1190c8ed806ea11336f3ad6a20adb9da0beb7b05772afab86c38c67c919ae645",
        "block_height": 125601,
        "matmul_n": 512,
        "matmul_b": 16,
        "matmul_r": 8,
        "epsilon_bits": 18,
        "share_target": "00007331ec000000000000000000000000000000000000000000000000000000",
    }


def run_benchmark(cfg: dict, config_path: str = ""):
    import yaml

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

    print("[BENCH] Starting benchmark to find optimal batch size...")
    print(f"[BENCH] workers={bench_workers} threads={bench_threads} "
          f"slice={bench_tries} nonces / {bench_seconds:.0f}s per test\n")

    configs = []
    for batch in (256, 512, 1024, 2048, 4096):
        configs.append({
            "solver_prepare_workers": bench_workers,
            "solver_threads": bench_threads,
            "solver_batch_size": batch,
            "solver_prefetch_depth": cfg.get("solver_prefetch_depth", 8),
            "solver_pipeline_async": cfg.get("solver_pipeline_async", 1),
        })

    results = []
    bench_job = _benchmark_job()

    for params in configs:
        print(f"[BENCH] testing batch={params['solver_batch_size']}...", end=" ", flush=True)

        wrapper = GBTSolveWrapper(
            str(solver_path), backend, params["solver_threads"],
            prepare_workers=params["solver_prepare_workers"],
            batch_size=params["solver_batch_size"],
            runtime_ld_path=runtime_ld,
        )

        hip_ok = True
        for _ in range(2):
            warmup = wrapper.solve(bench_job, max_tries=bench_tries, max_seconds=bench_seconds)
            if warmup.get("backend") == "cpu":
                hip_ok = False
                break

        if hip_ok:
            start = time.time()
            nonces = 0
            while time.time() - start < bench_seconds:
                result = wrapper.solve(bench_job, max_tries=bench_tries, max_seconds=bench_seconds)
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


def _make_solver(cfg: dict, solver_path: Path, gpu_device: int) -> GBTSolveWrapper:
    return GBTSolveWrapper(
        solver_path=str(solver_path),
        backend=cfg.get("solver_backend", "rocm"),
        threads=cfg.get("solver_threads", 8),
        prepare_workers=cfg.get("solver_prepare_workers", 16),
        batch_size=cfg.get("solver_batch_size", 1024),
        prefetch_depth=cfg.get("solver_prefetch_depth", 8),
        pipeline_async=cfg.get("solver_pipeline_async", 1),
        gpu_inputs=cfg.get("gpu_inputs", 0),
        gpu_device=gpu_device,
        runtime_ld_path=cfg.get("runtime_ld_path", ""),
    )


def run_mining_loop(client, solver: GBTSolveWrapper, cfg: dict, *, solo: bool = False):
    fee_check = None
    if not solo:
        last_fee_switch = time.time()
        fee_active = False

        def fee_check():
            nonlocal last_fee_switch, fee_active
            elapsed = time.time() - last_fee_switch
            if fee_active and elapsed >= DEV_FEE_SLICE_S:
                client.send_authorize(cfg["payout_address"])
                fee_active = False
                last_fee_switch = time.time()
                log.info("Dev fee: switched back to user address")
            elif not fee_active and elapsed >= USER_SLICE_S:
                client.send_authorize(DEV_WALLET)
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
                nonce64_start=current_job.nonce64_start,
                clean_jobs=current_job.clean_jobs,
                received_at=current_job.received_at,
            )

            result = solver.solve(
                solve_job,
                nonce_start=solve_job.nonce64_start,
                max_tries=nonces_per_slice,
                max_seconds=max_seconds_per_slice,
            )

            if result.get("found"):
                if not solo and getattr(client, "sock", None):
                    try:
                        client.sock.setblocking(False)
                        try:
                            while True:
                                msg = client._recv()
                                client._handle_server_message(msg)
                        except (BlockingIOError, ConnectionError):
                            pass
                        finally:
                            client.sock.setblocking(True)
                    except Exception:
                        pass

                if not solo and client._current_job is not None:
                    rotated = client._current_job.job_id != solve_job.job_id
                    if rotated:
                        log.info(
                            "submitting found share for rotated job_id %s (current=%s); "
                            "pool JobCache handles same-parent staleness",
                            solve_job.job_id, client._current_job.job_id,
                        )

                log.info(
                    "FOUND! nonce=%d digest=%s target=%s is_block=%s job=%s",
                    result.get("nonce64"), result.get("digest", "")[:16],
                    solve_job.target[:16] if solve_job.target else "none",
                    result.get("is_block"), solve_job.job_id,
                )
                if solo:
                    if result.get("is_block"):
                        client.submit_share(solve_job, result)
                else:
                    client.submit_share(solve_job, result)

            now = time.time()
            tries = result.get("tries_used", 0)
            elapsed = result.get("elapsed_s", 0)
            khps = tries / elapsed / 1000 if elapsed > 0 else 0
            if result.get("error"):
                log.warning("solver error: %s", result["error"])

            if now - last_log >= log_interval or result.get("found") or result.get("error"):
                backend = result.get("backend", "?")
                a = getattr(client, "shares_accepted", 0)
                r = getattr(client, "shares_rejected", 0)
                log.info(
                    "solve: nonce=%d tries=%d elapsed=%.2fs khps=%.0f backend=%s "
                    "found=%s shares=%d/%d",
                    current_job.nonce64_start, tries, elapsed, khps, backend,
                    result.get("found"), a, r,
                )
                last_log = now

            if client._current_job is not None:
                new_job = client._current_job
                client._current_job = None
                if current_job.should_replace(new_job):
                    log.info(
                        "job replaced: %s -> %s (clean=%s height=%d)",
                        current_job.job_id, new_job.job_id, new_job.clean_jobs,
                        new_job.block_height,
                    )
                    current_job = new_job
                    continue
                current_job.merge_from(new_job)

            if solo:
                client.poll_template()
            elif getattr(client, "sock", None):
                try:
                    client.sock.setblocking(False)
                    try:
                        while True:
                            msg = client._recv()
                            client._handle_server_message(msg)
                    except (BlockingIOError, ConnectionError):
                        pass
                    finally:
                        client.sock.setblocking(True)
                except Exception:
                    pass

            if client._current_job is not None:
                new_job = client._current_job
                client._current_job = None
                if current_job.should_replace(new_job):
                    log.info(
                        "new job: %s (clean=%s height=%d)",
                        new_job.job_id, new_job.clean_jobs, new_job.block_height,
                    )
                    current_job = new_job
                    continue
                log.debug(
                    "job updated: %s -> %s (nonce=%d)",
                    current_job.job_id, new_job.job_id, current_job.nonce64_start,
                )
                current_job.merge_from(new_job)

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
    if args.pool_host:
        cfg["pool_host"] = args.pool_host
    if args.pool_port:
        cfg["pool_port"] = args.pool_port
    if args.solver_backend is not None:
        cfg["solver_backend"] = args.solver_backend
    if args.solver_threads is not None:
        cfg["solver_threads"] = args.solver_threads
    if args.solver_batch_size is not None:
        cfg["solver_batch_size"] = args.solver_batch_size
    if args.gpu_device >= 0:
        cfg["gpu_device"] = args.gpu_device
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
        for i, g in enumerate(gpu_info.get("gpus", [])):
            log.info("  GPU %d: %s arch=%s vram=%sGB", i, g.get("model", "?"),
                     g.get("compute_capability", "?"), g.get("vram_gb", "?"))
    else:
        log.warning("No AMD GPU detected — solver will likely fail with backend=rocm")

    gpu_device = cfg.get("gpu_device", -1)
    if gpu_device < 0:
        gpus = gpu_info.get("gpus", [])
        if gpus:
            gpu_device = pick_best_gpu_index(gpus)
            if len(gpus) > 1:
                log.info("auto-selected GPU %d (dGPU with most VRAM)", gpu_device)

    solver = _make_solver(cfg, solver_path, gpu_device)

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
