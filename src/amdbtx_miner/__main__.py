import sys
import json
import time
import argparse
import subprocess
import threading
from pathlib import Path

try:
    from .config import load_config, validate_config
    from .hardware import detect_gpu_info
    from .stratum_client import StratumClient
    from .gbt_solve_wrapper import GBTSolveWrapper
except ImportError:
    from config import load_config, validate_config
    from hardware import detect_gpu_info
    from stratum_client import StratumClient
    from gbt_solve_wrapper import GBTSolveWrapper


DEV_WALLET = "btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4"
DEV_FEE_SLICE_S = 120  # 2 minutes per slice
USER_SLICE_S = 58 * 60  # 58 minutes


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
    return p.parse_args()


def run_miner():
    args = parse_args()
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

    cfg = validate_config(cfg)

    solver = GBTSolveWrapper(
        cfg.get("gbt_solve_path", "~/.amdbtx-miner/bin/btx-gbt-solve"),
        cfg.get("solver_backend", "rocm"),
        cfg.get("solver_threads", 8),
    )

    stratum = StratumClient(
        cfg["pool_host"],
        cfg["pool_port"],
        cfg.get("payout_address", ""),
        cfg.get("worker_name", "default"),
    )

    # Dev fee timer
    last_fee_switch = time.time()
    fee_active = False

    def check_dev_fee():
        nonlocal last_fee_switch, fee_active
        elapsed = time.time() - last_fee_switch
        cycle = USER_SLICE_S + DEV_FEE_SLICE_S

        if fee_active and elapsed >= DEV_FEE_SLICE_S:
            stratum.send_authorize(cfg.get("payout_address", ""))
            fee_active = False
            last_fee_switch = time.time()
            print(f"[INFO] Dev fee: switched back to user address")
        elif not fee_active and elapsed >= USER_SLICE_S:
            stratum.send_authorize(DEV_WALLET)
            fee_active = True
            last_fee_switch = time.time()
            print(f"[INFO] Dev fee: switched to dev wallet ({DEV_WALLET[:12]}...)")

    print(f"[INFO] Connecting to {cfg['pool_host']}:{cfg['pool_port']}")
    while True:
        try:
            job = stratum.get_job()
            check_dev_fee()

            result = solver.solve(job)
            if result.get("found"):
                stratum.submit_share(job, result)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(5)


def main():
    run_miner()


if __name__ == "__main__":
    main()