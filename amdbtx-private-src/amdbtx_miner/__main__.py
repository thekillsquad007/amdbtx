"""CLI entry point for amdbtx-miner."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from amdbtx_miner import USER_AGENT, __version__
from amdbtx_miner.config import MinerConfig, load_yaml_config
from amdbtx_miner.hardware import collect_static_hardware, hardware_summary_string
from amdbtx_miner.stratum_client import StratumClient

log = logging.getLogger("amdbtx_miner")


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amdbtx-miner",
        description="AMD GPU native stratum miner for BTX (MineBtx pool)",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"amdbtx-miner {__version__}",
    )
    parser.add_argument(
        "--config", "-c",
        default="~/.amdbtx-miner/config.yaml",
        help="Path to YAML config file (default: ~/.amdbtx-miner/config.yaml)",
    )
    parser.add_argument(
        "--pool-host",
        help="Pool hostname (overrides config)",
    )
    parser.add_argument(
        "--pool-port",
        type=int,
        help="Pool port (overrides config)",
    )
    parser.add_argument(
        "--pool-tls",
        action="store_true",
        default=None,
        help="Enable TLS to pool (overrides config)",
    )
    parser.add_argument(
        "--payout-address",
        help="BTX payout address (overrides config)",
    )
    parser.add_argument(
        "--worker-name",
        help="Worker name (overrides config)",
    )
    parser.add_argument(
        "--backend", "-b",
        default=None,
        choices=["rocm", "hip", "cpu"],
        help="Solver backend (default: rocm)",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        help="Solver threads per GPU (overrides config)",
    )
    parser.add_argument(
        "--solver-batch-size",
        type=int,
        help="Solver batch size (overrides config)",
    )
    parser.add_argument(
        "--solver-prepare-workers",
        type=int,
        help="Solver prepare workers (overrides config)",
    )
    parser.add_argument(
        "--gbt-solve-path",
        help="Path to btx-gbt-solve binary (overrides config)",
    )
    parser.add_argument(
        "--gpu-inputs",
        type=int,
        default=None,
        help="GPU input mode: 0=CPU-gen inputs (default: 0)",
    )
    parser.add_argument(
        "--nonces-per-slice",
        type=int,
        help="Nonces per solver slice (overrides config)",
    )
    parser.add_argument(
        "--max-seconds-per-slice",
        type=float,
        help="Max seconds per solver slice (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        help="Log level: DEBUG, INFO, WARNING, ERROR (overrides config)",
    )

    # Subcommands
    sub = parser.add_subparsers(dest="command")
    bench_parser = sub.add_parser("benchmark", help="Run solver benchmark")
    bench_parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Benchmark duration in seconds",
    )
    bench_parser.add_argument(
        "--backend", "-b",
        default="rocm",
        choices=["rocm", "hip", "cpu"],
        help="Solver backend for benchmark",
    )
    bench_parser.add_argument(
        "--config", "-c",
        default="~/.amdbtx-miner/config.yaml",
        help="Path to config YAML",
    )
    bench_parser.add_argument(
        "--slice-tries",
        type=int,
        default=2_000_000,
        help="Max tries per solver slice",
    )

    return parser


def _apply_cli_overrides(cfg: MinerConfig, args: argparse.Namespace) -> MinerConfig:
    """Apply CLI arguments that override config values."""
    if args.pool_host is not None:
        cfg.pool_host = args.pool_host
    if args.pool_port is not None:
        cfg.pool_port = args.pool_port
    if args.pool_tls is not None:
        cfg.pool_tls = args.pool_tls
    if args.payout_address is not None:
        cfg.payout_address = args.payout_address
    if args.worker_name is not None:
        cfg.worker_name = args.worker_name
    if args.backend is not None:
        cfg.solver_backend = args.backend
    if args.solver_threads is not None:
        cfg.solver_threads = args.solver_threads
    if args.solver_batch_size is not None:
        cfg.solver_batch_size = args.solver_batch_size
    if args.solver_prepare_workers is not None:
        cfg.solver_prepare_workers = args.solver_prepare_workers
    if args.gbt_solve_path is not None:
        cfg.gbt_solve_path = args.gbt_solve_path
    if args.gpu_inputs is not None:
        cfg.gpu_inputs = args.gpu_inputs
    if args.nonces_per_slice is not None:
        cfg.nonces_per_slice = args.nonces_per_slice
    if args.max_seconds_per_slice is not None:
        cfg.solver_max_seconds_per_slice = args.max_seconds_per_slice
    if args.log_level is not None:
        cfg.log_level = args.log_level
    return cfg


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Handle benchmark subcommand
    if args.command == "benchmark":
        from amdbtx_miner.benchmark import main as bench_main
        bench_main()
        return

    # Load config
    cfg = load_yaml_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    _setup_logging(cfg.log_level)

    # Validate payout address
    if not cfg.payout_address:
        print(
            "Error: payout_address is required. "
            "Set it in config.yaml or use --payout-address.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print startup banner
    log.info("%s starting", USER_AGENT)
    hw = collect_static_hardware(
        miner_version=USER_AGENT,
        cpu_threads_allocated=cfg.solver_threads,
        solver_path=cfg.gbt_solve_path,
    )
    log.info("hardware: %s", hardware_summary_string(hw))
    log.info(
        "pool=%s:%d tls=%s address=%s worker=%s backend=%s",
        cfg.pool_host, cfg.pool_port, cfg.pool_tls,
        cfg.payout_address, cfg.worker_name, cfg.solver_backend,
    )

    # Run the miner
    client = StratumClient(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown_handler(sig: int, frame: object) -> None:
        log.info("received signal %d, shutting down", sig)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        loop.run_until_complete(client.run_forever())
    except asyncio.CancelledError:
        log.info("miner stopped")
    except KeyboardInterrupt:
        log.info("miner interrupted")
    finally:
        loop.run_until_complete(client.stop())
        loop.close()

    log.info("exit")


if __name__ == "__main__":
    main()
