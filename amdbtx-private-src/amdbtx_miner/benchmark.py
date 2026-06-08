"""Benchmark tool for AMDBTX miner."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

from amdbtx_miner import USER_AGENT
from amdbtx_miner.config import MinerConfig, load_yaml_config
from amdbtx_miner.gbt_solve_wrapper import GbtSolveWrapper, SolveChallenge, SolverEnv

log = logging.getLogger(__name__)


def _make_bench_challenge() -> SolveChallenge:
    """Build a synthetic challenge for benchmarking."""
    return SolveChallenge(
        version=0x20000000,
        prev_hash="0000000000000000000000000000000000000000000000000000000000000000",
        merkle_root="",
        time=1779672814,
        bits="1d17c609",
        seed_a="0000000000000000000000000000000000000000000000000000000000000000",
        seed_b="0000000000000000000000000000000000000000000000000000000000000000",
        block_height=100000,
        share_target_hex="00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )


async def run_benchmark(
    cfg: MinerConfig,
    duration_s: float = 30.0,
    backend: str = "rocm",
    slice_tries: int = 2_000_000,
) -> None:
    """Run the solver for `duration_s` seconds and report hash rate."""
    solver_env = SolverEnv(
        batch_size=cfg.solver_batch_size,
        prefetch_depth=cfg.solver_prefetch_depth,
        prepare_workers=cfg.solver_prepare_workers,
        pipeline_async=cfg.solver_pipeline_async,
        gpu_inputs=cfg.gpu_inputs,
        solver_threads=cfg.solver_threads,
        backend=backend,
    )

    wrapper = GbtSolveWrapper(
        gbt_solve_path=cfg.gbt_solve_path,
        backend=backend,
        solver_env=solver_env,
    )

    challenge = _make_bench_challenge()

    print(f"AMDBTX Benchmark — backend={backend} duration={duration_s:.0f}s")
    print(f"  solver_threads={cfg.solver_threads} batch_size={cfg.solver_batch_size}")
    print(f"  prepare_workers={cfg.solver_prepare_workers} gpu_inputs={cfg.gpu_inputs}")
    print()

    total_tries = 0
    total_slices = 0
    wall_start = time.monotonic()
    next_report = wall_start + 5.0

    try:
        while True:
            wall_now = time.monotonic()
            elapsed = wall_now - wall_start
            if elapsed >= duration_s:
                break

            remaining = duration_s - elapsed
            max_s = min(remaining, cfg.solver_max_seconds_per_slice, 5.0)

            result = await wrapper.solve_slice(
                challenge,
                nonce_start=total_tries,
                max_tries=slice_tries,
                max_seconds=max_s,
            )

            total_tries += result.tries_used
            total_slices += 1

            # Report every 5 seconds
            if wall_now >= next_report:
                elapsed_now = time.monotonic() - wall_start
                nps = total_tries / elapsed_now if elapsed_now > 0 else 0
                slices_per_s = total_slices / elapsed_now if elapsed_now > 0 else 0
                print(
                    f"  [{elapsed_now:6.1f}s] "
                    f"tries={total_tries:,} "
                    f"slices={total_slices} "
                    f"nps={nps:,.0f} "
                    f"slices/s={slices_per_s:.1f} "
                    f"last_slice={result.tries_used:,} tries in {result.elapsed_s:.2f}s"
                )
                next_report = wall_now + 5.0

    except KeyboardInterrupt:
        print("\nbenchmark interrupted")
    finally:
        await wrapper.close()

    wall_total = time.monotonic() - wall_start
    avg_nps = total_tries / wall_total if wall_total > 0 else 0
    print()
    print(f"Results:")
    print(f"  Total tries:  {total_tries:,}")
    print(f"  Total slices: {total_slices}")
    print(f"  Wall time:    {wall_total:.2f}s")
    print(f"  Avg nps:      {avg_nps:,.0f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="amdbtx-miner benchmark",
        description="Benchmark the AMDBTX solver",
    )
    parser.add_argument(
        "--config", "-c",
        default="~/.amdbtx-miner/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Benchmark duration in seconds",
    )
    parser.add_argument(
        "--backend", "-b",
        default="rocm",
        choices=["rocm", "hip", "cpu"],
        help="Solver backend (default: rocm)",
    )
    parser.add_argument(
        "--slice-tries",
        type=int,
        default=2_000_000,
        help="Max tries per solver slice",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Log level",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_yaml_config(args.config)

    asyncio.run(
        run_benchmark(
            cfg=cfg,
            duration_s=args.duration,
            backend=args.backend,
            slice_tries=args.slice_tries,
        )
    )


if __name__ == "__main__":
    main()
