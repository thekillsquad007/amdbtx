"""YAML configuration loader for AMDBTX miner."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

_DEFAULT_GBT_SOLVE = "~/.amdbtx-miner/bin/btx-gbt-solve"


@dataclass
class MinerConfig:
    pool_host: str = "stratum.minebtx.com"
    pool_port: int = 3333
    pool_tls: bool = False

    payout_address: str = ""
    worker_name: str = "default"

    gbt_solve_path: str = _DEFAULT_GBT_SOLVE
    solver_backend: str = "rocm"
    solver_threads: int = 8
    solver_prepare_workers: int = 16
    solver_batch_size: int = 128
    solver_prefetch_depth: int = 8
    solver_pipeline_async: bool = True
    gpu_inputs: int = 0

    nonces_per_slice: int = 20_000_000
    solver_max_seconds_per_slice: float = 5.0

    reconnect_initial_s: float = 1.0
    reconnect_max_s: float = 60.0

    log_level: str = "INFO"


def fully_qualified_worker(cfg: MinerConfig) -> str:
    """Return ``payout_address.worker_name``."""
    return f"{cfg.payout_address}.{cfg.worker_name}"


def load_yaml_config(path: str | Path) -> MinerConfig:
    """Load a YAML config file and return a MinerConfig.

    Missing keys use dataclass defaults. Environment variable overrides:
        AMDBTX_POOL_HOST, AMDBTX_POOL_PORT, AMDBTX_PAYOUT_ADDRESS,
        AMDBTX_WORKER_NAME, AMDBTX_GBT_SOLVE_PATH, AMDBTX_SOLVER_BACKEND,
        AMDBTX_LOG_LEVEL.
    """
    cfg = MinerConfig()
    p = Path(path).expanduser()

    if p.exists():
        with open(p, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        for key, val in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        log.info("loaded config from %s", p)
    else:
        log.warning("config file %s not found, using defaults", p)

    # Environment variable overrides
    env_map = {
        "AMDBTX_POOL_HOST": ("pool_host", str),
        "AMDBTX_POOL_PORT": ("pool_port", int),
        "AMDBTX_PAYOUT_ADDRESS": ("payout_address", str),
        "AMDBTX_WORKER_NAME": ("worker_name", str),
        "AMDBTX_GBT_SOLVE_PATH": ("gbt_solve_path", str),
        "AMDBTX_SOLVER_BACKEND": ("solver_backend", str),
        "AMDBTX_LOG_LEVEL": ("log_level", str),
    }
    for env_key, (attr, cast) in env_map.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                setattr(cfg, attr, cast(env_val))
            except (ValueError, TypeError):
                log.warning("invalid env override %s=%s for %s", env_key, env_val, attr)

    return cfg
