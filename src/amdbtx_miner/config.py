import yaml
from pathlib import Path


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def validate_config(cfg: dict) -> dict:
    defaults = {
        "pool_host": "stratum.minebtx.com",
        "pool_port": 3333,
        "pool_tls": False,
        "payout_address": "",
        "worker_name": "default",
        "gbt_solve_path": "",
        "solver_backend": "rocm",
        "solver_threads": 8,
        "solver_prepare_workers": 16,
        "solver_batch_size": 128,
        "solver_prefetch_depth": 8,
        "solver_pipeline_async": 1,
        "gpu_device": -1,
        "gpu_inputs": 0,
        "nonces_per_slice": 20000000,
        "solver_max_seconds_per_slice": 5.0,
        "reconnect_initial_s": 1.0,
    "reconnect_max_s": 60.0,
    "log_level": "INFO",
    "runtime_ld_path": "",
}
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return cfg