import yaml
from pathlib import Path


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def validate_config(cfg: dict) -> dict:
    defaults = {
        "mining_mode": "pool",
        "pool_host": "stratum.minebtx.com",
        "pool_port": 3333,
        "pool_tls": False,
        "rpc_url": "http://127.0.0.1:19334",
        "rpc_user": "",
        "rpc_password": "",
        "rpc_cookie_file": "",
        "rpc_timeout": 120.0,
        "btx_cli_path": "",
        "coinbase_script_pubkey": "",
        "gbt_longpoll": True,
        "gbt_longpoll_timeout": 60.0,
        "payout_address": "",
        "worker_name": "default",
        "gbt_solve_path": "",
        "solver_backend": "rocm",
        "solver_threads": 8,
        "solver_prepare_workers": 16,
        "solver_batch_size": 1024,
        "solver_prefetch_depth": 8,
        "solver_pipeline_async": 1,
        "gpu_device": -1,
        "gpu_devices": None,
        "gpu_inputs": 0,
        "nonces_per_slice": 20000000,
        "solver_max_seconds_per_slice": 5.0,
        "reconnect_initial_s": 1.0,
    "reconnect_max_s": 60.0,
    "log_level": "INFO",
    "runtime_ld_path": "",
    "solo_dev_fee_bps": 200,
    "dev_wallet": "",
}
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return cfg