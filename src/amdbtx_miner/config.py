import yaml
from pathlib import Path


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def fully_qualified_worker(payout_address: str, worker_name: str) -> str:
    """The address.worker_name string used on mining.authorize and mining.submit."""
    if not payout_address:
        raise ValueError("payout_address must be set")
    name = (worker_name or "default").strip() or "default"
    return f"{payout_address}.{name}"


def validate_config(cfg: dict) -> dict:
    defaults = {
        "mining_mode": "pool",
        "pool_host": "127.0.0.1",
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
        "solver_threads": 16,
        "solver_prepare_workers": 16,
        "solver_batch_size": 4194304,
        "benchmark_batch_sizes": [
            131072, 262144, 524288, 1048576,
            2097152, 4194304, 8388608, 16777216,
        ],
        "benchmark_sweep_threads": False,
        "solver_prefetch_depth": 8,
        "solver_pipeline_async": 1,
        "gpu_device": -1,
        "gpu_devices": None,
        "gpu_inputs": 0,
        "nonces_per_slice": 20_000_000,
        "solver_max_seconds_per_slice": 5.0,
        # Pool: cap shares per 5s slice. 0 means submit every valid share
        # returned by the solver.
        "pool_max_shares_per_slice": 0,
        "experimental_rdna4_wmma": False,
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
    raw_pool_cap = cfg.get("pool_max_shares_per_slice")
    if raw_pool_cap is None:
        cfg["pool_max_shares_per_slice"] = 0
    else:
        cfg["pool_max_shares_per_slice"] = max(0, int(raw_pool_cap))
    return cfg
