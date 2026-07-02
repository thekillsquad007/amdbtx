"""Experimental RDNA4 WMMA flag wiring."""

from amdbtx_miner.gbt_solve_wrapper import apply_matmul_experimental_flags


def test_experimental_rdna4_sets_solver_env():
    env = {}
    apply_matmul_experimental_flags(env, False)
    assert "BTX_MATMUL_EXPERIMENTAL_GFX12_WMMA" not in env

    apply_matmul_experimental_flags(env, True)
    assert env["BTX_MATMUL_EXPERIMENTAL_GFX12_WMMA"] == "1"


def test_config_default_off():
    from amdbtx_miner.config import validate_config

    cfg = validate_config({})
    assert cfg["experimental_rdna4_wmma"] is False