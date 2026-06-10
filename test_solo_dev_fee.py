#!/usr/bin/env python3
"""Integration tests for solo dev-fee coinbase split (mocked RPC)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from amdbtx_miner import DEV_WALLET, DEFAULT_SOLO_DEV_FEE_BPS  # noqa: E402
from amdbtx_miner.block_builder import (  # noqa: E402
    _read_varint,
    assemble_block_hex,
    build_coinbase_tx,
    merkle_root,
    regenerate_witness_commitment,
    split_coinbase_value,
    txid_from_raw,
)
from amdbtx_miner.solo_client import SoloClient  # noqa: E402
from amdbtx_miner.stratum_client import Job  # noqa: E402

USER_SCRIPT = bytes.fromhex("76a914" + "11" * 20 + "88ac")
DEV_SCRIPT = bytes.fromhex("76a914" + "22" * 20 + "88ac")
WITNESS_SCRIPT = bytes.fromhex("6a24aa21a9ed" + "00" * 32)


def parse_vouts(tx: bytes) -> list[tuple[int, bytes]]:
    pos = 4
    vin_count, pos = _read_varint(tx, pos)
    for _ in range(vin_count):
        pos += 36
        slen, pos = _read_varint(tx, pos)
        pos += slen + 4
    vout_count, pos = _read_varint(tx, pos)
    vouts = []
    for _ in range(vout_count):
        value = int.from_bytes(tx[pos : pos + 8], "little", signed=True)
        pos += 8
        slen, script_start = _read_varint(tx, pos)
        vouts.append((value, tx[script_start : script_start + slen]))
        pos = script_start + slen
    return vouts


def _mock_rpc_responses(address: str) -> dict:
    if address == DEV_WALLET:
        return {"scriptPubKey": DEV_SCRIPT.hex()}
    return {"scriptPubKey": USER_SCRIPT.hex()}


def _sample_gbt(coinbase_value: int = 2_000_000_000) -> dict:
    gbt = {
        "version": 536870912,
        "previousblockhash": "0" * 64,
        "bits": "1d1ccc7b",
        "curtime": 1781000000,
        "height": 125601,
        "coinbasevalue": coinbase_value,
        "coinbaseaux": {},
        "default_witness_commitment": WITNESS_SCRIPT.hex(),
        "longpollid": "test",
        "transactions": [],
    }
    coinbase = build_coinbase_tx(
        gbt, USER_SCRIPT, dev_script=DEV_SCRIPT, dev_fee_bps=200,
    )
    coinbase = regenerate_witness_commitment(coinbase, [])
    merkle = merkle_root([txid_from_raw(coinbase)])
    gbt["merkleroot"] = merkle[::-1].hex()
    return gbt


def _sample_challenge(gbt: dict) -> dict:
    return {
        "header_context": {
            "version": gbt["version"],
            "previousblockhash": gbt["previousblockhash"],
            "merkleroot": gbt.get("merkleroot", "f" * 64),
            "time": gbt["curtime"],
            "bits": gbt["bits"],
            "height": gbt["height"],
            "seed_a": "a" * 64,
            "seed_b": "b" * 64,
        },
        "work_profile": {"pre_hash_lottery": {"epsilon_bits": 18}},
    }


def test_solo_client_resolves_dev_fee():
    cfg = {
        "payout_address": "btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx",
        "rpc_url": "http://127.0.0.1:19334",
        "rpc_user": "miner",
        "rpc_password": "miner",
        "solo_dev_fee_bps": 200,
        "gbt_longpoll": False,
    }

    def fake_call(method, params=None, timeout=None):
        if method == "getmininginfo":
            return {"blocks": 125601, "difficulty": 0.035}
        if method in ("validateaddress", "getaddressinfo"):
            return _mock_rpc_responses(params[0])
        raise AssertionError(f"unexpected RPC: {method}")

    with patch.object(SoloClient, "_check_node_ready", lambda self: None), \
         patch.object(SoloClient, "_init_dev_fee", lambda self: None):
        client = SoloClient(cfg)
    client.rpc.call = fake_call  # type: ignore[method-assign]
    client._init_dev_fee()

    assert client.dev_fee_bps == 200
    assert client._dev_script == DEV_SCRIPT
    print("  solo client dev fee init: OK")


def test_solo_client_disable_fee_on_bps_zero():
    cfg = {
        "payout_address": "btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx",
        "rpc_url": "http://127.0.0.1:19334",
        "rpc_user": "x",
        "rpc_password": "y",
        "solo_dev_fee_bps": 0,
    }
    with patch.object(SoloClient, "_check_node_ready", lambda self: None), \
         patch.object(SoloClient, "_init_dev_fee", lambda self: None):
        client = SoloClient(cfg)
    client._init_dev_fee()
    assert client.dev_fee_bps == 0
    assert client._dev_script is None
    print("  solo fee disable (bps=0): OK")


def test_assemble_block_with_dev_split():
    gbt = _sample_gbt()
    job = SoloClient._job_from_template(gbt, _sample_challenge(gbt))
    block_hex = assemble_block_hex(
        gbt, job, nonce64=42, digest_hex="f" * 64,
        payout_script=USER_SCRIPT, dev_script=DEV_SCRIPT, dev_fee_bps=200,
    )
    assert len(block_hex) > 200

    # Parse coinbase from block (header is fixed size for matmul; find first tx)
    # Block = header + varint(n_tx) + txs. MatMul header:
    # 4+32+32+4+4+8+32+2+32+32 = 182 bytes
    header_len = 182
    pos = header_len
    n_tx, pos = _read_varint(bytes.fromhex(block_hex), pos)
    assert n_tx == 1
    coinbase = bytes.fromhex(block_hex)[pos:]
    vouts = parse_vouts(coinbase)
    user_value, dev_value = split_coinbase_value(gbt["coinbasevalue"], 200)
    assert len(vouts) == 3  # user, dev, witness
    assert vouts[0] == (user_value, USER_SCRIPT)
    assert vouts[1] == (dev_value, DEV_SCRIPT)
    assert vouts[2][0] == 0
    assert vouts[0][0] + vouts[1][0] == gbt["coinbasevalue"]
    print("  assemble_block_hex dev split: OK")


def test_submit_block_mocked_rpc():
    gbt = _sample_gbt()
    challenge = _sample_challenge(gbt)
    cfg = {
        "payout_address": "btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx",
        "rpc_url": "http://127.0.0.1:19334",
        "rpc_user": "miner",
        "rpc_password": "miner",
        "solo_dev_fee_bps": DEFAULT_SOLO_DEV_FEE_BPS,
    }

    submitted: list[str] = []

    def fake_call(method, params=None, timeout=None):
        if method == "getmininginfo":
            return {"blocks": gbt["height"], "difficulty": 0.035}
        if method in ("validateaddress", "getaddressinfo"):
            return _mock_rpc_responses(params[0])
        if method == "submitblock":
            submitted.append(params[0])
            return None  # accepted
        raise AssertionError(f"unexpected RPC: {method}")

    with patch.object(SoloClient, "_check_node_ready", lambda self: None), \
         patch.object(SoloClient, "_init_dev_fee", lambda self: None):
        client = SoloClient(cfg)
    client.rpc.call = fake_call  # type: ignore[method-assign]
    client._init_dev_fee()

    job = SoloClient._job_from_template(gbt, challenge)
    client._gbt_by_job[job.job_id] = gbt

    ok = client.submit_block(job, {
        "is_block": True,
        "nonce64": 99,
        "digest": "a" * 64,
    })
    assert ok is True
    assert client.blocks_found == 1
    assert len(submitted) == 1

    block = bytes.fromhex(submitted[0])
    pos = 182
    _, pos = _read_varint(block, pos)
    vouts = parse_vouts(block[pos:])
    user_value, dev_value = split_coinbase_value(gbt["coinbasevalue"], 200)
    assert vouts[0][0] == user_value
    assert vouts[1][0] == dev_value
    print("  submit_block mocked RPC: OK")


def test_share_tier_skips_submit():
    gbt = _sample_gbt()
    cfg = {
        "payout_address": "btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx",
        "rpc_url": "http://127.0.0.1:19334",
        "rpc_user": "miner",
        "rpc_password": "miner",
        "solo_dev_fee_bps": 200,
    }
    with patch.object(SoloClient, "_check_node_ready", lambda self: None), \
         patch.object(SoloClient, "_init_dev_fee", lambda self: None):
        client = SoloClient(cfg)
    client.rpc.call = MagicMock()
    job = SoloClient._job_from_template(gbt, _sample_challenge(gbt))
    client._gbt_by_job[job.job_id] = gbt
    ok = client.submit_block(job, {"is_block": False, "nonce64": 1, "digest": "b" * 64})
    assert ok is False
    client.rpc.call.assert_not_called()
    print("  share-tier skip submit: OK")


def main():
    print("solo dev fee integration tests")
    test_solo_client_resolves_dev_fee()
    test_solo_client_disable_fee_on_bps_zero()
    test_assemble_block_with_dev_split()
    test_submit_block_mocked_rpc()
    test_share_tier_skips_submit()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())