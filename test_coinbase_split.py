#!/usr/bin/env python3
"""Unit tests for solo dev-fee coinbase splitting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from amdbtx_miner.block_builder import (  # noqa: E402
    build_coinbase_tx,
    regenerate_witness_commitment,
    split_coinbase_value,
    txid_from_raw,
)


def test_split_values():
    assert split_coinbase_value(1_000_000_000, 200) == (980_000_000, 20_000_000)
    assert split_coinbase_value(100, 200) == (98, 2)
    assert split_coinbase_value(100, 0) == (100, 0)
    assert split_coinbase_value(1, 200) == (1, 0)  # rounds down to 0 dev sats
    user, dev = split_coinbase_value(10_000, 10_000)
    assert user == 0 and dev == 10_000


def test_coinbase_outputs_with_fee_and_witness():
    user_script = bytes.fromhex("76a914" + "11" * 20 + "88ac")
    dev_script = bytes.fromhex("76a914" + "22" * 20 + "88ac")
    witness_script = bytes.fromhex(
        "6a24aa21a9ed" + "00" * 32
    )
    gbt = {
        "height": 125601,
        "coinbasevalue": 1_000_000_000,
        "coinbaseaux": {},
        "default_witness_commitment": witness_script.hex(),
    }
    tx = build_coinbase_tx(
        gbt, user_script, dev_script=dev_script, dev_fee_bps=200,
    )
    user_value, dev_value = split_coinbase_value(gbt["coinbasevalue"], 200)
    assert user_value + dev_value == gbt["coinbasevalue"]
    assert b"\x00\x01" in tx  # segwit serialization

    regen = regenerate_witness_commitment(tx, [])
    assert len(txid_from_raw(regen)) == 32


def main():
    test_split_values()
    test_coinbase_outputs_with_fee_and_witness()
    print("coinbase split tests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())