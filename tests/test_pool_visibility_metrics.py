import pytest

from amdbtx_miner.__main__ import (
    _expected_shares_from_gate,
    _target_probability_from_hex,
)


def test_target_probability_from_hex_uses_full_256_bit_target():
    assert _target_probability_from_hex("0" * 64) == 0.0
    assert _target_probability_from_hex("f" * 64) == pytest.approx(1.0)
    assert _target_probability_from_hex("8" + "0" * 63) == pytest.approx(0.5)
    assert _target_probability_from_hex("8000000000000000") == pytest.approx(0.5)


def test_expected_shares_from_gate_scales_with_share_target():
    target = "4" + "0" * 63
    assert _expected_shares_from_gate(8, target) == pytest.approx(2.0)
    assert _expected_shares_from_gate(8, "4000000000000000") == pytest.approx(2.0)
    assert _expected_shares_from_gate(0, target) == 0.0
    assert _expected_shares_from_gate(8, "") == 0.0
