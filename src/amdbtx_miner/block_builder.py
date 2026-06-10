"""Assemble MatMul blocks for submitblock (solo mining)."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

WITNESS_COMMIT_HEADER = bytes.fromhex("aa21a9ed")


def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def varint(n: int) -> bytes:
    if n < 0xFD:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", n)
    if n <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", n)
    return b"\xff" + struct.pack("<Q", n)


def hex_to_uint256_le(hex_str: str) -> bytes:
    hex_str = hex_str.zfill(64)
    out = bytearray(32)
    for i in range(32):
        out[31 - i] = int(hex_str[i * 2 : i * 2 + 2], 16)
    return bytes(out)


def uint256_to_display_hex(data: bytes) -> str:
    return "".join(f"{data[i]:02x}" for i in range(31, -1, -1))


def display_hex_to_le_bytes(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)[::-1]


def encode_bip34_height(height: int) -> bytes:
    out = bytearray()
    value = height
    while value > 0:
        out.append(value & 0xFF)
        value >>= 8
    return bytes(out)


def derive_v2_seed(
    prev_hash: str,
    height: int,
    version: int,
    merkle_root: str,
    time: int,
    bits_hex: str,
    nonce64: int,
    dim: int,
    which: int,
) -> bytes:
    tag = b"BTX_MATMUL_SEED_V2"
    buf = bytearray()
    buf.append(len(tag))
    buf += tag
    buf += hex_to_uint256_le(prev_hash)
    buf += struct.pack("<I", height)
    buf += struct.pack("<i", version)
    buf += hex_to_uint256_le(merkle_root)
    buf += struct.pack("<I", time)
    buf += struct.pack("<I", int(bits_hex, 16))
    buf += struct.pack("<Q", nonce64)
    buf += struct.pack("<H", dim)
    buf.append(which)
    return hashlib.sha256(buf).digest()


def resolve_header_seeds(job, nonce64: int) -> tuple[bytes, bytes]:
    dim = job.matmul_n or 512
    if job.block_height >= 125000:
        seed_a = derive_v2_seed(
            job.prev_hash, job.block_height, job.version, job.merkle_root,
            job.time, job.bits, nonce64, dim, 0,
        )
        seed_b = derive_v2_seed(
            job.prev_hash, job.block_height, job.version, job.merkle_root,
            job.time, job.bits, nonce64, dim, 1,
        )
        return seed_a, seed_b
    return hex_to_uint256_le(job.seed_a), hex_to_uint256_le(job.seed_b)


def merkle_root(hashes: list[bytes]) -> bytes:
    if not hashes:
        return b"\x00" * 32
    layer = list(hashes)
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        nxt = []
        for i in range(0, len(layer), 2):
            nxt.append(sha256d(layer[i] + layer[i + 1]))
        layer = nxt
    return layer[0]


def txid_from_raw(tx_bytes: bytes) -> bytes:
    if tx_bytes[4:6] == b"\x00\x01":
        stripped = tx_bytes[:4] + tx_bytes[6:]
        pos = 4
        vin_count, pos = _read_varint(stripped, pos)
        for _ in range(vin_count):
            pos += 36
            slen, pos = _read_varint(stripped, pos)
            pos += slen
            pos += 4
        vout_count, pos = _read_varint(stripped, pos)
        for _ in range(vout_count):
            pos += 8
            slen, pos = _read_varint(stripped, pos)
            pos += slen
        pos += 4
        legacy = stripped[:pos]
        return sha256d(legacy)
    return sha256d(tx_bytes)


def wtxid_from_raw(tx_bytes: bytes) -> bytes:
    if tx_bytes[4:6] == b"\x00\x01":
        return sha256d(tx_bytes)
    return sha256d(tx_bytes)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    prefix = data[pos]
    pos += 1
    if prefix < 0xFD:
        return prefix, pos
    if prefix == 0xFD:
        return struct.unpack_from("<H", data, pos)[0], pos + 2
    if prefix == 0xFE:
        return struct.unpack_from("<I", data, pos)[0], pos + 4
    return struct.unpack_from("<Q", data, pos)[0], pos + 8


def _write_outpoint_null() -> bytes:
    return b"\x00" * 32 + b"\xff\xff\xff\xff"


def _write_input(script_sig: bytes, sequence: int = 0xFFFFFFFF) -> bytes:
    return _write_outpoint_null() + varint(len(script_sig)) + script_sig + struct.pack("<I", sequence)


def _write_output(value: int, script: bytes) -> bytes:
    return struct.pack("<q", value) + varint(len(script)) + script


def _write_witness_stack(items: list[bytes]) -> bytes:
    out = varint(len(items))
    for item in items:
        out += varint(len(item)) + item
    return out


def _serialize_segwit_tx(
    version: int,
    script_sig: bytes,
    outputs: list[tuple[int, bytes]],
    witness_stack: list[bytes] | None,
    locktime: int = 0,
) -> bytes:
    tx = bytearray()
    tx += struct.pack("<i", version)
    tx += varint(1)
    tx += _write_input(script_sig)
    tx += varint(len(outputs))
    for value, script in outputs:
        tx += _write_output(value, script)
    if witness_stack is not None:
        tx += b"\x00\x01"
        tx += varint(1)
        tx += _write_witness_stack(witness_stack)
    tx += struct.pack("<I", locktime)
    return bytes(tx)


def split_coinbase_value(coinbase_value: int, dev_fee_bps: int) -> tuple[int, int]:
    """Return (user_value, dev_value) in satoshis; dev_fee_bps is basis points (200 = 2%)."""
    if dev_fee_bps <= 0 or coinbase_value <= 0:
        return coinbase_value, 0
    dev_fee_bps = min(dev_fee_bps, 10_000)
    dev_value = coinbase_value * dev_fee_bps // 10_000
    if dev_value <= 0:
        return coinbase_value, 0
    if dev_value >= coinbase_value:
        return 0, coinbase_value
    return coinbase_value - dev_value, dev_value


def build_coinbase_tx(
    gbt: dict[str, Any],
    payout_script: bytes,
    extranonce: bytes = b"",
    *,
    dev_script: bytes | None = None,
    dev_fee_bps: int = 0,
) -> bytes:
    height = int(gbt["height"])
    coinbase_value = int(gbt["coinbasevalue"])
    witness_commitment = gbt.get("default_witness_commitment")

    script_sig = bytearray()
    for _key, val in sorted((gbt.get("coinbaseaux") or {}).items()):
        script_sig += bytes.fromhex(val)
    script_sig += encode_bip34_height(height)
    script_sig += extranonce

    user_value, dev_value = split_coinbase_value(coinbase_value, dev_fee_bps)
    outputs: list[tuple[int, bytes]] = [(user_value, payout_script)]
    if dev_value > 0:
        if not dev_script:
            raise ValueError("dev_script required when dev_fee_bps > 0")
        outputs.append((dev_value, dev_script))
    witness_stack = None
    if witness_commitment:
        outputs.append((0, bytes.fromhex(witness_commitment)))
        witness_stack = [b"\x00" * 32]

    return _serialize_segwit_tx(2, bytes(script_sig), outputs, witness_stack)


def _find_witness_commitment_vout(coinbase_tx: bytes) -> tuple[int, int] | None:
    """Return (script_start, script_end) for the OP_RETURN witness commitment vout."""
    if len(coinbase_tx) < 10:
        return None
    pos = 4
    try:
        vin_count, pos = _read_varint(coinbase_tx, pos)
        for _ in range(vin_count):
            pos += 36
            slen, pos = _read_varint(coinbase_tx, pos)
            pos += slen + 4
        vout_count, pos = _read_varint(coinbase_tx, pos)
        for _ in range(vout_count):
            pos += 8
            script_len, script_start = _read_varint(coinbase_tx, pos)
            script_end = script_start + script_len
            script = coinbase_tx[script_start:script_end]
            if (len(script) >= 38 and script[:2] == bytes([0x6A, 0x24])
                    and script[2:6] == WITNESS_COMMIT_HEADER):
                return script_start, script_end
            pos = script_end
    except IndexError:
        return None
    return None


def regenerate_witness_commitment(coinbase_tx: bytes, tx_raw_list: list[bytes]) -> bytes:
    span = _find_witness_commitment_vout(coinbase_tx)
    if span is None:
        return coinbase_tx

    leaves = [b"\x00" * 32]
    for raw in tx_raw_list:
        leaves.append(wtxid_from_raw(raw))

    root = merkle_root(leaves)
    nonce = b"\x00" * 32
    commitment = sha256d(root + nonce)

    script_start, script_end = span
    old_script = coinbase_tx[script_start:script_end]
    if len(old_script) < 38 or old_script[:2] != bytes([0x6A, 0x24]):
        return coinbase_tx

    new_script = bytes([0x6A, 0x24]) + WITNESS_COMMIT_HEADER + commitment
    rebuilt = bytearray(coinbase_tx)
    rebuilt[script_start:script_end] = new_script
    return bytes(rebuilt)


def serialize_matmul_header(
    job,
    nonce64: int,
    digest_hex: str,
    seed_a: bytes,
    seed_b: bytes,
) -> bytes:
    header = bytearray()
    header += struct.pack("<I", job.version)
    header += display_hex_to_le_bytes(job.prev_hash)
    header += display_hex_to_le_bytes(job.merkle_root)
    header += struct.pack("<I", job.time)
    header += struct.pack("<I", int(job.bits, 16))
    header += struct.pack("<Q", nonce64)
    header += display_hex_to_le_bytes(digest_hex.zfill(64))
    header += struct.pack("<H", job.matmul_n or 512)
    header += seed_a
    header += seed_b
    return bytes(header)


def assemble_block_hex(
    gbt: dict[str, Any],
    job,
    nonce64: int,
    digest_hex: str,
    payout_script: bytes,
    *,
    dev_script: bytes | None = None,
    dev_fee_bps: int = 0,
) -> str:
    seed_a, seed_b = resolve_header_seeds(job, nonce64)
    coinbase = build_coinbase_tx(
        gbt, payout_script, dev_script=dev_script, dev_fee_bps=dev_fee_bps,
    )
    mempool_raw = [bytes.fromhex(tx["data"]) for tx in gbt.get("transactions", [])]

    if gbt.get("default_witness_commitment"):
        coinbase = regenerate_witness_commitment(coinbase, mempool_raw)

    tx_raw_list = [coinbase] + mempool_raw
    txids = [txid_from_raw(raw) for raw in tx_raw_list]
    computed_merkle = merkle_root(txids)
    expected_merkle = display_hex_to_le_bytes(job.merkle_root)
    if computed_merkle != expected_merkle:
        raise ValueError(
            f"merkle mismatch: built {computed_merkle[::-1].hex()} "
            f"expected {job.merkle_root}"
        )

    header = serialize_matmul_header(job, nonce64, digest_hex, seed_a, seed_b)
    block = bytearray(header)
    block += varint(len(tx_raw_list))
    for raw in tx_raw_list:
        block += raw
    return block.hex()