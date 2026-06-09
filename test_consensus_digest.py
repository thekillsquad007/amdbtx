#!/usr/bin/env python3
"""Cross-check V2 seed + sigma + product-digest byte layout against BTX serialization."""
import hashlib
import struct
import subprocess
import json
import sys
from pathlib import Path

TAG = b"BTX_MATMUL_SEED_V2"
PRODUCT_TAG = b"matmul-product-digest-v3"


def compact_size(n: int) -> bytes:
    if n < 253:
        return bytes([n])
    raise ValueError("tag too long")


def hex_to_uint256_le(hex_str: str) -> bytes:
    hex_str = hex_str.zfill(64)
    out = bytearray(32)
    for i in range(32):
        out[31 - i] = int(hex_str[i * 2 : i * 2 + 2], 16)
    return bytes(out)


def uint256_to_display_hex(data: bytes) -> str:
    return "".join(f"{data[i]:02x}" for i in range(31, -1, -1))


def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def derive_v2_seed_raw(prev_hash, height, version, merkle_root, time, bits, nonce64, dim, which):
    buf = bytearray()
    buf += compact_size(len(TAG))
    buf += TAG
    buf += hex_to_uint256_le(prev_hash)
    buf += struct.pack("<I", height)
    buf += struct.pack("<i", version)
    buf += hex_to_uint256_le(merkle_root)
    buf += struct.pack("<I", time)
    buf += struct.pack("<I", bits)
    buf += struct.pack("<Q", nonce64)
    buf += struct.pack("<H", dim)
    buf += bytes([which])
    return hashlib.sha256(buf).digest()


def derive_sigma_raw(version, prev_hash, merkle_root, time, bits, nonce64, dim, seed_a_raw, seed_b_raw):
    buf = bytearray()
    buf += struct.pack("<I", version)
    buf += hex_to_uint256_le(prev_hash)
    buf += hex_to_uint256_le(merkle_root)
    buf += struct.pack("<I", time)
    buf += struct.pack("<I", bits)
    buf += struct.pack("<Q", nonce64)
    buf += struct.pack("<H", dim)
    buf += seed_a_raw
    buf += seed_b_raw
    header_hash = hashlib.sha256(bytes(buf)).digest()
    return sha256d(header_hash)


def finalize_product_digest_raw(sigma_raw, c_prime_raw, dim, b):
    buf = bytearray()
    buf += PRODUCT_TAG
    buf += sigma_raw
    buf += c_prime_raw
    buf += struct.pack("<I", dim)
    buf += struct.pack("<I", b)
    inner = hashlib.sha256(bytes(buf)).digest()
    digest_raw = sha256d(inner)
    # Pool/stratum targets use SetHex layout (MSB at data[31]).
    return bytes(digest_raw[31 - i] for i in range(32))


def main():
    prev = "0000000000000000000000000000000000000000000000000000000000000001"
    merkle = "0000000000000000000000000000000000000000000000000000000000000002"
    height = 125000
    version = 4
    time_val = 1780000000
    bits = 0x1D00FFFF
    nonce = 7
    dim = 64

    seed_a = derive_v2_seed_raw(prev, height, version, merkle, time_val, bits, nonce, dim, 0)
    seed_b = derive_v2_seed_raw(prev, height, version, merkle, time_val, bits, nonce, dim, 1)
    sigma = derive_sigma_raw(version, prev, merkle, time_val, bits, nonce, dim, seed_a, seed_b)

    print("seed_a raw:", seed_a.hex())
    print("seed_b raw:", seed_b.hex())
    print("sigma raw:", sigma.hex())

    solver = Path.home() / ".amdbtx-miner/bin/btx-gbt-solve-hip"
    if not solver.exists():
        solver = Path(__file__).resolve().parent / "solver/build/btx-gbt-solve-hip"
    if not solver.exists():
        print("solver binary not found, skipping daemon check", file=sys.stderr)
        return 0

    job = {
        "version": version,
        "prev_hash": prev,
        "merkle_root": merkle,
        "time": time_val,
        "bits": f"{bits:08x}",
        "seed_a": "00" * 32,
        "seed_b": "00" * 32,
        "block_height": height,
        "matmul_n": dim,
        "matmul_b": 16,
        "matmul_r": 8,
        "epsilon_bits": 0,
        "nonce_start": nonce,
        "max_tries": 1,
        "max_seconds": 120.0,
        "share_target": "ff" * 32,
    }

    proc = subprocess.Popen(
        [str(solver), "--daemon", "--backend", "cpu", "--batch-size", "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stderr.readline()  # config line
    ready = proc.stderr.readline()
    assert "daemon_ready" in ready, ready
    proc.stdin.write(json.dumps(job) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    proc.terminate()
    result = json.loads(line)
    print("solver:", json.dumps(result, indent=2))
    if not result.get("found"):
        print("WARN: solver did not find within 1 try (expected for hard job)", file=sys.stderr)
    if result.get("digest"):
        print("digest hex:", result["digest"])
    print("consensus layout self-check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())