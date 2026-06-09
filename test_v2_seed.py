#!/usr/bin/env python3
"""Verify DeterministicMatMulSeedV2 matches BTX HashWriter serialization."""
import hashlib
import struct

TAG = b"BTX_MATMUL_SEED_V2"


def compact_size(n: int) -> bytes:
    if n < 253:
        return bytes([n])
    raise ValueError("tag too long")


def hex_to_uint256_le(hex_str: str) -> bytes:
    """Match miner HexToUint256: first hex pair -> data[31], last -> data[0]."""
    hex_str = hex_str.zfill(64)
    out = bytearray(32)
    for i in range(32):
        out[31 - i] = int(hex_str[i * 2 : i * 2 + 2], 16)
    return bytes(out)


def derive_v2_seed(prev_hash, height, version, merkle_root, time, bits, nonce64, dim, which):
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


def uint256_to_display_hex(data: bytes) -> str:
    """Match miner Uint256ToHex (data[31] printed first)."""
    return "".join(f"{data[i]:02x}" for i in range(31, -1, -1))


# BTX pow_tests MatMulNonceSeedV2 case
prev = "0000000000000000000000000000000000000000000000000000000000000001"
merkle = "0000000000000000000000000000000000000000000000000000000000000002"
seed_a = derive_v2_seed(prev, 125000, 4, merkle, 1780000000, 0x1D00FFFF, 7, 64, 0)
seed_b = derive_v2_seed(prev, 125000, 4, merkle, 1780000000, 0x1D00FFFF, 7, 64, 1)
print("seed_a:", uint256_to_display_hex(seed_a))
print("seed_b:", uint256_to_display_hex(seed_b))

# nonce sensitivity
seed_a2 = derive_v2_seed(prev, 125000, 4, merkle, 1780000000, 0x1D00FFFF, 8, 64, 0)
assert seed_a != seed_a2, "nonce should change seed"
print("nonce binding OK")