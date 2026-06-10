#!/usr/bin/env python3
"""Estimate sigma/header gate pass rates for pool jobs."""
import hashlib
import json
import socket
import struct


def hex_to_internal(hex_str: str) -> bytes:
    hex_str = hex_str.zfill(64)
    out = bytearray(32)
    for i in range(32):
        out[31 - i] = int(hex_str[i * 2 : i * 2 + 2], 16)
    return bytes(out)


def internal_to_msb(data: bytes) -> bytes:
    return bytes(data[31 - i] for i in range(32))


def int_from_internal(data: bytes) -> int:
    return int.from_bytes(internal_to_msb(data), "big")


def internal_from_int(n: int) -> bytes:
    n &= (1 << 256) - 1
    return bytes(n.to_bytes(32, "big")[31 - i] for i in range(32))


def derive_prehash_msb_shift(target_hex: str, epsilon: int) -> bytes:
    t = hex_to_internal(target_hex)
    be = bytearray(internal_to_msb(t))
    for _ in range(epsilon):
        overflow = 0
        for i in range(32):
            noverflow = 1 if be[i] & 0x80 else 0
            be[i] = ((be[i] << 1) | overflow) & 0xFF
            overflow = noverflow
        if overflow:
            return bytes([0xFF] * 32)
    out = bytearray(32)
    for i in range(32):
        out[31 - i] = be[i]
    return bytes(out)


def derive_prehash_arith_shift(target_hex: str, epsilon: int) -> bytes:
    n = int_from_internal(hex_to_internal(target_hex)) << epsilon
    return internal_from_int(n)


def passes_le(hash_bytes: bytes, prehash_internal: bytes) -> bool:
    for i in range(32):
        s = hash_bytes[i]
        t = prehash_internal[31 - i]
        if s < t:
            return True
        if s > t:
            return False
    return True


TAG = b"BTX_MATMUL_SEED_V2"


def v2_seed(prev, height, version, merkle, time, bits, nonce, dim, which):
    buf = bytearray([19]) + TAG
    buf += hex_to_internal(prev)
    buf += struct.pack("<I", height)
    buf += struct.pack("<i", version)
    buf += hex_to_internal(merkle)
    buf += struct.pack("<I", time)
    buf += struct.pack("<I", bits)
    buf += struct.pack("<Q", nonce)
    buf += struct.pack("<H", dim)
    buf += bytes([which])
    return hashlib.sha256(buf).digest()


def header_hash(version, prev, merkle, time, bits, nonce, dim, seed_a, seed_b):
    buf = bytearray()
    buf += struct.pack("<I", version)
    buf += hex_to_internal(prev)
    buf += hex_to_internal(merkle)
    buf += struct.pack("<I", time)
    buf += struct.pack("<I", bits)
    buf += struct.pack("<Q", nonce)
    buf += struct.pack("<H", dim)
    buf += seed_a + seed_b
    return hashlib.sha256(buf).digest()


def fetch_job():
    s = socket.create_connection(("stratum.minebtx.com", 3333), timeout=10)
    s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": ["probe", {}]}) + "\n").encode())
    buf = b""
    while b"mining.notify" not in buf:
        buf += s.recv(16384)
    s.close()
    for line in buf.split(b"\n"):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("method") == "mining.notify":
            p = m["params"]
            mat = p[8]
            return {
                "version": int(p[1]),
                "prev_hash": p[2],
                "merkle_root": p[3],
                "time": int(p[4]),
                "bits": int(p[5], 16),
                "share_target": p[6],
                "block_height": int(mat["block_height"]),
                "nonce_start": int(mat["nonce64_start"]),
            }
    raise RuntimeError("no notify")


def derive_block_target(bits: int) -> bytes:
    exponent = bits >> 24
    mantissa = bits & 0x7FFFFF
    if bits & 0x800000:
        mantissa |= 0x800000
    if exponent <= 3:
        mantissa >>= 8 * (3 - exponent)
        return internal_from_int(mantissa)
    return internal_from_int(mantissa << (8 * (exponent - 3)))


def count_passes(job, n, pre_fn, use_sigma, target_hex: str):
    pre = pre_fn(target_hex, 18)
    passes = 0
    ns = job["nonce_start"]
    for i in range(n):
        nonce = ns + i
        sa = v2_seed(
            job["prev_hash"], job["block_height"], job["version"], job["merkle_root"],
            job["time"], job["bits"], nonce, 512, 0,
        )
        sb = v2_seed(
            job["prev_hash"], job["block_height"], job["version"], job["merkle_root"],
            job["time"], job["bits"], nonce, 512, 1,
        )
        hh = header_hash(
            job["version"], job["prev_hash"], job["merkle_root"], job["time"],
            job["bits"], nonce, 512, sa, sb,
        )
        val = hashlib.sha256(hh).digest() if use_sigma else hh
        if passes_le(val, pre):
            passes += 1
    return passes


def main():
    job = fetch_job()
    n = 300_000
    block_hex = internal_to_msb(derive_block_target(job["bits"])).hex()
    print("job bits", hex(job["bits"]), "share", job["share_target"][:20], "nonce", job["nonce_start"])
    for tgt_name, tgt in [("block", block_hex), ("share", job["share_target"])]:
        for pre_name, pre_fn in [("msb", derive_prehash_msb_shift), ("arith", derive_prehash_arith_shift)]:
            for gate_name, use_sigma in [("header", False), ("sigma", True)]:
                p = count_passes(job, n, pre_fn, use_sigma, tgt)
                print(
                    f"{tgt_name:5} {pre_name:5} {gate_name:6} passes={p:6} / {n}  "
                    f"(~{n / (2**18):.1f} expected @ 2^-18)"
                )


if __name__ == "__main__":
    main()