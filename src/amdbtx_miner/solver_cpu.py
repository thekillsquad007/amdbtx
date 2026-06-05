#!/usr/bin/env python3
"""CPU-based BTX MatMul solver fallback for testing/developing on Windows."""
import sys
import json
import hashlib
import time
import argparse

# M31 field modulus
M31 = 2**31 - 1

def reduce64(x: int) -> int:
    x = (x & 0x7FFFFFFF) + (x >> 31)
    x = (x & 0x7FFFFFFF) + (x >> 31)
    if x >= 0x7FFFFFFF:
        x -= 0x7FFFFFFF
    return x

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def compute_digest(job: dict) -> bytes:
    """Simplified share check - returns SHA256d of job params."""
    data = (
        bytes.fromhex(job["prev_hash"]) +
        bytes.fromhex(job["merkle_root"]) +
        bytes.fromhex(job["seed_a"]) +
        bytes.fromhex(job["seed_b"]) +
        job["block_height"].to_bytes(4, "little")
    )
    return sha256d(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--max-tries", type=int, default=2000000)
    ap.add_argument("--max-seconds", type=float, default=5.0)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    start = time.time()
    tries = 0

    while tries < args.max_tries and (time.time() - start) < args.max_seconds:
        tries += 1
        time.sleep(0.00001)  # Simulate work

    # Report a result (simplified - real solver would do matmul PoW)
    result = {
        "found": tries % 100000 == 0,  # Fake found periodically
        "nonce64": tries,
        "nonce64_end": tries + 1000,
        "digest": "0" * 64,
        "elapsed_s": round(time.time() - start, 3),
        "tries_used": tries,
        "is_block": False,
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()