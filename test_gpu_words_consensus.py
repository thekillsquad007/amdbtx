#!/usr/bin/env python3
"""Scan gated nonces and ensure GPU path has no digest verify misses."""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


def _default_ld_path() -> str:
    parts = [os.path.join(Path.home(), ".amdbtx-miner", "runtime"), "/opt/rocm/lib"]
    for ver in ("7.2.3", "7.2.0", "6.4.0"):
        parts.append(f"/opt/rocm-{ver}/lib")
    return ":".join(parts)


LD = os.environ.get("LD_LIBRARY_PATH", _default_ld_path())


def run_slice(solver: Path, share_target: str, nonce_start: int, max_tries: int) -> tuple[dict, list[str]]:
    job = {
        "version": 536870912,
        "prev_hash": "51619e6d8d37ab84bf7b9b8a6a8100d6fc1b92d2a6473b2bf153681a416215a1",
        "merkle_root": "f58785dbeb5a7033daa54958364388273cbf363cb50e3bcb0d2879e18e8bfeff",
        "time": 1749000000,
        "bits": "1d1ccc7b",
        "seed_a": "43b5b748c3ad0928e56256e7c687c4907745220ba7053bc56905942c9a0fa1b2",
        "seed_b": "1190c8ed806ea11336f3ad6a20adb9da0beb7b05772afab86c38c67c919ae645",
        "block_height": 125874,
        "matmul_n": 512,
        "matmul_b": 16,
        "matmul_r": 8,
        "epsilon_bits": 18,
        "nonce_start": nonce_start,
        "max_tries": max_tries,
        "max_seconds": 120.0,
        "share_target": share_target,
    }
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD
    proc = subprocess.Popen(
        [str(solver), "--daemon", "--backend", "hip", "--batch-size", "512", "--epsilon-bits", "18"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stderr_lines: list[str] = []
    ready = threading.Event()

    def drain() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip()
            stderr_lines.append(line)
            if "daemon_ready" in line:
                ready.set()

    threading.Thread(target=drain, daemon=True).start()
    if not ready.wait(timeout=30.0):
        proc.kill()
        raise RuntimeError("solver failed to start")
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(job) + "\n")
    proc.stdin.flush()
    assert proc.stdout is not None
    line = proc.stdout.readline()
    proc.terminate()
    return json.loads(line), stderr_lines


def main() -> int:
    build = Path(__file__).resolve().parent / "solver/build/btx-gbt-solve-hip"
    installed = Path.home() / ".amdbtx-miner/bin/btx-gbt-solve-hip"
    solver = build if build.exists() else installed
    if not solver.exists():
        print(f"solver not found at {solver}", file=sys.stderr)
        return 1

    share_target = "00006ac210000000000000000000000000000000000000000000000000000"
    nonce_start = int(sys.argv[1]) if len(sys.argv) > 1 else 17076800000000
    max_tries = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000
    result, stderr = run_slice(solver, share_target, nonce_start, max_tries)
    misses = [l for l in stderr if "digest verify miss" in l]
    print("result:", json.dumps(result, indent=2))
    print(f"gate_passes={result.get('gate_passes', 0)} verify_misses={len(misses)}")
    if misses:
        print("sample misses:")
        for line in misses[:5]:
            print(" ", line)
        return 1
    print("PASS: no digest verify misses in eps=18 scan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())