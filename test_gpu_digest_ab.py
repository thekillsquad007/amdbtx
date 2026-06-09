#!/usr/bin/env python3
"""Verify HIP GPU digests match CPU on mainnet-sized jobs (n=512)."""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

def _default_ld_path() -> str:
    parts = [
        os.path.join(Path.home(), ".amdbtx-miner", "runtime"),
        "/opt/rocm/lib",
    ]
    for ver in ("7.2.3", "7.2.0", "6.4.0"):
        parts.append(f"/opt/rocm-{ver}/lib")
    return ":".join(parts)


LD = os.environ.get("LD_LIBRARY_PATH", _default_ld_path())

JOB = {
    "version": 536870912,
    "prev_hash": "51619e6d8d37ab84bf7b9b8a6a8100d6fc1b92d2a6473b2bf153681a416215a1",
    "merkle_root": "f58785dbeb5a7033daa54958364388273cbf363cb50e3bcb0d2879e18e8bfeff",
    "time": 1749000000,
    "bits": "1d1ccc7b",
    "seed_a": "43b5b748c3ad0928e56256e7c687c4907745220ba7053bc56905942c9a0fa1b2",
    "seed_b": "1190c8ed806ea11336f3ad6a20adb9da0beb7b05772afab86c38c67c919ae645",
    "block_height": 125601,
    "matmul_n": 512,
    "matmul_b": 16,
    "matmul_r": 8,
    "epsilon_bits": 0,
    "nonce_start": 1000,
    "max_tries": 3,
    "max_seconds": 120.0,
    "share_target": "ff" * 32,
}


def _wait_daemon_ready(proc: subprocess.Popen, backend: str, solver: Path) -> None:
    ready = threading.Event()
    stderr_lines: list[str] = []

    def drain_stderr() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())
            if "daemon_ready" in line:
                ready.set()

    threading.Thread(target=drain_stderr, daemon=True).start()
    timeout_s = 30.0 if backend == "hip" else 15.0
    if not ready.wait(timeout=timeout_s):
        proc.kill()
        tail = "\n".join(stderr_lines[-8:]) or "(no stderr)"
        raise RuntimeError(f"{solver} ({backend}) failed to start; stderr tail:\n{tail}")


def run_slice(solver: Path, backend: str) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD
    proc = subprocess.Popen(
        [str(solver), "--daemon", f"--backend", backend, "--batch-size", "128"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    _wait_daemon_ready(proc, backend, solver)
    proc.stdin.write(json.dumps(JOB) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    proc.terminate()
    return json.loads(line)


def main() -> int:
    hip = Path.home() / ".amdbtx-miner/bin/btx-gbt-solve-hip"
    build = Path(__file__).resolve().parent / "solver/build/btx-gbt-solve-hip"
    solver = build if build.exists() else hip
    if not solver.exists():
        print(f"solver not found at {solver}", file=sys.stderr)
        return 1

    cpu = run_slice(solver, "cpu")
    print("CPU:", json.dumps(cpu, indent=2))
    if not cpu.get("found"):
        print("CPU did not find (unexpected with eps=0 and easy target)", file=sys.stderr)
        return 1

    gpu = run_slice(solver, "hip")

    print("GPU:", json.dumps(gpu, indent=2))
    if not gpu.get("found"):
        print("FAIL: GPU did not find share that CPU found", file=sys.stderr)
        return 1
    if gpu.get("digest") != cpu.get("digest"):
        print("FAIL: digest mismatch", file=sys.stderr)
        print(f"  CPU: {cpu.get('digest')}", file=sys.stderr)
        print(f"  GPU: {gpu.get('digest')}", file=sys.stderr)
        return 1
    if gpu.get("nonce64") != cpu.get("nonce64"):
        print("FAIL: nonce mismatch", file=sys.stderr)
        return 1
    print("PASS: GPU digest == CPU digest on n=512 V2 job")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())