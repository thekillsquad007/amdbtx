#!/usr/bin/env python3
"""Deterministic local benchmark for btx-gbt-solve-hip.

Measures raw scan MN/s (impossible share target) and captures matmul_profile
stage timings. Run with the GPU miner stopped for clean numbers.

Usage:
  bench_solver.py [solver_path] [--tries N] [--runs K] [--env KEY=VAL ...]
"""
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

LD = os.environ.get(
    "LD_LIBRARY_PATH",
    ":".join([
        str(Path.home() / ".amdbtx-miner" / "runtime"),
        "/usr/lib/wsl/lib",
        "/opt/rocm-7.2.3/lib",
        "/opt/rocm-7.2.0/lib",
        "/opt/rocm/lib",
    ]),
)

BASE_JOB = {
    "version": 536870912,
    "prev_hash": "51619e6d8d37ab84bf7b9b8a6a8100d6fc1b92d2a6473b2bf153681a416215a1",
    "merkle_root": "f58785dbeb5a7033daa54958364388273cbf363cb50e3bcb0d2879e18e8bfeff",
    "time": 1749000000,
    "bits": "1d1ccc7b",
    "seed_a": "43b5b748c3ad0928e56256e7c687c4907745220ba7053bc56905942c9a0fa1b2",
    "seed_b": "1190c8ed806ea11336f3ad6a20adb9da0beb7b05772afab86c38c67c919ae645",
    "block_height": 147000,
    "parent_mtp": 1782910000,
    "matmul_n": 512, "matmul_b": 16, "matmul_r": 8, "epsilon_bits": 18,
}

PROF = re.compile(
    r"passed=(\d+).*?words_path=(\S+).*?sigma_ms=([0-9.]+).*?matrix_ms=([0-9.]+)"
    r".*?noise_seed_ms=([0-9.]+).*?noise_ms=([0-9.]+).*?compress_ms=([0-9.]+)"
    r".*?rhs_ms=([0-9.]+).*?rhs_a_ms=([0-9.]+).*?rhs_right_ms=([0-9.]+)"
    r".*?rhs_b_ms=([0-9.]+).*?words_ms=([0-9.]+).*?hash_compare_ms=([0-9.]+)"
)


def one_run(solver, tries, share_target, extra_env, batch="131072", bits=None, profile=True):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD
    env["HSA_ENABLE_DXG_DETECTION"] = "1"
    if profile:
        env["BTX_MATMUL_PROFILE"] = "1"
    if int(batch) > 131072:
        env.setdefault("BTX_MATMUL_MAX_SCAN_BATCH", batch)
    env.update(extra_env)
    p = subprocess.Popen(
        [solver, "--daemon", "--backend", "hip", "--batch-size", batch,
         "--epsilon-bits", "18"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    stderr = []
    ready = threading.Event()

    def drain():
        for line in p.stderr:
            stderr.append(line.rstrip())
            if "daemon_ready" in line:
                ready.set()

    threading.Thread(target=drain, daemon=True).start()
    if not ready.wait(25):
        p.kill()
        return {"error": "not_ready", "stderr": stderr[-6:]}
    job = dict(BASE_JOB)
    if bits:
        job["bits"] = bits
    job.update({"nonce_start": 17076800000000, "max_tries": tries,
                "max_seconds": 30.0, "max_results": 1,
                "share_target": share_target})
    t0 = time.time()
    p.stdin.write(json.dumps(job) + "\n")
    p.stdin.flush()
    out = p.stdout.readline().strip()
    wall = time.time() - t0
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
    res = json.loads(out)
    profs = [PROF.search(s) for s in stderr]
    profs = [m for m in profs if m]
    nps = res.get("tries_used", 0) / max(res.get("elapsed_s") or wall, 1e-9)
    row = {"backend": res.get("backend"), "nps": round(nps),
           "tries": res.get("tries_used"), "elapsed": res.get("elapsed_s"),
           "gate_passes": res.get("gate_passes", 0),
           "cpu_verify_misses": res.get("cpu_verify_misses", 0)}
    if profs:
        def avg(i):
            return round(statistics.mean(float(m.group(i)) for m in profs), 3)
        row.update({"path": profs[-1].group(2), "sigma_ms": avg(3),
                    "matrix_ms": avg(4), "noise_seed_ms": avg(5),
                    "noise_ms": avg(6), "compress_ms": avg(7),
                    "rhs_a_ms": avg(9), "rhs_right_ms": avg(10),
                    "rhs_b_ms": avg(11), "words_ms": avg(12)})
    return row


def bench(solver, tries, runs, extra_env):
    scan = "00" * 32  # impossible target -> pure scan
    rows = [one_run(solver, tries, scan, extra_env) for _ in range(runs)]
    good = [r for r in rows if r.get("backend") == "hip"]
    if not good:
        return {"solver": solver, "error": "no_hip", "sample": rows[:1]}
    med = round(statistics.median(r["nps"] for r in good))
    return {"solver": Path(solver).name, "runs": len(good),
            "median_nps": med, "path": good[-1].get("path"),
            "sigma_ms": good[-1].get("sigma_ms"),
            "rhs_a_ms": good[-1].get("rhs_a_ms"),
            "rhs_right_ms": good[-1].get("rhs_right_ms"),
            "rhs_b_ms": good[-1].get("rhs_b_ms"),
            "words_ms": good[-1].get("words_ms")}


def main():
    args = sys.argv[1:]
    solver = str(Path.home() / ".amdbtx-miner/bin/btx-gbt-solve-hip")
    tries, runs, batch, bits, profile = 4_000_000, 3, "131072", None, True
    extra = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tries":
            tries = int(args[i + 1]); i += 2
        elif a == "--runs":
            runs = int(args[i + 1]); i += 2
        elif a == "--batch":
            batch = args[i + 1]; i += 2
        elif a == "--bits":
            bits = args[i + 1]; i += 2
        elif a == "--no-profile":
            profile = False; i += 1
        elif a == "--env":
            k, v = args[i + 1].split("=", 1); extra[k] = v; i += 2
        else:
            solver = a; i += 1
    # rebuild one_run to use specified batch
    _one_run = lambda t, st, ee: one_run(solver, t, st, ee, batch, bits, profile)
    rows = [_one_run(tries, "00" * 32, extra) for _ in range(runs)]
    good = [r for r in rows if r.get("backend") == "hip"]
    if not good:
        print(json.dumps({"solver": solver, "error": "no_hip", "sample": rows[:1]}, sort_keys=True))
        return
    med = round(statistics.median(r["nps"] for r in good))
    print(json.dumps({"solver": Path(solver).name, "batch": batch,
                       "bits": bits or BASE_JOB["bits"],
                       "runs": len(good), "median_nps": med,
                       "gate_passes": good[-1].get("gate_passes"),
                       "cpu_verify_misses": good[-1].get("cpu_verify_misses"),
                       "path": good[-1].get("path"),
                       "sigma_ms": good[-1].get("sigma_ms"),
                       "matrix_ms": good[-1].get("matrix_ms"),
                       "noise_seed_ms": good[-1].get("noise_seed_ms"),
                       "noise_ms": good[-1].get("noise_ms"),
                       "compress_ms": good[-1].get("compress_ms"),
                       "rhs_a_ms": good[-1].get("rhs_a_ms"),
                       "rhs_right_ms": good[-1].get("rhs_right_ms"),
                       "rhs_b_ms": good[-1].get("rhs_b_ms"),
                       "words_ms": good[-1].get("words_ms")}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
