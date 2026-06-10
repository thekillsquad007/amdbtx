#!/usr/bin/env python3
"""Mine one slice against live pool and diagnose code-23 rejects."""
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from amdbtx_miner import PROTOCOL_CAPABILITIES, USER_AGENT, __version__
from amdbtx_miner.stratum_client import Job
from amdbtx_miner.hardware import collect_static_hardware, hardware_summary_string

HOST, PORT = "stratum.minebtx.com", 3333
PAYOUT = os.environ.get("BTX_PAYOUT", "btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4")
SOLVER = Path(os.environ.get("BTX_SOLVER", Path.home() / ".amdbtx-miner/bin/btx-gbt-solve-hip"))


def pool_exchange():
    s = socket.create_connection((HOST, PORT), timeout=15)
    buf = b""

    def recv():
        nonlocal buf
        while b"\n" not in buf:
            chunk = s.recv(8192)
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        line, buf = buf.split(b"\n", 1)
        return json.loads(line.decode())

    def call(mid, method, params):
        s.sendall((json.dumps({"id": mid, "method": method, "params": params}) + "\n").encode())
        while True:
            m = recv()
            if m.get("id") == mid:
                if m.get("error"):
                    raise RuntimeError(m["error"])
                return m.get("result")
            yield_notify(m)

    def yield_notify(m):
        if m.get("method") == "mining.notify":
            pass

    hw = collect_static_hardware(miner_version=__version__)
    ext = {"protocol_compliant": list(PROTOCOL_CAPABILITIES), "hardware": hw, "session_id": uuid.uuid4().hex}
    sub = None
    s.sendall((json.dumps({"id": 1, "method": "mining.subscribe", "params": [USER_AGENT, ext]}) + "\n").encode())
    while sub is None:
        m = recv()
        if m.get("id") == 1:
            sub = m.get("result")
    en1, en2_size = sub[1], int(sub[2])
    print("extranonce1", en1, "en2_size", en2_size)

    s.sendall((json.dumps({"id": 2, "method": "mining.authorize", "params": [PAYOUT, ""]}) + "\n").encode())
    while True:
        m = recv()
        if m.get("id") == 2:
            break

    job = None
    deadline = time.time() + 30
    while job is None and time.time() < deadline:
        m = recv()
        if m.get("method") == "mining.notify":
            job = Job.from_notify(m["params"])
    if job is None:
        raise RuntimeError("no job")
    print("job", job.job_id, "height", job.block_height, "time", job.time, "nonce_start", job.nonce64_start)
    print("target", job.target[:24], "bits", job.bits)
    return s, recv, job, en1, en2_size


def run_solver(job: Job, solver: Path, nonce_start: int, max_tries: int = 5_000_000):
    env = os.environ.copy()
    env["HSA_ENABLE_DXG_DETECTION"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([
        str(Path.home() / ".amdbtx-miner/runtime"),
        "/opt/rocm/lib",
        env.get("LD_LIBRARY_PATH", ""),
    ])
    env["BTX_MINER_HEADER_TIME_REFRESH_ATTEMPTS"] = "4294967295"
    backend = os.environ.get("BTX_BACKEND", "cpu")
    proc = subprocess.Popen(
        [str(solver), "--daemon", "--backend", backend, "--batch-size", "256", "--epsilon-bits", "18"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    proc.stderr.readline()
    assert "daemon_ready" in proc.stderr.readline()
    payload = {
        "version": job.version,
        "prev_hash": job.prev_hash,
        "merkle_root": job.merkle_root,
        "time": job.time,
        "bits": job.bits,
        "seed_a": job.seed_a,
        "seed_b": job.seed_b,
        "block_height": job.block_height,
        "matmul_n": job.matmul_n,
        "matmul_b": job.matmul_b,
        "matmul_r": job.matmul_r,
        "epsilon_bits": job.epsilon_bits,
        "nonce_start": nonce_start,
        "max_tries": max_tries,
        "max_seconds": 300.0,
        "share_target": job.target,
    }
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    proc.terminate()
    return json.loads(line)


def main():
    s, recv, job, en1, en2_size = pool_exchange()
    result = run_solver(job, SOLVER, job.nonce64_start)
    print("solver", json.dumps(result, indent=2))
    if not result.get("found"):
        print("no share in slice; increase max_tries or retry")
        return 1

    nonce_hex = f"{int(result['nonce64']):016x}"
    ntime = f"{job.time:08x}"
    en2 = "00" * en2_size
    params = [PAYOUT, job.job_id, en2, ntime, nonce_hex]
    print("submit", params)
    mid = 99
    s.sendall((json.dumps({"id": mid, "method": "mining.submit", "params": params}) + "\n").encode())
    deadline = time.time() + 30
    while time.time() < deadline:
        m = recv()
        if m.get("id") == mid:
            print("pool_response", m)
            return 0 if not m.get("error") else 2
    print("submit timeout")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())