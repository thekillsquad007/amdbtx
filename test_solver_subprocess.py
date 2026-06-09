import subprocess, json, os, sys, time

solver_path = os.path.expanduser("~/.amdbtx-miner/bin/btx-gbt-solve-hip")

env = os.environ.copy()
env["HSA_ENABLE_DXG_DETECTION"] = "1"

solver = subprocess.Popen(
    [solver_path, "--daemon", "--backend", "hip", "--batch-size", "128", "--epsilon-bits", "0"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env,
)

# Read ALL stderr lines to see debug output
import threading
def drain_stderr():
    for line in solver.stderr:
        print(f"[SOLVER_STDERR] {line.strip()}", flush=True)
t = threading.Thread(target=drain_stderr, daemon=True)
t.start()

# Wait a moment for solver to initialize
time.sleep(1)

job = {
    "version": 536870912,
    "prev_hash": "00000000000008a8000000000000000000000000000000000000000000000000",
    "merkle_root": "aabbccdd" * 8,
    "time": 1780981421,
    "bits": "1d5a4f84",
    "seed_a": "0000000000000000000000000000000000000000000000000000000000000001",
    "seed_b": "0000000000000000000000000000000000000000000000000000000000000002",
    "block_height": 125282,
    "nonce_start": 0,
    "max_tries": 1000,
    "max_seconds": 10.0,
    "share_target": "00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
}

print(f"Sending job...", flush=True)
solver.stdin.write(json.dumps(job) + "\n")
solver.stdin.flush()

t0 = time.time()
line = solver.stdout.readline()
elapsed = time.time() - t0
print(f"RESULT in {elapsed:.3f}s: '{line.strip()}'", flush=True)

solver.stdin.close()
time.sleep(0.5)
solver.terminate()
solver.wait()