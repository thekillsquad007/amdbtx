import subprocess, json, os, time, threading

solver_path = os.path.expanduser("~/.amdbtx-miner/bin/btx-gbt-solve-hip")
env = os.environ.copy()
env["HSA_ENABLE_DXG_DETECTION"] = "1"

# No LD_LIBRARY_PATH added

solver = subprocess.Popen(
    [solver_path, "--daemon", "--backend", "hip", "--batch-size", "128", "--epsilon-bits", "0"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
)

# Drain stderr in a thread
stderr_lines = []
def drain():
    for line in solver.stderr:
        stderr_lines.append(line.strip())
        print(f"[STDERR] {line.strip()}", flush=True)
t = threading.Thread(target=drain, daemon=True)
t.start()

time.sleep(2)
print(f"Alive: {solver.poll() is None}", flush=True)

job = {
    "version": 536870912,
    "prev_hash": "00000000000008a8000000000000000000000000000000000000000000000000",
    "merkle_root": "aabbccdd" * 8,
    "time": 1780981421,
    "bits": "1d5a4f84",
    "seed_a": "0000000000000000000000000000000000000000000000000000000000000001",
    "seed_b": "0000000000000000000000000000000000000000000000000000000000000002",
    "block_height": 125363,
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
print(f"RESULT in {elapsed:.3f}s: {line.strip()}", flush=True)

# Read any remaining stderr
time.sleep(1)
print(f"Dead: {solver.poll() is not None}, Return: {solver.poll()}", flush=True)
print(f"All stderr lines: {stderr_lines}", flush=True)

solver.terminate()
solver.wait()