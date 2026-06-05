import subprocess
import json
import os
import time
from pathlib import Path


class GBTSolveWrapper:
    def __init__(self, solver_path: str, backend: str = "rocm", threads: int = 8):
        self.solver_path = Path(solver_path).expanduser()
        self.backend = backend
        self.threads = threads
        self.proc: subprocess.Popen = None
        self._start()

    def _start(self):
        if not self.solver_path.exists():
            print(f"[ERROR] Solver not found at {self.solver_path}")
            print(f"[ERROR] Set gbt_solve_path in config or build solver binary")
            self.proc = None
            return

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["HSA_ENABLE_DXG_DETECTION"] = "1"
        try:
            self.proc = subprocess.Popen(
                [str(self.solver_path), "--daemon"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except FileNotFoundError:
            print(f"[ERROR] Solver binary not executable: {self.solver_path}")
            self.proc = None

    def solve(self, job: dict) -> dict:
        if self.proc is None:
            return {"found": False, "error": "solver not running"}

        payload = {
            "version": 536870912,
            "prev_hash": job["prev_hash"],
            "merkle_root": job["merkle_root"],
            "time": job["time"],
            "bits": job["bits"],
            "seed_a": job["seed_a"],
            "seed_b": job["seed_b"],
            "block_height": job["block_height"],
            "nonce_start": 0,
            "max_tries": int(job.get("max_tries", 2000000)),
            "max_seconds": 5.0,
            "share_target": "00" + "ff" * 31,
        }
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()

            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                try:
                    result = json.loads(line.strip())
                    if "found" in result:
                        return result
                except json.JSONDecodeError:
                    continue
        except (BrokenPipeError, OSError):
            return {"found": False, "error": "solver process died"}

        return {"found": False}