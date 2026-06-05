import subprocess
import json
import os
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class GBTSolveWrapper:
    def __init__(self, solver_path: str, backend: str = "rocm", threads: int = 8,
                 prepare_workers: int = 16, batch_size: int = 128,
                 prefetch_depth: int = 8, pipeline_async: int = 1,
                 gpu_inputs: int = 0,
                 runtime_ld_path: str = ""):
        self.solver_path = Path(solver_path).expanduser()
        self.backend = backend
        self.threads = threads
        self.prepare_workers = prepare_workers
        self.batch_size = batch_size
        self.prefetch_depth = prefetch_depth
        self.pipeline_async = pipeline_async
        self.gpu_inputs = gpu_inputs
        self.runtime_ld_path = runtime_ld_path
        self.proc: subprocess.Popen | None = None
        self.last_observed_nps = None
        self._start()

    def _build_ld_path(self) -> str:
        parts = []
        if self.runtime_ld_path:
            for p in self.runtime_ld_path.split(":"):
                if p and p not in parts:
                    parts.append(p)
        runtime_dir = str(Path.home() / ".amdbtx-miner" / "runtime")
        if os.path.isdir(runtime_dir) and runtime_dir not in parts:
            parts.append(runtime_dir)
        if os.path.isdir("/opt/rocm/lib") and "/opt/rocm/lib" not in parts:
            parts.append("/opt/rocm/lib")
        for d in sorted(Path("/opt").glob("rocm-*/lib")):
            s = str(d)
            if s not in parts:
                parts.append(s)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            for p in existing.split(":"):
                if p and p not in parts:
                    parts.append(p)
        return ":".join(parts)

    def _start(self):
        if not self.solver_path.exists():
            log.error("Solver not found at %s", self.solver_path)
            self.proc = None
            return

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = self._build_ld_path()
        env["HSA_ENABLE_DXG_DETECTION"] = "1"
        env["BTX_MATMUL_BACKEND"] = self.backend
        env["BTX_MATMUL_SOLVER_THREADS"] = str(self.threads)
        env["BTX_MATMUL_PREPARE_WORKERS"] = str(self.prepare_workers)
        env["BTX_MATMUL_SOLVE_BATCH_SIZE"] = str(self.batch_size)
        env["BTX_MATMUL_PREPARE_PREFETCH_DEPTH"] = str(self.prefetch_depth)
        env["BTX_MATMUL_PIPELINE_ASYNC"] = str(self.pipeline_async)
        env["BTX_MATMUL_GPU_INPUTS"] = str(self.gpu_inputs)

        cmd = [str(self.solver_path), "--daemon"]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            ready = self.proc.stderr.readline().strip() if self.proc.stderr else ""
            if "daemon_ready" in ready:
                log.info("solver daemon ready")
            else:
                log.info("solver started (ready=%s)", ready[:80] if ready else "?")
        except FileNotFoundError:
            log.error("Solver binary not executable: %s", self.solver_path)
            self.proc = None

    def solve(self, job, nonce_start: int = 0, max_tries: int = 20_000_000,
              max_seconds: float = 5.0) -> dict:
        if self.proc is None:
            return {"found": False, "error": "solver not running"}

        from .stratum_client import Job

        if isinstance(job, Job):
            share_target = job.target if job.target else ("00" + "ff" * 31)
            payload = {
                "version": job.version,
                "prev_hash": job.prev_hash,
                "merkle_root": job.merkle_root,
                "time": job.time,
                "bits": job.bits,
                "seed_a": job.seed_a,
                "seed_b": job.seed_b,
                "block_height": job.block_height,
                "nonce_start": nonce_start,
                "max_tries": max_tries,
                "max_seconds": max_seconds,
                "share_target": share_target,
            }
        elif isinstance(job, dict):
            share_target = job.get("target", job.get("share_target", "00" + "ff" * 31))
            payload = {
                "version": int(job.get("version", 536870912)),
                "prev_hash": job.get("prev_hash", "0" * 64),
                "merkle_root": job.get("merkle_root", "0" * 64),
                "time": int(job.get("time", 0)),
                "bits": job.get("bits", "1d17c609"),
                "seed_a": job.get("seed_a", "0" * 64),
                "seed_b": job.get("seed_b", "0" * 64),
                "block_height": int(job.get("block_height", 0)),
                "nonce_start": int(job.get("nonce_start", nonce_start)),
                "max_tries": int(job.get("max_tries", max_tries)),
                "max_seconds": float(job.get("max_seconds", max_seconds)),
                "share_target": share_target,
            }
        else:
            return {"found": False, "error": f"unsupported job type: {type(job)}"}

        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()

            t0 = time.time()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                try:
                    result = json.loads(line.strip())
                    if "found" in result:
                        elapsed = time.time() - t0
                        tries = result.get("tries_used", 0)
                        if elapsed > 0 and tries > 0:
                            self.last_observed_nps = tries / elapsed
                        return result
                except json.JSONDecodeError:
                    continue
        except (BrokenPipeError, OSError):
            return {"found": False, "error": "solver process died"}

        return {"found": False}

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
