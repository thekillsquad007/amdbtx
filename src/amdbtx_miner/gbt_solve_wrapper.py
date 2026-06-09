import subprocess
import json
import os
import time
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class GBTSolveWrapper:
    def __init__(self, solver_path: str, backend: str = "rocm", threads: int = 8,
                 prepare_workers: int = 16, batch_size: int = 1024,
                 prefetch_depth: int = 8, pipeline_async: int = 1,
                 gpu_inputs: int = 0, gpu_device: int = -1,
                 runtime_ld_path: str = ""):
        self.solver_path = Path(solver_path).expanduser()
        self.backend = backend
        self.threads = threads
        self.prepare_workers = prepare_workers
        self.batch_size = batch_size
        self.prefetch_depth = prefetch_depth
        self.pipeline_async = pipeline_async
        self.gpu_inputs = gpu_inputs
        self.gpu_device = gpu_device
        self.runtime_ld_path = runtime_ld_path
        self.proc: subprocess.Popen | None = None
        self.last_observed_nps = None
        self._stderr_thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._start(allow_cpu_fallback=True)

    def _build_ld_path(self) -> str:
        parts = []
        if self.runtime_ld_path:
            for p in self.runtime_ld_path.split(":"):
                if p and p not in parts:
                    parts.append(p)
        runtime_dir = str(Path.home() / ".amdbtx-miner" / "runtime")
        if os.path.isdir(runtime_dir) and runtime_dir not in parts:
            parts.append(runtime_dir)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            for p in existing.split(":"):
                if p and p not in parts:
                    parts.append(p)
        return ":".join(parts)

    def _stop_proc(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass
        self.proc = None

    def _start(self, *, allow_cpu_fallback: bool = False):
        if not self.solver_path.exists():
            log.error("Solver not found at %s", self.solver_path)
            self.proc = None
            return

        self._stop_proc()
        self._ready_event = threading.Event()

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
        # Post-125k the matmul seed folds nTime. Upstream btx-gbt-solve auto-refreshes
        # header time every 4096 attempts by default; the wrapper still submits the
        # original job ntime, so the pool recomputes a different digest (code-23).
        # dexbtx-miner v0.4.6 disables this; our standalone HIP solver ignores the
        # var, but set it when using dexbtx's prebuilt btx-gbt-solve binary.
        env.setdefault("BTX_MINER_HEADER_TIME_REFRESH_ATTEMPTS", "4294967295")
        if self.gpu_device >= 0:
            env["HIP_VISIBLE_DEVICES"] = str(self.gpu_device)

        backend_arg = "hip" if self.backend in ("rocm", "hip") else self.backend
        cmd = [
            str(self.solver_path),
            "--daemon",
            "--backend", backend_arg,
            "--batch-size", str(self.batch_size),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True)
            self._stderr_thread.start()
            log.info("solver started: %s", " ".join(cmd))
        except FileNotFoundError:
            log.error("Solver binary not executable: %s", self.solver_path)
            self.proc = None
            return

        if not self._ready_event.wait(timeout=8.0):
            rc = self.proc.poll() if self.proc is not None else None
            self._stop_proc()
            if (allow_cpu_fallback and backend_arg == "hip"
                    and self.backend in ("rocm", "hip")):
                log.warning(
                    "HIP solver failed to initialize (exit=%s); "
                    "falling back to CPU backend", rc,
                )
                self.backend = "cpu"
                self._start(allow_cpu_fallback=False)
            else:
                log.error("solver failed to start (exit=%s)", rc)

    def _drain_stderr(self):
        if self.proc is None or self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            line = line.strip()
            if not line:
                continue
            log.info("solver: %s", line)
            if "daemon_ready" in line:
                self._ready_event.set()

    def _ensure_running(self):
        if self.proc is not None and self.proc.poll() is None and self._ready_event.is_set():
            return True
        if self.proc is not None:
            rc = self.proc.returncode
            log.warning("solver exited (code=%s), restarting", rc)
        self._start(allow_cpu_fallback=False)
        return self.proc is not None and self._ready_event.is_set()

    def solve(self, job, nonce_start: int = 0, max_tries: int = 20_000_000,
              max_seconds: float = 5.0) -> dict:
        if not self._ensure_running():
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
                "matmul_n": job.matmul_n,
                "matmul_b": job.matmul_b,
                "matmul_r": job.matmul_r,
                "epsilon_bits": job.epsilon_bits,
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
                "matmul_n": int(job.get("matmul_n", 512)),
                "matmul_b": int(job.get("matmul_b", 16)),
                "matmul_r": int(job.get("matmul_r", 8)),
                "epsilon_bits": int(job.get("epsilon_bits", 18)),
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
        except (BrokenPipeError, OSError) as e:
            log.error("solver I/O error: %s", e)
            self.proc = None
            return {"found": False, "error": "solver process died"}

        log.warning("solver returned no result (process may have crashed)")
        self.proc = None
        return {"found": False, "error": "solver returned no result"}

    def stop(self):
        self._stop_proc()
