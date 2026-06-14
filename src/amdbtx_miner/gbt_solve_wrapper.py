import subprocess
import json
import os
import re
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.solver_version = self._probe_solver_version()
        self.supports_parent_mtp_v3 = self._supports_parent_mtp_v3(
            self.solver_version
        )
        self._compat_error_logged = False
        self._stderr_thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._start(allow_cpu_fallback=True)

    def _probe_solver_version(self) -> str:
        if not self.solver_path.exists():
            return ""
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = self._build_ld_path()
        try:
            result = subprocess.run(
                [str(self.solver_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return (result.stdout or result.stderr).strip()

    @staticmethod
    def _supports_parent_mtp_v3(version_output: str) -> bool:
        if "parent-MTP" in version_output:
            return True
        match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", version_output)
        return bool(match and tuple(map(int, match.groups())) >= (2, 1, 0))

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
            "--epsilon-bits", "18",
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
        from .stratum_client import Job

        block_height = (
            job.block_height if isinstance(job, Job)
            else int(job.get("block_height", 0)) if isinstance(job, dict)
            else 0
        )
        if block_height >= 130500 and not self.supports_parent_mtp_v3:
            error = (
                "solver is not BTX V3 compatible; install "
                "btx-gbt-solve-hip 2.1.0 or newer"
            )
            if not self._compat_error_logged:
                log.error(
                    "%s (reported version: %s)",
                    error,
                    self.solver_version or "unknown",
                )
                self._compat_error_logged = True
            return {"found": False, "error": error}

        if not self._ensure_running():
            return {"found": False, "error": "solver not running"}

        if isinstance(job, Job):
            # Pool must supply params[6]; hard fallback avoids false share hits.
            share_target = job.target if job.target else ("ff" * 64)
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
                # Result buffering is separate from daemon input prefetching.
                "max_results": 64,
                "share_target": share_target,
            }
            if job.parent_mtp is not None:
                payload["parent_mtp"] = job.parent_mtp
        elif isinstance(job, dict):
            share_target = job.get("target", job.get("share_target", "ff" * 64))
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
                "max_results": int(job.get("max_results", 64)),
                "share_target": share_target,
            }
            if job.get("parent_mtp") is not None:
                payload["parent_mtp"] = int(job["parent_mtp"])
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
                        tries = int(result.get("tries_used", 0) or 0)
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


class MultiGPUSolver:
    """One HIP solver subprocess per GPU; nonce ranges are partitioned each round."""

    def __init__(
        self,
        solver_path: str,
        backend: str = "rocm",
        threads: int = 8,
        prepare_workers: int = 16,
        batch_size: int = 1024,
        prefetch_depth: int = 8,
        pipeline_async: int = 1,
        gpu_inputs: int = 0,
        gpu_devices: list[int] | None = None,
        runtime_ld_path: str = "",
    ):
        self.gpu_devices = list(gpu_devices or [0])
        self.last_observed_nps: float | None = None
        self._peak_gate_nps: float = 0.0
        self.solvers: list[GBTSolveWrapper] = []
        for device in self.gpu_devices:
            self.solvers.append(
                GBTSolveWrapper(
                    solver_path=solver_path,
                    backend=backend,
                    threads=threads,
                    prepare_workers=prepare_workers,
                    batch_size=batch_size,
                    prefetch_depth=prefetch_depth,
                    pipeline_async=pipeline_async,
                    gpu_inputs=gpu_inputs,
                    gpu_device=device,
                    runtime_ld_path=runtime_ld_path,
                )
            )
        if len(self.solvers) > 1:
            log.info(
                "multi-GPU mining: %d solvers on devices %s",
                len(self.solvers), self.gpu_devices,
            )

    @property
    def num_gpus(self) -> int:
        return len(self.solvers)

    def _merge_results(self, results: list[dict], *, wall_elapsed: float) -> dict:
        total_tries = sum(int(r.get("tries_used", 0) or 0) for r in results)
        total_gate = sum(int(r.get("gate_passes", 0) or 0) for r in results)
        per_gpu_khps = []
        for r in results:
            elapsed = float(r.get("elapsed_s", 0) or 0)
            tries = int(r.get("tries_used", 0) or 0)
            per_gpu_khps.append(
                tries / elapsed / 1000 if elapsed > 0 else 0.0
            )

        merged = {
            "found": False,
            "tries_used": total_tries,
            "gate_passes": total_gate,
            "elapsed_s": wall_elapsed,
            "per_gpu_khps": per_gpu_khps,
            "gpu_devices": list(self.gpu_devices),
            "num_gpus": self.num_gpus,
        }
        if wall_elapsed > 0 and total_tries > 0:
            merged["nonce_khps_total"] = total_tries / wall_elapsed / 1000

        errors = [r.get("error") for r in results if r.get("error")]
        if errors and len(errors) == len(results):
            merged["error"] = errors[0]

        backends = {r.get("backend") for r in results if r.get("backend")}
        if backends:
            merged["backend"] = "+".join(sorted(backends))

        found = [r for r in results if r.get("found")]
        if found:
            blocks = [r for r in found if r.get("is_block")]
            winner = blocks[0] if blocks else found[0]
            solutions = [
                solution
                for result in found
                for solution in (result.get("solutions") or [result])
            ]
            merged.update(winner)
            merged["found"] = True
            merged["solutions"] = solutions
            merged["per_gpu_khps"] = per_gpu_khps
            merged["gpu_devices"] = list(self.gpu_devices)
            merged["num_gpus"] = self.num_gpus
            if wall_elapsed > 0 and total_tries > 0:
                merged["nonce_khps_total"] = total_tries / wall_elapsed / 1000
        return merged

    def solve(self, job, nonce_start: int = 0, max_tries: int = 20_000_000,
              max_seconds: float = 5.0) -> dict:
        if not self.solvers:
            return {"found": False, "error": "no solvers configured"}

        if len(self.solvers) == 1:
            return self.solvers[0].solve(
                job, nonce_start=nonce_start, max_tries=max_tries, max_seconds=max_seconds,
            )

        results: list[dict] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(self.solvers)) as pool:
            futures = {
                pool.submit(
                    solver.solve,
                    job,
                    nonce_start + idx * max_tries,
                    max_tries,
                    max_seconds,
                ): idx
                for idx, solver in enumerate(self.solvers)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    log.error("GPU %d solver failed: %s", self.gpu_devices[idx], e)
                    result = {"found": False, "error": str(e), "gpu_index": idx}
                else:
                    result["gpu_index"] = idx
                    result["gpu_device"] = self.gpu_devices[idx]
                results.append(result)

        merged = self._merge_results(results, wall_elapsed=time.time() - t0)
        khps = merged.get("nonce_khps_total")
        if khps:
            self.last_observed_nps = float(khps) * 1000.0
        return merged

    def stop(self):
        for solver in self.solvers:
            solver.stop()
