import subprocess
import json
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)


class GBTSolveWrapper:
    @staticmethod
    def _supports_parent_mtp_v3(version_output: str) -> bool:
        text = str(version_output or "").lower()
        if "parent-mtp" in text or "parent_mtp" in text:
            return True
        import re
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
        if not match:
            return False
        major, minor, patch = (int(part) for part in match.groups())
        return (major, minor, patch) >= (2, 1, 0)

    def __init__(self, solver_path: str, backend: str = "rocm", threads: int = 8,
                 prepare_workers: int = 16, batch_size: int = 81920,
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
        if self.batch_size > 131072:
            env.setdefault("BTX_MATMUL_MAX_SCAN_BATCH", str(self.batch_size))
        # Default trust-GPU to ON; pool re-verifies all shares, so a bad GPU
        # digest is just rejected.  Users can override via shell env.
        env.setdefault("BTX_MATMUL_TRUST_GPU_SHARES", "1")
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
        if not self._ensure_running():
            return {"found": False, "error": "solver not running"}

        from .stratum_client import Job

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
                "share_target": share_target,
            }
            if job.parent_mtp is not None:
                payload["parent_mtp"] = int(job.parent_mtp)
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
                        solver_elapsed = float(result.get("elapsed_s", 0) or 0)
                        wall_elapsed = time.time() - t0
                        elapsed = solver_elapsed if solver_elapsed > 0 else wall_elapsed
                        tries = int(result.get("tries_used", 0) or 0)
                        if elapsed > 0 and tries > 0:
                            self.last_observed_nps = tries / elapsed
                        solutions = result.get("solutions")
                        if isinstance(solutions, list) and solutions:
                            normalized = []
                            for item in solutions:
                                if not isinstance(item, dict):
                                    continue
                                entry = dict(item)
                                if entry.get("nonce64") is None and entry.get("nonce") is not None:
                                    nonce_val = entry["nonce"]
                                    entry["nonce64"] = (
                                        int(nonce_val, 16)
                                        if isinstance(nonce_val, str)
                                        and not nonce_val.lstrip("-").isdigit()
                                        else int(nonce_val)
                                    )
                                normalized.append(entry)
                            if normalized:
                                result["solutions"] = normalized
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
        batch_size: int = 81920,
        prefetch_depth: int = 8,
        pipeline_async: int = 1,
        gpu_inputs: int = 0,
        gpu_devices: list[int] | None = None,
        runtime_ld_path: str = "",
    ):
        self.gpu_devices = list(gpu_devices or [0])
        self.last_observed_nps: float | None = None
        self._peak_nonce_nps: float = 0.0
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
        per_gpu_matmul_khps = []
        for r in results:
            elapsed = float(r.get("elapsed_s", 0) or 0)
            gate = int(r.get("gate_passes", 0) or 0)
            tries = int(r.get("tries_used", 0) or 0)
            per_gpu_khps.append(tries / elapsed / 1000 if elapsed > 0 else 0.0)
            per_gpu_matmul_khps.append(
                gate / elapsed / 1000 if elapsed > 0 and gate else 0.0
            )

        merged = {
            "found": False,
            "tries_used": total_tries,
            "gate_passes": total_gate,
            "elapsed_s": wall_elapsed,
            "per_gpu_khps": per_gpu_khps,
            "per_gpu_matmul_khps": per_gpu_matmul_khps,
            "gpu_devices": list(self.gpu_devices),
            "num_gpus": self.num_gpus,
        }
        if wall_elapsed > 0:
            if total_tries > 0:
                merged["nonce_khps_total"] = total_tries / wall_elapsed / 1000
            if total_gate > 0:
                merged["matmul_khps_total"] = total_gate / wall_elapsed / 1000

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
            merged.update(winner)
            merged["found"] = True
            merged["per_gpu_khps"] = per_gpu_khps
            merged["per_gpu_matmul_khps"] = per_gpu_matmul_khps
            merged["gpu_devices"] = list(self.gpu_devices)
            merged["num_gpus"] = self.num_gpus
            if wall_elapsed > 0:
                if total_tries > 0:
                    merged["nonce_khps_total"] = total_tries / wall_elapsed / 1000
                if total_gate > 0:
                    merged["matmul_khps_total"] = total_gate / wall_elapsed / 1000
        return merged

    def solve(self, job, nonce_start: int = 0, max_tries: int = 20_000_000,
              max_seconds: float = 5.0) -> dict:
        if not self.solvers:
            return {"found": False, "error": "no solvers configured"}

        if len(self.solvers) == 1:
            result = self.solvers[0].solve(
                job, nonce_start=nonce_start, max_tries=max_tries, max_seconds=max_seconds,
            )
            observed_nps = self.solvers[0].last_observed_nps
            if not observed_nps:
                elapsed = float(result.get("elapsed_s", 0) or 0)
                tries = int(result.get("tries_used", 0) or 0)
                observed_nps = tries / elapsed if elapsed > 0 and tries > 0 else None
            self.last_observed_nps = observed_nps
            return result

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
