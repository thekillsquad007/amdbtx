"""Async daemon wrapper for btx-gbt-solve HIP solver binary."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class SolverEnv:
    batch_size: int = 128
    prefetch_depth: int = 8
    prepare_workers: int = 16
    pipeline_async: bool = True
    gpu_inputs: int = 0
    solver_threads: int = 8
    backend: str = "rocm"


@dataclass
class SolveChallenge:
    version: int = 536870912
    prev_hash: str = ""
    merkle_root: str = ""
    time: int = 0
    bits: str = "1d17c609"
    seed_a: str = ""
    seed_b: str = ""
    block_height: int = 0
    matmul_n: int = 512
    matmul_b: int = 16
    matmul_r: int = 8
    epsilon_bits: int = 18
    share_target_hex: str = ""


@dataclass
class SolveResult:
    found: bool = False
    tries_used: int = 0
    elapsed_s: float = 0.0
    nonce: int = 0
    digest_hex: str = ""
    ntime: int = 0
    raw_output: str = ""
    is_block: bool = False
    nonce_end: int = 0


class GbtSolveWrapper:
    """Manages the btx-gbt-solve daemon subprocess.

    Sends jobs via stdin JSON, reads results from stdout JSON.
    Lazily spawns daemon on first use. Handles daemon crashes by respawning.
    """

    def __init__(
        self,
        gbt_solve_path: str,
        backend: str = "rocm",
        solver_threads: Optional[int] = None,
        batch_size: Optional[int] = None,
        solver_env: Optional[SolverEnv] = None,
    ) -> None:
        self._binary_path = os.path.expanduser(gbt_solve_path)
        self._backend = backend
        self._solver_env = solver_env or SolverEnv()
        if solver_threads is not None:
            self._solver_env.solver_threads = solver_threads
        if batch_size is not None:
            self._solver_env.batch_size = batch_size

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._ready = False
        self._ready_event = asyncio.Event()
        self._pending_jobs: Dict[int, asyncio.Future[SolveResult]] = {}
        self._next_job_id = 0
        self._lock = asyncio.Lock()
        self._closed = False

    def _verify_binary(self) -> None:
        """Check that the solver binary exists and is executable."""
        p = Path(self._binary_path)
        if not p.exists():
            raise FileNotFoundError(
                f"solver binary not found: {self._binary_path}. "
                "Install the HIP solver or set gbt_solve_path in config."
            )
        if not os.access(self._binary_path, os.X_OK):
            raise PermissionError(
                f"solver binary not executable: {self._binary_path}"
            )

    async def _ensure_daemon(self) -> None:
        """Lazy-spawn daemon process with HIP args if not already running."""
        if self._proc is not None and self._proc.returncode is None:
            return

        async with self._lock:
            # Double-check under lock
            if self._proc is not None and self._proc.returncode is None:
                return

            self._verify_binary()

            args = [
                self._binary_path,
                "--daemon",
                "--backend", self._backend,
                "--threads", str(self._solver_env.solver_threads),
                "--batch-size", str(self._solver_env.batch_size),
                "--prefetch-depth", str(self._solver_env.prefetch_depth),
                "--prepare-workers", str(self._solver_env.prepare_workers),
            ]

            if self._solver_env.pipeline_async:
                args.append("--pipeline-async")

            if self._solver_env.gpu_inputs > 0:
                args.extend(["--gpu-inputs", str(self._solver_env.gpu_inputs)])

            env = os.environ.copy()
            env["BTX_MATMUL_BACKEND"] = self._backend
            if self._backend in ("rocm", "hip"):
                env["HSA_OVERRIDE_GFX_VERSION"] = env.get("HSA_OVERRIDE_GFX_VERSION", "")
            # Ensure ROCm library path is in LD_LIBRARY_PATH
            for rocm_lib in ("/opt/rocm-6.0.0/lib", "/opt/rocm/lib"):
                try:
                    if os.path.isdir(rocm_lib):
                        existing = env.get("LD_LIBRARY_PATH", "")
                        if rocm_lib not in existing:
                            env["LD_LIBRARY_PATH"] = f"{rocm_lib}:{existing}" if existing else rocm_lib
                except Exception:
                    pass
            log.info("solver env LD_LIBRARY_PATH=%s", env.get("LD_LIBRARY_PATH", "(unset)"))

            log.info("spawning solver daemon: %s", " ".join(args))

            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            self._ready = False
            self._ready_event.clear()

            # Start reader task
            self._reader_task = asyncio.create_task(self._reader_loop())

            # Wait for daemon_ready on stderr (with timeout)
            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                # Check if process died — collect remaining stderr
                stderr_text = ""
                if self._proc and self._proc.returncode is not None:
                    try:
                        remaining = await asyncio.wait_for(
                            self._proc.stderr.read(), timeout=2.0
                        )
                        stderr_text = remaining.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                if stderr_text:
                    log.warning(
                        "solver daemon exited (code %s): %s",
                        self._proc.returncode, stderr_text.strip(),
                    )
                else:
                    log.warning("solver daemon did not signal ready within 15s")

    async def _shutdown_daemon(self) -> None:
        """Cleanly shut down the solver daemon."""
        self._closed = True

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                log.warning("solver daemon did not exit, sending SIGKILL")
                try:
                    self._proc.kill()
                except Exception:
                    pass
            except ProcessLookupError:
                pass

        self._proc = None
        self._ready = False

    async def _reader_loop(self) -> None:
        """Read stdout lines from daemon (results + events)."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        stderr_task = asyncio.create_task(self._stderr_reader())
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    log.debug("non-JSON from solver stdout: %s", text[:200])
                    continue

                # Daemon ready event (can also come on stdout)
                if record.get("event") == "daemon_ready":
                    self._ready = True
                    self._ready_event.set()
                    log.info("solver daemon ready")
                    continue

                # Result message — resolve pending future
                job_id = record.get("job_id")
                if job_id is not None and job_id in self._pending_jobs:
                    fut = self._pending_jobs.pop(job_id)
                    if not fut.done():
                        fut.set_result(self._result_from_record(record))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("solver reader error: %s", exc)
        finally:
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

    async def _stderr_reader(self) -> None:
        """Read stderr from daemon for logging and ready signal."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                try:
                    record = json.loads(text)
                    if record.get("event") == "daemon_ready":
                        self._ready = True
                        self._ready_event.set()
                        log.info("solver daemon ready (via stderr)")
                        continue
                except json.JSONDecodeError:
                    pass

                log.warning("solver stderr: %s", text[:500])
        except asyncio.CancelledError:
            pass

    def _result_from_record(self, record: Dict[str, Any]) -> SolveResult:
        """Parse a JSON result record from the solver daemon."""
        return SolveResult(
            found=record.get("found", False),
            tries_used=record.get("tries_used", 0),
            elapsed_s=record.get("elapsed_s", 0.0),
            nonce=record.get("nonce64", record.get("nonce", 0)),
            digest_hex=record.get("digest", record.get("digest_hex", "")),
            ntime=record.get("ntime", 0),
            raw_output=json.dumps(record),
            is_block=record.get("is_block", False),
            nonce_end=record.get("nonce64_end", record.get("nonce_end", 0)),
        )

    async def solve_slice(
        self,
        challenge: SolveChallenge,
        nonce_start: int = 0,
        max_tries: int = 20_000_000,
        max_seconds: float = 5.0,
    ) -> SolveResult:
        """Send a job to the solver daemon and return the result.

        Retries once on daemon crash (respawns the daemon).
        """
        await self._ensure_daemon()
        return await self._send_job_with_retry(challenge, nonce_start, max_tries, max_seconds)

    async def _send_job_with_retry(
        self,
        challenge: SolveChallenge,
        nonce_start: int,
        max_tries: int,
        max_seconds: float,
    ) -> SolveResult:
        """Bounded retry with daemon respawn on crash."""
        MAX_RETRIES = 2
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self._send_job(challenge, nonce_start, max_tries, max_seconds)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                last_error = exc
                log.warning(
                    "solver daemon crash (attempt %d/%d): %s",
                    attempt + 1, MAX_RETRIES + 1, exc,
                )
                await self._shutdown_daemon()
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(0.5)
                    await self._ensure_daemon()

        raise RuntimeError(f"solver daemon failed after {MAX_RETRIES + 1} attempts: {last_error}")

    async def _send_job(
        self,
        challenge: SolveChallenge,
        nonce_start: int,
        max_tries: int,
        max_seconds: float,
    ) -> SolveResult:
        """Send a single job and wait for result."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("solver daemon not running")

        job_id = self._next_job_id
        self._next_job_id += 1

        job: Dict[str, Any] = {
            "job_id": job_id,
            "version": challenge.version,
            "prev_hash": challenge.prev_hash,
            "merkle_root": challenge.merkle_root,
            "time": challenge.time,
            "bits": challenge.bits,
            "seed_a": challenge.seed_a,
            "seed_b": challenge.seed_b,
            "block_height": challenge.block_height,
            "nonce_start": nonce_start,
            "max_tries": max_tries,
            "max_seconds": max_seconds,
            "share_target": challenge.share_target_hex,
        }

        # Optional matmul params
        if challenge.matmul_n:
            job["matmul_n"] = challenge.matmul_n
        if challenge.matmul_b:
            job["matmul_b"] = challenge.matmul_b
        if challenge.matmul_r:
            job["matmul_r"] = challenge.matmul_r
        if challenge.epsilon_bits:
            job["epsilon_bits"] = challenge.epsilon_bits

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[SolveResult] = loop.create_future()
        self._pending_jobs[job_id] = fut

        line = json.dumps(job) + "\n"
        try:
            self._proc.stdin.write(line.encode("utf-8"))
            await self._proc.stdin.drain()
        except (BrokenPipeError, OSError) as exc:
            self._pending_jobs.pop(job_id, None)
            raise

        try:
            result = await asyncio.wait_for(fut, timeout=max_seconds + 30.0)
        except asyncio.TimeoutError:
            self._pending_jobs.pop(job_id, None)
            log.warning("solver job %d timed out", job_id)
            return SolveResult(
                found=False,
                tries_used=0,
                elapsed_s=max_seconds,
                nonce=nonce_start,
            )

        return result

    async def close(self) -> None:
        """Shut down the daemon and clean up."""
        await self._shutdown_daemon()

    async def __aenter__(self) -> GbtSolveWrapper:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
