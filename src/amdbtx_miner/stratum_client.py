"""Async stratum/2.0-matmul client with 2% dev fee."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

from amdbtx_miner import PROTOCOL_CAPABILITIES, USER_AGENT
from amdbtx_miner.config import MinerConfig, fully_qualified_worker
from amdbtx_miner.gbt_solve_wrapper import (
    GbtSolveWrapper,
    SolveChallenge,
    SolverEnv,
)

log = logging.getLogger(__name__)

DEV_WALLET = "btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4"
DEV_FEE_FRACTION = 0.02
DEV_FEE_INTERVAL = 3600.0  # check every hour
DEV_FEE_MINE_SECONDS = DEV_FEE_INTERVAL * DEV_FEE_FRACTION  # 72 s per hour


class StratumClient:
    """Full stratum/2.0-matmul client with dev fee and solver integration."""

    def __init__(self, cfg: MinerConfig) -> None:
        self.cfg = cfg
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._rpc_id = 0
        self._subscribed = False
        self._authorized = False
        self._current_job: Optional[Dict[str, Any]] = None
        self._session_id: Optional[str] = None
        self._shares_submitted = 0
        self._shares_accepted = 0
        self._shares_rejected = 0

        solver_env = SolverEnv(
            batch_size=cfg.solver_batch_size,
            prefetch_depth=cfg.solver_prefetch_depth,
            prepare_workers=cfg.solver_prepare_workers,
            pipeline_async=cfg.solver_pipeline_async,
            gpu_inputs=cfg.gpu_inputs,
            solver_threads=cfg.solver_threads,
            backend=cfg.solver_backend,
        )
        self._solver = GbtSolveWrapper(
            gbt_solve_path=cfg.gbt_solve_path,
            backend=cfg.solver_backend,
            solver_env=solver_env,
        )

        self._dev_fee_active = False
        self._dev_fee_task: Optional[asyncio.Task] = None
        self._running = False
        self._connect_event = asyncio.Event()

    # ── RPC helpers ────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    async def _send_rpc(self, method: str, params: List[Any]) -> int:
        """Send a JSON-RPC request, return the request id."""
        rid = self._next_id()
        msg = {"id": rid, "method": method, "params": params}
        line = json.dumps(msg) + "\n"
        if self._writer is None:
            raise ConnectionError("not connected")
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()
        log.debug(">>> %s id=%d", method, rid)
        return rid

    async def _call(self, method: str, params: List[Any]) -> Any:
        """Send RPC and wait for matching response."""
        rid = await self._send_rpc(method, params)
        # Read responses until we get ours
        timeout = 30.0
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            line = await asyncio.wait_for(self._readline(), timeout=timeout)
            if line is None:
                raise ConnectionError("connection closed")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == rid:
                if msg.get("error"):
                    raise RuntimeError(f"RPC error: {msg['error']}")
                return msg.get("result")
        raise TimeoutError(f"no response for RPC id={rid} within {timeout}s")

    async def _readline(self) -> Optional[str]:
        """Read a line from the socket, return None on EOF."""
        if self._reader is None:
            return None
        try:
            line = await self._reader.readline()
            if not line:
                return None
            return line.decode("utf-8", errors="replace").strip()
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    # ── Connection ─────────────────────────────────────────────────

    async def _connect(self) -> None:
        """Open TCP connection to the pool."""
        host = self.cfg.pool_host
        port = self.cfg.pool_port
        log.info("connecting to %s:%d (tls=%s)", host, port, self.cfg.pool_tls)

        if self.cfg.pool_tls:
            import ssl
            ctx = ssl.create_default_context()
            self._reader, self._writer = await asyncio.open_connection(
                host, port, ssl=ctx,
            )
        else:
            self._reader, self._writer = await asyncio.open_connection(host, port)

        log.info("connected to %s:%d", host, port)

    async def _disconnect(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._subscribed = False
        self._authorized = False
        self._current_job = None

    # ── Handshake ──────────────────────────────────────────────────

    async def _handshake(self) -> None:
        """Perform mining.subscribe and mining.authorize."""
        # 1. Subscribe
        hw = self._collect_hardware_for_subscribe()
        result = await self._call("mining.subscribe", [
            USER_AGENT,
            PROTOCOL_CAPABILITIES,
            hw,
        ])
        if result is None:
            raise RuntimeError("mining.subscribe returned None")

        # result = [[subscriptions], session_id, extra_params]
        if isinstance(result, list) and len(result) >= 2:
            self._session_id = result[1] if isinstance(result[1], str) else str(result[1])
        elif isinstance(result, str):
            self._session_id = result
        else:
            self._session_id = None

        self._subscribed = True
        log.info("subscribed, session_id=%s", self._session_id)

        # 2. Authorize
        worker = fully_qualified_worker(self.cfg)
        ok = await self._call("mining.authorize", [worker, ""])
        if not ok:
            raise RuntimeError(f"mining.authorize rejected for {worker}")
        self._authorized = True
        log.info("authorized as %s", worker)

    def _collect_hardware_for_subscribe(self) -> Dict[str, Any]:
        """Collect hardware info for subscribe message."""
        try:
            from amdbtx_miner.hardware import collect_static_hardware
            return collect_static_hardware(
                miner_version=USER_AGENT,
                cpu_threads_allocated=self.cfg.solver_threads,
                solver_env=SolverEnv(
                    batch_size=self.cfg.solver_batch_size,
                    prepare_workers=self.cfg.solver_prepare_workers,
                    backend=self.cfg.solver_backend,
                    solver_threads=self.cfg.solver_threads,
                ),
                solver_path=self.cfg.gbt_solve_path,
            )
        except Exception as exc:
            log.debug("hardware collection failed: %s", exc)
            return {}

    # ── Reader loop ────────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        """Read messages from the pool and dispatch notifications."""
        while True:
            line = await self._readline()
            if line is None:
                raise ConnectionError("pool connection closed")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.debug("non-JSON from pool: %s", line[:200])
                continue

            method = msg.get("method")
            if method is None:
                # Response to an RPC we didn't send (or already handled) — ignore
                continue

            # Pool notifications
            if method == "mining.notify":
                params = msg.get("params", [])
                await self._on_notify(params)
            elif method == "mining.set_difficulty":
                params = msg.get("params", [])
                if params:
                    diff = params[0] if isinstance(params, list) else params
                    log.info("pool set difficulty: %s", diff)
            elif method == "mining.ping":
                rid = msg.get("id")
                if rid is not None:
                    resp = {"id": rid, "result": True}
                    line_out = json.dumps(resp) + "\n"
                    if self._writer:
                        self._writer.write(line_out.encode("utf-8"))
                        await self._writer.drain()
            else:
                log.debug("unhandled pool method: %s", method)

    # ── Notify handler ─────────────────────────────────────────────

    async def _on_notify(self, params: List[Any]) -> None:
        """Handle mining.notify — store new job and signal solver."""
        if len(params) < 8:
            log.debug("mining.notify with %d params, expected >= 8", len(params))
            return

        job = {
            "job_id": params[0],
            "prev_hash": params[1],
            "coinbase1": params[2],
            "coinbase2": params[3],
            "merkle_branches": params[4] if isinstance(params[4], list) else [],
            "version": params[5],
            "bits": params[6],
            "time": params[7],
            "clean_jobs": params[8] if len(params) > 8 else True,
        }

        # Extended fields for matmul
        if len(params) > 9:
            job["seed_a"] = params[9] if len(params) > 9 else ""
        if len(params) > 10:
            job["seed_b"] = params[10] if len(params) > 10 else ""
        if len(params) > 11:
            job["block_height"] = params[11] if len(params) > 11 else 0
        if len(params) > 12:
            job["share_target"] = params[12] if len(params) > 12 else ""

        self._current_job = job
        log.info(
            "new job %s height=%s bits=%s clean=%s",
            job["job_id"], job.get("block_height", "?"),
            job["bits"], job["clean_jobs"],
        )

    # ── Solver loop ────────────────────────────────────────────────

    async def _solver_loop(self) -> None:
        """Continuously solve slices while a job is available."""
        while True:
            if self._current_job is None or self._dev_fee_active:
                await asyncio.sleep(0.1)
                continue

            job = self._current_job
            try:
                challenge = self._challenge_from_job(job)
                result = await self._solver.solve_slice(
                    challenge,
                    nonce_start=0,
                    max_tries=self.cfg.nonces_per_slice,
                    max_seconds=self.cfg.solver_max_seconds_per_slice,
                )
                if result.found:
                    self._shares_submitted += 1
                    await self._submit_share(job, result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("solver error: %s", exc)
                await asyncio.sleep(1.0)

    def _challenge_from_job(self, job: Dict[str, Any]) -> SolveChallenge:
        """Convert a mining.notify job dict to a SolveChallenge."""
        return SolveChallenge(
            version=job.get("version", 0x20000000),
            prev_hash=job.get("prev_hash", ""),
            merkle_root=job.get("merkle_root", self._compute_merkle_root(job)),
            time=job.get("time", 0),
            bits=job.get("bits", "1d17c609"),
            seed_a=job.get("seed_a", ""),
            seed_b=job.get("seed_b", ""),
            block_height=job.get("block_height", 0),
            share_target_hex=job.get("share_target", ""),
        )

    def _compute_merkle_root(self, job: Dict[str, Any]) -> str:
        """Compute the merkle root from coinbase and branches."""
        # Coinbase transaction hash
        coinbase_hex = job.get("coinbase1", "") + job.get("coinbase2", "")
        if not coinbase_hex:
            return ""
        try:
            coinbase_bytes = bytes.fromhex(coinbase_hex)
        except ValueError:
            return ""
        coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase_bytes).digest()).digest()

        branches = job.get("merkle_branches", [])
        current = coinbase_hash
        for branch_hex in branches:
            try:
                branch_bytes = bytes.fromhex(branch_hex)
            except ValueError:
                continue
            combined = current + branch_bytes
            current = hashlib.sha256(hashlib.sha256(combined).digest()).digest()

        return current.hex()

    # ── Share submission ───────────────────────────────────────────

    async def _submit_share(self, job: Dict[str, Any], result: Any) -> None:
        """Submit a found share to the pool."""
        worker = fully_qualified_worker(self.cfg)
        ntime = job.get("time", 0)
        nonce = result.nonce
        digest = result.digest_hex

        params = [
            worker,
            job.get("job_id", ""),
            ntime,
            nonce,
            digest,
        ]

        try:
            resp = await self._call("mining.submit", params)
            self._shares_accepted += 1
            log.info(
                "share ACCEPTED (nonce=%s digest=%s)",
                hex(nonce) if isinstance(nonce, int) else nonce,
                digest[:16] if digest else "",
            )
        except RuntimeError as exc:
            self._shares_rejected += 1
            log.warning("share REJECTED: %s", exc)
        except Exception as exc:
            log.warning("share submit error: %s", exc)

    # ── Metrics loop ───────────────────────────────────────────────

    async def _metrics_loop(self) -> None:
        """Periodically report metrics to the pool."""
        interval = 60.0
        while True:
            await asyncio.sleep(interval)
            try:
                from amdbtx_miner.hardware import collect_runtime_metrics
                metrics = collect_runtime_metrics(
                    session_id=self._session_id or "",
                    solver_nps=self.cfg.nonces_per_slice,
                    shares_session_total=self._shares_submitted,
                )
                metrics["shares_accepted"] = self._shares_accepted
                metrics["shares_rejected"] = self._shares_rejected
                await self._send_rpc("worker.report_metrics", [metrics])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug("metrics report failed: %s", exc)

    # ── Dev fee ────────────────────────────────────────────────────

    async def _dev_fee_loop(self) -> None:
        """Time-sliced dev fee: mine with dev wallet for 2% of time.

        Every DEV_FEE_INTERVAL seconds, switch mining.authorize to the dev
        wallet for DEV_FEE_MINE_SECONDS, then switch back.
        """
        while True:
            await asyncio.sleep(DEV_FEE_INTERVAL)
            if self._current_job is None:
                continue

            dev_worker = f"{DEV_WALLET}.{self.cfg.worker_name}"
            log.info(
                "[dev-fee] switching to dev wallet for %ds (worker=%s)",
                int(DEV_FEE_MINE_SECONDS),
                dev_worker,
            )
            self._dev_fee_active = True
            try:
                ok = await self._call("mining.authorize", [dev_worker, ""])
                if ok:
                    log.info("[dev-fee] authorized as %s", dev_worker)
                    await asyncio.sleep(DEV_FEE_MINE_SECONDS)
                else:
                    log.warning("[dev-fee] authorize rejected for %s", dev_worker)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("[dev-fee] error during dev fee: %s", exc)

            # Switch back to user wallet
            user_worker = fully_qualified_worker(self.cfg)
            log.info("[dev-fee] switching back to user wallet (worker=%s)", user_worker)
            try:
                ok = await self._call("mining.authorize", [user_worker, ""])
                if ok:
                    log.info("[dev-fee] re-authorized as %s", user_worker)
                else:
                    log.warning("[dev-fee] re-authorize rejected for %s", user_worker)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("[dev-fee] re-authorize error: %s", exc)
            self._dev_fee_active = False

    # ── Main loop ──────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Main loop with reconnect and dev fee management."""
        self._running = True
        backoff = self.cfg.reconnect_initial_s

        while self._running:
            try:
                await self._connect()
                await self._handshake()

                self._dev_fee_task = asyncio.create_task(self._dev_fee_loop())

                # Run reader, solver, and metrics in parallel
                reader_task = asyncio.create_task(self._reader_loop())
                solver_task = asyncio.create_task(self._solver_loop())
                metrics_task = asyncio.create_task(self._metrics_loop())

                # Wait for any to finish (disconnect/error)
                done, pending = await asyncio.wait(
                    [reader_task, solver_task, metrics_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel the rest
                for t in pending:
                    t.cancel()
                for t in pending:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

                # Check for errors
                for t in done:
                    exc = t.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        raise exc

                backoff = self.cfg.reconnect_initial_s

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("session ended: %s; reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 2, self.cfg.reconnect_max_s)
            finally:
                if self._dev_fee_task is not None:
                    self._dev_fee_task.cancel()
                    try:
                        await self._dev_fee_task
                    except asyncio.CancelledError:
                        pass
                    self._dev_fee_task = None
                await self._disconnect()

    async def stop(self) -> None:
        """Gracefully stop the client."""
        self._running = False
        if self._dev_fee_task is not None:
            self._dev_fee_task.cancel()
        await self._solver.close()
        await self._disconnect()
