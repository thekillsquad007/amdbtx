import socket
import json
import time
import uuid
import logging
import random
import threading
from collections import deque
from typing import Any

from . import PROTOCOL_CAPABILITIES, USER_AGENT, __version__
from .config import fully_qualified_worker
from .hardware import (
    collect_runtime_metrics,
    collect_static_hardware,
    detect_gpu_info,
    hardware_summary_string,
    solver_sha256_hex,
)

METRICS_REPORT_INTERVAL_SEC = 60.0

log = logging.getLogger(__name__)


class Job:
    __slots__ = (
        "job_id", "version", "prev_hash", "merkle_root", "time",
        "bits", "target", "seed_a", "seed_b", "block_height",
        "matmul_n", "matmul_b", "matmul_r", "epsilon_bits",
        "parent_mtp", "nonce64_start", "clean_jobs", "received_at",
        "luckypool_nonce_bits",
    )

    def __init__(self, **kwargs):
        for s in self.__slots__:
            default = 0 if s in ("version", "time", "block_height", "matmul_n", "matmul_b", "matmul_r", "epsilon_bits", "nonce64_start", "luckypool_nonce_bits") else (
                None if s == "parent_mtp" else (
                "0" * 64 if s in ("prev_hash", "merkle_root", "seed_a", "seed_b", "target") else (
                    "1d17c609" if s == "bits" else (
                        False if s == "clean_jobs" else (
                            time.time() if s == "received_at" else "")))))
            setattr(self, s, kwargs.get(s, default))

    @classmethod
    def _validate_matmul(cls, matmul: dict, *, context: str) -> None:
        missing = [k for k in ("seed_a", "seed_b", "block_height") if k not in matmul]
        if missing:
            raise ValueError(
                f"notify missing required matmul fields: {missing} "
                f"(got keys: {sorted(matmul.keys())}); {context}"
            )
        height = int(matmul.get("block_height", 0) or 0)
        if height >= 130500 and matmul.get("parent_mtp") is None:
            raise ValueError(
                f"notify missing parent_mtp at block_height={height} "
                f"(matmul_parent_mtp_seed_v3); {context}"
            )

    @classmethod
    def from_notify(cls, params) -> "Job":
        if isinstance(params, list) and len(params) >= 6:
            matmul = params[8] if len(params) > 8 and isinstance(params[8], dict) else {}
            cls._validate_matmul(matmul, context="refusing placeholder seeds")
            return cls(
                job_id=params[0],
                version=int(params[1]),
                prev_hash=params[2],
                merkle_root=params[3],
                time=int(params[4]),
                bits=params[5],
                target=params[6] if len(params) > 6 else "",
                clean_jobs=bool(params[7]) if len(params) > 7 else False,
                seed_a=matmul.get("seed_a", "0" * 64),
                seed_b=matmul.get("seed_b", "0" * 64),
                block_height=int(matmul.get("block_height", 0)),
                matmul_n=int(matmul.get("matmul_n", 512)),
                matmul_b=int(matmul.get("matmul_b", 16)),
                matmul_r=int(matmul.get("matmul_r", 8)),
                epsilon_bits=int(matmul.get("epsilon_bits", 18)),
                parent_mtp=int(matmul["parent_mtp"]) if matmul.get("parent_mtp") is not None else None,
                nonce64_start=int(matmul.get("nonce64_start", 0)),
            )
        if isinstance(params, dict):
            matmul = params.get("matmul", {})
            if isinstance(matmul, dict):
                params = {**params, **matmul}
            cls._validate_matmul(params, context="refusing placeholder seeds")
            return cls(
                job_id=params.get("job_id", ""),
                version=int(params.get("version", 0)),
                prev_hash=params.get("prev_hash", "0" * 64),
                merkle_root=params.get("merkle_root", "0" * 64),
                time=int(params.get("time", 0)),
                bits=params.get("bits", "1d17c609"),
                target=params.get("target", params.get("share_target", "")),
                seed_a=params.get("seed_a", "0" * 64),
                seed_b=params.get("seed_b", "0" * 64),
                block_height=int(params.get("block_height", 0)),
                matmul_n=int(params.get("matmul_n", 512)),
                matmul_b=int(params.get("matmul_b", 16)),
                matmul_r=int(params.get("matmul_r", 8)),
                epsilon_bits=int(params.get("epsilon_bits", 18)),
                parent_mtp=int(params["parent_mtp"]) if params.get("parent_mtp") is not None else None,
                nonce64_start=int(params.get("nonce64_start", 0)),
            )
        raise ValueError(f"cannot parse notify params: type={type(params)}")

    @classmethod
    def from_luckypool(cls, params: dict) -> "Job":
        """Parse LuckyPool's login/job JSON-RPC dialect."""
        nonce_bits = int(params.get("nonceBits", 0) or 0)
        nonce_prefix = str(params.get("noncePrefix", "") or "0")
        try:
            prefix_value = int(nonce_prefix, 10 if nonce_prefix.isdigit() else 16)
        except ValueError:
            prefix_value = int(nonce_prefix, 10)
        return cls(
            job_id=str(params.get("jobId", "")),
            version=int(params.get("nVersion", 0)),
            prev_hash=str(params.get("prevHash", "0" * 64)),
            merkle_root=str(params.get("merkleRoot", "0" * 64)),
            time=int(params.get("nTime", 0)),
            bits=str(params.get("nBits", "1d17c609")),
            target=str(params.get("shareTarget", "")),
            seed_a="0" * 64,
            seed_b="0" * 64,
            block_height=int(params.get("height", 0)),
            matmul_n=int(params.get("matmulDim", 512)),
            matmul_b=int(params.get("b", 16)),
            matmul_r=int(params.get("r", 8)),
            epsilon_bits=int(params.get("epsilonBits", 18)),
            parent_mtp=int(params["parentMtp"]) if params.get("parentMtp") is not None else None,
            nonce64_start=prefix_value << nonce_bits,
            luckypool_nonce_bits=nonce_bits,
            clean_jobs=bool(params.get("cleanJobs", False)),
        )

    @staticmethod
    def infer_luckypool_nonce_bits(nonce64_start: int) -> int:
        """Infer LuckyPool suffix width from an aligned nonce prefix."""
        if nonce64_start <= 0:
            return 0
        trailing = 0
        value = int(nonce64_start)
        while trailing < 64 and (value & 1) == 0:
            trailing += 1
            value >>= 1
        # LuckyPool BTX jobs observed so far use a 40-bit nonce suffix.
        # Only infer common byte-aligned suffix sizes to avoid masking a full nonce
        # by accident if a pool ever sends an unaligned start.
        return trailing if trailing in (32, 40, 48) else 0

    def should_replace(self, other: "Job") -> bool:
        # Same-height notify rotations (clean=true) update job_id/target only.
        return other.block_height != self.block_height

    def merge_from(self, other: "Job") -> None:
        """Apply same-height pool updates without resetting our nonce counter."""
        saved_nonce = self.nonce64_start
        self.job_id = other.job_id
        self.version = other.version
        self.prev_hash = other.prev_hash
        self.merkle_root = other.merkle_root
        self.time = other.time
        self.bits = other.bits
        self.target = other.target
        self.seed_a = other.seed_a
        self.seed_b = other.seed_b
        self.block_height = other.block_height
        self.matmul_n = other.matmul_n
        self.matmul_b = other.matmul_b
        self.matmul_r = other.matmul_r
        self.epsilon_bits = other.epsilon_bits
        self.parent_mtp = other.parent_mtp
        self.clean_jobs = other.clean_jobs
        self.received_at = other.received_at
        self.luckypool_nonce_bits = other.luckypool_nonce_bits
        self.nonce64_start = saved_nonce


class StratumClient:
    def __init__(self, host: str, port: int, payout_address: str, worker_name: str,
                 cfg: dict | None = None):
        self.host = host
        self.port = port
        self.payout_address = payout_address
        self.operator_worker_name = worker_name
        self._submit_worker = fully_qualified_worker(payout_address, worker_name)
        self.worker_name = self._submit_worker
        self._canonical_worker_name = ""
        self._protocol = "stratum"
        self.cfg = cfg or {}
        self.sock: socket.socket | None = None
        self._buf = b""
        self._msg_id = 0
        self._extranonce1 = ""
        self._extranonce2_size = 4
        self._difficulty = 1.0
        self._session_id = uuid.uuid4().hex
        self.shares_accepted = 0
        self.shares_rejected = 0
        self.blocks_found = 0
        self._pending_submits: dict[int, dict[str, Any]] = {}
        self._accepted_share_events: deque[tuple[float, float]] = deque()
        self._current_job: Job | None = None
        self._metrics_stop = threading.Event()
        self._metrics_thread: threading.Thread | None = None
        self._solver_ref: Any = None
        self._connect()

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _connect(self):
        log.info("connecting to %s:%d", self.host, self.port)
        self.sock = socket.create_connection((self.host, self.port))
        self._handshake()

    def _handshake(self):
        hw = collect_static_hardware(
            miner_version=__version__,
            cpu_threads_allocated=self.cfg.get("solver_threads"),
            solver_env=self._build_solver_env(),
            solver_path=self.cfg.get("gbt_solve_path"),
        )
        log.info("hardware: %s", hardware_summary_string(hw))
        gpus = hw.get("gpus") or []
        if not gpus:
            log.warning(
                "no GPUs in hardware handshake — pool will assign CPU-* canonical name "
                "and vardiff may stay at minimum; ensure rocm-smi works with "
                "HSA_ENABLE_DXG_DETECTION=1 before mining"
            )
        else:
            for i, g in enumerate(gpus):
                log.info(
                    "pool gpu[%d]: model=%s uuid=%s arch=%s",
                    i,
                    g.get("model", "?"),
                    g.get("gpu_uuid", "?"),
                    g.get("compute_capability", "?"),
                )

        extension = {
            "protocol_compliant": list(PROTOCOL_CAPABILITIES),
            "hardware": hw,
            "operator_label": self.operator_worker_name,
            "session_id": self._session_id,
        }

        try:
            sub = self._call("mining.subscribe", [USER_AGENT, extension])
        except RuntimeError as e:
            if "Method not found" in str(e):
                log.info(
                    "pool does not support mining.subscribe; trying LuckyPool login protocol"
                )
                self._lucky_handshake()
                return
            raise
        if isinstance(sub, list) and len(sub) >= 3:
            self._extranonce1 = sub[1]
            self._extranonce2_size = int(sub[2])
        else:
            raise RuntimeError(f"bad subscribe response: {sub!r}")
        log.info("subscribed; extranonce1=%s en2_size=%d", self._extranonce1[:8], self._extranonce2_size)

        # dexbtx-miner uses address.worker_name for authorize AND submit; the
        # pool keys vardiff/report_metrics to that identity. Canonical names are
        # dashboard labels only (mining.set_canonical_name).
        ok = self._call("mining.authorize", [self._submit_worker, ""])
        if not ok:
            raise RuntimeError(f"authorize rejected for worker={self._submit_worker}")
        log.info("authorized as %s", self._submit_worker)

        # Drain handshake messages (set_canonical_name, set_difficulty, etc.).
        self._drain_handshake_messages(3.0)

        if self._canonical_worker_name:
            log.info("canonical worker name: %s", self._canonical_worker_name)
            if self._canonical_worker_name.upper().startswith("CPU-"):
                log.warning(
                    "pool assigned CPU canonical name (%s) — vardiff may not ramp; "
                    "restart after GPU detection is fixed (see pool gpu[] lines above)",
                    self._canonical_worker_name,
                )
            else:
                log.info(
                    "if pool dashboard still lists a stale CPU-* worker with the same "
                    "shares, stop all other miner instances and wait ~10 min for it to age out"
                )
        else:
            log.info("no canonical name assigned, using address-only")
        log.info(
            "pool vardiff: difficulty=%s (dashboard hashrate scales with this, not matmul_khps)",
            self._difficulty,
        )

    def _lucky_handshake(self) -> None:
        self._protocol = "luckypool"
        ok = self._call(
            "login",
            {
                "login": self._submit_worker,
                "pass": "x",
                "agent": USER_AGENT,
                "algo": "btx",
            },
        )
        if not ok:
            raise RuntimeError(f"luckypool login rejected for worker={self._submit_worker}")
        # LuckyPool currently advertises BTX vardiff start difficulty 0.0002.
        self._difficulty = float(self.cfg.get("pool_difficulty", 0.0002) or 0.0002)
        log.info(
            "luckypool login OK as %s; waiting for job messages",
            self._submit_worker,
        )

    def start_metrics_reporter(self, solver: Any) -> None:
        """Background worker.report_metrics heartbeats (dexbtx-miner parity)."""
        if self._protocol == "luckypool":
            log.info("luckypool protocol: worker.report_metrics disabled")
            return
        self._solver_ref = solver
        self._metrics_stop.clear()
        if self._metrics_thread is not None and self._metrics_thread.is_alive():
            return
        self._metrics_thread = threading.Thread(
            target=self._metrics_loop, name="pool-metrics", daemon=True,
        )
        self._metrics_thread.start()

    def stop_metrics_reporter(self) -> None:
        self._metrics_stop.set()

    def _metrics_loop(self) -> None:
        time.sleep(random.uniform(5.0, METRICS_REPORT_INTERVAL_SEC))
        while not self._metrics_stop.is_set():
            try:
                solver = self._solver_ref
                solver_nps = getattr(solver, "last_observed_nps", None) if solver else None
                if solver_nps and solver_nps > 0:
                    self.report_metrics(
                        float(solver_nps),
                        self.shares_accepted + self.shares_rejected,
                    )
            except Exception as e:
                log.debug("metrics report failed (non-fatal): %s", e)
            if self._metrics_stop.wait(METRICS_REPORT_INTERVAL_SEC):
                break

    def _drain_handshake_messages(self, seconds: float):
        if self.sock is None:
            return
        deadline = time.time() + seconds
        old_timeout = self.sock.gettimeout()
        try:
            self.sock.settimeout(0.2)
            while time.time() < deadline:
                try:
                    msg = self._recv()
                except socket.timeout:
                    continue
                self._dispatch_message(msg)
        finally:
            self.sock.settimeout(old_timeout)

    def _build_solver_env(self) -> dict[str, Any]:
        backend = self.cfg.get("solver_backend", "rocm")
        if backend in ("rocm", "hip"):
            backend = "hip"
        env: dict[str, Any] = {
            "BTX_MATMUL_BACKEND": backend,
            "BTX_MATMUL_GPU_INPUTS": self.cfg.get("gpu_inputs", 0),
            "BTX_MATMUL_SOLVE_BATCH_SIZE": self.cfg.get("solver_batch_size", 1024),
            "BTX_MATMUL_PREPARE_PREFETCH_DEPTH": self.cfg.get("solver_prefetch_depth", 8),
            "BTX_MATMUL_PREPARE_WORKERS": self.cfg.get("solver_prepare_workers", 16),
            "BTX_MATMUL_PIPELINE_ASYNC": self.cfg.get("solver_pipeline_async", 1),
            "BTX_MATMUL_SOLVER_THREADS": self.cfg.get("solver_threads", 8),
        }
        import os as _os
        for k, v in _os.environ.items():
            if k.startswith("BTX_MATMUL_") and k not in env:
                env[k] = v
        return env

    def _send(self, msg: dict):
        if self.sock is None:
            raise ConnectionError("not connected")
        line = json.dumps(msg) + "\n"
        self.sock.sendall(line.encode())

    def _recv(self) -> dict:
        while b"\n" not in self._buf:
            if self.sock is None:
                raise ConnectionError("not connected")
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("stratum connection closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode())

    def _complete_pending_submit(self, msg_id: int, resp: dict) -> None:
        meta = self._pending_submits.pop(msg_id, None)
        if not meta:
            return
        job_id = meta["job_id"]
        nonce_hex = meta["nonce_hex"]
        if resp.get("error"):
            self.shares_rejected += 1
            log.info(
                "share REJECTED job=%s nonce=%s (a/r=%d/%d) error=%s",
                job_id, nonce_hex,
                self.shares_accepted, self.shares_rejected, resp["error"],
            )
            return
        self.shares_accepted += 1
        difficulty = float(meta.get("difficulty", self._difficulty) or self._difficulty)
        self._accepted_share_events.append((time.time(), difficulty))
        self._prune_accepted_share_events(time.time() - 600.0)
        is_block = meta.get("is_block", False)
        if is_block:
            self.blocks_found += 1
        log.info(
            "share OK job=%s nonce=%s is_block=%s (a/r/b=%d/%d/%d)",
            job_id, nonce_hex, is_block,
            self.shares_accepted, self.shares_rejected, self.blocks_found,
        )

    def _dispatch_message(self, msg: dict) -> bool:
        """Route one pool line. Returns True if it was a pending submit response."""
        msg_id = msg.get("id")
        if isinstance(msg_id, str):
            try:
                msg_id = int(msg_id, 10)
            except ValueError:
                pass
        if msg_id is not None and msg_id in self._pending_submits:
            self._complete_pending_submit(msg_id, msg)
            return True
        if msg.get("method"):
            self._handle_server_message(msg)
        elif msg_id is not None:
            log.debug("unmatched pool response id=%r msg=%r", msg.get("id"), msg)
        return False

    def process_available_messages(self) -> int:
        """Non-blocking drain: apply notify/vardiff and finish async submit responses."""
        if self.sock is None:
            return 0
        count = 0
        old_timeout = self.sock.gettimeout()
        try:
            self.sock.setblocking(False)
            while True:
                try:
                    msg = self._recv()
                except (BlockingIOError, ConnectionError):
                    break
                self._dispatch_message(msg)
                count += 1
        finally:
            self.sock.setblocking(True)
            self.sock.settimeout(old_timeout)
        return count

    def _handle_server_message(self, msg: dict):
        method = msg.get("method")
        if method == "mining.notify":
            params = msg.get("params", [])
            try:
                job = Job.from_notify(params)
                # dexbtx v0.4.2: same-parent rotations must not reset nonce progress,
                # even when clean=true (pool job_id rotation / vardiff retarget).
                if (
                    self._current_job is not None
                    and self._current_job.prev_hash == job.prev_hash
                    and self._current_job.block_height == job.block_height
                ):
                    job.nonce64_start = self._current_job.nonce64_start
                self._current_job = job
                log.info("notify job=%s height=%d clean=%s",
                         self._current_job.job_id, self._current_job.block_height,
                         self._current_job.clean_jobs)
            except (IndexError, KeyError, ValueError) as e:
                log.warning("malformed notify: %s", e)
        elif method == "job":
            params = msg.get("params", {})
            try:
                job = Job.from_luckypool(params)
                if (
                    job.luckypool_nonce_bits <= 0
                    and self._current_job is not None
                    and self._current_job.luckypool_nonce_bits > 0
                ):
                    job.luckypool_nonce_bits = self._current_job.luckypool_nonce_bits
                if (
                    self._current_job is not None
                    and self._current_job.prev_hash == job.prev_hash
                    and self._current_job.block_height == job.block_height
                ):
                    job.nonce64_start = self._current_job.nonce64_start
                    job.luckypool_nonce_bits = self._current_job.luckypool_nonce_bits
                self._current_job = job
                log.info(
                    "luckypool job=%s height=%d clean=%s nonce_start=%d nonce_bits=%d",
                    job.job_id, job.block_height, job.clean_jobs,
                    job.nonce64_start, job.luckypool_nonce_bits,
                )
            except (TypeError, ValueError) as e:
                log.warning("malformed luckypool job: %s; params=%r", e, params)
        elif method == "mining.set_difficulty":
            params = msg.get("params", [])
            if params:
                try:
                    new_diff = float(params[0])
                    old_diff = self._difficulty
                    self._difficulty = new_diff
                    if new_diff != old_diff:
                        log.info(
                            "pool vardiff: difficulty %.6g -> %.6g "
                            "(higher = harder shares = higher dashboard hashrate per share)",
                            old_diff,
                            new_diff,
                        )
                except (TypeError, ValueError):
                    pass
        elif method == "mining.set_extranonce":
            params = msg.get("params", [])
            if params:
                try:
                    self._extranonce1 = params[0]
                    self._extranonce2_size = int(params[1])
                except (IndexError, TypeError, ValueError):
                    pass
        elif method == "mining.set_canonical_name":
            params = msg.get("params")
            canonical = self._extract_canonical_name(params)
            if canonical:
                self._canonical_worker_name = canonical
            log.info("canonical name: %s", params)

    @staticmethod
    def _extract_canonical_name(params: Any) -> str:
        if isinstance(params, list) and params:
            first = params[0]
            if isinstance(first, dict):
                return str(first.get("canonical_name", "") or "")
        if isinstance(params, dict):
            return str(params.get("canonical_name", "") or "")
        return ""

    def _call(self, method: str, params: Any) -> Any:
        msg_id = self._next_id()
        self._send({"id": msg_id, "method": method, "params": params})
        deadline = time.time() + 30.0
        while time.time() < deadline:
            msg = self._recv()
            if msg.get("id") == msg_id:
                if msg.get("error") is not None:
                    raise RuntimeError(f"pool error: {msg['error']}")
                return msg.get("result")
            self._dispatch_message(msg)
        raise RuntimeError(f"{method} timed out")

    def send_authorize(self, payout_address: str, worker_name: str | None = None):
        if "." in payout_address and worker_name is None:
            worker = payout_address
        else:
            worker = fully_qualified_worker(
                payout_address, worker_name or self.operator_worker_name,
            )
        self._submit_worker = worker
        self.worker_name = worker
        msg_id = self._next_id()
        self._send({"id": msg_id, "method": "mining.authorize", "params": [worker, ""]})
        log.info("re-authorized as %s", worker)

    def get_job(self) -> Job:
        if self._current_job is not None:
            job = self._current_job
            self._current_job = None
            return job
        while True:
            msg = self._recv()
            method = msg.get("method")
            if method == "mining.notify":
                params = msg.get("params", [])
                try:
                    return Job.from_notify(params)
                except (IndexError, KeyError, ValueError) as e:
                    log.warning("malformed notify: %s; params=%r", e, params)
            else:
                self._dispatch_message(msg)
                if self._current_job is not None:
                    job = self._current_job
                    self._current_job = None
                    return job

    def wait_for_job(self) -> Job:
        while True:
            msg = self._recv()
            self._dispatch_message(msg)
            if self._current_job is not None:
                job = self._current_job
                self._current_job = None
                return job

    def submit_share(self, job: Job, result: dict, *, wait: bool = False):
        if self._protocol == "luckypool":
            return self._submit_luckypool_share(job, result, wait=wait)
        worker = self._submit_worker or self.worker_name or self.payout_address
        nonce_hex = f"{int(result['nonce64']):016x}" if "nonce64" in result else ""
        ntime_val = int(result.get("ntime") or job.time)
        ntime = f"{ntime_val:08x}"
        extranonce2 = "00" * self._extranonce2_size
        params = [worker, job.job_id, extranonce2, ntime, nonce_hex]
        log.info(
            "submit job=%s ntime=%s nonce=%s digest=%s target=%s",
            job.job_id, ntime, nonce_hex,
            result.get("digest", ""), (job.target or "")[:16],
        )
        msg_id = self._next_id()
        is_block = bool(result.get("is_block", False))
        self._pending_submits[msg_id] = {
            "job_id": job.job_id,
            "nonce_hex": nonce_hex,
            "is_block": is_block,
            "difficulty": float(self._difficulty or 0.0),
            "sent_at": time.time(),
        }
        self._send({"id": msg_id, "method": "mining.submit", "params": params})
        if not wait:
            return
        deadline = time.time() + 30.0
        old_timeout = self.sock.gettimeout() if self.sock is not None else None
        try:
            if self.sock is not None:
                self.sock.settimeout(1.0)
            while time.time() < deadline:
                if msg_id not in self._pending_submits:
                    return
                try:
                    resp = self._recv()
                except socket.timeout:
                    continue
                if self._dispatch_message(resp):
                    return
        finally:
            if self.sock is not None:
                self.sock.settimeout(old_timeout)
        self._pending_submits.pop(msg_id, None)
        log.warning("submit timed out job=%s nonce=%s", job.job_id, nonce_hex)

    def _submit_luckypool_share(self, job: Job, result: dict, *, wait: bool = False):
        if "nonce64" in result:
            nonce64 = int(result["nonce64"])
            nonce_bits = int(getattr(job, "luckypool_nonce_bits", 0) or 0)
            if nonce_bits <= 0:
                nonce_bits = Job.infer_luckypool_nonce_bits(
                    int(getattr(job, "nonce64_start", 0) or 0)
                )
            if nonce_bits > 0:
                nonce_mask = (1 << nonce_bits) - 1
                nonce_width = (nonce_bits + 3) // 4
                nonce_hex = f"{nonce64 & nonce_mask:0{nonce_width}x}"
            else:
                nonce_hex = f"{nonce64:016x}"
        else:
            nonce_hex = ""
        digest_hex = str(result.get("digest", ""))
        log.info(
            "submit luckypool job=%s nonce=%s digest=%s target=%s",
            job.job_id, nonce_hex, digest_hex, (job.target or "")[:16],
        )
        msg_id = self._next_id()
        is_block = bool(result.get("is_block", False))
        self._pending_submits[msg_id] = {
            "job_id": job.job_id,
            "nonce_hex": nonce_hex,
            "is_block": is_block,
            "difficulty": float(self._difficulty or 0.0),
            "sent_at": time.time(),
        }
        self._send({
            "id": msg_id,
            "method": "submit",
            "params": {
                "jobId": job.job_id,
                "nonce": nonce_hex,
                "result": digest_hex,
            },
        })
        if not wait:
            return
        deadline = time.time() + 30.0
        old_timeout = self.sock.gettimeout() if self.sock is not None else None
        try:
            if self.sock is not None:
                self.sock.settimeout(1.0)
            while time.time() < deadline:
                if msg_id not in self._pending_submits:
                    return
                try:
                    resp = self._recv()
                except socket.timeout:
                    continue
                if self._dispatch_message(resp):
                    return
        finally:
            if self.sock is not None:
                self.sock.settimeout(old_timeout)
        self._pending_submits.pop(msg_id, None)
        log.warning("submit timed out job=%s nonce=%s", job.job_id, nonce_hex)

    def _prune_accepted_share_events(self, cutoff: float) -> None:
        while self._accepted_share_events and self._accepted_share_events[0][0] < cutoff:
            self._accepted_share_events.popleft()

    def pool_credit_stats(self, window_sec: float = 60.0) -> dict[str, float]:
        """Rolling accepted-share estimate using submit-time vardiff credit."""
        now = time.time()
        cutoff = now - window_sec
        self._prune_accepted_share_events(now - max(window_sec, 600.0))
        events = [(ts, diff) for ts, diff in self._accepted_share_events if ts >= cutoff]
        accepted = len(events)
        credit = sum(diff for _, diff in events)
        return {
            "window_sec": float(window_sec),
            "accepted": float(accepted),
            "credit": float(credit),
            "credit_per_min": credit * 60.0 / window_sec if window_sec > 0 else 0.0,
            "avg_diff": credit / accepted if accepted else 0.0,
        }

    def report_metrics(self, solver_nps: float, shares_total: int = 0) -> None:
        if solver_nps <= 0 or self.sock is None:
            return
        solver_path = self.cfg.get("gbt_solve_path")
        backend = self.cfg.get("solver_backend", "rocm")
        if backend in ("rocm", "hip"):
            backend = "hip"
        payload = collect_runtime_metrics(
            session_id=self._session_id,
            solver_nps=solver_nps,
            shares_session_total=shares_total,
            wrapper_version=__version__,
            solver_sha256=solver_sha256_hex(solver_path),
            solver_backend=backend,
        )
        # dexbtx-miner sends params[0] as a dict; a JSON string is ignored by
        # the pool and vardiff stays at minimum (~0.34 nps/share credit).
        self._send({
            "method": "worker.report_metrics",
            "params": [payload],
        })
        log.info(
            "report_metrics solver_nps=%.0f shares=%d pool_diff=%.6g "
            "(pool uses this + share rate to adjust vardiff)",
            solver_nps,
            shares_total,
            self._difficulty,
        )
