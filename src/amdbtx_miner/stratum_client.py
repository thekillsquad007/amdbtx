import socket
import json
import time
import uuid
import logging
from typing import Any

from . import PROTOCOL_CAPABILITIES, USER_AGENT, __version__
from .hardware import collect_static_hardware, detect_gpu_info, hardware_summary_string

log = logging.getLogger(__name__)


class Job:
    __slots__ = (
        "job_id", "version", "prev_hash", "merkle_root", "time",
        "bits", "target", "seed_a", "seed_b", "block_height",
        "matmul_n", "matmul_b", "matmul_r", "epsilon_bits",
        "nonce64_start", "clean_jobs", "received_at",
    )

    def __init__(self, **kwargs):
        for s in self.__slots__:
            default = 0 if s in ("version", "time", "block_height", "matmul_n", "matmul_b", "matmul_r", "epsilon_bits", "nonce64_start") else (
                "0" * 64 if s in ("prev_hash", "merkle_root", "seed_a", "seed_b", "target") else (
                    "1d17c609" if s == "bits" else (
                        False if s == "clean_jobs" else (
                            time.time() if s == "received_at" else ""))))
            setattr(self, s, kwargs.get(s, default))

    @classmethod
    def from_notify(cls, params) -> "Job":
        if isinstance(params, list) and len(params) >= 6:
            matmul = params[8] if len(params) > 8 and isinstance(params[8], dict) else {}
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
                nonce64_start=int(matmul.get("nonce64_start", 0)),
            )
        if isinstance(params, dict):
            matmul = params.get("matmul", {})
            if isinstance(matmul, dict):
                params = {**params, **matmul}
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
                nonce64_start=int(params.get("nonce64_start", 0)),
            )
        raise ValueError(f"cannot parse notify params: type={type(params)}")

    def should_replace(self, other: "Job") -> bool:
        return bool(other.clean_jobs) or other.block_height != self.block_height

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
        self.clean_jobs = other.clean_jobs
        self.received_at = other.received_at
        self.nonce64_start = saved_nonce


class StratumClient:
    def __init__(self, host: str, port: int, payout_address: str, worker_name: str,
                 cfg: dict | None = None):
        self.host = host
        self.port = port
        self.payout_address = payout_address
        self.operator_worker_name = worker_name
        self.worker_name = ""
        self._canonical_worker_name = ""
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
        self._current_job: Job | None = None
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

        extension = {
            "protocol_compliant": list(PROTOCOL_CAPABILITIES),
            "hardware": hw,
            "operator_label": self.operator_worker_name,
            "session_id": self._session_id,
        }

        sub = self._call("mining.subscribe", [USER_AGENT, extension])
        if isinstance(sub, list) and len(sub) >= 3:
            self._extranonce1 = sub[1]
            self._extranonce2_size = int(sub[2])
        else:
            raise RuntimeError(f"bad subscribe response: {sub!r}")
        log.info("subscribed; extranonce1=%s en2_size=%d", self._extranonce1[:8], self._extranonce2_size)

        # Authorize once with the payout address. No re-authorize with a
        # full "address.worker" identity — that can create a separate PPLNS
        # sub-account on the pool's side.
        ok = self._call("mining.authorize", [self.payout_address, ""])
        if not ok:
            raise RuntimeError(f"authorize rejected for address={self.payout_address}")
        log.info("authorized as %s", self.payout_address)

        # Drain handshake messages (set_canonical_name, set_difficulty, etc.).
        # The canonical worker name is used as a label for share submissions,
        # not for re-authorization.
        self._drain_handshake_messages(2.0)

        if self._canonical_worker_name:
            self.worker_name = self._canonical_worker_name
            log.info("canonical worker name: %s", self.worker_name)
        else:
            log.info("no canonical name assigned, using address-only")

    def _drain_handshake_messages(self, seconds: float):
        if self.sock is None:
            return
        deadline = time.time() + seconds
        old_timeout = self.sock.gettimeout()
        try:
            self.sock.settimeout(0.2)
            while time.time() < deadline and not self._canonical_worker_name:
                try:
                    msg = self._recv()
                except socket.timeout:
                    continue
                self._handle_server_message(msg)
        finally:
            self.sock.settimeout(old_timeout)

    def _build_solver_env(self) -> dict[str, Any]:
        env: dict[str, Any] = {
            "BTX_MATMUL_BACKEND": self.cfg.get("solver_backend", "rocm"),
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

    def _handle_server_message(self, msg: dict):
        method = msg.get("method")
        if method == "mining.notify":
            params = msg.get("params", [])
            try:
                self._current_job = Job.from_notify(params)
                log.info("notify job=%s height=%d clean=%s",
                         self._current_job.job_id, self._current_job.block_height,
                         self._current_job.clean_jobs)
            except (IndexError, KeyError, ValueError) as e:
                log.warning("malformed notify: %s", e)
        elif method == "mining.set_difficulty":
            params = msg.get("params", [])
            if params:
                try:
                    self._difficulty = float(params[0])
                    log.info("difficulty set to %s", self._difficulty)
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
                self.worker_name = canonical
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

    def _call(self, method: str, params: list) -> Any:
        msg_id = self._next_id()
        self._send({"id": msg_id, "method": method, "params": params})
        deadline = time.time() + 30.0
        while time.time() < deadline:
            msg = self._recv()
            if msg.get("id") == msg_id:
                if msg.get("error") is not None:
                    raise RuntimeError(f"pool error: {msg['error']}")
                return msg.get("result")
            self._handle_server_message(msg)
        raise RuntimeError(f"{method} timed out")

    def send_authorize(self, address: str):
        msg_id = self._next_id()
        self._send({"id": msg_id, "method": "mining.authorize", "params": [address, ""]})

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
                self._handle_server_message(msg)

    def wait_for_job(self) -> Job:
        while True:
            msg = self._recv()
            self._handle_server_message(msg)
            if self._current_job is not None:
                job = self._current_job
                self._current_job = None
                return job

    def submit_share(self, job: Job, result: dict):
        worker = self.worker_name or self.payout_address
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
        self._send({"id": msg_id, "method": "mining.submit", "params": params})
        deadline = time.time() + 30.0
        while time.time() < deadline:
            resp = self._recv()
            if resp.get("id") == msg_id:
                if resp.get("error"):
                    self.shares_rejected += 1
                    log.info("share REJECTED job=%s nonce=%s (a/r=%d/%d) error=%s",
                             job.job_id, nonce_hex,
                             self.shares_accepted, self.shares_rejected, resp["error"])
                else:
                    self.shares_accepted += 1
                    is_block = result.get("is_block", False)
                    if is_block:
                        self.blocks_found += 1
                    log.info("share OK job=%s nonce=%s is_block=%s (a/r/b=%d/%d/%d)",
                             job.job_id, nonce_hex, is_block, self.shares_accepted, self.shares_rejected, self.blocks_found)
                return
            self._handle_server_message(resp)
        log.warning("submit timed out")
