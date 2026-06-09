"""Solo mining against a local btxd node (GBT + submitblock)."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

from .block_builder import assemble_block_hex
from .rpc_client import BtxRpcClient, RpcError
from .stratum_client import Job

log = logging.getLogger(__name__)


class SoloClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.payout_address = cfg["payout_address"]
        self.rpc = BtxRpcClient(
            url=cfg.get("rpc_url", "http://127.0.0.1:19334"),
            rpc_user=cfg.get("rpc_user", ""),
            rpc_password=cfg.get("rpc_password", ""),
            cookie_file=cfg.get("rpc_cookie_file", ""),
            timeout=float(cfg.get("rpc_timeout", 120.0)),
        )
        self._longpollid = ""
        self._current_job: Job | None = None
        self._gbt_by_job: dict[str, dict[str, Any]] = {}
        self._payout_script: bytes | None = None
        self.blocks_found = 0
        self.blocks_rejected = 0
        self._check_node_ready()

    def _check_node_ready(self):
        try:
            info = self.rpc.call("getmininginfo", timeout=10.0)
        except Exception as e:
            raise RuntimeError(f"cannot reach btxd RPC: {e}") from e

        guard = info.get("chain_guard") or {}
        if guard.get("should_pause_mining"):
            log.warning(
                "node chain_guard suggests pausing mining: %s",
                guard.get("recommended_action", "unknown"),
            )
        log.info(
            "solo: connected to node height=%s difficulty=%s",
            info.get("blocks"), info.get("difficulty"),
        )

    def _resolve_payout_script(self) -> bytes:
        if self._payout_script is not None:
            return self._payout_script

        configured = self.cfg.get("coinbase_script_pubkey", "")
        if configured:
            self._payout_script = bytes.fromhex(configured)
            return self._payout_script

        for method, params in (
            ("validateaddress", [self.payout_address]),
            ("getaddressinfo", [self.payout_address]),
        ):
            try:
                info = self.rpc.call(method, params, timeout=10.0)
                spk = info.get("scriptPubKey")
                if spk:
                    self._payout_script = bytes.fromhex(spk)
                    log.info("solo: payout script resolved via %s", method)
                    return self._payout_script
            except RpcError:
                continue

        cli = self.cfg.get("btx_cli_path", "")
        if cli:
            try:
                out = subprocess.check_output(
                    [cli, "validateaddress", self.payout_address],
                    text=True,
                    timeout=15,
                )
                import json
                info = json.loads(out)
                spk = info.get("scriptPubKey")
                if spk:
                    self._payout_script = bytes.fromhex(spk)
                    log.info("solo: payout script resolved via btx-cli")
                    return self._payout_script
            except Exception as e:
                log.debug("btx-cli validateaddress failed: %s", e)

        raise RuntimeError(
            "cannot resolve coinbase script for payout address; set coinbase_script_pubkey "
            "in config or ensure validateaddress works on your node"
        )

    @staticmethod
    def _job_from_template(gbt: dict[str, Any], challenge: dict[str, Any]) -> Job:
        hc = challenge.get("header_context") or {}
        matmul = gbt.get("matmul") or {}
        wp = challenge.get("work_profile") or {}
        epsilon = int(
            wp.get("pre_hash_lottery", {}).get("epsilon_bits")
            or matmul.get("epsilon_bits")
            or gbt.get("epsilon_bits")
            or 18
        )
        height = int(gbt.get("height", hc.get("height", 0)))
        prev_hash = hc.get("previousblockhash") or gbt.get("previousblockhash", "")
        job_id = f"solo-{height}-{prev_hash[:16]}"

        return Job(
            job_id=job_id,
            version=int(hc.get("version", gbt.get("version", 0))),
            prev_hash=prev_hash,
            merkle_root=hc.get("merkleroot", ""),
            time=int(hc.get("time", gbt.get("curtime", 0))),
            bits=hc.get("bits", gbt.get("bits", "")),
            target=gbt.get("target", ""),
            seed_a=hc.get("seed_a", matmul.get("seed_a", gbt.get("seed_a", "0" * 64))),
            seed_b=hc.get("seed_b", matmul.get("seed_b", gbt.get("seed_b", "0" * 64))),
            block_height=height,
            matmul_n=int(matmul.get("n", gbt.get("matmul_n", 512))),
            matmul_b=int(matmul.get("b", gbt.get("matmul_b", 16))),
            matmul_r=int(matmul.get("r", gbt.get("matmul_r", 8))),
            epsilon_bits=epsilon,
            nonce64_start=0,
            clean_jobs=False,
            received_at=time.time(),
        )

    def _fetch_template(self, longpoll: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        params: dict[str, Any] = {"rules": ["segwit"]}
        if longpoll and self._longpollid:
            params["longpollid"] = self._longpollid
        timeout = float(self.cfg.get("gbt_longpoll_timeout", 60.0)) if longpoll else 30.0
        gbt = self.rpc.call("getblocktemplate", [params], timeout=timeout)
        challenge = self.rpc.call("getmatmulchallenge", [], timeout=30.0)
        self._longpollid = gbt.get("longpollid", self._longpollid)
        return gbt, challenge

    def get_job(self) -> Job:
        if self._current_job is not None:
            job = self._current_job
            self._current_job = None
            return job

        gbt, challenge = self._fetch_template(longpoll=bool(self.cfg.get("gbt_longpoll", True)))
        job = self._job_from_template(gbt, challenge)
        self._gbt_by_job[job.job_id] = gbt
        log.info(
            "solo template job=%s height=%d prev=%s... merkle=%s...",
            job.job_id, job.block_height, job.prev_hash[:16], job.merkle_root[:16],
        )
        return job

    def poll_template(self) -> None:
        try:
            gbt, challenge = self._fetch_template(longpoll=False)
            job = self._job_from_template(gbt, challenge)
            self._gbt_by_job[job.job_id] = gbt
            self._current_job = job
        except Exception as e:
            log.debug("solo template poll failed: %s", e)

    def submit_share(self, job: Job, result: dict):
        self.submit_block(job, result)

    def submit_block(self, job: Job, result: dict) -> bool:
        if not result.get("is_block"):
            log.info("solo: found share-tier hit (not a block); skipping submitblock")
            return False

        gbt = self._gbt_by_job.get(job.job_id)
        if gbt is None:
            log.warning("solo: missing cached GBT for job=%s; cannot submit", job.job_id)
            self.blocks_rejected += 1
            return False

        nonce64 = int(result["nonce64"])
        digest = result.get("digest", "")
        try:
            payout_script = self._resolve_payout_script()
            block_hex = assemble_block_hex(gbt, job, nonce64, digest, payout_script)
        except Exception as e:
            log.error("solo: block assembly failed: %s", e)
            self.blocks_rejected += 1
            return False

        try:
            submit_result = self.rpc.call("submitblock", [block_hex], timeout=60.0)
        except RpcError as e:
            log.error("solo: submitblock RPC error: %s", e)
            self.blocks_rejected += 1
            return False

        if submit_result in (None, "null", ""):
            self.blocks_found += 1
            log.info("solo: BLOCK ACCEPTED height=%d nonce=%d digest=%s",
                     job.block_height, nonce64, digest[:16])
            return True

        log.warning("solo: submitblock rejected: %s", submit_result)
        self.blocks_rejected += 1
        return False