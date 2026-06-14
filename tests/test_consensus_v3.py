import unittest
from unittest.mock import Mock

from amdbtx_miner.block_builder import (
    derive_v2_seed,
    derive_v3_seed,
    resolve_header_seeds,
    uint256_to_display_hex,
)
from amdbtx_miner.gbt_solve_wrapper import GBTSolveWrapper
from amdbtx_miner.stratum_client import Job
from amdbtx_miner.stratum_client import StratumClient


HEADER = {
    "prev_hash": "00" * 31 + "06",
    "version": 4,
    "merkle_root": "00" * 31 + "07",
    "time": 1_780_000_020,
    "bits_hex": "1d00ffff",
    "nonce64": 11,
    "dim": 512,
}


class ConsensusV3Tests(unittest.TestCase):
    def test_activation_boundary_uses_v2_then_v3(self):
        before = Job(
            prev_hash=HEADER["prev_hash"],
            version=HEADER["version"],
            merkle_root=HEADER["merkle_root"],
            time=HEADER["time"],
            bits=HEADER["bits_hex"],
            block_height=130499,
            matmul_n=HEADER["dim"],
            parent_mtp=1_780_000_000,
        )
        at_fork = Job(
            prev_hash=HEADER["prev_hash"],
            version=HEADER["version"],
            merkle_root=HEADER["merkle_root"],
            time=HEADER["time"],
            bits=HEADER["bits_hex"],
            block_height=130500,
            matmul_n=HEADER["dim"],
            parent_mtp=1_780_000_000,
        )

        self.assertEqual(
            resolve_header_seeds(before, HEADER["nonce64"])[0],
            derive_v2_seed(
                **HEADER,
                height=130499,
                which=0,
            ),
        )
        self.assertEqual(
            resolve_header_seeds(at_fork, HEADER["nonce64"])[0],
            derive_v3_seed(
                **HEADER,
                parent_mtp=1_780_000_000,
                height=130500,
                which=0,
            ),
        )

    def test_v3_consensus_vectors(self):
        seed_a = derive_v3_seed(
            **HEADER,
            parent_mtp=1_780_000_000,
            height=130500,
            which=0,
        )
        seed_b = derive_v3_seed(
            **HEADER,
            parent_mtp=1_780_000_000,
            height=130500,
            which=1,
        )
        self.assertEqual(
            uint256_to_display_hex(seed_a),
            "db97f6761a7a9cccee62655e3b1bac75d680e22845b6d8febd0f82108e3c5d2d",
        )
        self.assertEqual(
            uint256_to_display_hex(seed_b),
            "49cde7a89a892807fd1292376de093b36b1562e847206abb2abd21f0c43bd0ae",
        )

    def test_v3_requires_and_commits_parent_mtp(self):
        job = Job(block_height=130500, matmul_n=512)
        with self.assertRaisesRegex(ValueError, "parent_mtp"):
            resolve_header_seeds(job, 0)

        first = derive_v3_seed(
            **HEADER,
            parent_mtp=1_780_000_000,
            height=130500,
            which=0,
        )
        second = derive_v3_seed(
            **HEADER,
            parent_mtp=1_780_000_001,
            height=130500,
            which=0,
        )
        self.assertNotEqual(first, second)

    def test_stratum_parent_mtp_propagation_and_reset(self):
        first = Job.from_notify({
            "job_id": "v3",
            "block_height": 130500,
            "matmul": {"parent_mtp": 1_780_000_000},
        })
        second = Job.from_notify({
            "job_id": "old",
            "block_height": 130499,
            "matmul": {},
        })
        self.assertEqual(first.parent_mtp, 1_780_000_000)
        first.merge_from(second)
        self.assertIsNone(first.parent_mtp)

    def test_solver_version_gate(self):
        self.assertFalse(
            GBTSolveWrapper._supports_parent_mtp_v3(
                "btx-gbt-solve-hip 2.0.0"
            )
        )
        self.assertTrue(
            GBTSolveWrapper._supports_parent_mtp_v3(
                "btx-gbt-solve-hip 2.1.0 (BTX V3 parent-MTP)"
            )
        )

    def test_share_dedupe_spans_rotated_job_ids(self):
        client = StratumClient.__new__(StratumClient)
        client._submit_worker = "address.worker"
        client.worker_name = "address.worker"
        client.payout_address = "address"
        client._extranonce2_size = 4
        client._pending_submits = {}
        client._submitted_share_keys = set()
        client._submitted_share_order = __import__("collections").deque()
        client._next_id = Mock(side_effect=[1, 2])
        client._send = Mock()

        first = Job(
            job_id="job.b",
            prev_hash="ab" * 32,
            time=1_781_441_921,
        )
        rotated = Job(
            job_id="job.c",
            prev_hash=first.prev_hash,
            time=first.time,
        )
        result = {"nonce64": 1234, "digest": "00" * 32}

        client.submit_share(first, result)
        client.submit_share(rotated, result)

        self.assertEqual(client._send.call_count, 1)


if __name__ == "__main__":
    unittest.main()
