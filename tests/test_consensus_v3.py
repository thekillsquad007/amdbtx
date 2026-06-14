import unittest

from amdbtx_miner.block_builder import (
    derive_v2_seed,
    derive_v3_seed,
    resolve_header_seeds,
    uint256_to_display_hex,
)
from amdbtx_miner.gbt_solve_wrapper import GBTSolveWrapper
from amdbtx_miner.stratum_client import Job


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


if __name__ == "__main__":
    unittest.main()
