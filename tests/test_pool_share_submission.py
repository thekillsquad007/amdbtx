import time
from collections import deque

from amdbtx_miner.__main__ import _solve_slice_continuous, _submitted_share_keys
from amdbtx_miner.config import validate_config
from amdbtx_miner.gbt_solve_wrapper import MultiGPUSolver
from amdbtx_miner.stratum_client import Job, StratumClient


class DummySolver:
    num_gpus = 1

    def solve(self, job, nonce_start=0, max_tries=0, max_seconds=0):
        return {
            "found": True,
            "nonce64": nonce_start,
            "nonce64_end": nonce_start + max_tries - 1,
            "tries_used": max_tries,
            "elapsed_s": 0.1,
            "gate_passes": 3,
            "words_hits": 3,
            "backend": "hip",
            "solutions": [
                {
                    "nonce64": nonce_start + 1,
                    "digest": "01" * 32,
                    "ntime": job.time,
                    "is_block": False,
                },
                {
                    "nonce64": nonce_start + 2,
                    "digest": "02" * 32,
                    "ntime": job.time,
                    "is_block": False,
                },
                {
                    "nonce64": nonce_start + 3,
                    "digest": "03" * 32,
                    "ntime": job.time,
                    "is_block": False,
                },
            ],
        }


class DummyClient:
    _current_job = None
    sock = None

    def __init__(self):
        self.submitted = []

    def submit_share(self, job, result, *, wait=False):
        self.submitted.append(result)


class DummyInnerSolver:
    def __init__(self, last_observed_nps):
        self.last_observed_nps = last_observed_nps

    def solve(self, job, nonce_start=0, max_tries=0, max_seconds=0):
        return {
            "found": False,
            "tries_used": max_tries,
            "elapsed_s": 2.0,
        }


def _job():
    return Job(
        job_id="job-1",
        version=536870912,
        prev_hash="11" * 32,
        merkle_root="22" * 32,
        time=1234567890,
        bits="1d00ffff",
        target="ff" * 32,
        seed_a="33" * 32,
        seed_b="44" * 32,
        block_height=125874,
        matmul_n=512,
        matmul_b=16,
        matmul_r=8,
        epsilon_bits=18,
    )


def test_slice_submits_all_solver_solutions_when_uncapped():
    _submitted_share_keys.clear()
    client = DummyClient()
    result = _solve_slice_continuous(
        DummySolver(),
        client,
        _job(),
        solo=False,
        nonce_start=100,
        nonces_per_slice=1000,
        max_seconds_per_slice=5.0,
        max_shares_per_slice=0,
    )

    assert [share["nonce64"] for share in client.submitted] == [101, 102, 103]
    assert result["shares_in_slice"] == 3


def test_slice_share_cap_applies_to_submitted_solutions():
    _submitted_share_keys.clear()
    client = DummyClient()
    result = _solve_slice_continuous(
        DummySolver(),
        client,
        _job(),
        solo=False,
        nonce_start=100,
        nonces_per_slice=1000,
        max_seconds_per_slice=5.0,
        max_shares_per_slice=2,
    )

    assert [share["nonce64"] for share in client.submitted] == [101, 102]
    assert result["shares_in_slice"] == 2


def test_pool_share_cap_zero_remains_unlimited():
    cfg = validate_config({"pool_max_shares_per_slice": 0})
    assert cfg["pool_max_shares_per_slice"] == 0


def test_single_gpu_solver_propagates_observed_nps_for_pool_metrics():
    solver = MultiGPUSolver.__new__(MultiGPUSolver)
    solver.gpu_devices = [0]
    solver.solvers = [DummyInnerSolver(123_456_789.0)]
    solver.last_observed_nps = None

    solver.solve(_job(), nonce_start=0, max_tries=1000, max_seconds=1.0)

    assert solver.last_observed_nps == 123_456_789.0


def test_single_gpu_solver_derives_observed_nps_when_child_metric_missing():
    solver = MultiGPUSolver.__new__(MultiGPUSolver)
    solver.gpu_devices = [0]
    solver.solvers = [DummyInnerSolver(None)]
    solver.last_observed_nps = None

    solver.solve(_job(), nonce_start=0, max_tries=1000, max_seconds=1.0)

    assert solver.last_observed_nps == 500.0


def test_share_accept_records_submit_difficulty_for_pool_credit():
    client = StratumClient.__new__(StratumClient)
    client._difficulty = 1.0
    client._pending_submits = {
        7: {
            "job_id": "job-1",
            "nonce_hex": "00",
            "is_block": False,
            "difficulty": 0.25,
        }
    }
    client._accepted_share_events = deque()
    client.shares_accepted = 0
    client.shares_rejected = 0
    client.blocks_found = 0

    client._complete_pending_submit(7, {"id": 7, "result": True})

    assert client.shares_accepted == 1
    assert list(client._accepted_share_events)[0][1] == 0.25


def test_pool_credit_stats_reports_recent_credit_per_minute():
    client = StratumClient.__new__(StratumClient)
    now = time.time()
    client._accepted_share_events = deque([
        (now - 10.0, 0.5),
        (now - 70.0, 4.0),
    ])

    stats = client.pool_credit_stats(60.0)

    assert stats["accepted"] == 1.0
    assert stats["credit"] == 0.5
    assert stats["avg_diff"] == 0.5
    assert stats["credit_per_min"] == 0.5


def test_luckypool_job_parser_maps_nonce_prefix_and_v3_fields():
    job = Job.from_luckypool({
        "jobId": "4f4",
        "height": 147654,
        "nVersion": 536870912,
        "prevHash": "11" * 32,
        "merkleRoot": "22" * 32,
        "nTime": 1782965958,
        "nBits": "1c477239",
        "matmulDim": 512,
        "b": 16,
        "r": 8,
        "epsilonBits": 18,
        "noncePrefix": "9266998",
        "nonceBits": 40,
        "shareTarget": "00" + "ff" * 31,
        "cleanJobs": True,
        "parentMtp": 1782965509,
    })

    assert job.job_id == "4f4"
    assert job.block_height == 147654
    assert job.parent_mtp == 1782965509
    assert job.nonce64_start == (9266998 << 40)
    assert job.luckypool_nonce_bits == 40
    assert job.target == "00" + "ff" * 31


def test_luckypool_submit_uses_login_protocol_payload_with_nonce_suffix():
    sent = []
    client = StratumClient.__new__(StratumClient)
    client._protocol = "luckypool"
    client._msg_id = 0
    client._pending_submits = {}
    client._difficulty = 0.0002
    client._send = sent.append
    job = Job.from_luckypool({
        "jobId": "504",
        "height": 147662,
        "nVersion": 536870912,
        "prevHash": "11" * 32,
        "merkleRoot": "22" * 32,
        "nTime": 1782967143,
        "nBits": "1c4c2e02",
        "noncePrefix": "9266998",
        "nonceBits": 40,
        "shareTarget": "00001387ec780000",
        "parentMtp": 1782967000,
    })

    client.submit_share(
        job,
        {
            "nonce64": 0x8d67420074026c11,
            "digest": "ab" * 32,
            "is_block": False,
        },
    )

    assert sent == [{
        "id": 1,
        "method": "submit",
        "params": {
            "jobId": "504",
            "nonce": "0074026c11",
            "result": "ab" * 32,
        },
    }]


def test_luckypool_same_height_rotation_preserves_nonce_suffix_width():
    sent = []
    client = StratumClient.__new__(StratumClient)
    client._protocol = "luckypool"
    client._msg_id = 0
    client._pending_submits = {}
    client._difficulty = 0.0002
    client._send = sent.append
    client._current_job = Job.from_luckypool({
        "jobId": "554",
        "height": 147726,
        "nVersion": 536870912,
        "prevHash": "11" * 32,
        "merkleRoot": "22" * 32,
        "nTime": 1782972536,
        "nBits": "1c4c2e02",
        "noncePrefix": "7368840",
        "nonceBits": 40,
        "shareTarget": "00001387ec780000",
        "parentMtp": 1782972000,
    })

    client._handle_server_message({
        "method": "job",
        "params": {
            "jobId": "555",
            "height": 147726,
            "nVersion": 536870912,
            "prevHash": "11" * 32,
            "merkleRoot": "33" * 32,
            "nTime": 1782972566,
            "nBits": "1c4c2e02",
            "shareTarget": "00001387ec780000",
            "parentMtp": 1782972000,
        },
    })

    assert client._current_job.luckypool_nonce_bits == 40

    client.submit_share(
        client._current_job,
        {
            "nonce64": 0x7069e703235b9f46,
            "digest": "ab" * 32,
            "is_block": False,
        },
    )

    assert sent[0]["params"]["nonce"] == "03235b9f46"


def test_luckypool_sparse_job_keeps_or_infers_nonce_suffix_width():
    sent = []
    client = StratumClient.__new__(StratumClient)
    client._protocol = "luckypool"
    client._msg_id = 0
    client._pending_submits = {}
    client._difficulty = 0.0002
    client._send = sent.append
    client._current_job = Job.from_luckypool({
        "jobId": "56b",
        "height": 147744,
        "nVersion": 536870912,
        "prevHash": "11" * 32,
        "merkleRoot": "22" * 32,
        "nTime": 1782974168,
        "nBits": "1c4c2e02",
        "noncePrefix": "1717641",
        "nonceBits": 40,
        "shareTarget": "00001387ec780000",
        "parentMtp": 1782974000,
    })

    client._handle_server_message({
        "method": "job",
        "params": {
            "jobId": "56c",
            "height": 147745,
            "nVersion": 536870912,
            "prevHash": "33" * 32,
            "merkleRoot": "44" * 32,
            "nTime": 1782974198,
            "nBits": "1c4c2e02",
            "noncePrefix": "1717641",
            "shareTarget": "00001387ec780000",
            "parentMtp": 1782974000,
        },
    })

    assert client._current_job.luckypool_nonce_bits == 40

    client.submit_share(
        client._current_job,
        {
            "nonce64": 0x1a35890093a6bdf4,
            "digest": "ab" * 32,
            "is_block": False,
        },
    )

    assert sent[0]["params"]["nonce"] == "0093a6bdf4"


def test_luckypool_submit_infers_suffix_width_from_aligned_nonce_start():
    sent = []
    client = StratumClient.__new__(StratumClient)
    client._protocol = "luckypool"
    client._msg_id = 0
    client._pending_submits = {}
    client._difficulty = 0.0002
    client._send = sent.append
    job = Job.from_luckypool({
        "jobId": "56c",
        "height": 147745,
        "nVersion": 536870912,
        "prevHash": "33" * 32,
        "merkleRoot": "44" * 32,
        "nTime": 1782974198,
        "nBits": "1c4c2e02",
        "noncePrefix": "1717641",
        "nonceBits": 40,
        "shareTarget": "00001387ec780000",
        "parentMtp": 1782974000,
    })
    job.luckypool_nonce_bits = 0

    client.submit_share(
        job,
        {
            "nonce64": 0x1a35890093a6bdf4,
            "digest": "ab" * 32,
            "is_block": False,
        },
    )

    assert sent[0]["params"]["nonce"] == "0093a6bdf4"
