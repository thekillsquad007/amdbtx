from amdbtx_miner.__main__ import _solve_slice_continuous, _submitted_share_keys
from amdbtx_miner.config import validate_config
from amdbtx_miner.gbt_solve_wrapper import MultiGPUSolver
from amdbtx_miner.stratum_client import Job


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
