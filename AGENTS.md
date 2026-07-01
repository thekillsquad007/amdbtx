# Agent Notes

## Current Context

- Workspace: `E:\Business\amdbtx`
- WSL path: `/mnt/e/Business/amdbtx`
- Target: AMD BTX miner for `btxchain/btx`, usually mining through local pool/proxy `127.0.0.1:3333` or `pool.minebtx.com`.
- Primary GPU observed: RX 7800 XT / RDNA3 `gfx1101`.
- Runtime is WSL GPU via `/dev/dxg`; `/dev/kfd` and `/dev/dri` are not exposed. Normal Linux ROCm profilers such as `rocprof`/`rocprofv2` are not installed/available in this WSL environment.

## Installed Solver

- Installed path: `/home/aravindthana/.amdbtx-miner/bin/btx-gbt-solve-hip`
- Version: `btx-gbt-solve-hip 2.1.0 (BTX V3 parent-MTP)`
- Current installed SHA256: `28f8f08b8c4b427be580261b9c47641745004c229cc0333a40e27d007e08bcdf`
- Installed after trust-GPU solver-side fix on `2026-07-01 08:22`.
- Backup made before install:
  - `/home/aravindthana/.amdbtx-miner/bin/btx-gbt-solve-hip.bak-trust-gpu-20260701-082202`
- Previous useful backups from earlier work may also exist:
  - `/home/aravindthana/.amdbtx-miner/bin/btx-gbt-solve-hip.bak-fusedA-default-20260701-075232`
  - `/home/aravindthana/.amdbtx-miner/bin/btx-gbt-solve-hip.bak-main-perf-20260701-074022`

The currently running miner keeps using its already-open executable until it is restarted. Restart is required to pick up the installed solver above.

## Live Miner Profile Findings

The miner was profiled live with:

```bash
BTX_MATMUL_PROFILE=1 amdbtx-miner --config ~/.amdbtx-miner/config.yaml 2>&1 | tee ~/amdbtx-live-profile.log
```

Fresh profile log path:

```bash
/home/aravindthana/amdbtx-live-profile.log
```

Observed live process:

- Miner process: `/home/aravindthana/.local/bin/amdbtx-miner --config /home/aravindthana/.amdbtx-miner/config.yaml`
- Solver process: `/home/aravindthana/.amdbtx-miner/bin/btx-gbt-solve-hip --daemon --backend hip --batch-size 131072 --epsilon-bits 18`
- Solver env included:
  - `BTX_MATMUL_WMMA=1`
  - `BTX_MATMUL_TRUST_GPU_SHARES=1`
  - `BTX_MATMUL_SOLVE_BATCH_SIZE=131072`
  - `BTX_MATMUL_PREPARE_WORKERS=24`
  - `BTX_MATMUL_SOLVER_THREADS=8`
  - `BTX_MATMUL_PIPELINE_ASYNC=1`

Profile summary from `~/amdbtx-live-profile.log`:

- `profile_rows`: 19596
- Full `131072` scan rows: 18372
- Internal GPU-stage implied speed: about `100-102 MN/s`
- Pool-facing recent solve lines: mostly around `5.4 kN/s`, with some noisy high/low slices.
- `words_path=wmma`, `arch=gfx1101`.
- No rejects or duplicates were seen in the sampled profile log:
  - `share OK`: 11
  - `reject`: 0
  - `duplicate`: 0

Mean full-batch stage split at `batch=131072`:

| Stage | Mean ms | Notes |
| --- | ---: | --- |
| `rhs_ms` | ~0.507 | Biggest current GPU-stage cost. |
| `hash_compare_ms` | ~0.353 | Second/third major cost. |
| `sigma_ms` | ~0.282 | Epsilon gate. |
| `noise_ms` | ~0.052 | Small. |
| `words_ms` | ~0.041 | WMMA path is already tiny. |
| `matrix_ms` | ~0.025 | Small. |
| `noise_seed_ms` | ~0.019 | Small. |
| `compress_ms` | ~0.022 | Small. |

Important conclusion: WMMA is no longer the bottleneck. The next serious performance work should target `rhs_ms`, `hash_compare_ms`, `sigma_ms`, and host/device synchronization boundaries.

## WSL Impact Estimate

Do not assume WSL is the whole problem. Based on live profile:

- Pure GPU kernels are fast and stable, roughly `100 MN/s` internally.
- WSL likely costs practical performance through HIP launch/sync/copy overhead and `/dev/dxg` runtime behavior.
- Rough estimate excluding native Linux comparison:
  - WSL tax likely `10-30%`, worst case maybe `40%`.
  - Remaining solver architecture headroom excluding WSL is probably `20-40%`, not an easy multi-x gain.

The live solver used about `3.7-3.8` CPU cores with 6 threads while Python was mostly sleeping, so Python feeding is not the current bottleneck.

## Current Config Observed

Home config at `/home/aravindthana/.amdbtx-miner/config.yaml` was observed with:

```yaml
pool_host: 127.0.0.1
pool_port: 3333
pool_tls: false
solver_backend: rocm
solver_batch_size: 131072
solver_prepare_workers: 24
solver_threads: 8
solver_prefetch_depth: 8
solver_pipeline_async: 1
gpu_inputs: 0
worker_name: AMD-RX7800XT
```

Repo defaults currently differ in some places and were previously moved toward `81920`. The live miner was using `131072`.

## Current Source Changes Worth Keeping

- `solver/CMakeLists.txt`
  - Adds HIP arch flags from `HIP_ARCHS`.
  - Adds ROCm runpath for `/opt/rocm/lib:/opt/rocm-7.2.0/lib`.

- `solver/src/matmul_kernel.hip`
  - Grouped right-noise RHS kernel:
    - `batch_build_right_noise_rhs_grouped_512x16x8_kernel`
    - Default path unless `BTX_MATMUL_OLD_RIGHT_RHS=1`.
  - Precomputed seed midstates for A/B:
    - `ComputeSeedMidstates`
    - `GenerateMatrixFromMidstate512`
  - Fused A perturb path currently default for `512x16x8`:
    - Disable with `BTX_MATMUL_NO_FUSED_A=1`.
    - Try 512-thread variant with `BTX_MATMUL_FUSED_A_512T=1`.
  - Profile fields include:
    - `arch`
    - `words_path`
    - `rhs_a_ms`
    - `rhs_right_ms`
    - `rhs_b_ms`

- `src/amdbtx_miner/gbt_solve_wrapper.py`
  - Sets `BTX_MATMUL_TRUST_GPU_SHARES=1` by default.
  - This is intentional: the pool verifies shares; a bad GPU share is rejected by the pool.

- `solver/src/solve_gpu.hip`
  - New solver-side support for `BTX_MATMUL_TRUST_GPU_SHARES`.
  - Before this fix, the wrapper set the env var but the C++ solver still CPU-verified every GPU-found share.
  - With trust enabled, found shares use the GPU digest directly.
  - With trust disabled, old CPU verification behavior remains.

## Recent Experiment: Hash+Compare Fusion

Tried fusing `HashTranscriptKernel` and `CompareDigestsKernel` to avoid one launch and one digest reread.

Result: reverted. It increased register pressure and was slower/noisier, especially when many candidates passed the sigma gate. Do not reintroduce unless there is a more careful occupancy/register plan.

## Recent Fix: Trust-GPU Actually Works

Root issue:

- Python wrapper already set `BTX_MATMUL_TRUST_GPU_SHARES=1`.
- C++ `SolveGPU` ignored it and still ran:
  - `PrepareNonceSeeds`
  - `ComputeDigestForNonce`
  - CPU digest target check
  for every GPU-found share.

Fix:

- Added env check in `solver/src/solve_gpu.hip`.
- When `BTX_MATMUL_TRUST_GPU_SHARES=1`, use the GPU digest from `digests[bi]` directly.
- When unset/false, keep old CPU-verify path.

Why this matters:

- It does not change internal GPU-stage MN/s much.
- It should reduce pool-facing latency and CPU stalls on found-share-heavy slices.
- Easy-target smoke with trust enabled returned GPU-found shares directly.
- The old no-trust easy-target path became heavy enough under live mining contention that it had to be killed, which is consistent with the suspected CPU verify stall.

## Experiments Tried And Reverted Earlier

- Fused hash+compare: slower due register pressure.
- Launch bounds on perturb/fused-B: slower.
- Cold/noinline oracle fallback: slower.
- Oracle `!= kFieldModulus` fast check: neutral; reverted.
- `GenerateMatrixFromMidstate512` 512-thread default: slower.
- Shared seed cache: slower.
- Fused-B 128-thread variant: incorrect.
- Fused-B reduce interval `8`: incorrect.
- Shared `s_w` fused-B cache: slightly slower.

## Benchmarking

Benchmark script:

```bash
/mnt/e/Business/amdbtx/bench_solver.py
```

Use from WSL:

```bash
cd /mnt/e/Business/amdbtx
python3 bench_solver.py --runs 5 --batch 131072
python3 bench_solver.py --runs 5 --batch 81920
```

When the live miner is running, benchmarks are contaminated by GPU contention. Prefer live `BTX_MATMUL_PROFILE=1` logs or stop the miner for clean synthetic numbers.

Single-arch build for RX 7800 XT:

```bash
cd /mnt/e/Business/amdbtx/solver
rm -rf build-test
cmake -S . -B build-test -DHIP_ARCHS=gfx1101
cmake --build build-test -j$(nproc)
```

Full multi-arch build with runpath:

```bash
cd /mnt/e/Business/amdbtx/solver
rm -rf build-rpath
cmake -S . -B build-rpath
cmake --build build-rpath -j$(nproc)
```

Install with backup:

```bash
src=/mnt/e/Business/amdbtx/solver/build-rpath/btx-gbt-solve-hip
dst=$HOME/.amdbtx-miner/bin/btx-gbt-solve-hip
backup=$dst.bak-$(date +%Y%m%d-%H%M%S)
cp -a "$dst" "$backup"
install -m 0755 "$src" "$dst"
sha256sum "$dst"
"$dst" --version
```

Clean generated build dirs from PowerShell:

```powershell
Remove-Item -LiteralPath 'E:\Business\amdbtx\solver\build-test','E:\Business\amdbtx\solver\build-rpath' -Recurse -Force -ErrorAction SilentlyContinue
```

## Verification

Python compile sanity passed after the trust-GPU solver change:

```bash
python -m compileall -q src tests
```

Full `pytest` was not claimed passing in the latest state. Earlier, main-branch Python/stratum tests had unrelated failures around rotated-job dedupe, solver version gate, and parent-MTP test fixture behavior.

## Worktree State Notes

As of the latest update, `git status --short` showed:

```text
 M README.md
 M config.example.yaml
 M solver/CMakeLists.txt
 M solver/src/matmul_kernel.hip
 M solver/src/solve_gpu.hip
 M src/amdbtx_miner/config.py
 M src/amdbtx_miner/gbt_solve_wrapper.py
?? bench_solver.py
```

`AGENTS.md` itself may be ignored by git. Do not assume it appears in `git status`.

## Next Performance Targets

1. Reduce `rhs_ms`, especially `rhs_a_ms` and `rhs_b_ms`.
2. Investigate `sigma_ms` SHA path and register pressure/occupancy.
3. Reduce host/device sync boundaries:
   - Current flow includes scan, sync/copy gate count, process passed nonces, sync/copy results.
4. Consider native Linux profiling later if possible, but pool can remain on Windows if accessed over LAN. On this same physical machine, native Linux cannot run the Windows pool simultaneously.

Be careful with attractive micro-optimizations. Several looked promising but were slower or incorrect. Always run correctness checks and median benchmarks before installing.
