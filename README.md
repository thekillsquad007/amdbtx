# AMDBTX — AMD GPU Miner for BTX (Pool + Solo)

A native AMD GPU miner for BTX MatMul PoW using ROCm/HIP. Mine on the
[MineBtx pool](https://minebtx.com) **or solo** against your own `btxd` node.

- **Pool**: `stratum+tcp://stratum.minebtx.com:3333`
- **Solo**: `getblocktemplate` + `getmatmulchallenge` + `submitblock` via JSON-RPC
- **Algorithm**: MatMul PoW (n=512, b=16, r=8, M31 field, sigma gate)
- **Dev fee**: 2% transparent — time-sliced in pool mode, coinbase split in solo
- **Dev wallet**: `btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4`

---

## Quick Start

### Linux (Native Ubuntu 22.04+)

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- --address btx1z...YOUR_ADDRESS --yes
```

Launch:

```bash
amdbtx-miner --config ~/.amdbtx-miner/config.yaml
```

### Windows (WSL2 with AMD GPU)

> Requires: Windows 11 with WSL2, AMD GPU (RDNA 2+ recommended), latest AMD Adrenalin driver.

```powershell
# From PowerShell in the repo folder:
.\install_amd.cmd btx1z...YOUR_ADDRESS
```

Launch:

```powershell
wsl -e amdbtx-miner --config ~/.amdbtx-miner/config.yaml
```

---

## Installation Details

### Linux

**Prerequisites**: AMD GPU (GCN 4+), Ubuntu 22.04+. The installer auto-detects your Ubuntu version and installs the correct ROCm runtime (6.4 on 22.04, 7.2 on 24.04+), Python venv, solver binary, and config.

Custom options:

```bash
# With worker name and custom pool
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- \
  --address btx1z... --worker myrig --pool stratum.minebtx.com:3333 --yes
```

### Windows (WSL2)

The `.cmd` launcher runs `install_amd.ps1` with `-ExecutionPolicy Bypass`, so no system policy changes are needed. It sets `HSA_ENABLE_DXG_DETECTION=1` for WSL GPU passthrough, installs the solver and Python wrapper inside WSL, and writes the config.

### HiveOS

HiveOS manages the AMD driver/ROCm stack itself. The installer detects HiveOS and skips ROCm package installation automatically.

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- --address btx1z...YOUR_ADDRESS --yes
```

If GPU detection fails on HiveOS, check that Hive exposes ROCm/HIP libraries and devices:

```bash
ldconfig -p | grep libamdhip64
ls -la /dev/kfd /dev/dri
rocminfo | grep -m1 gfx
```

### Manual / Advanced

```bash
git clone https://github.com/thekillsquad007/amdbtx.git
cd amdbtx
bash install_amd.sh --address btx1z... --skip-rocm
# Edit ~/.amdbtx-miner/config.yaml if needed
amdbtx-miner
```

---

## Configuration

Edit `~/.amdbtx-miner/config.yaml` (generated during install). See
[`config.example.yaml`](config.example.yaml) for all options including solo mode.

**Pool mode** (default):

```yaml
mining_mode: "pool"
pool_host: "stratum.minebtx.com"
pool_port: 3333
payout_address: "btx1z..."
worker_name: "7800XT-ALPHA-1"
gbt_solve_path: "/home/user/.amdbtx-miner/bin/btx-gbt-solve-hip"
solver_backend: "rocm"     # "rocm" or "cpu"
solver_threads: 8
solver_prepare_workers: 16
solver_batch_size: 128
solver_prefetch_depth: 8
solver_pipeline_async: 1
gpu_device: -1           # -1 = auto (single best GPU), 0/1/.. = force one GPU
# gpu_devices: "all"     # multi-GPU: "all", "0,1", or [0, 1] — hashrate stacks
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
```

CLI flags override config values:

```bash
amdbtx-miner --payout-address btx1z... --worker-name myrig --solver-backend rocm
```

### Multi-GPU

On rigs with multiple AMD GPUs, the miner can run **one HIP solver per card**.
Each GPU searches a disjoint nonce range in parallel — **effective hashrate stacks**
for both pool and solo (2× 0.30 kH/s cards ≈ 0.60 kH/s total).

#### How to enable multi-GPU

**1. Confirm GPUs are visible**

```bash
rocminfo | grep -E 'Marketing Name|gfx'
# or
rocm-smi --showproductname
```

You should see one entry per card (e.g. `gfx1100` for RX 7800 XT). Note the
device indices — usually `0`, `1`, … in probe order.

**2. Edit config** (`~/.amdbtx-miner/config.yaml`)

```yaml
# Mine on every detected AMD GPU:
gpu_devices: "all"

# Or pick specific cards (skip an iGPU or a busy display GPU):
# gpu_devices: "0,1"
```

**3. Start the miner** (pool or solo — same `gpu_devices` setting)

Pool:

```bash
amdbtx-miner --config ~/.amdbtx-miner/config.yaml
```

Solo:

```bash
amdbtx-miner --solo \
  --rpc-url http://192.168.1.15:19334 \
  --rpc-user miner --rpc-password YOUR_PASSWORD \
  --payout-address btx1z... \
  --gpu-devices all
```

CLI overrides config: `--gpu-devices 0,1` is equivalent to `gpu_devices: "0,1"`.

**4. Verify in logs**

Startup should show:

```
multi-GPU mining on devices [0, 1] (hashrate stacks)
multi-GPU mining: 2 solvers on devices [0, 1]
```

During mining, look for combined hashrate:

```
matmul_khps=0.58 total (0.29+0.29 per GPU) backend=hip gpus=2
```

**Behaviour**

| Mode | Connection | Hashrate |
|------|------------|----------|
| Pool | One stratum worker; all GPUs submit shares | Stacks — pool sees sum of work |
| Solo | One `btxd` RPC; first GPU to find a block submits | Stacks — ~2× faster expected block time |

Leave `gpu_devices` unset to keep single-GPU mode: auto-pick the best card
(useful on laptops with iGPU + dGPU where you only want the dGPU).

### Performance branch (`perf/miner-gpu-optimization`)

Active GPU solver work targets the HIP path in `solver/src/solve_gpu.hip`:

- Persistent device memory pool (no per-slice `hipFree`/`hipMalloc`)
- GPU transcript digest filter (`HashTranscriptKernel` + `CompareDigestsKernel`)
- V2 seeds/sigma re-derived on CPU after sigma gate (required for pool consensus)
- Inner-loop unroll in `ComputeCompressedWordsFusedKernel`

Rebuild after pulling:

```bash
cd solver && bash build.sh
cp build/btx-gbt-solve-hip ~/.amdbtx-miner/bin/
```

---

## Solo Mining

Solo mode mines directly against a synced BTX full node (`btxd`). The miner
fetches block templates over JSON-RPC, runs the same HIP MatMul solver, and
submits full blocks with `submitblock` when a solution meets **network**
difficulty (not pool share difficulty). A **2% dev fee** is taken from the
coinbase reward (split outputs) on every block found — same rate as pool mode,
applied differently.

### Requirements

- A synced `btxd` node on **[BTX v0.32.3+](https://github.com/btxchain/btx/releases)**
- RPC enabled with `getblocktemplate`, `getmatmulchallenge`, and `submitblock`
- A BTX payout address (`btx1z...` or `btx1q...`) — block rewards go to this address
- The same HIP solver binary used for pool mining

### Local node (same machine)

If `btxd` runs locally with cookie auth (default when no `rpcuser` is set):

```yaml
mining_mode: solo
rpc_url: "http://127.0.0.1:19334"
rpc_cookie_file: "~/.btx/.cookie"
payout_address: "btx1z...YOUR_ADDRESS..."
```

```bash
amdbtx-miner --solo --payout-address btx1z...YOUR_ADDRESS...
```

### Remote node (LAN or VPS)

Use `rpcuser` / `rpcpassword` from the node's `btx.conf`. Cookie files only work
on the machine where `btxd` created them.

```yaml
mining_mode: solo
rpc_url: "http://192.168.1.15:19334"
rpc_user: "miner"
rpc_password: "your_rpc_password"
payout_address: "btx1z...YOUR_ADDRESS..."
gbt_longpoll: true
gbt_longpoll_timeout: 60.0
```

```bash
amdbtx-miner --solo \
  --rpc-url http://192.168.1.15:19334 \
  --rpc-user miner \
  --rpc-password your_rpc_password \
  --payout-address btx1z...YOUR_ADDRESS...
```

### How it works

1. `getblocktemplate` + `getmatmulchallenge` — fetch the current block template and MatMul challenge (seeds, height, epsilon, merkle root).
2. HIP solver — search nonces against **network** `bits` / target (same digest path as pool mining).
3. `submitblock` — when `is_block=true`, assemble coinbase + mempool txs + MatMul header and submit the full block hex to your node.

The miner polls for template updates each loop (new transactions, refreshed `nTime`)
without resetting your nonce counter on the same block height.

### Pool vs solo

| | Pool | Solo |
|---|------|------|
| Work source | Stratum `mining.notify` | `getblocktemplate` + `getmatmulchallenge` |
| Difficulty | Pool share target | Full network block target |
| Submit | `mining.submit` (nonce + ntime) | `submitblock` (full block) |
| Payout | Pool (PPLNS) | Full block reward to your address |
| Dev fee | 2% time-sliced (stratum) | 2% coinbase split per block |

### Expected logs

```
solo: connected to node height=125601 difficulty=0.035...
solo template job=solo-125601-51619e6d8d37ab84 height=125601 ...
FOUND! nonce=... digest=... is_block=true ...
solo: BLOCK ACCEPTED height=125601 nonce=... digest=...
```

If you see `is_block=false`, the solver found a share-tier hit (not a valid block)
and solo mode correctly skips `submitblock`.

### When solo makes sense

Solo competes against **total network hashrate**, not pool hashrate. It can be
worth trying when network difficulty is low relative to your GPU's `matmul_khps`,
or when you want the full block reward without pool fees. At typical single-GPU
speeds on mainnet, blocks may be rare — check your `matmul_khps` in the solve logs
and compare to network conditions on [BTXplorer](https://explorer.minebtx.com).

### Solo dev fee

By default, **2%** of each block's `coinbasevalue` is paid to the dev wallet
as a second coinbase output; the rest goes to your `payout_address`. Disable or
adjust in config:

```yaml
solo_dev_fee_bps: 200   # 200 = 2.00%; set 0 to disable
# dev_wallet: "btx1z..."  # optional override (defaults to built-in dev wallet)
```

On startup you should see:

```
solo: dev fee 200 bps (2.00% of coinbase) -> btx1zdcnts8q7...
```

When a block is accepted:

```
solo: BLOCK ACCEPTED ... reward user=1960000000 dev=40000000 sats (2.00% fee)
```

### Solo troubleshooting

| Symptom | Fix |
|---------|-----|
| `cannot reach btxd RPC` | Check `rpc_url`, firewall, and that `btxd` is running |
| `no RPC credentials` | Set `rpc_user`/`rpc_password` (remote) or `rpc_cookie_file` (local) |
| `cannot resolve coinbase script` | Ensure `validateaddress` works on the node, or set `coinbase_script_pubkey` in config |
| `solo: submitblock rejected` | Node may have moved to a new tip — usually resolves on next template; check `btxd` logs |
| `merkle mismatch` | Template changed while assembling — miner will pick up the new template next loop |

---

## GPU Tuning

| GPU Family | Arch | GFX | Workers | Threads | Batch |
|------------|------|-----|---------|---------|-------|
| RX 470/480/570/580 | GCN 4 | gfx803 | 8 | 4 | 64 |
| RX Vega 56/64 | GCN 5 | gfx900 | 8 | 4 | 64 |
| Radeon VII | GCN 5 | gfx906 | 12 | 8 | 128 |
| RX 5500/5600/5700 | RDNA 1 | gfx1010 | 12 | 8 | 128 |
| RX 6600/6700/6800/6900 | RDNA 2 | gfx1030 | 16 | 8 | 128 |
| RX 7600/7700/7800/7900 | RDNA 3 | gfx1100 | 16 | 8 | 128 |
| RX 9070 | RDNA 4 | gfx1102 | 16 | 8 | 128 |

Tuning is auto-detected during install. Override in config for optimization.

---

## Building Solver from Source

Required if the pre-built binary doesn't support your GPU:

```bash
bash build_solver.sh
```

Output: `amdbtx-private-solver/build/btx-gbt-solve-hip`. Point `gbt_solve_path` in config to this path.

---

## Dev Fee (pool mode only)

Transparent 2% dev fee: time-sliced in pool mode, coinbase split in solo mode.

1. Mine with your address for ~58 minutes
2. Switch authorization to dev wallet for ~2 minutes
3. Switch back to your address
4. All switches logged at `INFO` level

Dev wallet: `btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4`

---

## Pool Information

- **Pool Dashboard**: https://pool.minebtx.com
- **Stratum**: `stratum+tcp://stratum.minebtx.com:3333`
- **Algorithm**: MatMul PoW (BTX spec, n=512, b=16, r=8, M31 field)
- **Pool Fee**: 2.5% (PPLNS, weekly payouts)
- **Telegram**: @btxdexbot (`/stats`, `/mybalance`, `/help`)

### Getting a BTX Address

The pool does **not** create wallets. Visit https://easybtx.com/wallet to create a BTX wallet and get a payout address (starts with `btx1z...` or `btx1q...`).

---

## Troubleshooting

| Symptom | Fix |
|---------|------|
| `invalid payout address` | Config has placeholder — edit `~/.amdbtx-miner/config.yaml` |
| Solver binary does nothing | Build from source: `bash build_solver.sh` |
| `rocm-smi` not found | Install ROCm or `export PATH=/opt/rocm/bin:$PATH` |
| `/dev/kfd` permission denied | `sudo chmod 666 /dev/kfd /dev/dri/*` |
| GPU not detected in WSL | Set `HSA_ENABLE_DXG_DETECTION=1` in Windows env, `wsl --shutdown`, restart |
| Low GPU utilization | Bump `solver_prepare_workers` and `solver_threads` |
| Share rejected (code 21) | Normal after reconnect, wait 1–2 minutes |

---

## Links

- **Pool**: https://minebtx.com
- **Dashboard**: https://pool.minebtx.com
- **Telegram**: @btxdexbot (`/stats`, `/mybalance`, `/help`)
- **GitHub**: https://github.com/thekillsquad007/amdbtx
