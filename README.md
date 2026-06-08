# AMDBTX — AMD GPU Miner for MineBtx Pool

A native AMD GPU miner for the [MineBtx](https://minebtx.com) pool using
ROCm/HIP. Provides a HIP solver binary and a Python stratum wrapper.

- **Pool**: `stratum+tcp://stratum.minebtx.com:3333`
- **Algorithm**: MatMul PoW (n=512, b=16, r=8, M31 field, sigma gate)
- **Dev fee**: 2% time-sliced (transparent, logged)
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

Edit `~/.amdbtx-miner/config.yaml` (generated during install):

```yaml
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
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
```

CLI flags override config values:

```bash
amdbtx-miner --payout-address btx1z... --worker-name myrig --backend rocm
```

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

## Dev Fee

Transparent 2% time-sliced dev fee (industry-standard approach):

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
