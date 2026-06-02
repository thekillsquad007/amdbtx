# AMDBTX — AMD GPU Miner for MineBtx Pool

A native AMD GPU miner for the [MineBtx](https://minebtx.com) pool using
ROCm/HIP. Provides a HIP solver binary and a Python stratum wrapper.

- **Pool**: `stratum+tcp://stratum.minebtx.com:3333`
- **Algorithm**: MatMul PoW (n=512, b=16, r=8, M31 field, sigma gate)
- **Dev fee**: 2% time-sliced (transparent, logged)

## Prerequisites

- AMD GPU (GCN 4+ / RDNA 1-4)
- Linux (Ubuntu 22.04+ recommended)
- ROCm 6.0+ installed at `/opt/rocm`

## Quick Start

### 1. Clone

```bash
git clone https://github.com/thekillsquad007/amdbtx.git
cd amdbtx
```

### 2. Set your payout address

```bash
export PAYOUT=btx1...YOUR_ADDRESS_HERE...
```

### 3. Run (builds solver + installs wheel + configures)

```bash
PAYOUT_ADDRESS=$PAYOUT bash start_mining.sh
```

This single command:
1. Builds the HIP solver from source (or downloads pre-built binary)
2. Installs the Python stratum wrapper
3. Writes a tuned config for your GPU
4. Prints ready message

### 4. Launch the miner

```bash
amdbtx-miner
```

## Alternative install methods

### One-liner (pre-built binary, no source needed)

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- --address btx1...
```

### Manual install with installer script

```bash
git clone https://github.com/thekillsquad007/amdbtx.git
cd amdbtx
bash install_amd.sh --skip-rocm
```

Then edit `~/.amdbtx-miner/config.yaml`, set `payout_address`, and run:

```bash
amdbtx-miner
```

## Configuration

Edit `~/.amdbtx-miner/config.yaml` or use CLI flags:

```bash
amdbtx-miner \
  --payout-address btx1... \
  --worker-name myrig \
  --backend rocm \
  --solver-threads 8 \
  --solver-batch-size 128
```

See [config.example.yaml](config.example.yaml) for all options.

## Usage

```bash
# Foreground (Ctrl+C to stop)
amdbtx-miner

# As a daemon via tmux
tmux new -d -s amdbtx 'amdbtx-miner 2>&1 | tee ~/.amdbtx-miner/miner.log'
tmux attach -t amdbtx

# Benchmark
amdbtx-miner benchmark
```

## Building the solver from source

Required if the pre-built binary doesn't support your GPU:

```bash
bash build_solver.sh
```

Output: `amdbtx-private-solver/build/btx-gbt-solve`.

Point `gbt_solve_path` in config to this path and re-launch.

## GPU Tuning

| GPU family | GPU architecture | `workers` | `threads` | `batch` |
|------------|-----------------|-----------|-----------|---------|
| RX 470/480/570/580 | GCN 4 (gfx803) | 8 | 4 | 64 |
| RX Vega 56/64 | GCN 5 (gfx900) | 8 | 4 | 64 |
| Radeon VII | GCN 5 (gfx906) | 12 | 8 | 128 |
| RX 5500/5600/5700 | RDNA 1 (gfx1010) | 12 | 8 | 128 |
| RX 6600/6700/6800/6900 | RDNA 2 (gfx1030) | 16 | 8 | 128 |
| RX 7600/7700/7800/7900 | RDNA 3 (gfx1100) | 16 | 8 | 128 |
| RX 9070 | RDNA 4 (gfx1102) | 16 | 8 | 128 |

## Dev Fee

Transparent 2% time-sliced dev fee (standard practice):

1. Mine with your address for ~58 minutes
2. Switch authorization to dev wallet for ~2 minutes
3. Switch back to your address
4. All switches logged at `INFO` level

Dev wallet: `btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4`

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `401 Miner does not declare pre_hash_block_tier_v18` | Upgrade to latest wheel: `pip install --user --force-reinstall https://github.com/thekillsquad007/amdbtx/releases/download/amdbtx-prebuilds-v1.0/amdbtx_miner-1.0.0-py3-none-any.whl` |
| Solver binary times out / does nothing | Build from source: `bash build_solver.sh`, update `gbt_solve_path` |
| `rocm-smi` not found | Install ROCm or set `solver_backend: cpu` |
| `/dev/kfd` permission denied | `sudo chmod 666 /dev/kfd /dev/dri/*` or run as root in container |
| Low GPU utilization | Bump `solver_prepare_workers` and `solver_threads` in config |
| Share rejected (code 21) | Normal after reconnect, wait 1–2 minutes |

## Links

- **Pool**: https://minebtx.com
- **Dashboard**: https://pool.minebtx.com
- **Telegram**: @btxdexbot (`/stats`, `/mybalance`, `/help`)
- **GitHub**: https://github.com/thekillsquad007/amdbtx
