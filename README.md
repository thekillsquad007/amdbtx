# AMDBTX — AMD GPU Miner for MineBtx Pool

A native AMD GPU miner for the [MineBtx](https://minebtx.com) pool using
ROCm/HIP. Provides a pre-built HIP solver binary and a Python stratum wrapper.

- **Pool**: `stratum+tcp://stratum.minebtx.com:3333`
- **Algorithm**: MatMul PoW (n=512, b=16, r=8, M31 field, sigma gate)
- **Dev fee**: 2% time-sliced (transparent, logged)

## Quick Install

### Prerequisites

- AMD GPU with ROCm support (GCN 4+ / RDNA 1-4)
- Linux (Ubuntu 22.04+ recommended inside distrobox or Docker)
- ROCm 6.0+ installed (see [ROCm install docs](https://rocm.docs.amd.com))

### One-liner

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash
```

Or with your payout address:

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- --address btx1...
```

### Manual install

```bash
# Clone
git clone https://github.com/thekillsquad007/amdbtx.git
cd amdbtx

# Run installer (auto-detects GPU, downloads pre-built solver + wheel)
bash install_amd.sh

# Or skip ROCm installation if already present
bash install_amd.sh --skip-rocm
```

After install, edit `~/.amdbtx-miner/config.yaml` to set your payout address,
then launch:

```bash
amdbtx-miner
```

### Building solver from source (developers)

If you want to build the HIP solver binary yourself instead of using the
pre-built binary from releases:

```bash
# Ensure ROCm 6.0+ is installed and /opt/rocm exists
bash build_solver.sh
```

The binary is written to `amdbtx-private-solver/build/btx-gbt-solve`.
Point `gbt_solve_path` in config to this path.

## Configuration

Edit `~/.amdbtx-miner/config.yaml` (created by installer) or use CLI flags:

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
# Run in foreground (ctrl-c to stop)
amdbtx-miner

# Run as daemon via tmux
tmux new -d -s amdbtx 'amdbtx-miner 2>&1 | tee ~/.amdbtx-miner/miner.log'
tmux attach -t amdbtx

# Benchmark mode
amdbtx-miner benchmark
```

## GPU Tuning

The installer auto-detects your GPU and applies optimal settings. Override in
config:

| GPU family | `prepare_workers` | `solver_threads` | `batch_size` |
|------------|------------------:|-----------------:|-------------:|
| RX 470–580 (GCN 4) | 8 | 4 | 64 |
| RX Vega (GCN 5) | 8 | 4 | 64 |
| Radeon VII | 12 | 8 | 128 |
| RX 5500–5700 (RDNA 1) | 12 | 8 | 128 |
| RX 6600–6900 (RDNA 2) | 16 | 8 | 128 |
| RX 7600–7900 (RDNA 3) | 16 | 8 | 128 |
| RX 9070 (RDNA 4) | 16 | 8 | 128 |

## Dev Fee

This miner includes a transparent 2% dev fee using time-sliced worker switching
(standard practice used by Claymore, Phoenix, etc.):

- Mines with your address for ~58 minutes
- Switches to the dev wallet for ~2 minutes
- Switches back to your address
- All switches logged at `INFO` level

Dev wallet: `btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4`

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `rocm-smi` not found | Install ROCm or set `solver_backend: cpu` |
| `/dev/kfd` permission denied | `sudo chmod 666 /dev/kfd /dev/dri/*` or run as root in container |
| Low GPU utilization | Increase `solver_prepare_workers` and `solver_threads` |
| Share rejected (code 21) | Normal after reconnect, wait 1–2 minutes |
| `401` on connect | Upgrade to latest release (v5.0+ protocol required) |
| `/opt/rocm` not found | Set `CMAKE_PREFIX_PATH` or `--rocm-path` if building from source |

## Links

- **Pool dashboard**: https://pool.minebtx.com
- **Telegram bot**: @btxdexbot (`/stats`, `/mybalance`, `/help`)
- **GitHub**: https://github.com/thekillsquad007/amdbtx
