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

One-liner install (downloads pre-built binaries):

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- --address btx1z...
```

Launch:

```bash
amdbtx-miner
```

### Windows (WSL2 with AMD GPU)

> **Requires**: Windows 11 with WSL2, AMD GPU (RDNA 2+ recommended).

```powershell
# 1. Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# 2. Enable AMD GPU detection (required!)
$env:HSA_ENABLE_DXG_DETECTION=1
# Or add permanently: System Properties > Environment Variables

# 3. Run the Windows installer
.\install_amd.ps1 -Address "btx1z..."

# 4. Launch miner
wsl -d Ubuntu-22.04 -e amdbtx-miner
```

---

## Platform Guides

### 🐧 Linux (Native)

#### Prerequisites

- AMD GPU (GCN 4+ / RDNA 1-4)
- Ubuntu 22.04+ (or equivalent with ROCm support)
- ROCm 6.0+ installed at `/opt/rocm`

#### Option A: One-liner (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- --address btx1z...YOUR_ADDRESS --worker myrig
```

#### Option B: From source

```bash
git clone https://github.com/thekillsquad007/amdbtx.git
cd amdbtx
export PAYOUT_ADDRESS=btx1z...
bash start_mining.sh
```

#### Option C: Manual

```bash
git clone https://github.com/thekillsquad007/amdbtx.git
cd amdbtx
bash install_amd.sh --skip-rocm
# Edit ~/.amdbtx-miner/config.yaml, set payout_address
amdbtx-miner
```

### 🪟 Windows (WSL2)

#### Prerequisites

- Windows 11 22H2+ (or Windows 10 21H2+)
- AMD GPU with WSL GPU support (RDNA 2+/RX 6600+ recommended)
- WSL2 with Ubuntu 22.04

#### Step-by-step

**1. Install WSL2 with Ubuntu**

```powershell
# From PowerShell as Admin
wsl --install -d Ubuntu-22.04
# Restart if prompted
```

**2. Install AMD GPU drivers**

Download and install the latest AMD Adrenalin driver from [AMD.com](https://www.amd.com/en/support).

**3. Enable AMD GPU detection in WSL**

```powershell
# Set environment variable (temporary)
$env:HSA_ENABLE_DXG_DETECTION=1

# OR set permanently (recommended):
# System Properties > Advanced > Environment Variables
# Add: HSA_ENABLE_DXG_DETECTION=1
```

**4. Clone and install**

```powershell
cd E:\Business\amdbtx
.\install_amd.ps1 -Address "btx1z..." -Worker "myrig"
```

**5. Launch the miner**

```powershell
wsl -d Ubuntu-22.04 -e amdbtx-miner
```

Or for persistent mining (via tmux inside WSL):

```powershell
wsl -d Ubuntu-22.04 tmux new -d -s amdbtx 'amdbtx-miner 2>&1 | tee ~/.amdbtx-miner/miner.log'
wsl -d Ubuntu-22.04 tmux attach -t amdbtx
```

### 🐳 Container (Docker/Proxmox)

For containerized environments, ensure `--device=/dev/kfd --device=/dev/dri` are passed:

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  -v /opt/rocm:/opt/rocm \
  ubuntu:22.04 bash
```

Then run the one-liner installer inside the container.

---

## Worker Names

The pool assigns worker names based on detected GPU model and a canonical group
(ALPHA, BRAVO, CHARLIE, DELTA, etc.):

| GPU | Worker Name | Canonical Group |
|-----|-------------|-----------------|
| RX 7800 XT | `7800XT-ALPHA-1` | ALPHA |
| RX 7900 XTX | `7900XTX-BRAVO-1` | BRAVO |
| RX 6800 XT | `6800XT-CHARLIE-1` | CHARLIE |
| RX 5700 XT | `5700XT-DELTA-1` | DELTA |
| RX 6600 | `6600-ECHO-1` | ECHO |

- The `-1` suffix is the GPU index (increment for multi-GPU rigs: `-1`, `-2`, etc.)
- The canonical group rotates per GPU model for dashboard organization
- Custom names can be set with `--worker` flag during install

---

## Configuration

Edit `~/.amdbtx-miner/config.yaml`:

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
gpu_inputs: 0
nonces_per_slice: 20000000
solver_max_seconds_per_slice: 5.0
reconnect_initial_s: 1.0
reconnect_max_s: 60.0
log_level: "INFO"
venv_path: "/home/user/.amdbtx-miner/venv"
```

CLI flags override config file values:

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

Output: `amdbtx-private-solver/build/btx-gbt-solve-hip`.

Point `gbt_solve_path` in config to this path and re-launch.

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

1. Visit https://pool.minebtx.com
2. Register an account or use a compatible BTX wallet
3. Copy your BTX deposit address (starts with `btx1z...` or `btx1q...`)

---

## Troubleshooting

| Symptom | Fix |
|---------|------|
| `invalid payout address` | Config has placeholder address — edit `~/.amdbtx-miner/config.yaml` |
| `401 Miner does not declare pre_hash_block_tier_v18` | Upgrade wheel: `pip install --force-reinstall ~/.amdbtx-miner/amdbtx_miner.whl` |
| Solver binary does nothing | Build from source: `bash build_solver.sh` |
| `rocm-smi` not found | Install ROCm or `export PATH=/opt/rocm/bin:$PATH` |
| `/dev/kfd` permission denied | `sudo chmod 666 /dev/kfd /dev/dri/*` |
| GPU not detected in WSL | Set `HSA_ENABLE_DXG_DETECTION=1` in Windows env, `wsl --shutdown`, restart |
| Low GPU utilization | Bump `solver_prepare_workers` and `solver_threads` |
| Share rejected (code 21) | Normal after reconnect, wait 1–2 minutes |
| `--break-system-packages` error | Use venv: `python3 -m venv ~/.amdbtx-miner/venv` then install there |

---

## Links

- **Pool**: https://minebtx.com
- **Dashboard**: https://pool.minebtx.com
- **Telegram**: @btxdexbot (`/stats`, `/mybalance`, `/help`)
- **GitHub**: https://github.com/thekillsquad007/amdbtx
- **Pool Stratum Check**: `nc stratum.minebtx.com 3333`
