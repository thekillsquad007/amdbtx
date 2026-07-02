# AMDBTX — AMD GPU Miner for BTX (Pool + Solo)

A native AMD GPU miner for BTX MatMul PoW using ROCm/HIP. Mine on
[BitMinerPool](https://bitminerpool.xyz), another supported pool, or solo against
your own `btxd` node.

- **Default pool**: `stratum+tcp://btx-sg.lproute.com:8660` (LuckyPool)
- **Solo**: mine directly against a synced `btxd` node
- **Dev fee**: 2% transparent (time-sliced in pool mode, coinbase split in solo)

---

## Quick Start

### Linux (Ubuntu 22.04+)

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/amdbtx/main/install_amd.sh | bash -s -- --address btx1z...YOUR_ADDRESS --yes
amdbtx-miner --config ~/.amdbtx-miner/config.yaml
```

### Windows (WSL2 + AMD GPU)

Requires Windows 11, WSL2, and a recent AMD Adrenalin driver.

```powershell
.\install_amd.cmd btx1z...YOUR_ADDRESS
wsl -e amdbtx-miner --config ~/.amdbtx-miner/config.yaml
```

### Install options

```bash
# Default: compile solver for your GPU (best compatibility)
bash install_amd.sh --address btx1z... --yes

# Fast install: download prebuilt release (no compiler needed)
bash install_amd.sh --address btx1z... --use-prebuilt --yes

# Custom pool and worker name
bash install_amd.sh --address btx1z... --worker myrig --pool btx-sg.lproute.com:8660 --yes
```

Prebuilt releases: https://github.com/thekillsquad007/amdbtx-releases

---

## Configuration

Edit `~/.amdbtx-miner/config.yaml` after install. See
[`config.example.yaml`](config.example.yaml) for all options.

**Pool mode** (default):

```yaml
mining_mode: "pool"
pool_host: "btx-sg.lproute.com"
pool_port: 8660
payout_address: "btx1z..."
worker_name: "myrig"
solver_backend: "rocm"
solver_threads: 16
solver_batch_size: 4194304   # installer sets a GPU-specific default; tune with --benchmark
gpu_device: -1               # -1 = auto; 0/1/... = force one GPU
# gpu_devices: "all"         # multi-GPU: "all", "0,1", or [0, 1]
log_level: "INFO"
```

CLI overrides:

```bash
amdbtx-miner --payout-address btx1z... --worker-name myrig
```

### Multi-GPU

Set `gpu_devices: "all"` (or `"0,1"`) in config. Each GPU mines in parallel and
effective hashrate stacks.

```bash
rocminfo | grep -E 'Marketing Name|gfx'   # confirm cards are visible
amdbtx-miner --config ~/.amdbtx-miner/config.yaml
```

Or on the command line: `--gpu-devices all`

### Performance tuning

The installer picks sensible defaults for your GPU. To find the best batch size:

```bash
amdbtx-miner --benchmark --config ~/.amdbtx-miner/config.yaml
```

Then copy the recommended `solver_batch_size` into your config.

| GPU | Typical batch size |
|-----|-------------------|
| RX 470/480/570/580 | 64 |
| RX Vega / RDNA 1 | 4096 |
| RX 6600–6900 (RDNA 2) | 1048576 |
| RX 7600–7900 (RDNA 3) | 4194304 |
| RX 9060/9070 (RDNA 4) | run `--benchmark` |

---

## Solo Mining

Mine against your own synced BTX node instead of a pool. Requires
[BTX v0.32.3+](https://github.com/btxchain/btx/releases) with RPC enabled.

**Local node** (cookie auth):

```yaml
mining_mode: solo
rpc_url: "http://127.0.0.1:19334"
rpc_cookie_file: "~/.btx/.cookie"
payout_address: "btx1z..."
```

```bash
amdbtx-miner --solo --payout-address btx1z...
```

**Remote node** (username/password):

```yaml
mining_mode: solo
rpc_url: "http://192.168.1.15:19334"
rpc_user: "miner"
rpc_password: "your_rpc_password"
payout_address: "btx1z..."
```

Solo uses the same GPU solver as pool mode but submits full blocks when network
difficulty is met. A 2% dev fee is taken from the coinbase on each block found
(set `solo_dev_fee_bps: 0` in config to disable).

---

## Pools

### LuckyPool (default)

- **Dashboard**: https://luckypool.org
- **Stratum**: `stratum+tcp://btx-sg.lproute.com:8660`
- **Pool fee**: 0.5% (PPLNS)

### BitMinerPool

Point your config at a BitMinerPool endpoint (may need a local proxy):

```yaml
pool_host: "stratum.bitminerpool.xyz"
pool_port: 3333
```
pool_port: 8660
```

On LuckyPool hosts (`*.lproute.com`) the miner auto-switches to the LuckyPool
login/submit dialect. Other pools (BitMinerPool, etc.) always use standard
stratum. Force LuckyPool with `pool_protocol: luckypool` in config.

### Wallet

Pools do not create wallets. Get a BTX address at https://easybtx.com/wallet
(`btx1z...` or `btx1q...`).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `invalid payout address` | Replace the placeholder in `~/.amdbtx-miner/config.yaml` |
| Solver not working | Re-run install, or `bash build_solver.sh` from a cloned repo |
| `rocm-smi` not found | Install ROCm or `export PATH=/opt/rocm/bin:$PATH` |
| `/dev/kfd` permission denied | `sudo chmod 666 /dev/kfd /dev/dri/*` |
| GPU not detected in WSL | Set `HSA_ENABLE_DXG_DETECTION=1`, then `wsl --shutdown` and restart |
| Low hashrate | Run `--benchmark` and update `solver_batch_size` |
| Share rejected after reconnect | Wait 1–2 minutes for the pool to sync |

---

## Links

- **Pool**: https://bitminerpool.xyz
- **Releases**: https://github.com/thekillsquad007/amdbtx-releases
- **GitHub**: https://github.com/thekillsquad007/amdbtx
- **Telegram**: @btxdexbot