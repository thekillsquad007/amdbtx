# AGENTS.md — AMD BTX Miner (amdbtx)

## Project Overview

An AMD GPU miner for the [MineBtx](https://minebtx.com) pool. Forks the
[dexbtx-miner](https://github.com/dexbtx/minebtx) Python stratum client and
adds a standalone HIP/ROCm solver binary for AMD GPUs.

- **Pool stratum**: `stratum+tcp://stratum.minebtx.com:3333`
- **Pool fee**: 2.5% (PPLNS, weekly payouts)
- **Dev fee**: 2% (time-sliced, transparent)
- **Dev wallet**: `btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4`

## Architecture

```
amdbtx-miner (Python)
  ├── stratum_client.py    ← pool communication (stratum/2.0-matmul)
  ├── gbt_solve_wrapper.py ← drives solver daemon via stdin/stdout JSON
  ├── hardware.py          ← AMD GPU detection (rocm-smi)
  ├── config.py            ← YAML config loader
  └── __main__.py          ← CLI entry point

btx-gbt-solve-hip (C++/HIP)
  ├── main.cpp             ← CLI + daemon mode
  ├── field.h              ← M31 field arithmetic (q = 2^31 - 1)
  ├── matmul_kernel.hip    ← GPU matmul kernel (HIP/ROCm)
  ├── noise.h              ← Noise generation from sigma
  ├── transcript.h         ← Transcript compression + SHA-256d
  └── sha256.h             ← SHA-256 implementation
```

## MatMul PoW Algorithm (BTX Spec)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Matrix dimension (n) | 512 | O(n³) = ~134M field muls per attempt |
| Transcript block size (b) | 16 | (n/b)³ = 32,768 intermediate blocks |
| Noise rank (r) | 8 | Security parameter |
| Field modulus (q) | 2³¹ − 1 | Mersenne prime M31 |
| Pre-hash epsilon (ε) | 18 | Sigma gate: target « ε |
| Freivalds rounds (k) | 2 | False-positive < 2⁻⁶² |

### Solve Pipeline (per nonce)

1. **Sigma gate**: SHA-256 pre-hash; skip if digest ≥ target « ε
2. **Noise gen**: Derive EL, ER, FL, FR from sigma via SHA-256 PRF
3. **Noisy MatMul**: C' = (A+EL·ER) · (B+FL·FR), block decomposition b=16
4. **Compression**: Inner-product per b×b block → ~131KB total
5. **SHA-256d**: Rolling hash on compressed transcript
6. **Check**: H(transcript) < target → found share

### M31 Reduction

```c
// Double Mersenne fold, safe for all uint64 inputs
inline uint32_t reduce64(uint64_t x) {
    x = (x & 0x7FFFFFFF) + (x >> 31);
    x = (x & 0x7FFFFFFF) + (x >> 31);
    if (x >= 0x7FFFFFFF) x -= 0x7FFFFFFF;
    return (uint32_t)x;
}
```

## Solver Binary Protocol (Daemon Mode)

### Input (stdin, one JSON per line)

```json
{
  "version": 536870912,
  "prev_hash": "hex64",
  "merkle_root": "hex64",
  "time": 1779672814,
  "bits": "1d17c609",
  "seed_a": "hex64",
  "seed_b": "hex64",
  "block_height": 110806,
  "nonce_start": 0,
  "max_tries": 2000000,
  "max_seconds": 5.0,
  "share_target": "hex64"
}
```

### Output (stdout, one JSON per line)

```json
{
  "found": true,
  "nonce64": 12345678,
  "nonce64_end": 12346000,
  "digest": "hex64",
  "elapsed_s": 3.14,
  "tries_used": 1000000,
  "is_block": false
}
```

### Handshake

Solver writes `{"event":"daemon_ready"}` to stderr on startup.

## Dev Fee Mechanism

Time-sliced worker switching (standard approach used by Claymore/Phoenix/etc.):

1. Mine normally with user address for 58 minutes
2. Switch `mining.authorize` to dev wallet for 2 minutes
3. Re-authorize with user address
4. Log all switches at INFO level

The pool credits shares based on the address in `payout_address.worker_name`.
Switching the authorize message is sufficient — no reconnect needed.

Dev wallet: `btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4`

## AMD GPU Tuning Profiles

| GPU | Arch | GFX | workers | threads | batch |
|-----|------|-----|---------|---------|-------|
| RX 470/480/570/580 | GCN 4 | gfx803 | 8 | 4 | 64 |
| RX Vega 56/64 | GCN 5 | gfx900 | 8 | 4 | 64 |
| Radeon VII | GCN 5 | gfx906 | 12 | 8 | 128 |
| RX 5500/5600/5700 | RDNA 1 | gfx1010 | 12 | 8 | 128 |
| RX 6600/6700/6800/6900 | RDNA 2 | gfx1030 | 16 | 8 | 128 |
| RX 7600/7700/7800/7900 | RDNA 3 | gfx1100 | 16 | 8 | 128 |
| RX 9070 | RDNA 4 | gfx1102 | 16 | 8 | 128 |
| **Universal default** | | | **16** | **8** | **128** |

## Build Requirements

### Container/Host

- Ubuntu 22.04+ (the user's existing container)
- ROCm 6.0+ (`rocm-dev`, `rocm-hip-runtime`, `hipsolver`, `hipblas`)
- Python 3.10+
- CMake 3.22+
- GCC 11+ or Clang 16+

### Build Commands

```bash
# Install ROCm (in container)
curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 jammy main' > /etc/apt/sources.list.d/rocm.list
apt-get update && apt-get install -y rocm-dev hipsolver hipblas

# Build solver
cd solver && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
make -j$(nproc)

# Install Python wrapper
cd /var/home/bazzite/amdbtx
pip install --user -e .
```

## File Inventory

| File | Purpose | Lines |
|------|---------|-------|
| `AGENTS.md` | This file | ~200 |
| `solver/CMakeLists.txt` | Build system | ~80 |
| `solver/src/main.cpp` | CLI + daemon entry | ~400 |
| `solver/src/field.h` | M31 field ops | ~100 |
| `solver/src/matmul_kernel.hip` | GPU matmul kernel | ~500 |
| `solver/src/noise.h` | Noise generation | ~120 |
| `solver/src/transcript.h` | Transcript compression | ~200 |
| `solver/src/sha256.h` | SHA-256 | ~150 |
| `src/amdbtx_miner/__init__.py` | Package init | ~20 |
| `src/amdbtx_miner/__main__.py` | CLI entry | ~120 |
| `src/amdbtx_miner/config.py` | Config loader | ~80 |
| `src/amdbtx_miner/stratum_client.py` | Stratum + dev fee | ~450 |
| `src/amdbtx_miner/gbt_solve_wrapper.py` | Solver wrapper | ~250 |
| `src/amdbtx_miner/hardware.py` | AMD GPU detection | ~200 |
| `src/amdbtx_miner/benchmark.py` | Benchmark tool | ~200 |
| `install_amd.sh` | AMD installer | ~300 |
| `pyproject.toml` | Package definition | ~30 |
| `config.example.yaml` | Config template | ~50 |

## Testing

1. **Unit test field arithmetic**: Verify M31 reduce64, add, mul, dot product
2. **Smoke test solver**: Run on known test vectors, verify digest matches CPU
3. **Daemon mode test**: Send JSON job, verify JSON result
4. **Pool integration**: Connect to testnet pool, verify accepted shares
5. **Dev fee verification**: Log watch for authorize switches

## Common Pitfalls

- **ROCm not detecting GPU**: Ensure `/dev/kfd` and `/dev/dri` are accessible in container
- **Low GPU util**: Bump `solver_prepare_workers` and `solver_threads` together
- **Share rejected (code 21)**: Normal after reconnect, wait 1-2 minutes
- **Silent CPU fallback**: Check `rocm-smi` shows GPU in use during mining

## Pool Contact

- **Dashboard**: https://pool.minebtx.com
- **Telegram**: @btxdexbot (`/stats`, `/mybalance`, `/help`)
- **GitHub**: https://github.com/dexbtx/minebtx
