# AMDBTX HiveOS custom miner

This package is a lightweight HiveOS wrapper for AMDBTX `v1.2.0`.
It does not bundle the miner or replace the HiveOS ROCm stack. On first run,
`h-run.sh` downloads the upstream installer and runs it with `--skip-rocm`.

## Build the package

From the repository root:

```bash
bash hiveos/build-package.sh
```

The archive will be written to `dist/amdbtx-1.2.0_hiveos.tar.gz`.

## HiveOS Flight Sheet

- Miner: Custom
- Miner name: `amdbtx`
- Installation URL: direct URL to `amdbtx-1.2.0_hiveos.tar.gz`
- Wallet and worker template: `%WAL%.%WORKER_NAME%`
- Pool URL: `stratum+tcp://btx-sg.lproute.com:8660`
- Pass: `x`

The generated AMDBTX config uses LuckyPool by default and enables all AMD GPUs:

```yaml
pool_protocol: "luckypool"
gpu_devices: "all"
```

## Extra config

Use HiveOS "Extra config arguments" as YAML lines when you want to override the
generated config:

```yaml
solver_batch_size: 1048576
solver_threads: 12
gpu_devices: "0,1"
```

Simple CLI flags are also supported, for example:

```bash
--log-level DEBUG
```

## Installation mode

Default first-run install mode is prebuilt:

```bash
AMDBTX_HIVE_INSTALL_MODE=prebuilt
```

For GPUs not covered by the upstream prebuilt solver, set this environment
variable to `source`. That preserves HiveOS ROCm packages but requires a working
HIP compiler on the rig:

```bash
AMDBTX_HIVE_INSTALL_MODE=source
```
