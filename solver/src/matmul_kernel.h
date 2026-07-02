#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

#include "solve.h"
#include <cstddef>
#include <cstdint>
#include <vector>

struct FusedMatMulJob {
    PowState state;
    uint32_t n = 0;
    uint32_t b = 0;
    uint32_t r = 0;
    uint32_t epsilon_bits = 0;
    Uint256 share_target;
    Uint256 block_target;
};

struct MatMulBatchHit {
    uint64_t nonce = 0;
    Uint256 digest;
    uint8_t found = 0;
};

size_t FusedWorkspaceBytesPerNonce(uint32_t n, uint32_t r, uint32_t b, bool use_v2_seeds);

int AutoBatchSizeForDevice(
    int device,
    uint32_t n,
    uint32_t r,
    uint32_t b,
    int32_t block_height,
    int max_cap = 2048);

extern "C" bool LaunchMatMulTranscriptBatch(
    int device,
    const FusedMatMulJob& job,
    const std::vector<uint64_t>& nonces,
    std::vector<Uint256>& out_digests,
    std::vector<bool>& out_found,
    std::vector<bool>* out_gate_passed = nullptr);

extern "C" bool LaunchMatMulTranscriptCompact(
    int device,
    const FusedMatMulJob& job,
    uint64_t nonce_start,
    size_t nonce_count,
    std::vector<MatMulBatchHit>& out_hits,
    uint64_t* out_gate_passes = nullptr);

extern "C" bool HipVerifyAgainstCpu(
    const FusedMatMulJob& job,
    uint64_t nonce);

#endif
