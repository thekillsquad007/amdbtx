#ifndef SOLVE_H
#define SOLVE_H

#include "matrix.h"
#include "noise.h"
#include "transcript.h"
#include "uint256.h"
#include <cstdint>

struct PowState {
    int32_t version;
    Uint256 previous_block_hash;
    Uint256 merkle_root;
    uint32_t time;
    uint32_t bits;
    Uint256 seed_a;
    Uint256 seed_b;
    uint64_t nonce;
    uint16_t matmul_dim;
    int32_t height;
    Uint256 digest;
};

Uint256 DeriveSigma(const PowState& state);
Uint256 DeriveBlockTarget(uint32_t bits);
bool Uint256LE(const Uint256& a, const Uint256& b);
void Uint256ToMsbBytes(const Uint256& v, uint8_t out[32]);
Uint256 DerivePreHashTarget(const Uint256& target, uint32_t epsilon_bits);

// BTX v0.32.3 DeterministicMatMulSeedV2 (per-nonce seed derivation for height >= 125000)
Uint256 DeterministicMatMulSeedV2(
    const Uint256& prev_hash, int32_t height,
    int32_t version, const Uint256& merkle_root,
    uint32_t time, uint32_t bits,
    uint64_t nonce64, uint16_t matmul_dim,
    uint8_t which);

bool ComputeDigestForNonce(PowState& state, uint32_t n, uint32_t b, uint32_t r, Uint256& out_digest);

bool SolveCPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
              const Uint256& block_target, const Uint256& share_target,
              uint64_t& max_tries, double max_seconds,
              uint64_t& tries_used, double& elapsed_s,
              uint32_t epsilon_bits = 18);

bool SolveGPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
              const Uint256& block_target, const Uint256& share_target,
              uint64_t& max_tries, double max_seconds,
              uint64_t& tries_used, double& elapsed_s,
              uint32_t batch_size = 128, uint32_t epsilon_bits = 18,
              bool* cpu_fallback = nullptr);

#endif
