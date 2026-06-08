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
    Uint256 digest;
};

Uint256 DeriveSigma(const PowState& state);
Uint256 DeriveBlockTarget(uint32_t bits);
bool Uint256LE(const Uint256& a, const Uint256& b);

bool SolveCPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
              const Uint256& block_target, const Uint256& share_target,
              uint64_t& max_tries, double max_seconds,
              uint64_t& tries_used, double& elapsed_s);

bool SolveGPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
              const Uint256& block_target, const Uint256& share_target,
              uint64_t& max_tries, double max_seconds,
              uint64_t& tries_used, double& elapsed_s,
              uint32_t batch_size = 128, uint32_t epsilon_bits = 18);

#endif
