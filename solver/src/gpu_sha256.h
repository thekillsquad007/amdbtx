#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "field.h"

namespace gpasha {

// Block header structure for sigma derivation.
// version/time/bits/nonce/dim: LE (matches BTX wire serialization)
// prev_hash/merkle_root: raw Uint256.data() bytes (LSB-first for SetHex, MSB-first for hash)
// seed_a/seed_b: raw Uint256.data() bytes (LSB-first for V1 hex, MSB-first for V2 hash)
// This matches BTX HashWriter serialization which writes data[0..31] raw bytes.
struct SigmaHeader {
    uint8_t version[4];       // LE32
    uint8_t prev_hash[32];    // raw Uint256.data() bytes
    uint8_t merkle_root[32];  // raw Uint256.data() bytes
    uint8_t time[4];          // LE32
    uint8_t bits[4];          // LE32
    uint8_t nonce_lo[4];      // nonce bits 0-31 (LE)
    uint8_t nonce_hi[4];      // nonce bits 32-63 (LE)
    uint8_t dim[2];           // LE16
    uint8_t seed_a[32];       // raw Uint256.data() bytes
    uint8_t seed_b[32];       // raw Uint256.data() bytes
};

void DeriveSigmaGateKernel_launch(
    const SigmaHeader* header,
    uint64_t nonce_start,
    const uint8_t* sigma_gate_threshold_be,
    uint32_t epsilon_bits,
    uint64_t batch_size,
    uint32_t* gate_count,
    uint64_t* sigma_nonces,
    uint8_t* sigma_batch,
    hipStream_t stream = 0);

// V2 batched sigma gate: takes an array of headers (one per nonce).
// Each header already has the correct nonce and seeds.
void DeriveSigmaGateKernel_Batched_launch(
    const SigmaHeader* headers,
    const uint8_t* sigma_gate_threshold_be,
    uint32_t epsilon_bits,
    uint64_t batch_size,
    uint32_t* gate_count,
    uint64_t* sigma_nonces,
    uint8_t* sigma_batch,
    hipStream_t stream = 0);

void DeriveNoiseSeedsKernel_launch(
    const uint8_t* sigma_batch,
    uint8_t* noise_seeds,
    uint8_t* compress_seeds,
    uint32_t batch_size,
    hipStream_t stream = 0);

void GenerateNoiseKernel_launch(
    const uint8_t* noise_seeds,
    uint32_t batch_size,
    uint32_t num_elements,
    uint32_t seed_index,
    Element* output,
    hipStream_t stream = 0);

void GenerateCompressKernel_launch(
    const uint8_t* compress_seeds,
    uint32_t batch_size,
    uint32_t num_elements,
    Element* output,
    hipStream_t stream = 0);

void HashTranscriptKernel_launch(
    const Element* compressed_words,
    const uint8_t* sigma_batch,
    uint32_t words_per_nonce,
    uint32_t n,
    uint32_t b,
    uint32_t batch_size,
    uint8_t* digest_batch,
    hipStream_t stream = 0);

void CompareDigestsKernel_launch(
    const uint8_t* digest_batch,
    const uint8_t* block_target,
    const uint8_t* share_target,
    uint32_t batch_size,
    int32_t* results,
    hipStream_t stream = 0);

// Generate matrix elements from per-nonce seeds.
// seeds: batch_size * 32 bytes (one 32-byte seed per nonce, LE)
// output: batch_size * n * n elements
void GenerateMatrixKernel_launch(
    const uint8_t* seeds,
    uint32_t batch_size,
    uint32_t n,
    Element* output,
    hipStream_t stream = 0);

// V2 nonce-bound seed parameters (common across all nonces in a batch).
struct V2SeedParams {
    uint8_t prev_hash[32];    // LSB-first (BTX SetHex internal storage)
    uint32_t height;          // block height
    uint8_t version[4];       // LE32
    uint8_t merkle_root[32];  // LSB-first (BTX SetHex internal storage)
    uint8_t time[4];          // LE32
    uint8_t bits[4];          // LE32
    uint8_t dim[2];           // LE16
};

// Combined V2 seed derivation + sigma gate kernel.
// For each nonce: computes V2 seed_a, seed_b, then sigma from header with those seeds.
// Output: sigma_batch (only for gate-passing nonces), sigma_nonces, and gate_count.
void DeriveV2SeedsAndSigmaGateKernel_launch(
    const V2SeedParams* params,
    uint64_t nonce_start,
    const uint8_t* sigma_gate_threshold_be,
    uint32_t epsilon_bits,
    uint64_t batch_size,
    uint32_t* gate_count,
    uint64_t* sigma_nonces,
    uint8_t* sigma_batch,
    uint8_t* seed_a_batch,
    uint8_t* seed_b_batch,
    hipStream_t stream = 0);

}
