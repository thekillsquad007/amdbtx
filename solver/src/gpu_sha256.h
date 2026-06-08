#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "field.h"

namespace gpasha {

// Block header structure for sigma derivation.
// All fields are stored in little-endian (matching CPU DeriveSigma).
struct SigmaHeader {
    uint8_t version[4];       // LE32
    uint8_t prev_hash[32];    // canonical bytes (big-endian reversed → LE)
    uint8_t merkle_root[32];  // canonical bytes
    uint8_t time[4];          // LE32
    uint8_t bits[4];          // LE32
    uint8_t nonce_lo[4];      // nonce bits 0-31 (LE) — unused, kernel uses nonce_start
    uint8_t nonce_hi[4];      // nonce bits 32-63 (LE)
    uint8_t dim[2];           // LE16
    uint8_t seed_a[32];       // canonical bytes
    uint8_t seed_b[32];       // canonical bytes
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

}
