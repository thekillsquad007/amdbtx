#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "field.h"

namespace gpasha {

struct SigmaHeader;

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
