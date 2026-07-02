// HIP implementation for BTX MatMul PoW transcript search.
// One HIP block per nonce: full sigma derivation, noise, blocked matmul,
// running transcript compression, and target check on device.
// Host uploads constant A/B (FromSeed) once per job; CPU VerifySolution
// cross-checks any hits before submission.

#include "solve.h"
#include "matmul_kernel.h"
#include "matrix.h"
#include "field.h"
#include "gpu_sha256.h"
#include "uint256.h"



#ifdef __HIPCC__

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kFieldModulus = 0x7FFFFFFFU;
constexpr int kBlockThreads = 256;

struct HipJobParams {
    int32_t version;
    uint32_t time;
    uint32_t bits;
    uint32_t n;
    uint32_t b;
    uint32_t r;
    uint32_t epsilon_bits;
    uint32_t block_height;
    uint32_t fast_v3_scan;
    int64_t parent_mtp;
    uint8_t prev_hash[32];
    uint8_t merkle_root[32];
    uint8_t seed_a[32];
    uint8_t seed_b[32];
    uint8_t target[32];
    uint8_t pre_hash_target[32];
};

struct Sha256State {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t data[64];
    uint32_t datalen;
};

__device__ __forceinline__ uint32_t d_rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t d_ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t d_maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t d_ep0(uint32_t x)
{
    return d_rotr(x, 2) ^ d_rotr(x, 13) ^ d_rotr(x, 22);
}

__device__ __forceinline__ uint32_t d_ep1(uint32_t x)
{
    return d_rotr(x, 6) ^ d_rotr(x, 11) ^ d_rotr(x, 25);
}

__device__ __forceinline__ uint32_t d_sig0(uint32_t x)
{
    return d_rotr(x, 7) ^ d_rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t d_sig1(uint32_t x)
{
    return d_rotr(x, 17) ^ d_rotr(x, 19) ^ (x >> 10);
}

__device__ void d_sha256_transform(Sha256State* ctx, const uint8_t block[64])
{
    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    uint32_t m[64];
    for (int i = 0; i < 16; ++i) {
        m[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        m[i] = d_sig1(m[i - 2]) + m[i - 7] + d_sig0(m[i - 15]) + m[i - 16];
    }

    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + K[i] + m[i];
        uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

__device__ void d_sha256_init(Sha256State* ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

__device__ void d_sha256_update(Sha256State* ctx, const uint8_t* data, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            d_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void d_sha256_final(Sha256State* ctx, uint8_t hash[32])
{
    uint32_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) {
            ctx->data[i++] = 0;
        }
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) {
            ctx->data[i++] = 0;
        }
        d_sha256_transform(ctx, ctx->data);
        for (uint32_t j = 0; j < 56; ++j) {
            ctx->data[j] = 0;
        }
        i = 56;
    }

    ctx->bitlen += static_cast<uint64_t>(ctx->datalen) * 8;
    ctx->data[63] = static_cast<uint8_t>(ctx->bitlen);
    ctx->data[62] = static_cast<uint8_t>(ctx->bitlen >> 8);
    ctx->data[61] = static_cast<uint8_t>(ctx->bitlen >> 16);
    ctx->data[60] = static_cast<uint8_t>(ctx->bitlen >> 24);
    ctx->data[59] = static_cast<uint8_t>(ctx->bitlen >> 32);
    ctx->data[58] = static_cast<uint8_t>(ctx->bitlen >> 40);
    ctx->data[57] = static_cast<uint8_t>(ctx->bitlen >> 48);
    ctx->data[56] = static_cast<uint8_t>(ctx->bitlen >> 56);
    d_sha256_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i] = static_cast<uint8_t>((ctx->state[0] >> (24 - i * 8)) & 0xff);
        hash[i + 4] = static_cast<uint8_t>((ctx->state[1] >> (24 - i * 8)) & 0xff);
        hash[i + 8] = static_cast<uint8_t>((ctx->state[2] >> (24 - i * 8)) & 0xff);
        hash[i + 12] = static_cast<uint8_t>((ctx->state[3] >> (24 - i * 8)) & 0xff);
        hash[i + 16] = static_cast<uint8_t>((ctx->state[4] >> (24 - i * 8)) & 0xff);
        hash[i + 20] = static_cast<uint8_t>((ctx->state[5] >> (24 - i * 8)) & 0xff);
        hash[i + 24] = static_cast<uint8_t>((ctx->state[6] >> (24 - i * 8)) & 0xff);
        hash[i + 28] = static_cast<uint8_t>((ctx->state[7] >> (24 - i * 8)) & 0xff);
    }
}

__device__ void d_to_canonical(const uint8_t* internal, uint8_t* canonical)
{
    for (int i = 0; i < 32; ++i) {
        canonical[i] = internal[31 - i];
    }
}

__device__ __constant__ uint32_t kSha256K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

__device__ __forceinline__ void d_set_sha_byte(uint32_t w[16], uint32_t offset, uint32_t byte)
{
    const uint32_t word_index = offset >> 2U;
    const uint32_t shift = (3U - (offset & 3U)) * 8U;
    w[word_index] |= (byte & 0xffU) << shift;
}

__device__ __forceinline__ uint32_t d_bswap32(uint32_t x)
{
    return ((x & 0x000000ffU) << 24U) |
           ((x & 0x0000ff00U) << 8U) |
           ((x & 0x00ff0000U) >> 8U) |
           ((x & 0xff000000U) >> 24U);
}

__device__ __forceinline__ uint32_t d_sha_word4(
    uint32_t b0,
    uint32_t b1,
    uint32_t b2,
    uint32_t b3)
{
    return ((b0 & 0xffU) << 24U) |
           ((b1 & 0xffU) << 16U) |
           ((b2 & 0xffU) << 8U) |
           (b3 & 0xffU);
}

__device__ __forceinline__ uint32_t d_digest_byte(
    const uint32_t words[8],
    uint32_t byte_index)
{
    const uint32_t word = words[byte_index >> 2U];
    const uint32_t shift = (3U - (byte_index & 3U)) * 8U;
    return (word >> shift) & 0xffU;
}

__device__ __forceinline__ uint32_t d_digest_word4(
    const uint32_t words[8],
    uint32_t byte_index)
{
    return d_sha_word4(
        d_digest_byte(words, byte_index),
        d_digest_byte(words, byte_index + 1U),
        d_digest_byte(words, byte_index + 2U),
        d_digest_byte(words, byte_index + 3U));
}

__device__ __forceinline__ void d_sha256_init_words(uint32_t state[8])
{
    state[0] = 0x6a09e667U;
    state[1] = 0xbb67ae85U;
    state[2] = 0x3c6ef372U;
    state[3] = 0xa54ff53aU;
    state[4] = 0x510e527fU;
    state[5] = 0x9b05688cU;
    state[6] = 0x1f83d9abU;
    state[7] = 0x5be0cd19U;
}

__device__ __forceinline__ void d_sha256_compress(uint32_t state[8], uint32_t w[16])
{
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    #pragma unroll
    for (uint32_t t = 0; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = d_sig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] + d_sig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + kSha256K[t] + wt;
        const uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void d_sha256_bytes(const uint8_t* message, uint32_t message_len, uint8_t out[32])
{
    uint32_t state[8];
    d_sha256_init_words(state);

    const uint32_t total_blocks = (message_len + 9U + 63U) / 64U;
    const uint64_t bit_len = static_cast<uint64_t>(message_len) * 8U;
    for (uint32_t block = 0; block < total_blocks; ++block) {
        uint32_t w[16] = {};
        for (uint32_t word = 0; word < 16; ++word) {
            uint32_t packed = 0;
            for (uint32_t byte = 0; byte < 4; ++byte) {
                const uint32_t message_index = block * 64U + word * 4U + byte;
                uint8_t value = 0;
                if (message_index < message_len) {
                    value = message[message_index];
                } else if (message_index == message_len) {
                    value = 0x80U;
                } else {
                    const uint32_t length_start = total_blocks * 64U - 8U;
                    if (message_index >= length_start) {
                        const uint32_t shift = (7U - (message_index - length_start)) * 8U;
                        value = static_cast<uint8_t>((bit_len >> shift) & 0xffU);
                    }
                }
                packed = (packed << 8U) | value;
            }
            w[word] = packed;
        }
        d_sha256_compress(state, w);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
}

__device__ void d_sha256_block0_midstate(const uint8_t* message, uint32_t state[8])
{
    d_sha256_init_words(state);
    uint32_t w[16];
    for (uint32_t word = 0; word < 16; ++word) {
        uint32_t packed = 0;
        for (uint32_t byte = 0; byte < 4; ++byte) {
            packed = (packed << 8U) | message[word * 4U + byte];
        }
        w[word] = packed;
    }
    d_sha256_compress(state, w);
}

__device__ void d_sha256_bytes_from_midstate(
    const uint32_t midstate[8],
    const uint8_t* message,
    uint32_t message_len,
    uint8_t out[32])
{
    uint32_t state[8];
    for (uint32_t i = 0; i < 8; ++i) {
        state[i] = midstate[i];
    }

    const uint32_t total_blocks = (message_len + 9U + 63U) / 64U;
    const uint64_t bit_len = static_cast<uint64_t>(message_len) * 8U;
    for (uint32_t block = 1; block < total_blocks; ++block) {
        uint32_t w[16] = {};
        for (uint32_t word = 0; word < 16; ++word) {
            uint32_t packed = 0;
            for (uint32_t byte = 0; byte < 4; ++byte) {
                const uint32_t message_index = block * 64U + word * 4U + byte;
                uint8_t value = 0;
                if (message_index < message_len) {
                    value = message[message_index];
                } else if (message_index == message_len) {
                    value = 0x80U;
                } else {
                    const uint32_t length_start = total_blocks * 64U - 8U;
                    if (message_index >= length_start) {
                        const uint32_t shift = (7U - (message_index - length_start)) * 8U;
                        value = static_cast<uint8_t>((bit_len >> shift) & 0xffU);
                    }
                }
                packed = (packed << 8U) | value;
            }
            w[word] = packed;
        }
        d_sha256_compress(state, w);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
}

__device__ __forceinline__ void d_sha256_state_to_bytes(
    const uint32_t state[8],
    uint8_t out[32])
{
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
}

__device__ __forceinline__ void d_sha256_once_state_words_to_state(
    const uint32_t words[8],
    uint32_t out_state[8])
{
    uint32_t w[16];
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        w[i] = words[i];
    }
    w[8] = 0x80000000U;
    #pragma unroll
    for (uint32_t i = 9U; i < 15U; ++i) {
        w[i] = 0U;
    }
    w[15] = 32U * 8U;
    d_sha256_init_words(out_state);
    d_sha256_compress(out_state, w);
}

__device__ __forceinline__ void d_matmul_seed_v3_midstate_words(
    const uint32_t seed_midstate[8],
    const HipJobParams& job,
    uint64_t nonce64,
    uint32_t which,
    uint32_t out_words[8])
{
    uint32_t state[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        state[i] = seed_midstate[i];
    }

    uint32_t w[16];
    w[0] = d_sha_word4(
        (static_cast<uint32_t>(job.version) >> 8U) & 0xffU,
        (static_cast<uint32_t>(job.version) >> 16U) & 0xffU,
        (static_cast<uint32_t>(job.version) >> 24U) & 0xffU,
        job.merkle_root[0]);
    #pragma unroll
    for (uint32_t i = 1U; i < 8U; ++i) {
        const uint32_t base = 1U + (i - 1U) * 4U;
        w[i] = d_sha_word4(
            job.merkle_root[base],
            job.merkle_root[base + 1U],
            job.merkle_root[base + 2U],
            job.merkle_root[base + 3U]);
    }
    w[8] = d_sha_word4(
        job.merkle_root[29],
        job.merkle_root[30],
        job.merkle_root[31],
        job.time & 0xffU);
    w[9] = d_sha_word4(
        (job.time >> 8U) & 0xffU,
        (job.time >> 16U) & 0xffU,
        (job.time >> 24U) & 0xffU,
        job.bits & 0xffU);
    w[10] = d_sha_word4(
        (job.bits >> 8U) & 0xffU,
        (job.bits >> 16U) & 0xffU,
        (job.bits >> 24U) & 0xffU,
        nonce64 & 0xffU);
    w[11] = d_sha_word4(
        (nonce64 >> 8U) & 0xffU,
        (nonce64 >> 16U) & 0xffU,
        (nonce64 >> 24U) & 0xffU,
        (nonce64 >> 32U) & 0xffU);
    w[12] = d_sha_word4(
        (nonce64 >> 40U) & 0xffU,
        (nonce64 >> 48U) & 0xffU,
        (nonce64 >> 56U) & 0xffU,
        job.n & 0xffU);
    w[13] = d_sha_word4((job.n >> 8U) & 0xffU, which, 0x80U, 0U);
    w[14] = 0U;
    w[15] = 118U * 8U;
    d_sha256_compress(state, w);

    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        out_words[i] = state[i];
    }
}

__device__ __forceinline__ void d_matmul_seed_pair_v3_midstate_words(
    const uint32_t seed_midstate[8],
    const HipJobParams& job,
    uint64_t nonce64,
    uint32_t seed_a_words[8],
    uint32_t seed_b_words[8])
{
    d_matmul_seed_v3_midstate_words(seed_midstate, job, nonce64, 0U, seed_a_words);
    d_matmul_seed_v3_midstate_words(seed_midstate, job, nonce64, 1U, seed_b_words);
}

__device__ __forceinline__ void d_derive_sigma_v3_midstate_words_state(
    const uint32_t header_midstate[8],
    const HipJobParams& job,
    uint64_t nonce64,
    const uint32_t seed_a_words[8],
    const uint32_t seed_b_words[8],
    uint32_t sigma_state[8])
{
    uint32_t state[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8U; ++i) {
        state[i] = header_midstate[i];
    }

    uint32_t w[16];
    w[0] = d_sha_word4(
        job.merkle_root[28],
        job.merkle_root[29],
        job.merkle_root[30],
        job.merkle_root[31]);
    w[1] = d_bswap32(job.time);
    w[2] = d_bswap32(job.bits);
    w[3] = d_bswap32(static_cast<uint32_t>(nonce64));
    w[4] = d_bswap32(static_cast<uint32_t>(nonce64 >> 32U));
    w[5] = d_sha_word4(
        job.n & 0xffU,
        (job.n >> 8U) & 0xffU,
        d_digest_byte(seed_a_words, 0U),
        d_digest_byte(seed_a_words, 1U));
    w[6] = d_digest_word4(seed_a_words, 2U);
    w[7] = d_digest_word4(seed_a_words, 6U);
    w[8] = d_digest_word4(seed_a_words, 10U);
    w[9] = d_digest_word4(seed_a_words, 14U);
    w[10] = d_digest_word4(seed_a_words, 18U);
    w[11] = d_digest_word4(seed_a_words, 22U);
    w[12] = d_digest_word4(seed_a_words, 26U);
    w[13] = d_sha_word4(
        d_digest_byte(seed_a_words, 30U),
        d_digest_byte(seed_a_words, 31U),
        d_digest_byte(seed_b_words, 0U),
        d_digest_byte(seed_b_words, 1U));
    w[14] = d_digest_word4(seed_b_words, 2U);
    w[15] = d_digest_word4(seed_b_words, 6U);
    d_sha256_compress(state, w);

    w[0] = d_digest_word4(seed_b_words, 10U);
    w[1] = d_digest_word4(seed_b_words, 14U);
    w[2] = d_digest_word4(seed_b_words, 18U);
    w[3] = d_digest_word4(seed_b_words, 22U);
    w[4] = d_digest_word4(seed_b_words, 26U);
    w[5] = d_sha_word4(
        d_digest_byte(seed_b_words, 30U),
        d_digest_byte(seed_b_words, 31U),
        0x80U,
        0U);
    #pragma unroll
    for (uint32_t i = 6U; i < 15U; ++i) {
        w[i] = 0U;
    }
    w[15] = 150U * 8U;
    d_sha256_compress(state, w);
    d_sha256_once_state_words_to_state(state, sigma_state);
}

__device__ __forceinline__ bool d_sigma_state_below_prehash(
    const uint32_t sigma_state[8],
    const uint8_t target_arith[32])
{
    #pragma unroll
    for (uint32_t word = 0; word < 8U; ++word) {
        const uint32_t lhs = sigma_state[word];
        const uint32_t base = word * 4U;
        const uint32_t rhs = d_sha_word4(
            target_arith[31U - base],
            target_arith[30U - base],
            target_arith[29U - base],
            target_arith[28U - base]);
        if (lhs < rhs) return true;
        if (lhs > rhs) return false;
    }
    return true;
}

__device__ __forceinline__ void d_pack_seed_words(const uint8_t* seed_internal, uint32_t seed_w[8])
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        seed_w[i] = 0U;
    }
    for (uint32_t i = 0; i < 32; ++i) {
        d_set_sha_byte(seed_w, i, seed_bytes[i]);
    }
}

__device__ void d_compute_seed_midstate(const uint8_t* seed_internal, uint32_t midstate[16])
{
    uint32_t w[8];
    d_pack_seed_words(seed_internal, w);
    uint32_t a = 0x6a09e667U, b = 0xbb67ae85U, c = 0x3c6ef372U, d = 0xa54ff53aU;
    uint32_t e = 0x510e527fU, f = 0x9b05688cU, g = 0x1f83d9abU, h = 0x5be0cd19U;
    #pragma unroll
    for (uint32_t t = 0; t < 8; ++t) {
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + kSha256K[t] + w[t];
        const uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    midstate[0] = w[0];
    midstate[1] = w[1];
    midstate[2] = w[2];
    midstate[3] = w[3];
    midstate[4] = w[4];
    midstate[5] = w[5];
    midstate[6] = w[6];
    midstate[7] = w[7];
    midstate[8] = a;
    midstate[9] = b;
    midstate[10] = c;
    midstate[11] = d;
    midstate[12] = e;
    midstate[13] = f;
    midstate[14] = g;
    midstate[15] = h;
}

__device__ __forceinline__ uint32_t d_candidate_from_midstate(const uint32_t* mb, uint32_t index)
{
    uint32_t w[16];
    w[0] = mb[0];
    w[1] = mb[1];
    w[2] = mb[2];
    w[3] = mb[3];
    w[4] = mb[4];
    w[5] = mb[5];
    w[6] = mb[6];
    w[7] = mb[7];
    #pragma unroll
    for (uint32_t i = 8; i < 16; ++i) {
        w[i] = 0U;
    }
    d_set_sha_byte(w, 32U, index & 0xffU);
    d_set_sha_byte(w, 33U, (index >> 8U) & 0xffU);
    d_set_sha_byte(w, 34U, (index >> 16U) & 0xffU);
    d_set_sha_byte(w, 35U, (index >> 24U) & 0xffU);
    d_set_sha_byte(w, 36U, 0x80U);
    w[15] = 36U * 8U;

    uint32_t a = mb[8], b = mb[9], c = mb[10], d = mb[11];
    uint32_t e = mb[12], f = mb[13], g = mb[14], h = mb[15];
    #pragma unroll
    for (uint32_t t = 8; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = d_sig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] + d_sig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + kSha256K[t] + wt;
        const uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    return d_bswap32(0x6a09e667U + a) & kFieldModulus;
}

__device__ __forceinline__ uint32_t d_candidate_from_seed_index(
    const uint8_t* seed_internal,
    uint32_t index,
    bool with_retry,
    uint32_t retry)
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);

    uint32_t w[16] = {};
    for (uint32_t i = 0; i < 32; ++i) {
        d_set_sha_byte(w, i, seed_bytes[i]);
    }
    d_set_sha_byte(w, 32U, index & 0xffU);
    d_set_sha_byte(w, 33U, (index >> 8U) & 0xffU);
    d_set_sha_byte(w, 34U, (index >> 16U) & 0xffU);
    d_set_sha_byte(w, 35U, (index >> 24U) & 0xffU);

    uint32_t message_len = 36U;
    if (with_retry) {
        d_set_sha_byte(w, 36U, retry & 0xffU);
        d_set_sha_byte(w, 37U, (retry >> 8U) & 0xffU);
        d_set_sha_byte(w, 38U, (retry >> 16U) & 0xffU);
        d_set_sha_byte(w, 39U, (retry >> 24U) & 0xffU);
        message_len = 40U;
    }

    d_set_sha_byte(w, message_len, 0x80U);
    w[15] = message_len * 8U;

    uint32_t state[8];
    d_sha256_init_words(state);
    d_sha256_compress(state, w);
    return d_bswap32(state[0]) & kFieldModulus;
}

__device__ __forceinline__ uint32_t d_fallback_candidate(const uint8_t* seed_internal, uint32_t index)
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);

    uint32_t w[16] = {};
    for (uint32_t i = 0; i < 32; ++i) {
        d_set_sha_byte(w, i, seed_bytes[i]);
    }
    d_set_sha_byte(w, 32U, index & 0xffU);
    d_set_sha_byte(w, 33U, (index >> 8U) & 0xffU);
    d_set_sha_byte(w, 34U, (index >> 16U) & 0xffU);
    d_set_sha_byte(w, 35U, (index >> 24U) & 0xffU);

    constexpr uint8_t fallback_tag[15] = {
        'o', 'r', 'a', 'c', 'l', 'e', '-', 'f', 'a', 'l', 'l', 'b', 'a', 'c', 'k'
    };
    for (uint32_t i = 0; i < 15; ++i) {
        d_set_sha_byte(w, 36U + i, fallback_tag[i]);
    }

    d_set_sha_byte(w, 51U, 0x80U);
    w[15] = 51U * 8U;

    uint32_t state[8];
    d_sha256_init_words(state);
    d_sha256_compress(state, w);
    return d_bswap32(state[0]) % kFieldModulus;
}

__device__ __forceinline__ uint32_t d_from_oracle_fast(const uint8_t* seed_internal, uint32_t index)
{
    for (uint32_t retry = 0; retry < 256; ++retry) {
        const uint32_t candidate = retry == 0
            ? d_candidate_from_seed_index(seed_internal, index, false, 0U)
            : d_candidate_from_seed_index(seed_internal, index, true, retry);
        if (candidate < kFieldModulus) {
            return candidate;
        }
    }
    return d_fallback_candidate(seed_internal, index);
}

__device__ __forceinline__ uint32_t d_from_oracle_midstate(
    const uint32_t* midstate,
    const uint8_t* seed_internal,
    uint32_t index)
{
    const uint32_t candidate = d_candidate_from_midstate(midstate, index);
    return candidate < kFieldModulus ? candidate : d_from_oracle_fast(seed_internal, index);
}

__device__ __forceinline__ void d_append_byte(uint8_t* message, uint32_t& offset, uint8_t value)
{
    message[offset++] = value;
}

__device__ __forceinline__ void d_append_bytes(
    uint8_t* message,
    uint32_t& offset,
    const uint8_t* data,
    uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        message[offset++] = data[i];
    }
}

__device__ __forceinline__ void d_append_le16(uint8_t* message, uint32_t& offset, uint16_t value)
{
    d_append_byte(message, offset, static_cast<uint8_t>(value & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 8U) & 0xffU));
}

__device__ __forceinline__ void d_append_le32(uint8_t* message, uint32_t& offset, uint32_t value)
{
    d_append_byte(message, offset, static_cast<uint8_t>(value & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 8U) & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 16U) & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 24U) & 0xffU));
}

__device__ __forceinline__ void d_append_le64(uint8_t* message, uint32_t& offset, uint64_t value)
{
    for (uint32_t i = 0; i < 8; ++i) {
        d_append_byte(message, offset, static_cast<uint8_t>((value >> (i * 8U)) & 0xffU));
    }
}

__device__ uint32_t d_build_matmul_seed_message(
    const HipJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    uint8_t message[128])
{
    uint32_t offset = 0;
    static const char kTagV2[] = "BTX_MATMUL_SEED_V2";
    static const char kTagV3[] = "BTX_MATMUL_SEED_V3";
    const bool use_v3 = job.block_height >= 130500U;
    const char* tag = use_v3 ? kTagV3 : kTagV2;
    d_append_byte(message, offset, 18U);
    d_append_bytes(message, offset, reinterpret_cast<const uint8_t*>(tag), 18U);
    d_append_bytes(message, offset, job.prev_hash, 32U);
    if (use_v3) {
        d_append_le64(message, offset, static_cast<uint64_t>(job.parent_mtp));
    }
    d_append_le32(message, offset, job.block_height);
    d_append_le32(message, offset, static_cast<uint32_t>(job.version));
    d_append_bytes(message, offset, job.merkle_root, 32U);
    d_append_le32(message, offset, job.time);
    d_append_le32(message, offset, job.bits);
    d_append_le64(message, offset, nonce64);
    d_append_le16(message, offset, static_cast<uint16_t>(job.n));
    d_append_byte(message, offset, which);
    return offset;
}

__device__ uint32_t d_build_matmul_header_message(
    const HipJobParams& job,
    uint64_t nonce64,
    const uint8_t* seed_a,
    const uint8_t* seed_b,
    uint8_t message[150])
{
    uint32_t offset = 0;
    d_append_le32(message, offset, static_cast<uint32_t>(job.version));
    d_append_bytes(message, offset, job.prev_hash, 32U);
    d_append_bytes(message, offset, job.merkle_root, 32U);
    d_append_le32(message, offset, job.time);
    d_append_le32(message, offset, job.bits);
    d_append_le64(message, offset, nonce64);
    d_append_le16(message, offset, static_cast<uint16_t>(job.n));
    d_append_bytes(message, offset, seed_a, 32U);
    d_append_bytes(message, offset, seed_b, 32U);
    return offset;
}

__device__ void d_matmul_seed_fast(
    const uint32_t seed_midstate[8],
    const HipJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    uint8_t out_internal[32])
{
    uint8_t message[128];
    const uint32_t len = d_build_matmul_seed_message(job, nonce64, which, message);
    d_sha256_bytes_from_midstate(seed_midstate, message, len, out_internal);
}

__device__ void d_derive_sigma_fast(
    const uint32_t header_midstate[8],
    const HipJobParams& job,
    uint64_t nonce64,
    const uint8_t* seed_a,
    const uint8_t* seed_b,
    uint8_t* sigma_internal)
{
    uint8_t message[150];
    const uint32_t len = d_build_matmul_header_message(job, nonce64, seed_a, seed_b, message);
    uint8_t header_hash[32];
    d_sha256_bytes_from_midstate(header_midstate, message, len, header_hash);
    d_sha256_bytes(header_hash, 32U, sigma_internal);
}

__device__ void d_init_scan_midstates(
    const HipJobParams& job,
    uint32_t seed_midstate[8],
    uint32_t header_midstate[8])
{
    uint8_t prefix[150];
    d_build_matmul_seed_message(job, 0U, 0U, prefix);
    d_sha256_block0_midstate(prefix, seed_midstate);
    d_build_matmul_header_message(job, 0U, job.prev_hash, job.prev_hash, prefix);
    d_sha256_block0_midstate(prefix, header_midstate);
}

__device__ __forceinline__ uint32_t d_add(uint32_t a, uint32_t b)
{
    uint32_t s = a + b;
    return (s >= kFieldModulus) ? (s - kFieldModulus) : s;
}

__device__ __forceinline__ uint32_t d_mul(uint32_t a, uint32_t b)
{
    uint64_t p = static_cast<uint64_t>(a) * b;
    uint64_t fold = (p & kFieldModulus) + (p >> 31);
    uint32_t r = static_cast<uint32_t>((fold & kFieldModulus) + (fold >> 31));
    if (r >= kFieldModulus) {
        r -= kFieldModulus;
    }
    return r;
}

constexpr uint32_t kReduceInterval = 4;

__device__ __forceinline__ uint64_t d_reduce64(uint64_t acc)
{
    uint64_t fold = (acc & kFieldModulus) + (acc >> 31);
    return (fold & kFieldModulus) + (fold >> 31);
}

__host__ __device__ __forceinline__ bool d_use_factored_path(uint32_t n, uint32_t bsz)
{
    if ((n % 32U) != 0U) {
        return false;
    }
    const uint32_t blocks_per_axis = n / bsz;
    return (blocks_per_axis % 2U) == 0U;
}

__device__ __forceinline__ uint32_t d_dot(const uint32_t* a, const uint32_t* b, uint32_t len)
{
    uint64_t acc = 0;
    uint32_t pending = 0;
    for (uint32_t i = 0; i < len; ++i) {
        acc += static_cast<uint64_t>(a[i]) * b[i];
        if (++pending == 4) {
            uint64_t fold = (acc & kFieldModulus) + (acc >> 31);
            acc = (fold & kFieldModulus) + (fold >> 31);
            pending = 0;
        }
    }
    uint64_t fold = (acc & kFieldModulus) + (acc >> 31);
    uint32_t r = static_cast<uint32_t>((fold & kFieldModulus) + (fold >> 31));
    if (r >= kFieldModulus) {
        r -= kFieldModulus;
    }
    return r;
}

__device__ uint32_t d_from_oracle(const uint8_t* seed_internal, uint32_t index)
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);

    for (uint32_t retry = 0; retry < 256; ++retry) {
        Sha256State st;
        d_sha256_init(&st);
        d_sha256_update(&st, seed_bytes, 32);

        uint8_t idx_le[4] = {
            static_cast<uint8_t>(index & 0xff),
            static_cast<uint8_t>((index >> 8) & 0xff),
            static_cast<uint8_t>((index >> 16) & 0xff),
            static_cast<uint8_t>((index >> 24) & 0xff),
        };
        d_sha256_update(&st, idx_le, 4);

        if (retry > 0) {
            uint8_t r_le[4] = {
                static_cast<uint8_t>(retry & 0xff),
                static_cast<uint8_t>((retry >> 8) & 0xff),
                static_cast<uint8_t>((retry >> 16) & 0xff),
                static_cast<uint8_t>((retry >> 24) & 0xff),
            };
            d_sha256_update(&st, r_le, 4);
        }

        uint8_t hash[32];
        d_sha256_final(&st, hash);

        const uint32_t candidate = static_cast<uint32_t>(hash[0]) |
                                   (static_cast<uint32_t>(hash[1]) << 8) |
                                   (static_cast<uint32_t>(hash[2]) << 16) |
                                   (static_cast<uint32_t>(hash[3]) << 24);
        const uint32_t masked = candidate & kFieldModulus;
        if (masked < kFieldModulus) {
            return masked;
        }
    }

    Sha256State fb;
    d_sha256_init(&fb);
    d_sha256_update(&fb, seed_bytes, 32);
    uint8_t idx_le[4] = {
        static_cast<uint8_t>(index & 0xff),
        static_cast<uint8_t>((index >> 8) & 0xff),
        static_cast<uint8_t>((index >> 16) & 0xff),
        static_cast<uint8_t>((index >> 24) & 0xff),
    };
    d_sha256_update(&fb, idx_le, 4);
    const char* tag = "oracle-fallback";
    d_sha256_update(&fb, reinterpret_cast<const uint8_t*>(tag), 15);
    uint8_t fbh[32];
    d_sha256_final(&fb, fbh);
    const uint32_t v = static_cast<uint32_t>(fbh[0]) | (static_cast<uint32_t>(fbh[1]) << 8) |
                       (static_cast<uint32_t>(fbh[2]) << 16) | (static_cast<uint32_t>(fbh[3]) << 24);
    return v % kFieldModulus;
}

__device__ void d_derive_noise_seed(const char* domain_tag, const uint8_t* sigma_internal, uint8_t* out_internal)
{
    uint8_t sigma_canonical[32];
    d_to_canonical(sigma_internal, sigma_canonical);

    Sha256State st;
    d_sha256_init(&st);
    d_sha256_update(&st, reinterpret_cast<const uint8_t*>(domain_tag), 18);
    d_sha256_update(&st, sigma_canonical, 32);
    uint8_t digest[32];
    d_sha256_final(&st, digest);

    for (int i = 0; i < 32; ++i) {
        out_internal[i] = digest[31 - i];
    }
}

__device__ void d_write_compact_size(Sha256State& hasher, uint64_t val)
{
    if (val < 253) {
        const uint8_t c = static_cast<uint8_t>(val);
        d_sha256_update(&hasher, &c, 1);
    } else if (val < 0x10000) {
        const uint8_t buf[3] = {
            0xFD,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
        };
        d_sha256_update(&hasher, buf, 3);
    } else if (val < 0x100000000ULL) {
        const uint8_t buf[5] = {
            0xFE,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
            static_cast<uint8_t>(val >> 16),
            static_cast<uint8_t>(val >> 24),
        };
        d_sha256_update(&hasher, buf, 5);
    } else {
        const uint8_t buf[9] = {
            0xFF,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
            static_cast<uint8_t>(val >> 16),
            static_cast<uint8_t>(val >> 24),
            static_cast<uint8_t>(val >> 32),
            static_cast<uint8_t>(val >> 40),
            static_cast<uint8_t>(val >> 48),
            static_cast<uint8_t>(val >> 56),
        };
        d_sha256_update(&hasher, buf, 9);
    }
}

__device__ void d_matmul_seed_v2(
    const HipJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    uint8_t out_internal[32])
{
    Sha256State hasher;
    d_sha256_init(&hasher);
    static const char kTag[] = "BTX_MATMUL_SEED_V2";
    d_write_compact_size(hasher, sizeof(kTag) - 1);
    d_sha256_update(&hasher, reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
    d_sha256_update(&hasher, job.prev_hash, 32);

    uint8_t le[8];
    le[0] = static_cast<uint8_t>(job.block_height & 0xff);
    le[1] = static_cast<uint8_t>((job.block_height >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.block_height >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.block_height >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    le[0] = static_cast<uint8_t>(job.version & 0xff);
    le[1] = static_cast<uint8_t>((job.version >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.version >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.version >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    d_sha256_update(&hasher, job.merkle_root, 32);
    le[0] = static_cast<uint8_t>(job.time & 0xff);
    le[1] = static_cast<uint8_t>((job.time >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.time >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.time >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    le[0] = static_cast<uint8_t>(job.bits & 0xff);
    le[1] = static_cast<uint8_t>((job.bits >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.bits >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.bits >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    for (int i = 0; i < 8; ++i) {
        le[i] = static_cast<uint8_t>((nonce64 >> (i * 8)) & 0xff);
    }
    d_sha256_update(&hasher, le, 8);
    const uint16_t dim = static_cast<uint16_t>(job.n);
    le[0] = static_cast<uint8_t>(dim & 0xff);
    le[1] = static_cast<uint8_t>((dim >> 8) & 0xff);
    d_sha256_update(&hasher, le, 2);
    d_sha256_update(&hasher, &which, 1);
    d_sha256_final(&hasher, out_internal);
}

__device__ void d_derive_sigma(
    const HipJobParams& job,
    uint64_t nonce64,
    const uint8_t* seed_a,
    const uint8_t* seed_b,
    uint8_t* sigma_internal)
{
    Sha256State h;
    d_sha256_init(&h);

    uint8_t ver[4] = {
        static_cast<uint8_t>(job.version & 0xff),
        static_cast<uint8_t>((job.version >> 8) & 0xff),
        static_cast<uint8_t>((job.version >> 16) & 0xff),
        static_cast<uint8_t>((job.version >> 24) & 0xff),
    };
    uint8_t t[4] = {
        static_cast<uint8_t>(job.time & 0xff),
        static_cast<uint8_t>((job.time >> 8) & 0xff),
        static_cast<uint8_t>((job.time >> 16) & 0xff),
        static_cast<uint8_t>((job.time >> 24) & 0xff),
    };
    uint8_t bi[4] = {
        static_cast<uint8_t>(job.bits & 0xff),
        static_cast<uint8_t>((job.bits >> 8) & 0xff),
        static_cast<uint8_t>((job.bits >> 16) & 0xff),
        static_cast<uint8_t>((job.bits >> 24) & 0xff),
    };
    uint8_t n64[8];
    for (int i = 0; i < 8; ++i) {
        n64[i] = static_cast<uint8_t>((nonce64 >> (i * 8)) & 0xff);
    }
    const uint16_t dim = static_cast<uint16_t>(job.n);
    uint8_t dm[2] = {static_cast<uint8_t>(dim & 0xff), static_cast<uint8_t>((dim >> 8) & 0xff)};

    d_sha256_update(&h, ver, 4);
    d_sha256_update(&h, job.prev_hash, 32);
    d_sha256_update(&h, job.merkle_root, 32);
    d_sha256_update(&h, t, 4);
    d_sha256_update(&h, bi, 4);
    d_sha256_update(&h, n64, 8);
    d_sha256_update(&h, dm, 2);
    d_sha256_update(&h, seed_a, 32);
    d_sha256_update(&h, seed_b, 32);

    uint8_t first[32];
    d_sha256_final(&h, first);

    Sha256State h2;
    d_sha256_init(&h2);
    d_sha256_update(&h2, first, 32);
    d_sha256_final(&h2, sigma_internal);
}

__device__ bool d_sigma_below_prehash(const uint8_t sigma[32], const uint8_t target_arith[32])
{
    for (int i = 0; i < 32; ++i) {
        const uint8_t s = sigma[i];
        const uint8_t t = target_arith[31 - i];
        if (s < t) return true;
        if (s > t) return false;
    }
    return true;
}

__device__ void d_fill_rect(
    uint32_t* out,
    uint32_t rows,
    uint32_t cols,
    const uint8_t* seed_internal,
    const uint32_t* seed_midstate)
{
    const uint32_t total = rows * cols;
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        out[idx] = seed_midstate
            ? d_from_oracle_midstate(seed_midstate, seed_internal, idx)
            : d_from_oracle_fast(seed_internal, idx);
    }
}

__device__ void d_zero_u32(uint32_t* out, uint32_t count)
{
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        out[i] = 0;
    }
}

__device__ void d_mat_get(const uint32_t* m, uint32_t n, uint32_t row, uint32_t col, uint32_t* out, uint32_t bsz)
{
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t r = idx / bsz;
        const uint32_t c = idx % bsz;
        out[idx] = m[(row + r) * n + (col + c)];
    }
}

__device__ void d_block_lowrank(
    uint32_t* out,
    const uint32_t* L,
    const uint32_t* R,
    uint32_t l_rows,
    uint32_t l_cols,
    uint32_t r_cols,
    uint32_t row0,
    uint32_t col0,
    uint32_t bsz)
{
    (void)l_rows;
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t i = idx / bsz;
        const uint32_t j = idx % bsz;
        uint32_t acc = 0;
        for (uint32_t k = 0; k < l_cols; ++k) {
            const uint32_t a = L[(row0 + i) * l_cols + k];
            const uint32_t b = R[k * r_cols + (col0 + j)];
            acc = d_add(acc, d_mul(a, b));
        }
        out[idx] = acc;
    }
}

__device__ void d_block_matmul(uint32_t* out, const uint32_t* A, const uint32_t* B, uint32_t bsz)
{
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t i = idx / bsz;
        const uint32_t j = idx % bsz;
        uint32_t acc = 0;
        for (uint32_t k = 0; k < bsz; ++k) {
            acc = d_add(acc, d_mul(A[i * bsz + k], B[k * bsz + j]));
        }
        out[idx] = acc;
    }
}

__device__ void d_add_block(uint32_t* dst, const uint32_t* src, uint32_t bsz)
{
    const uint32_t n = bsz * bsz;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = d_add(dst[i], src[i]);
    }
}

__device__ void d_write_c_block(uint32_t* C, uint32_t job_n, uint32_t row0, uint32_t col0,
                                const uint32_t* cblk, uint32_t bsz)
{
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t x = idx / bsz;
        const uint32_t y = idx % bsz;
        C[(row0 + x) * job_n + (col0 + y)] = cblk[idx];
    }
}

__device__ void d_canonicalize_sha256(const uint8_t raw[32], uint8_t out[32])
{
    for (int i = 0; i < 32; ++i) {
        out[i] = raw[31 - i];
    }
}

// Match btx Uint256LE / pool share_validator: data[31] is most significant.
__device__ bool d_uint256_le(const uint8_t digest[32], const uint8_t target[32])
{
    for (int i = 31; i >= 0; --i) {
        if (digest[i] < target[i]) {
            return true;
        }
        if (digest[i] > target[i]) {
            return false;
        }
    }
    return true;
}

__device__ void d_hash_matrix_words(const uint32_t* words, uint32_t count, uint8_t out[32])
{
    Sha256State hasher;
    d_sha256_init(&hasher);
    if (count > 0) {
        d_sha256_update(&hasher, reinterpret_cast<const uint8_t*>(words),
                        static_cast<size_t>(count) * sizeof(uint32_t));
    }
    uint8_t inner[32];
    d_sha256_final(&hasher, inner);

    Sha256State outer;
    d_sha256_init(&outer);
    d_sha256_update(&outer, inner, 32);
    d_sha256_final(&outer, out);
}

__device__ void d_finalize_product_digest(
    const uint8_t* sigma_data,
    const uint8_t* c_prime_data,
    uint32_t n,
    uint32_t bsz,
    uint8_t digest_out[32])
{
    Sha256State outer;
    d_sha256_init(&outer);
    static const char kTag[] = "matmul-product-digest-v3";
    d_sha256_update(&outer, reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
    d_sha256_update(&outer, sigma_data, 32);
    d_sha256_update(&outer, c_prime_data, 32);
    uint8_t dim_le[4] = {
        static_cast<uint8_t>(n & 0xff),
        static_cast<uint8_t>((n >> 8) & 0xff),
        static_cast<uint8_t>((n >> 16) & 0xff),
        static_cast<uint8_t>((n >> 24) & 0xff),
    };
    uint8_t b_le[4] = {
        static_cast<uint8_t>(bsz & 0xff),
        static_cast<uint8_t>((bsz >> 8) & 0xff),
        static_cast<uint8_t>((bsz >> 16) & 0xff),
        static_cast<uint8_t>((bsz >> 24) & 0xff),
    };
    d_sha256_update(&outer, dim_le, 4);
    d_sha256_update(&outer, b_le, 4);

    uint8_t inner[32];
    d_sha256_final(&outer, inner);

    Sha256State outer2;
    d_sha256_init(&outer2);
    d_sha256_update(&outer2, inner, 32);
    d_sha256_final(&outer2, digest_out);
}

__device__ void d_build_compress_vec(const uint8_t* sigma_internal, uint32_t bsz, uint32_t* compress_v)
{
    __shared__ uint8_t s_seed_internal[32];
    __shared__ uint32_t s_seed_midstate[16];

    if (threadIdx.x == 0) {
        uint8_t sigma_canonical[32];
        d_to_canonical(sigma_internal, sigma_canonical);

        uint8_t message[64];
        uint32_t offset = 0;
        static const char kTag[] = "matmul-compress-v1";
        d_append_bytes(message, offset, reinterpret_cast<const uint8_t*>(kTag), 18U);
        d_append_bytes(message, offset, sigma_canonical, 32U);
        uint8_t seedb[32];
        d_sha256_bytes(message, offset, seedb);

        for (int i = 0; i < 32; ++i) {
            s_seed_internal[i] = seedb[31 - i];
        }
        d_compute_seed_midstate(s_seed_internal, s_seed_midstate);
    }
    __syncthreads();

    const uint32_t len = bsz * bsz;
    for (uint32_t k = threadIdx.x; k < len; k += blockDim.x) {
        compress_v[k] = d_from_oracle_midstate(s_seed_midstate, s_seed_internal, k);
    }
}

__device__ void d_perturb_matrix(
    uint32_t* matrix,
    const uint32_t* base,
    const uint32_t* noise_left,
    const uint32_t* noise_right,
    uint32_t n,
    uint32_t r)
{
    const uint32_t total = n * n;
    for (uint32_t gid = threadIdx.x; gid < total; gid += blockDim.x) {
        const uint32_t row = gid / n;
        const uint32_t col = gid % n;
        uint64_t acc = 0;
        uint32_t pending = 0;
        for (uint32_t k = 0; k < r; ++k) {
            acc += static_cast<uint64_t>(noise_left[row * r + k]) * noise_right[k * n + col];
            if (++pending == kReduceInterval) {
                acc = d_reduce64(acc);
                pending = 0;
            }
        }
        matrix[gid] = d_add(base[gid], static_cast<uint32_t>(d_reduce64(acc)));
    }
}

__device__ void d_build_factored_rhs(
    const uint32_t* matrix_b,
    const uint32_t* compress_v,
    uint32_t* rhs,
    uint32_t n,
    uint32_t bsz,
    uint32_t blocks_per_axis)
{
    const uint32_t rhs_elems = bsz * n * blocks_per_axis;
    for (uint32_t gid = threadIdx.x; gid < rhs_elems; gid += blockDim.x) {
        const uint32_t m = gid % n;
        const uint32_t jx = gid / n;
        const uint32_t x = jx % bsz;
        const uint32_t j = jx / bsz;
        const uint32_t* b_row = matrix_b + static_cast<size_t>(m) * n + j * bsz;
        const uint32_t* w_row = compress_v + x * bsz;
        uint64_t acc = 0;
        uint32_t pending = 0;
        for (uint32_t y = 0; y < bsz; ++y) {
            acc += static_cast<uint64_t>(w_row[y]) * b_row[y];
            if (++pending == kReduceInterval) {
                acc = d_reduce64(acc);
                pending = 0;
            }
        }
        rhs[gid] = static_cast<uint32_t>(d_reduce64(acc));
    }
}

__device__ void d_compute_factored_words(
    const uint32_t* matrix_a,
    const uint32_t* rhs,
    uint32_t* output,
    uint32_t n,
    uint32_t bsz,
    uint32_t blocks_per_axis)
{
    const uint32_t tiles_per_axis = blocks_per_axis >> 1U;
    const uint32_t total_tiles = tiles_per_axis * tiles_per_axis;
    const uint32_t warps_per_block = blockDim.x >> 5U;
    __shared__ uint64_t s_lane_partials[8][32][4];

    for (uint32_t tile_base = 0; tile_base < total_tiles; tile_base += warps_per_block) {
        const uint32_t lane = threadIdx.x & 31U;
        const uint32_t warp_id = threadIdx.x >> 5U;
        const uint32_t tile_index = tile_base + warp_id;
        const bool active = tile_index < total_tiles;

        uint64_t acc00 = 0;
        uint64_t acc01 = 0;
        uint64_t acc10 = 0;
        uint64_t acc11 = 0;
        if (active) {
            const uint32_t j0 = (tile_index % tiles_per_axis) * 2U;
            const uint32_t i0 = (tile_index / tiles_per_axis) * 2U;
            uint32_t pending = 0;
            for (uint32_t x = 0; x < bsz; ++x) {
                const uint32_t* a_row0 = matrix_a + static_cast<size_t>(i0 * bsz + x) * n;
                const uint32_t* a_row1 = a_row0 + static_cast<size_t>(bsz) * n;
                const uint32_t* d_row0 = rhs + static_cast<size_t>(j0 * bsz + x) * n;
                const uint32_t* d_row1 = d_row0 + static_cast<size_t>(bsz) * n;
                for (uint32_t m = lane; m < n; m += 32U) {
                    const uint64_t a0 = a_row0[m];
                    const uint64_t a1 = a_row1[m];
                    const uint64_t d0 = d_row0[m];
                    const uint64_t d1 = d_row1[m];
                    acc00 += a0 * d0;
                    acc01 += a0 * d1;
                    acc10 += a1 * d0;
                    acc11 += a1 * d1;
                    if (++pending == kReduceInterval) {
                        acc00 = d_reduce64(acc00);
                        acc01 = d_reduce64(acc01);
                        acc10 = d_reduce64(acc10);
                        acc11 = d_reduce64(acc11);
                        pending = 0;
                    }
                }
            }
            s_lane_partials[warp_id][lane][0] = d_reduce64(acc00);
            s_lane_partials[warp_id][lane][1] = d_reduce64(acc01);
            s_lane_partials[warp_id][lane][2] = d_reduce64(acc10);
            s_lane_partials[warp_id][lane][3] = d_reduce64(acc11);
        }
        __syncthreads();

        if (lane == 0 && active) {
            const uint32_t j0 = (tile_index % tiles_per_axis) * 2U;
            const uint32_t i0 = (tile_index / tiles_per_axis) * 2U;
            uint64_t sum00 = 0;
            uint64_t sum01 = 0;
            uint64_t sum10 = 0;
            uint64_t sum11 = 0;
            for (uint32_t i = 0; i < 32U; ++i) {
                sum00 += s_lane_partials[warp_id][i][0];
                sum01 += s_lane_partials[warp_id][i][1];
                sum10 += s_lane_partials[warp_id][i][2];
                sum11 += s_lane_partials[warp_id][i][3];
            }
            output[i0 * blocks_per_axis + j0] = static_cast<uint32_t>(d_reduce64(sum00));
            output[i0 * blocks_per_axis + j0 + 1U] = static_cast<uint32_t>(d_reduce64(sum01));
            output[(i0 + 1U) * blocks_per_axis + j0] = static_cast<uint32_t>(d_reduce64(sum10));
            output[(i0 + 1U) * blocks_per_axis + j0 + 1U] = static_cast<uint32_t>(d_reduce64(sum11));
        }
        __syncthreads();
    }
}

__device__ void d_copy_matrix(uint32_t* dst, const uint32_t* src, uint32_t count)
{
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__device__ bool d_solve_nonce(
    const HipJobParams& job,
    uint32_t* A,
    uint32_t* B,
    const uint32_t* A_base,
    const uint32_t* B_base,
    uint64_t nonce64,
    uint32_t* C,
    uint32_t* E_L,
    uint32_t* E_R,
    uint32_t* F_L,
    uint32_t* F_R,
    uint32_t* compress_v,
    uint32_t* factored_rhs,
    uint32_t* ablk,
    uint32_t* bblk,
    uint32_t* noise_blk,
    uint32_t* prod,
    uint32_t* cblk,
    const uint8_t* reuse_seed_a,
    const uint8_t* reuse_seed_b,
    uint8_t* digest_out)
{
    __shared__ uint8_t s_sigma[32];
    __shared__ uint8_t s_seed_el[32];
    __shared__ uint8_t s_seed_er[32];
    __shared__ uint8_t s_seed_fl[32];
    __shared__ uint8_t s_seed_fr[32];
    __shared__ bool s_hit;
    __shared__ bool s_sigma_pass;
    __shared__ uint8_t s_seed_a[32];
    __shared__ uint8_t s_seed_b[32];
    __shared__ uint32_t s_scan_seed_midstate[8];
    __shared__ uint32_t s_scan_header_midstate[8];
    __shared__ uint32_t s_mat_a_midstate[16];
    __shared__ uint32_t s_mat_b_midstate[16];
    __shared__ uint32_t s_noise_midstate[16];

    const bool use_v2 = job.block_height >= 125000;
    const bool use_factored = d_use_factored_path(job.n, job.b);

    if (threadIdx.x == 0 && use_v2) {
        d_init_scan_midstates(job, s_scan_seed_midstate, s_scan_header_midstate);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (reuse_seed_a && reuse_seed_b) {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = reuse_seed_a[i];
                s_seed_b[i] = reuse_seed_b[i];
            }
        } else if (use_v2) {
            d_matmul_seed_fast(s_scan_seed_midstate, job, nonce64, 0, s_seed_a);
            d_matmul_seed_fast(s_scan_seed_midstate, job, nonce64, 1, s_seed_b);
        } else {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = job.seed_a[i];
                s_seed_b[i] = job.seed_b[i];
            }
        }
        if (use_v2) {
            d_derive_sigma_fast(
                s_scan_header_midstate, job, nonce64, s_seed_a, s_seed_b, s_sigma);
        } else {
            d_derive_sigma(job, nonce64, s_seed_a, s_seed_b, s_sigma);
        }
        s_sigma_pass = job.epsilon_bits == 0 ||
                       d_sigma_below_prehash(s_sigma, job.pre_hash_target);
    }
    __syncthreads();
    if (!s_sigma_pass) {
        return false;
    }

    if (use_v2 && A && B) {
        if (threadIdx.x == 0) {
            d_compute_seed_midstate(s_seed_a, s_mat_a_midstate);
            d_compute_seed_midstate(s_seed_b, s_mat_b_midstate);
        }
        __syncthreads();
        d_fill_rect(A, job.n, job.n, s_seed_a, s_mat_a_midstate);
        __syncthreads();
        d_fill_rect(B, job.n, job.n, s_seed_b, s_mat_b_midstate);
        __syncthreads();
    } else if (use_factored && A && B && A_base && B_base) {
        d_copy_matrix(A, A_base, job.n * job.n);
        __syncthreads();
        d_copy_matrix(B, B_base, job.n * job.n);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_derive_noise_seed("matmul_noise_EL_v1", s_sigma, s_seed_el);
        d_derive_noise_seed("matmul_noise_ER_v1", s_sigma, s_seed_er);
        d_derive_noise_seed("matmul_noise_FL_v1", s_sigma, s_seed_fl);
        d_derive_noise_seed("matmul_noise_FR_v1", s_sigma, s_seed_fr);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_el, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(E_L, job.n, job.r, s_seed_el, s_noise_midstate);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_er, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(E_R, job.r, job.n, s_seed_er, s_noise_midstate);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_fl, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(F_L, job.n, job.r, s_seed_fl, s_noise_midstate);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_fr, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(F_R, job.r, job.n, s_seed_fr, s_noise_midstate);
    __syncthreads();

    const uint32_t bsz = job.b;
    const uint32_t N = job.n / bsz;
    const uint32_t word_count = N * N;

    d_zero_u32(C, word_count);
    __syncthreads();

    d_build_compress_vec(s_sigma, job.b, compress_v);
    __syncthreads();

    if (use_factored && A && B && factored_rhs) {
        d_perturb_matrix(A, A, E_L, E_R, job.n, job.r);
        __syncthreads();
        d_perturb_matrix(B, B, F_L, F_R, job.n, job.r);
        __syncthreads();
        d_build_factored_rhs(B, compress_v, factored_rhs, job.n, bsz, N);
        __syncthreads();
        d_compute_factored_words(A, factored_rhs, C, job.n, bsz, N);
        __syncthreads();
    } else {
        const uint32_t total_steps = N * N * N;

        // Product-committed digest: sum compress(A'[i,ell]*B'[ell,j]) over ell per (i,j).
        for (uint32_t step = 0; step < total_steps; ++step) {
            const uint32_t bi = step / (N * N);
            const uint32_t rem = step % (N * N);
            const uint32_t bj = rem / N;
            const uint32_t ell = rem % N;
            const uint32_t row0 = bi * bsz;
            const uint32_t col0 = bj * bsz;
            const uint32_t mid0 = ell * bsz;

            d_mat_get(A, job.n, row0, mid0, ablk, bsz);
            __syncthreads();
            d_block_lowrank(noise_blk, E_L, E_R, job.n, job.r, job.n, row0, mid0, bsz);
            __syncthreads();
            d_add_block(ablk, noise_blk, bsz);
            __syncthreads();

            d_mat_get(B, job.n, mid0, col0, bblk, bsz);
            __syncthreads();
            d_block_lowrank(noise_blk, F_L, F_R, job.n, job.r, job.n, mid0, col0, bsz);
            __syncthreads();
            d_add_block(bblk, noise_blk, bsz);
            __syncthreads();

            d_block_matmul(prod, ablk, bblk, bsz);
            __syncthreads();

            if (threadIdx.x == 0) {
                const uint32_t word_idx = bi * N + bj;
                const uint32_t term = d_dot(prod, compress_v, bsz * bsz);
                C[word_idx] = d_add(C[word_idx], term);
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        uint8_t c_prime[32];
        d_hash_matrix_words(C, word_count, c_prime);
        d_finalize_product_digest(s_sigma, c_prime, job.n, bsz, digest_out);
        s_hit = d_uint256_le(digest_out, job.target);
    }
    __syncthreads();
    return s_hit;
}

__global__ void matmul_nonce_kernel(
    HipJobParams params,
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ B,
    const uint64_t* __restrict__ nonces,
    uint32_t* __restrict__ workspaces,
    uint8_t* __restrict__ out_digests,
    uint8_t* __restrict__ out_found)
{
    const size_t idx = static_cast<size_t>(blockIdx.x);
    if (idx >= gridDim.x) {
        return;
    }

    const uint32_t n = params.n;
    const uint32_t r = params.r;
    const uint32_t bsz = params.b;
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t noise_elems = 2 * (static_cast<size_t>(n) * r + static_cast<size_t>(r) * n);
    const size_t scratch_elems = static_cast<size_t>(bsz) * bsz;
    const size_t compress_elems = scratch_elems;
    const bool use_v2 = params.block_height >= 125000;
    const bool use_factored = d_use_factored_path(n, bsz);
    const size_t blocks_per_axis = static_cast<size_t>(n / bsz);
    const size_t matrix_elems = use_v2 ? (2 * nn) : (use_factored ? (2 * nn) : 0);
    const size_t factored_rhs_elems = use_factored ? (static_cast<size_t>(bsz) * n * blocks_per_axis) : 0;
    const size_t legacy_scratch_elems = use_factored ? 0 : (scratch_elems * 5);
    const size_t per_nonce_elems =
        matrix_elems + nn + noise_elems + compress_elems + factored_rhs_elems + legacy_scratch_elems;

    // Per-nonce workspace layout:
    // [A nn][B nn] (V2 or factored non-V2) [C nn][noise...][compress][factored_rhs?][legacy scratch?]
    size_t off = idx * per_nonce_elems;

    uint32_t* A_local = (use_v2 || use_factored) ? (workspaces + off) : nullptr;
    if (use_v2 || use_factored) off += nn;
    uint32_t* B_local = (use_v2 || use_factored) ? (workspaces + off) : nullptr;
    if (use_v2 || use_factored) off += nn;

    uint32_t* C = workspaces + off;
    off += nn;
    uint32_t* E_L = workspaces + off;
    off += static_cast<size_t>(n) * r;
    uint32_t* E_R = workspaces + off;
    off += static_cast<size_t>(r) * n;
    uint32_t* F_L = workspaces + off;
    off += static_cast<size_t>(n) * r;
    uint32_t* F_R = workspaces + off;
    off += static_cast<size_t>(r) * n;
    uint32_t* compress_v = workspaces + off;
    off += compress_elems;
    uint32_t* factored_rhs = use_factored ? (workspaces + off) : nullptr;
    if (use_factored) off += factored_rhs_elems;
    uint32_t* ablk = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* bblk = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* noise_blk = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* prod = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* cblk = use_factored ? nullptr : (workspaces + off);

    __shared__ uint8_t s_v2_seed_a[32];
    __shared__ uint8_t s_v2_seed_b[32];
    __shared__ uint32_t s_scan_seed_midstate[8];
    __shared__ uint32_t s_scan_header_midstate[8];

    if (use_v2 && threadIdx.x == 0) {
        d_init_scan_midstates(params, s_scan_seed_midstate, s_scan_header_midstate);
        d_matmul_seed_fast(s_scan_seed_midstate, params, nonces[idx], 0, s_v2_seed_a);
        d_matmul_seed_fast(s_scan_seed_midstate, params, nonces[idx], 1, s_v2_seed_b);
    }
    __syncthreads();

    uint32_t* use_a = (use_v2 || use_factored) ? A_local : const_cast<uint32_t*>(A);
    uint32_t* use_b = (use_v2 || use_factored) ? B_local : const_cast<uint32_t*>(B);

    uint8_t digest[32];
    const uint8_t* reuse_a = use_v2 ? s_v2_seed_a : nullptr;
    const uint8_t* reuse_b = use_v2 ? s_v2_seed_b : nullptr;
    const bool hit = d_solve_nonce(
        params, use_a, use_b, A, B, nonces[idx], C, E_L, E_R, F_L, F_R, compress_v,
        factored_rhs, ablk, bblk, noise_blk, prod, cblk, reuse_a, reuse_b, digest);

    if (threadIdx.x == 0) {
        if (hit) {
            out_found[idx] = 1;
            for (int i = 0; i < 32; ++i) {
                out_digests[idx * 32 + i] = digest[i];
            }
        }
    }
}

__global__ void init_scan_midstates_kernel(
    HipJobParams params,
    uint32_t* seed_midstate,
    uint32_t* header_midstate)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_init_scan_midstates(params, seed_midstate, header_midstate);
    }
}

__global__ void sigma_gate_kernel(
    HipJobParams params,
    uint64_t nonce_start,
    size_t nonce_count,
    const uint32_t* __restrict__ seed_midstate,
    const uint32_t* __restrict__ header_midstate,
    uint32_t* __restrict__ out_count,
    uint64_t* __restrict__ out_nonces,
    uint8_t* __restrict__ out_sigma,
    uint8_t* __restrict__ out_seed_a,
    uint8_t* __restrict__ out_seed_b)
{
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= nonce_count) {
        return;
    }

    uint8_t seed_a[32];
    uint8_t seed_b[32];
    uint8_t sigma[32];
    const uint64_t nonce = nonce_start + idx;

    if (params.block_height >= 130500 && params.fast_v3_scan != 0U) {
        uint32_t seed_a_words[8];
        uint32_t seed_b_words[8];
        uint32_t sigma_state[8];
        d_matmul_seed_pair_v3_midstate_words(
            seed_midstate, params, nonce, seed_a_words, seed_b_words);
        d_derive_sigma_v3_midstate_words_state(
            header_midstate, params, nonce, seed_a_words, seed_b_words,
            sigma_state);

        if (params.epsilon_bits != 0 &&
            !d_sigma_state_below_prehash(sigma_state, params.pre_hash_target)) {
            return;
        }

        const uint32_t compact_index = atomicAdd(out_count, 1u);
        out_nonces[compact_index] = nonce;
        d_sha256_state_to_bytes(sigma_state, out_sigma + compact_index * 32);
        d_sha256_state_to_bytes(seed_a_words, out_seed_a + compact_index * 32);
        d_sha256_state_to_bytes(seed_b_words, out_seed_b + compact_index * 32);
        return;
    }

    if (params.block_height >= 125000) {
        d_matmul_seed_fast(seed_midstate, params, nonce, 0, seed_a);
        d_matmul_seed_fast(seed_midstate, params, nonce, 1, seed_b);
        d_derive_sigma_fast(
            header_midstate, params, nonce, seed_a, seed_b, sigma);
    } else {
        for (uint32_t i = 0; i < 32; ++i) {
            seed_a[i] = params.seed_a[i];
            seed_b[i] = params.seed_b[i];
        }
        d_derive_sigma(params, nonce, seed_a, seed_b, sigma);
    }

    if (params.epsilon_bits != 0 &&
        !d_sigma_below_prehash(sigma, params.pre_hash_target)) {
        return;
    }

    const uint32_t compact_index = atomicAdd(out_count, 1u);
    out_nonces[compact_index] = nonce;
    for (uint32_t i = 0; i < 32; ++i) {
        out_sigma[compact_index * 32 + i] = sigma[i];
        out_seed_a[compact_index * 32 + i] = seed_a[i];
        out_seed_b[compact_index * 32 + i] = seed_b[i];
    }
}

__global__ void batch_perturb_matrix_kernel(
    const uint32_t* __restrict__ base,
    const uint32_t* __restrict__ noise_left,
    const uint32_t* __restrict__ noise_right,
    uint32_t n,
    uint32_t r,
    size_t matrix_elements,
    size_t total_elements,
    uint32_t* __restrict__ output)
{
    const size_t gid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = gid / matrix_elements;
    const uint32_t local_index = static_cast<uint32_t>(gid % matrix_elements);
    const uint32_t row = local_index / n;
    const uint32_t col = local_index % n;
    const uint32_t* batch_base = base + batch_index * matrix_elements;
    const uint32_t* left = noise_left + batch_index * n * r;
    const uint32_t* right = noise_right + batch_index * r * n;

    uint64_t acc = 0;
    uint32_t pending = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        if (k < r) {
            acc += static_cast<uint64_t>(left[row * r + k]) *
                   right[k * n + col];
            if (++pending == kReduceInterval) {
                acc = d_reduce64(acc);
                pending = 0;
            }
        }
    }
    output[gid] = d_add(
        batch_base[local_index],
        static_cast<uint32_t>(d_reduce64(acc)));
}

__global__ void batch_perturb_matrix_512x8_kernel(
    const uint32_t* __restrict__ base,
    const uint32_t* __restrict__ noise_left,
    const uint32_t* __restrict__ noise_right,
    size_t total_elements,
    uint32_t* __restrict__ output)
{
    const size_t gid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    constexpr uint32_t kN = 512;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr size_t kNoiseElements = size_t{kN} * 8;
    const size_t batch_index = gid >> 18U;
    const uint32_t local_index =
        static_cast<uint32_t>(gid & (kMatrixElements - 1U));
    const uint32_t row = local_index >> 9U;
    const uint32_t col = local_index & (kN - 1U);
    const uint32_t* batch_base = base + batch_index * kMatrixElements;
    const uint32_t* left = noise_left + batch_index * kNoiseElements;
    const uint32_t* right = noise_right + batch_index * kNoiseElements;

    uint64_t acc = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        acc += static_cast<uint64_t>(left[(row << 3U) + k]) *
               right[k * kN + col];
        if ((k & 3U) == 3U) {
            acc = d_reduce64(acc);
        }
    }
    output[gid] = d_add(
        batch_base[local_index],
        static_cast<uint32_t>(d_reduce64(acc)));
}

__global__ void batch_generate_perturb_matrix_512x8_kernel(
    const uint8_t* __restrict__ seeds,
    const uint32_t* __restrict__ midstates,
    const uint32_t* __restrict__ noise_left,
    const uint32_t* __restrict__ noise_right,
    size_t total_elements,
    uint32_t* __restrict__ output)
{
    constexpr uint32_t kN = 512;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr size_t kNoiseElements = size_t{kN} * 8;
    __shared__ uint32_t s_midstate[16];

    const size_t tile_start =
        static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t gid = tile_start + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = tile_start >> 18U;
    const uint32_t local_base =
        static_cast<uint32_t>(tile_start & (kMatrixElements - 1U));
    const uint32_t local_index = local_base + threadIdx.x;
    const uint32_t row = local_index >> 9U;
    const uint32_t col = local_index & (kN - 1U);

    const uint8_t* seed = seeds + batch_index * 32U;
    if (threadIdx.x < 16U) {
        s_midstate[threadIdx.x] =
            midstates[batch_index * 16U + threadIdx.x];
    }
    const uint32_t* left = noise_left + batch_index * kNoiseElements;
    __syncthreads();

    const uint32_t* right = noise_right + batch_index * kNoiseElements;

    uint64_t acc = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        acc += static_cast<uint64_t>(left[(row << 3U) + k]) *
               right[k * kN + col];
        if ((k & 3U) == 3U) {
            acc = d_reduce64(acc);
        }
    }
    const uint32_t base =
        d_from_oracle_midstate(s_midstate, seed, local_index);
    output[gid] = d_add(base, static_cast<uint32_t>(d_reduce64(acc)));
}

// One workgroup per output word. Threads map to the 16x16 compression
// coordinates and accumulate all 32 inner blocks for that word.
__global__ void batch_compressed_words_kernel(
    const uint32_t* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const uint32_t* __restrict__ compress,
    uint32_t n,
    uint32_t bsz,
    uint32_t blocks_per_axis,
    uint32_t words_per_nonce,
    size_t matrix_elements,
    uint32_t* __restrict__ output)
{
    __shared__ uint32_t partials[256];

    const uint32_t global_word = blockIdx.x;
    const uint32_t batch_index = global_word / words_per_nonce;
    const uint32_t word_index = global_word % words_per_nonce;
    const uint32_t bi = word_index / blocks_per_axis;
    const uint32_t bj = word_index % blocks_per_axis;
    const uint32_t tid = threadIdx.x;
    const bool active = tid < bsz * bsz;
    const uint32_t x = active ? tid / bsz : 0;
    const uint32_t y = active ? tid % bsz : 0;

    const uint32_t* A = matrix_a + static_cast<size_t>(batch_index) * matrix_elements;
    const uint32_t* B = matrix_b + static_cast<size_t>(batch_index) * matrix_elements;
    const uint32_t* W = compress + static_cast<size_t>(batch_index) * bsz * bsz;

    uint32_t compressed_sum = 0;
    for (uint32_t ell = 0; ell < blocks_per_axis; ++ell) {
        uint64_t dot = 0;
        uint32_t pending = 0;
        if (active) {
            const uint32_t row = bi * bsz + x;
            const uint32_t col = bj * bsz + y;
            const uint32_t middle = ell * bsz;
            #pragma unroll
            for (uint32_t k = 0; k < 16; ++k) {
                if (k < bsz) {
                    dot += static_cast<uint64_t>(
                        A[static_cast<size_t>(row) * n + middle + k]) *
                        B[static_cast<size_t>(middle + k) * n + col];
                    if (++pending == kReduceInterval) {
                        dot = d_reduce64(dot);
                        pending = 0;
                    }
                }
            }
            partials[tid] = d_mul(
                static_cast<uint32_t>(d_reduce64(dot)), W[tid]);
        } else {
            partials[tid] = 0;
        }
        __syncthreads();

        for (uint32_t stride = 128; stride > 0; stride >>= 1U) {
            if (tid < stride) {
                partials[tid] = d_add(partials[tid], partials[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            compressed_sum = d_add(compressed_sum, partials[0]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[static_cast<size_t>(batch_index) * words_per_nonce + word_index] =
            compressed_sum;
    }
}

__global__ void batch_compressed_words_512x16_kernel(
    const uint32_t* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const uint32_t* __restrict__ compress,
    uint32_t* __restrict__ output)
{
    __shared__ uint32_t partials[256];

    constexpr uint32_t kN = 512;
    constexpr uint32_t kWordsPerNonce = 32 * 32;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    const uint32_t global_word = blockIdx.x;
    const uint32_t batch_index = global_word >> 10U;
    const uint32_t word_index = global_word & (kWordsPerNonce - 1U);
    const uint32_t bi = word_index >> 5U;
    const uint32_t bj = word_index & 31U;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = (bi << 4U) + (tid >> 4U);
    const uint32_t col = (bj << 4U) + (tid & 15U);

    const uint32_t* A =
        matrix_a + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t* B =
        matrix_b + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t weight =
        compress[(static_cast<size_t>(batch_index) << 8U) + tid];

    uint32_t compressed_sum = 0;
    for (uint32_t ell = 0; ell < 32; ++ell) {
        const uint32_t middle = ell << 4U;
        uint64_t dot = 0;
        #pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            dot += static_cast<uint64_t>(
                A[static_cast<size_t>(row) * kN + middle + k]) *
                B[static_cast<size_t>(middle + k) * kN + col];
            if ((k & 3U) == 3U) {
                dot = d_reduce64(dot);
            }
        }
        partials[tid] = d_mul(
            static_cast<uint32_t>(d_reduce64(dot)), weight);
        __syncthreads();

        for (uint32_t stride = 128; stride > 0; stride >>= 1U) {
            if (tid < stride) {
                partials[tid] = d_add(partials[tid], partials[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            compressed_sum = d_add(compressed_sum, partials[0]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[
            (static_cast<size_t>(batch_index) << 10U) + word_index] =
            compressed_sum;
    }
}

__global__ void batch_build_factored_rhs_512x16_kernel(
    const uint32_t* __restrict__ matrix_b,
    const uint32_t* __restrict__ compress,
    size_t total_elements,
    uint32_t* __restrict__ rhs)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kBlocksPerAxis = 32;
    constexpr size_t kMatrixElements = size_t{kN} * kN;

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = gid >> 18U;
    const uint32_t local = static_cast<uint32_t>(gid & (kMatrixElements - 1U));
    const uint32_t m = local & (kN - 1U);
    const uint32_t jx = local >> 9U;
    const uint32_t x = jx & (kB - 1U);
    const uint32_t j = jx >> 4U;

    const uint32_t* B = matrix_b + batch_index * kMatrixElements;
    const uint32_t* W = compress + (batch_index << 8U) + (x << 4U);
    const uint32_t* b_row = B + static_cast<size_t>(m) * kN + j * kB;

    uint64_t acc = 0;
    #pragma unroll
    for (uint32_t y = 0; y < kB; ++y) {
        acc += static_cast<uint64_t>(W[y]) * b_row[y];
        if ((y & 3U) == 3U) {
            acc = d_reduce64(acc);
        }
    }
    rhs[gid] = static_cast<uint32_t>(d_reduce64(acc));
}

__global__ void batch_build_right_noise_rhs_512x16x8_kernel(
    const uint32_t* __restrict__ noise_fr,
    const uint32_t* __restrict__ compress,
    size_t total_elements,
    uint32_t* __restrict__ right_rhs)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kR = 8;
    constexpr uint32_t kBlocksPerAxis = 32;
    constexpr size_t kNoiseElements = size_t{kR} * kN;

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = gid >> 12U;
    const uint32_t local = static_cast<uint32_t>(gid & 4095U);
    const uint32_t k = local & (kR - 1U);
    const uint32_t x = (local >> 3U) & (kB - 1U);
    const uint32_t j = local >> 7U;

    const uint32_t* FR = noise_fr + batch_index * kNoiseElements;
    const uint32_t* W = compress + (batch_index << 8U) + (x << 4U);
    const uint32_t* fr_row = FR + k * kN + j * kB;

    uint64_t acc = 0;
    #pragma unroll
    for (uint32_t y = 0; y < kB; ++y) {
        acc += static_cast<uint64_t>(W[y]) * fr_row[y];
        if ((y & 3U) == 3U) {
            acc = d_reduce64(acc);
        }
    }
    right_rhs[gid] = static_cast<uint32_t>(d_reduce64(acc));
}

__global__ void batch_build_right_noise_rhs_grouped_512x16x8_kernel(
    const uint32_t* __restrict__ noise_fr,
    const uint32_t* __restrict__ compress,
    size_t total_pairs,
    uint32_t* __restrict__ right_rhs)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kR = 8;
    constexpr uint32_t kPairsPerNonce = 32U * kB;
    constexpr size_t kNoiseElements = size_t{kR} * kN;
    constexpr size_t kRightRhsElements = 32U * kB * kR;

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_pairs) {
        return;
    }

    const size_t batch_index = gid / kPairsPerNonce;
    const uint32_t local = static_cast<uint32_t>(gid & (kPairsPerNonce - 1U));
    const uint32_t x = local & (kB - 1U);
    const uint32_t j = local >> 4U;

    const uint32_t* FR = noise_fr + batch_index * kNoiseElements;
    const uint32_t* W = compress + (batch_index << 8U) + (x << 4U);

    uint64_t acc0 = 0;
    uint64_t acc1 = 0;
    uint64_t acc2 = 0;
    uint64_t acc3 = 0;
    uint64_t acc4 = 0;
    uint64_t acc5 = 0;
    uint64_t acc6 = 0;
    uint64_t acc7 = 0;

    #pragma unroll
    for (uint32_t y = 0; y < kB; ++y) {
        const uint64_t w = W[y];
        const uint32_t col = (j << 4U) + y;
        acc0 += w * FR[col];
        acc1 += w * FR[kN + col];
        acc2 += w * FR[2U * kN + col];
        acc3 += w * FR[3U * kN + col];
        acc4 += w * FR[4U * kN + col];
        acc5 += w * FR[5U * kN + col];
        acc6 += w * FR[6U * kN + col];
        acc7 += w * FR[7U * kN + col];
        if ((y & 3U) == 3U) {
            acc0 = d_reduce64(acc0);
            acc1 = d_reduce64(acc1);
            acc2 = d_reduce64(acc2);
            acc3 = d_reduce64(acc3);
            acc4 = d_reduce64(acc4);
            acc5 = d_reduce64(acc5);
            acc6 = d_reduce64(acc6);
            acc7 = d_reduce64(acc7);
        }
    }

    uint32_t* out = right_rhs + batch_index * kRightRhsElements +
        ((j << 4U) + x) * kR;
    out[0] = static_cast<uint32_t>(d_reduce64(acc0));
    out[1] = static_cast<uint32_t>(d_reduce64(acc1));
    out[2] = static_cast<uint32_t>(d_reduce64(acc2));
    out[3] = static_cast<uint32_t>(d_reduce64(acc3));
    out[4] = static_cast<uint32_t>(d_reduce64(acc4));
    out[5] = static_cast<uint32_t>(d_reduce64(acc5));
    out[6] = static_cast<uint32_t>(d_reduce64(acc6));
    out[7] = static_cast<uint32_t>(d_reduce64(acc7));
}

__global__ void batch_build_factored_rhs_direct_512x16x8_kernel(
    const uint32_t* __restrict__ base_b,
    const uint32_t* __restrict__ noise_fl,
    const uint32_t* __restrict__ right_rhs,
    const uint32_t* __restrict__ compress,
    size_t total_elements,
    uint32_t* __restrict__ rhs)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kR = 8;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr size_t kNoiseLeftElements = size_t{kN} * kR;
    constexpr size_t kRightRhsElements = 32U * kB * kR;

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = gid >> 18U;
    const uint32_t local = static_cast<uint32_t>(gid & (kMatrixElements - 1U));
    const uint32_t m = local & (kN - 1U);
    const uint32_t jx = local >> 9U;
    const uint32_t x = jx & (kB - 1U);
    const uint32_t j = jx >> 4U;

    const uint32_t* B = base_b + batch_index * kMatrixElements;
    const uint32_t* FL = noise_fl + batch_index * kNoiseLeftElements;
    const uint32_t* RR = right_rhs + batch_index * kRightRhsElements;
    const uint32_t* W = compress + (batch_index << 8U) + (x << 4U);
    const uint32_t* b_row = B + static_cast<size_t>(m) * kN + j * kB;
    const uint32_t* fl_row = FL + static_cast<size_t>(m) * kR;
    const uint32_t* rr_row = RR + ((j << 4U) + x) * kR;

    uint64_t acc = 0;
    uint32_t pending = 0;
    #pragma unroll
    for (uint32_t y = 0; y < kB; ++y) {
        acc += static_cast<uint64_t>(W[y]) * b_row[y];
        if (++pending == kReduceInterval) {
            acc = d_reduce64(acc);
            pending = 0;
        }
    }
    #pragma unroll
    for (uint32_t k = 0; k < kR; ++k) {
        acc += static_cast<uint64_t>(fl_row[k]) * rr_row[k];
        if (++pending == kReduceInterval) {
            acc = d_reduce64(acc);
            pending = 0;
        }
    }
    rhs[gid] = static_cast<uint32_t>(d_reduce64(acc));
}

__global__ void batch_build_factored_rhs_fused_b_512x16x8_kernel(
    const uint8_t* __restrict__ seed_b,
    const uint32_t* __restrict__ seed_b_midstates,
    const uint32_t* __restrict__ noise_fl,
    const uint32_t* __restrict__ right_rhs,
    const uint32_t* __restrict__ compress,
    uint32_t* __restrict__ rhs)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kR = 8;
    constexpr uint32_t kBlocksPerAxis = 32;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr size_t kNoiseLeftElements = size_t{kN} * kR;
    constexpr size_t kRightRhsElements = kBlocksPerAxis * kB * kR;
    constexpr uint32_t kTilesPerNonce = kBlocksPerAxis * kBlocksPerAxis;

    __shared__ uint32_t s_midstate[16];
    __shared__ uint32_t s_b[kB][kB];

    const uint32_t tile = blockIdx.x;
    const uint32_t batch_index = tile / kTilesPerNonce;
    const uint32_t local_tile = tile & (kTilesPerNonce - 1U);
    const uint32_t j = local_tile >> 5U;
    const uint32_t m_tile = local_tile & (kBlocksPerAxis - 1U);
    const uint32_t tid = threadIdx.x;

    const uint8_t* seed = seed_b + static_cast<size_t>(batch_index) * 32U;
    if (tid < 16U) {
        s_midstate[tid] =
            seed_b_midstates[static_cast<size_t>(batch_index) * 16U + tid];
    }
    __syncthreads();

    const uint32_t b_local_m = tid >> 4U;
    const uint32_t y = tid & (kB - 1U);
    const uint32_t m_for_b = (m_tile << 4U) + b_local_m;
    const uint32_t b_index = m_for_b * kN + (j << 4U) + y;
    s_b[b_local_m][y] = d_from_oracle_midstate(s_midstate, seed, b_index);
    __syncthreads();

    const uint32_t x = tid >> 4U;
    const uint32_t m_local = tid & (kB - 1U);
    const uint32_t m = (m_tile << 4U) + m_local;
    const uint32_t* W = compress + (static_cast<size_t>(batch_index) << 8U) + (x << 4U);
    const uint32_t* FL = noise_fl + static_cast<size_t>(batch_index) * kNoiseLeftElements;
    const uint32_t* RR = right_rhs + static_cast<size_t>(batch_index) * kRightRhsElements;
    const uint32_t* fl_row = FL + static_cast<size_t>(m) * kR;
    const uint32_t* rr_row = RR + ((j << 4U) + x) * kR;

    uint64_t acc = 0;
    uint32_t pending = 0;
    #pragma unroll
    for (uint32_t yy = 0; yy < kB; ++yy) {
        acc += static_cast<uint64_t>(W[yy]) * s_b[m_local][yy];
        if (++pending == kReduceInterval) {
            acc = d_reduce64(acc);
            pending = 0;
        }
    }
    #pragma unroll
    for (uint32_t k = 0; k < kR; ++k) {
        acc += static_cast<uint64_t>(fl_row[k]) * rr_row[k];
        if (++pending == kReduceInterval) {
            acc = d_reduce64(acc);
            pending = 0;
        }
    }

    const size_t out_index =
        static_cast<size_t>(batch_index) * kMatrixElements +
        static_cast<size_t>((j << 4U) + x) * kN + m;
    rhs[out_index] = static_cast<uint32_t>(d_reduce64(acc));
}

__global__ void batch_factored_words_512x16_kernel(
    const uint32_t* __restrict__ matrix_a,
    const uint32_t* __restrict__ rhs,
    uint32_t* __restrict__ output)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kBlocksPerAxis = 32;
    constexpr uint32_t kTilesPerAxis = 16;
    constexpr uint32_t kTilesPerNonce = kTilesPerAxis * kTilesPerAxis;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    __shared__ uint64_t partials[256][4];

    const uint32_t global_tile = blockIdx.x;
    const uint32_t batch_index = global_tile / kTilesPerNonce;
    const uint32_t tile_index = global_tile & (kTilesPerNonce - 1U);
    const uint32_t j0 = (tile_index & (kTilesPerAxis - 1U)) << 1U;
    const uint32_t i0 = (tile_index >> 4U) << 1U;
    const uint32_t tid = threadIdx.x;

    const uint32_t* A = matrix_a + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t* D = rhs + static_cast<size_t>(batch_index) * kMatrixElements;

    uint64_t acc00 = 0;
    uint64_t acc01 = 0;
    uint64_t acc10 = 0;
    uint64_t acc11 = 0;
    uint32_t pending = 0;
    for (uint32_t idx = tid; idx < kB * kN; idx += blockDim.x) {
        const uint32_t x = idx >> 9U;
        const uint32_t m = idx & (kN - 1U);
        const uint32_t* a_row0 = A + static_cast<size_t>(i0 * kB + x) * kN;
        const uint32_t* a_row1 = a_row0 + static_cast<size_t>(kB) * kN;
        const uint32_t* d_row0 = D + static_cast<size_t>(j0 * kB + x) * kN;
        const uint32_t* d_row1 = d_row0 + static_cast<size_t>(kB) * kN;
        const uint64_t a0 = a_row0[m];
        const uint64_t a1 = a_row1[m];
        const uint64_t d0 = d_row0[m];
        const uint64_t d1 = d_row1[m];
        acc00 += a0 * d0;
        acc01 += a0 * d1;
        acc10 += a1 * d0;
        acc11 += a1 * d1;
        if (++pending == kReduceInterval) {
            acc00 = d_reduce64(acc00);
            acc01 = d_reduce64(acc01);
            acc10 = d_reduce64(acc10);
            acc11 = d_reduce64(acc11);
            pending = 0;
        }
    }

    partials[tid][0] = d_reduce64(acc00);
    partials[tid][1] = d_reduce64(acc01);
    partials[tid][2] = d_reduce64(acc10);
    partials[tid][3] = d_reduce64(acc11);
    __syncthreads();

    for (uint32_t stride = 128; stride > 0; stride >>= 1U) {
        if (tid < stride) {
            partials[tid][0] += partials[tid + stride][0];
            partials[tid][1] += partials[tid + stride][1];
            partials[tid][2] += partials[tid + stride][2];
            partials[tid][3] += partials[tid + stride][3];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const size_t out_base = (static_cast<size_t>(batch_index) << 10U);
        output[out_base + i0 * kBlocksPerAxis + j0] =
            static_cast<uint32_t>(d_reduce64(partials[0][0]));
        output[out_base + i0 * kBlocksPerAxis + j0 + 1U] =
            static_cast<uint32_t>(d_reduce64(partials[0][1]));
        output[out_base + (i0 + 1U) * kBlocksPerAxis + j0] =
            static_cast<uint32_t>(d_reduce64(partials[0][2]));
        output[out_base + (i0 + 1U) * kBlocksPerAxis + j0 + 1U] =
            static_cast<uint32_t>(d_reduce64(partials[0][3]));
    }
}

__global__ void batch_factored_words_wmma_512x16_kernel(
    const uint32_t* __restrict__ matrix_a,
    const uint32_t* __restrict__ rhs,
    uint32_t* __restrict__ output)
{
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__)
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kBlocksPerAxis = 32;
    constexpr uint32_t kK = kB * kN;
    constexpr uint32_t kTileCount = 4;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr uint32_t kWeights[7] = {
        1U, 256U, 65536U, 16777216U, 2U, 512U, 131072U};

    using i32x4 = int __attribute__((ext_vector_type(4)));
    using i32x8 = int __attribute__((ext_vector_type(8)));

    const uint32_t global_tile = blockIdx.x;
    const uint32_t batch_index = global_tile / kTileCount;
    const uint32_t tile_index = global_tile & (kTileCount - 1U);
    const uint32_t row_base = (tile_index >> 1U) << 4U;
    const uint32_t col_base = (tile_index & 1U) << 4U;
    const uint32_t lane = threadIdx.x & 15U;
    const uint32_t lane_group = threadIdx.x >> 4U;

    const uint32_t* A = matrix_a + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t* D = rhs + static_cast<size_t>(batch_index) * kMatrixElements;
    uint64_t acc[8] = {};

    #pragma unroll
    for (uint32_t limb_a = 0; limb_a < 4; ++limb_a) {
        #pragma unroll
        for (uint32_t limb_b = 0; limb_b < 4; ++limb_b) {
            i32x8 frag_c = {};
            for (uint32_t k0 = 0; k0 < kK; k0 += 16U) {
                i32x4 frag_a;
                i32x4 frag_b;
                #pragma unroll
                for (uint32_t pack = 0; pack < 4; ++pack) {
                    uint32_t packed_a = 0;
                    uint32_t packed_b = 0;
                    #pragma unroll
                    for (uint32_t q = 0; q < 4; ++q) {
                        const uint32_t kk = k0 + pack * 4U + q;
                        const uint32_t x = kk >> 9U;
                        const uint32_t m = kk & (kN - 1U);
                        const uint32_t a_val = A[static_cast<size_t>((row_base + lane) * kB + x) * kN + m];
                        const uint32_t d_val = D[static_cast<size_t>((col_base + lane) * kB + x) * kN + m];
                        const uint32_t a_limb = (a_val >> (limb_a * 8U)) & (limb_a == 3U ? 0x7fU : 0xffU);
                        const uint32_t b_limb = (d_val >> (limb_b * 8U)) & (limb_b == 3U ? 0x7fU : 0xffU);
                        packed_a |= a_limb << (q * 8U);
                        packed_b |= b_limb << (q * 8U);
                    }
                    frag_a[pack] = static_cast<int>(packed_a);
                    frag_b[pack] = static_cast<int>(packed_b);
                }
                frag_c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    false, frag_a, false, frag_b, frag_c, false);
            }

            const uint64_t weight = kWeights[limb_a + limb_b];
            #pragma unroll
            for (uint32_t e = 0; e < 8; ++e) {
                acc[e] += weight * static_cast<uint32_t>(frag_c[e]);
                acc[e] = d_reduce64(acc[e]);
            }
        }
    }

    #pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        const uint32_t row = row_base + e * 2U + lane_group;
        const uint32_t col = col_base + lane;
        output[(static_cast<size_t>(batch_index) << 10U) + row * kBlocksPerAxis + col] =
            static_cast<uint32_t>(d_reduce64(acc[e]));
    }
#endif
}

__global__ void batch_factored_words_wmma_partial_512x16_kernel(
    const uint32_t* __restrict__ matrix_a,
    const uint32_t* __restrict__ rhs,
    uint32_t* __restrict__ partials)
{
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__)
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kK = kB * kN;
    constexpr uint32_t kTileCount = 4;
    constexpr uint32_t kGroups = 64;
    constexpr uint32_t kGroupK = kK / kGroups;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr uint32_t kWeights[7] = {
        1U, 256U, 65536U, 16777216U, 2U, 512U, 131072U};

    using i32x4 = int __attribute__((ext_vector_type(4)));
    using i32x8 = int __attribute__((ext_vector_type(8)));

    const uint32_t global = blockIdx.x;
    const uint32_t group = global & (kGroups - 1U);
    const uint32_t tile_global = global >> 6U;
    const uint32_t batch_index = tile_global / kTileCount;
    const uint32_t tile_index = tile_global & (kTileCount - 1U);
    const uint32_t row_base = (tile_index >> 1U) << 4U;
    const uint32_t col_base = (tile_index & 1U) << 4U;
    const uint32_t lane = threadIdx.x & 15U;
    const uint32_t lane_group = threadIdx.x >> 4U;

    const uint32_t* A = matrix_a + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t* D = rhs + static_cast<size_t>(batch_index) * kMatrixElements;
    uint64_t acc[8] = {};
    const uint32_t k_begin = group * kGroupK;
    const uint32_t k_end = k_begin + kGroupK;

    i32x8 frag_c00 = {};
    i32x8 frag_c01 = {};
    i32x8 frag_c02 = {};
    i32x8 frag_c03 = {};
    i32x8 frag_c10 = {};
    i32x8 frag_c11 = {};
    i32x8 frag_c12 = {};
    i32x8 frag_c13 = {};
    i32x8 frag_c20 = {};
    i32x8 frag_c21 = {};
    i32x8 frag_c22 = {};
    i32x8 frag_c23 = {};
    i32x8 frag_c30 = {};
    i32x8 frag_c31 = {};
    i32x8 frag_c32 = {};
    i32x8 frag_c33 = {};
    for (uint32_t k0 = k_begin; k0 < k_end; k0 += 16U) {
        i32x4 frag_a0;
        i32x4 frag_a1;
        i32x4 frag_a2;
        i32x4 frag_a3;
        i32x4 frag_b0;
        i32x4 frag_b1;
        i32x4 frag_b2;
        i32x4 frag_b3;
        #pragma unroll
        for (uint32_t pack = 0; pack < 4; ++pack) {
            uint32_t packed_a0 = 0;
            uint32_t packed_a1 = 0;
            uint32_t packed_a2 = 0;
            uint32_t packed_a3 = 0;
            uint32_t packed_b0 = 0;
            uint32_t packed_b1 = 0;
            uint32_t packed_b2 = 0;
            uint32_t packed_b3 = 0;
            #pragma unroll
            for (uint32_t q = 0; q < 4; ++q) {
                const uint32_t kk = k0 + pack * 4U + q;
                const uint32_t x = kk >> 9U;
                const uint32_t m = kk & (kN - 1U);
                const uint32_t a_val = A[static_cast<size_t>((row_base + lane) * kB + x) * kN + m];
                const uint32_t d_val = D[static_cast<size_t>((col_base + lane) * kB + x) * kN + m];
                const uint32_t shift = q * 8U;
                packed_a0 |= ((a_val & 0xffU) << shift);
                packed_a1 |= (((a_val >> 8U) & 0xffU) << shift);
                packed_a2 |= (((a_val >> 16U) & 0xffU) << shift);
                packed_a3 |= (((a_val >> 24U) & 0x7fU) << shift);
                packed_b0 |= ((d_val & 0xffU) << shift);
                packed_b1 |= (((d_val >> 8U) & 0xffU) << shift);
                packed_b2 |= (((d_val >> 16U) & 0xffU) << shift);
                packed_b3 |= (((d_val >> 24U) & 0x7fU) << shift);
            }
            frag_a0[pack] = static_cast<int>(packed_a0);
            frag_a1[pack] = static_cast<int>(packed_a1);
            frag_a2[pack] = static_cast<int>(packed_a2);
            frag_a3[pack] = static_cast<int>(packed_a3);
            frag_b0[pack] = static_cast<int>(packed_b0);
            frag_b1[pack] = static_cast<int>(packed_b1);
            frag_b2[pack] = static_cast<int>(packed_b2);
            frag_b3[pack] = static_cast<int>(packed_b3);
        }
        frag_c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a0, false, frag_b0, frag_c00, false);
        frag_c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a0, false, frag_b1, frag_c01, false);
        frag_c02 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a0, false, frag_b2, frag_c02, false);
        frag_c03 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a0, false, frag_b3, frag_c03, false);
        frag_c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a1, false, frag_b0, frag_c10, false);
        frag_c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a1, false, frag_b1, frag_c11, false);
        frag_c12 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a1, false, frag_b2, frag_c12, false);
        frag_c13 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a1, false, frag_b3, frag_c13, false);
        frag_c20 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a2, false, frag_b0, frag_c20, false);
        frag_c21 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a2, false, frag_b1, frag_c21, false);
        frag_c22 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a2, false, frag_b2, frag_c22, false);
        frag_c23 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a2, false, frag_b3, frag_c23, false);
        frag_c30 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a3, false, frag_b0, frag_c30, false);
        frag_c31 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a3, false, frag_b1, frag_c31, false);
        frag_c32 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a3, false, frag_b2, frag_c32, false);
        frag_c33 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, frag_a3, false, frag_b3, frag_c33, false);
    }

    #pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        acc[e] += static_cast<uint32_t>(frag_c00[e]) * static_cast<uint64_t>(kWeights[0]);
        acc[e] += static_cast<uint32_t>(frag_c01[e]) * static_cast<uint64_t>(kWeights[1]);
        acc[e] += static_cast<uint32_t>(frag_c02[e]) * static_cast<uint64_t>(kWeights[2]);
        acc[e] += static_cast<uint32_t>(frag_c03[e]) * static_cast<uint64_t>(kWeights[3]);
        acc[e] += static_cast<uint32_t>(frag_c10[e]) * static_cast<uint64_t>(kWeights[1]);
        acc[e] += static_cast<uint32_t>(frag_c11[e]) * static_cast<uint64_t>(kWeights[2]);
        acc[e] += static_cast<uint32_t>(frag_c12[e]) * static_cast<uint64_t>(kWeights[3]);
        acc[e] += static_cast<uint32_t>(frag_c13[e]) * static_cast<uint64_t>(kWeights[4]);
        acc[e] += static_cast<uint32_t>(frag_c20[e]) * static_cast<uint64_t>(kWeights[2]);
        acc[e] += static_cast<uint32_t>(frag_c21[e]) * static_cast<uint64_t>(kWeights[3]);
        acc[e] += static_cast<uint32_t>(frag_c22[e]) * static_cast<uint64_t>(kWeights[4]);
        acc[e] += static_cast<uint32_t>(frag_c23[e]) * static_cast<uint64_t>(kWeights[5]);
        acc[e] += static_cast<uint32_t>(frag_c30[e]) * static_cast<uint64_t>(kWeights[3]);
        acc[e] += static_cast<uint32_t>(frag_c31[e]) * static_cast<uint64_t>(kWeights[4]);
        acc[e] += static_cast<uint32_t>(frag_c32[e]) * static_cast<uint64_t>(kWeights[5]);
        acc[e] += static_cast<uint32_t>(frag_c33[e]) * static_cast<uint64_t>(kWeights[6]);
        acc[e] = d_reduce64(acc[e]);
    }

    const size_t partial_base =
        (((static_cast<size_t>(batch_index) * kTileCount + tile_index) * kGroups + group) << 8U);
    #pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        const uint32_t local_row = e * 2U + lane_group;
        partials[partial_base + local_row * 16U + lane] =
            static_cast<uint32_t>(d_reduce64(acc[e]));
    }
#endif
}

__global__ void batch_reduce_wmma_partials_512x16_kernel(
    const uint32_t* __restrict__ partials,
    size_t total_words,
    uint32_t* __restrict__ output)
{
    constexpr uint32_t kBlocksPerAxis = 32;
    constexpr uint32_t kTileCount = 4;
    constexpr uint32_t kGroups = 64;
    constexpr uint32_t kWordsPerNonce = 1024;
    constexpr uint32_t kWordsPerTile = 256;

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_words) {
        return;
    }
    const size_t batch_index = gid >> 10U;
    const uint32_t word = static_cast<uint32_t>(gid & (kWordsPerNonce - 1U));
    const uint32_t row = word >> 5U;
    const uint32_t col = word & 31U;
    const uint32_t tile = ((row >> 4U) << 1U) | (col >> 4U);
    const uint32_t local = ((row & 15U) << 4U) | (col & 15U);

    uint64_t acc = 0;
    #pragma unroll
    for (uint32_t group = 0; group < kGroups; ++group) {
        const size_t idx =
            (((static_cast<size_t>(batch_index) * kTileCount + tile) * kGroups + group) *
             kWordsPerTile) + local;
        acc += partials[idx];
        acc = d_reduce64(acc);
    }
    output[batch_index * kWordsPerNonce + row * kBlocksPerAxis + col] =
        static_cast<uint32_t>(d_reduce64(acc));
}

__global__ void batch_build_left_noise_projection_512x16x8_kernel(
    const uint32_t* __restrict__ noise_er,
    const uint32_t* __restrict__ rhs,
    size_t total_elements,
    uint32_t* __restrict__ left_proj)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kR = 8;
    constexpr size_t kNoiseRightElements = size_t{kR} * kN;
    constexpr size_t kRhsElements = size_t{kN} * kN;

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = gid >> 12U;
    const uint32_t local = static_cast<uint32_t>(gid & 4095U);
    const uint32_t k = local & (kR - 1U);
    const uint32_t x = (local >> 3U) & (kB - 1U);
    const uint32_t j = local >> 7U;

    const uint32_t* ER = noise_er + batch_index * kNoiseRightElements;
    const uint32_t* D = rhs + batch_index * kRhsElements;
    const uint32_t* er_row = ER + k * kN;
    const uint32_t* d_row = D + ((j << 4U) + x) * kN;

    uint64_t acc = 0;
    uint32_t pending = 0;
    for (uint32_t m = 0; m < kN; ++m) {
        acc += static_cast<uint64_t>(er_row[m]) * d_row[m];
        if (++pending == kReduceInterval) {
            acc = d_reduce64(acc);
            pending = 0;
        }
    }
    left_proj[gid] = static_cast<uint32_t>(d_reduce64(acc));
}

__global__ void batch_factored_words_direct_a_512x16x8_kernel(
    const uint32_t* __restrict__ base_a,
    const uint32_t* __restrict__ noise_el,
    const uint32_t* __restrict__ left_proj,
    const uint32_t* __restrict__ rhs,
    uint32_t* __restrict__ output)
{
    constexpr uint32_t kN = 512;
    constexpr uint32_t kB = 16;
    constexpr uint32_t kR = 8;
    constexpr uint32_t kBlocksPerAxis = 32;
    constexpr uint32_t kTilesPerAxis = 16;
    constexpr uint32_t kTilesPerNonce = kTilesPerAxis * kTilesPerAxis;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr size_t kNoiseLeftElements = size_t{kN} * kR;
    constexpr size_t kLeftProjElements = 32U * kB * kR;
    __shared__ uint64_t partials[256][4];

    const uint32_t global_tile = blockIdx.x;
    const uint32_t batch_index = global_tile / kTilesPerNonce;
    const uint32_t tile_index = global_tile & (kTilesPerNonce - 1U);
    const uint32_t j0 = (tile_index & (kTilesPerAxis - 1U)) << 1U;
    const uint32_t i0 = (tile_index >> 4U) << 1U;
    const uint32_t tid = threadIdx.x;

    const uint32_t* A = base_a + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t* EL = noise_el + static_cast<size_t>(batch_index) * kNoiseLeftElements;
    const uint32_t* LP = left_proj + static_cast<size_t>(batch_index) * kLeftProjElements;
    const uint32_t* D = rhs + static_cast<size_t>(batch_index) * kMatrixElements;

    uint64_t acc00 = 0;
    uint64_t acc01 = 0;
    uint64_t acc10 = 0;
    uint64_t acc11 = 0;
    uint32_t pending = 0;
    for (uint32_t idx = tid; idx < kB * kN; idx += blockDim.x) {
        const uint32_t x = idx >> 9U;
        const uint32_t m = idx & (kN - 1U);
        const uint32_t* a_row0 = A + static_cast<size_t>(i0 * kB + x) * kN;
        const uint32_t* a_row1 = a_row0 + static_cast<size_t>(kB) * kN;
        const uint32_t* d_row0 = D + static_cast<size_t>(j0 * kB + x) * kN;
        const uint32_t* d_row1 = d_row0 + static_cast<size_t>(kB) * kN;
        const uint64_t a0 = a_row0[m];
        const uint64_t a1 = a_row1[m];
        const uint64_t d0 = d_row0[m];
        const uint64_t d1 = d_row1[m];
        acc00 += a0 * d0;
        acc01 += a0 * d1;
        acc10 += a1 * d0;
        acc11 += a1 * d1;
        if (++pending == kReduceInterval) {
            acc00 = d_reduce64(acc00);
            acc01 = d_reduce64(acc01);
            acc10 = d_reduce64(acc10);
            acc11 = d_reduce64(acc11);
            pending = 0;
        }
    }
    for (uint32_t idx = tid; idx < kB * kR; idx += blockDim.x) {
        const uint32_t x = idx >> 3U;
        const uint32_t k = idx & (kR - 1U);
        const uint32_t el0 = EL[static_cast<size_t>(i0 * kB + x) * kR + k];
        const uint32_t el1 = EL[static_cast<size_t>((i0 + 1U) * kB + x) * kR + k];
        const uint32_t lp0 = LP[((j0 << 4U) + x) * kR + k];
        const uint32_t lp1 = LP[(((j0 + 1U) << 4U) + x) * kR + k];
        acc00 += static_cast<uint64_t>(el0) * lp0;
        acc01 += static_cast<uint64_t>(el0) * lp1;
        acc10 += static_cast<uint64_t>(el1) * lp0;
        acc11 += static_cast<uint64_t>(el1) * lp1;
        if (++pending == kReduceInterval) {
            acc00 = d_reduce64(acc00);
            acc01 = d_reduce64(acc01);
            acc10 = d_reduce64(acc10);
            acc11 = d_reduce64(acc11);
            pending = 0;
        }
    }

    partials[tid][0] = d_reduce64(acc00);
    partials[tid][1] = d_reduce64(acc01);
    partials[tid][2] = d_reduce64(acc10);
    partials[tid][3] = d_reduce64(acc11);
    __syncthreads();

    for (uint32_t stride = 128; stride > 0; stride >>= 1U) {
        if (tid < stride) {
            partials[tid][0] += partials[tid + stride][0];
            partials[tid][1] += partials[tid + stride][1];
            partials[tid][2] += partials[tid + stride][2];
            partials[tid][3] += partials[tid + stride][3];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const size_t out_base = (static_cast<size_t>(batch_index) << 10U);
        output[out_base + i0 * kBlocksPerAxis + j0] =
            static_cast<uint32_t>(d_reduce64(partials[0][0]));
        output[out_base + i0 * kBlocksPerAxis + j0 + 1U] =
            static_cast<uint32_t>(d_reduce64(partials[0][1]));
        output[out_base + (i0 + 1U) * kBlocksPerAxis + j0] =
            static_cast<uint32_t>(d_reduce64(partials[0][2]));
        output[out_base + (i0 + 1U) * kBlocksPerAxis + j0 + 1U] =
            static_cast<uint32_t>(d_reduce64(partials[0][3]));
    }
}

__global__ void batch_generate_matrix_midstate_512_kernel(
    const uint8_t* __restrict__ seeds,
    uint32_t* __restrict__ output)
{
    constexpr uint32_t kN = 512;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    __shared__ uint8_t s_seed[32];
    __shared__ uint32_t s_midstate[16];

    const size_t tile_start = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t batch_index = tile_start >> 18U;
    const uint32_t local_base = static_cast<uint32_t>(tile_start & (kMatrixElements - 1U));
    const uint8_t* seed = seeds + batch_index * 32;

    if (threadIdx.x < 32U) {
        s_seed[threadIdx.x] = seed[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed, s_midstate);
    }
    __syncthreads();

    const uint32_t local_index = local_base + threadIdx.x;
    if (local_index < kMatrixElements) {
        output[tile_start + threadIdx.x] =
            d_from_oracle_midstate(s_midstate, s_seed, local_index);
    }
}

void GenerateMatrixMidstate512(
    const uint8_t* seeds,
    size_t batch,
    uint32_t* output)
{
    constexpr uint32_t kThreads = 256;
    constexpr size_t kMatrixElements = size_t{512} * 512;
    const uint32_t blocks = static_cast<uint32_t>(
        (batch * kMatrixElements + kThreads - 1) / kThreads);
    batch_generate_matrix_midstate_512_kernel<<<blocks, kThreads>>>(
        seeds, output);
}

__global__ void batch_generate_matrix_from_midstate_512_kernel(
    const uint8_t* __restrict__ seeds,
    const uint32_t* __restrict__ midstates,
    size_t total_elements,
    uint32_t* __restrict__ output)
{
    constexpr uint32_t kN = 512;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    __shared__ uint32_t s_midstate[16];

    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = gid >> 18U;
    if (threadIdx.x < 16U) {
        s_midstate[threadIdx.x] =
            midstates[batch_index * 16U + threadIdx.x];
    }
    __syncthreads();

    const uint32_t local_index =
        static_cast<uint32_t>(gid & (kMatrixElements - 1U));
    output[gid] = d_from_oracle_midstate(
        s_midstate, seeds + batch_index * 32U, local_index);
}

void GenerateMatrixFromMidstate512(
    const uint8_t* seeds,
    const uint32_t* midstates,
    size_t batch,
    uint32_t* output)
{
    constexpr uint32_t kThreads = 256;
    constexpr size_t kMatrixElements = size_t{512} * 512;
    const size_t total_elements = batch * kMatrixElements;
    const uint32_t blocks = static_cast<uint32_t>(
        (total_elements + kThreads - 1) / kThreads);
    batch_generate_matrix_from_midstate_512_kernel<<<blocks, kThreads>>>(
        seeds, midstates, total_elements, output);
}

__global__ void batch_compute_seed_midstates_kernel(
    const uint8_t* __restrict__ seeds,
    size_t batch,
    uint32_t* __restrict__ midstates)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }
    d_compute_seed_midstate(
        seeds + static_cast<size_t>(idx) * 32U,
        midstates + static_cast<size_t>(idx) * 16U);
}

void ComputeSeedMidstates(
    const uint8_t* seeds,
    size_t batch,
    uint32_t* midstates)
{
    constexpr uint32_t kThreads = 256;
    const uint32_t blocks = static_cast<uint32_t>(
        (batch + kThreads - 1) / kThreads);
    batch_compute_seed_midstates_kernel<<<blocks, kThreads>>>(
        seeds, batch, midstates);
}

struct DeviceMatrixCache {
    int device = -1;
    uint8_t seed_a[32]{};
    uint8_t seed_b[32]{};
    uint32_t n = 0;
    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
};

DeviceMatrixCache g_matrix_cache[16];

struct DeviceLaunchPool {
    int device = -1;
    uint32_t* d_workspace = nullptr;
    uint32_t* d_gate_count = nullptr;
    uint64_t* d_passed_nonces = nullptr;
    uint8_t* d_passed_sigma = nullptr;
    uint8_t* d_passed_seed_a = nullptr;
    uint8_t* d_passed_seed_b = nullptr;
    uint32_t* d_seed_midstate = nullptr;
    uint32_t* d_header_midstate = nullptr;
    size_t nonce_cap = 0;
    size_t ws_bytes = 0;
    size_t digest_cap = 0;
};

DeviceLaunchPool g_launch_pool[16];

struct BatchedMatMulPool {
    int device = -1;
    uint8_t* d_seed_a = nullptr;
    uint8_t* d_seed_b = nullptr;
    uint32_t* d_seed_a_midstate = nullptr;
    uint32_t* d_seed_b_midstate = nullptr;
    uint8_t* d_sigma = nullptr;
    uint8_t* d_noise_seeds = nullptr;
    uint8_t* d_compress_seeds = nullptr;
    uint32_t* d_base_a = nullptr;
    uint32_t* d_base_b = nullptr;
    uint32_t* d_matrix_a = nullptr;
    uint32_t* d_matrix_b = nullptr;
    uint32_t* d_noise_el = nullptr;
    uint32_t* d_noise_er = nullptr;
    uint32_t* d_noise_fl = nullptr;
    uint32_t* d_noise_fr = nullptr;
    uint32_t* d_compress = nullptr;
    uint32_t* d_words = nullptr;
    uint8_t* d_digests = nullptr;
    int32_t* d_results = nullptr;
    uint8_t* d_block_target = nullptr;
    uint8_t* d_share_target = nullptr;
    size_t capacity = 0;
};

BatchedMatMulPool g_batched_pool[16];

bool EnvEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

bool DeviceIsGfx11(int device)
{
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        return false;
    }
    return std::strncmp(prop.gcnArchName, "gfx11", 5) == 0;
}

struct MatmulProfile {
    bool enabled = false;
    size_t scan_batch = 0;
    size_t passed = 0;
    const char* words_path = "none";
    char arch[64] = "unknown";
    float sigma_ms = 0.0f;
    float matrix_ms = 0.0f;
    float noise_seed_ms = 0.0f;
    float noise_ms = 0.0f;
    float compress_ms = 0.0f;
    float rhs_ms = 0.0f;
    float rhs_a_ms = 0.0f;
    float rhs_right_ms = 0.0f;
    float rhs_b_ms = 0.0f;
    float words_ms = 0.0f;
    float hash_compare_ms = 0.0f;
};

struct GpuStageTimer {
    bool enabled = false;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    explicit GpuStageTimer(bool enable) : enabled(enable)
    {
        if (!enabled) {
            return;
        }
        if (hipEventCreate(&start) != hipSuccess ||
            hipEventCreate(&stop) != hipSuccess ||
            hipEventRecord(start, 0) != hipSuccess) {
            enabled = false;
        }
    }

    float finish()
    {
        float ms = 0.0f;
        if (enabled &&
            hipEventRecord(stop, 0) == hipSuccess &&
            hipEventSynchronize(stop) == hipSuccess) {
            (void)hipEventElapsedTime(&ms, start, stop);
        }
        if (start) {
            (void)hipEventDestroy(start);
        }
        if (stop) {
            (void)hipEventDestroy(stop);
        }
        return ms;
    }
};

void FreeBatchedPool(BatchedMatMulPool& pool)
{
    hipFree(pool.d_seed_a);
    hipFree(pool.d_seed_b);
    hipFree(pool.d_seed_a_midstate);
    hipFree(pool.d_seed_b_midstate);
    hipFree(pool.d_sigma);
    hipFree(pool.d_noise_seeds);
    hipFree(pool.d_compress_seeds);
    hipFree(pool.d_base_a);
    hipFree(pool.d_base_b);
    hipFree(pool.d_matrix_a);
    hipFree(pool.d_matrix_b);
    hipFree(pool.d_noise_el);
    hipFree(pool.d_noise_er);
    hipFree(pool.d_noise_fl);
    hipFree(pool.d_noise_fr);
    hipFree(pool.d_compress);
    hipFree(pool.d_words);
    hipFree(pool.d_digests);
    hipFree(pool.d_results);
    hipFree(pool.d_block_target);
    hipFree(pool.d_share_target);
    pool = {};
}

template <typename T>
bool HipAlloc(T*& ptr, size_t count)
{
    return hipMalloc(&ptr, count * sizeof(T)) == hipSuccess;
}

bool EnsureBatchedPool(
    int device,
    size_t batch,
    uint32_t n,
    uint32_t r,
    uint32_t bsz)
{
    if (device < 0 || device >= 16) {
        return false;
    }
    BatchedMatMulPool& pool = g_batched_pool[device];
    if (pool.device == device && pool.capacity >= batch) {
        return true;
    }
    FreeBatchedPool(pool);
    pool.device = device;
    pool.capacity = batch;

    const size_t matrix = batch * static_cast<size_t>(n) * n;
    const size_t noise_left = batch * static_cast<size_t>(n) * r;
    const size_t noise_right = batch * static_cast<size_t>(r) * n;
    const size_t compress = batch * static_cast<size_t>(bsz) * bsz;
    const size_t words = batch * static_cast<size_t>(n / bsz) * (n / bsz);

    const bool ok =
        HipAlloc(pool.d_seed_a, batch * 32) &&
        HipAlloc(pool.d_seed_b, batch * 32) &&
        HipAlloc(pool.d_seed_a_midstate, batch * 16) &&
        HipAlloc(pool.d_seed_b_midstate, batch * 16) &&
        HipAlloc(pool.d_sigma, batch * 32) &&
        HipAlloc(pool.d_noise_seeds, batch * 4 * 32) &&
        HipAlloc(pool.d_compress_seeds, batch * 32) &&
        HipAlloc(pool.d_base_a, matrix) &&
        HipAlloc(pool.d_base_b, matrix) &&
        HipAlloc(pool.d_matrix_a, matrix) &&
        HipAlloc(pool.d_matrix_b, matrix) &&
        HipAlloc(pool.d_noise_el, noise_left) &&
        HipAlloc(pool.d_noise_er, noise_right) &&
        HipAlloc(pool.d_noise_fl, noise_left) &&
        HipAlloc(pool.d_noise_fr, noise_right) &&
        HipAlloc(pool.d_compress, compress) &&
        HipAlloc(pool.d_words, words) &&
        HipAlloc(pool.d_digests, batch * 32) &&
        HipAlloc(pool.d_results, batch) &&
        HipAlloc(pool.d_block_target, 32) &&
        HipAlloc(pool.d_share_target, 32);
    if (!ok) {
        FreeBatchedPool(pool);
    }
    return ok;
}

void FreeLaunchPool(DeviceLaunchPool& pool)
{
    if (pool.d_workspace) hipFree(pool.d_workspace);
    if (pool.d_gate_count) hipFree(pool.d_gate_count);
    if (pool.d_passed_nonces) hipFree(pool.d_passed_nonces);
    if (pool.d_passed_sigma) hipFree(pool.d_passed_sigma);
    if (pool.d_passed_seed_a) hipFree(pool.d_passed_seed_a);
    if (pool.d_passed_seed_b) hipFree(pool.d_passed_seed_b);
    if (pool.d_seed_midstate) hipFree(pool.d_seed_midstate);
    if (pool.d_header_midstate) hipFree(pool.d_header_midstate);
    pool = {};
}

bool EnsureLaunchPool(int device, size_t batch, size_t ws_bytes)
{
    if (device < 0 || device >= 16) return false;
    DeviceLaunchPool& pool = g_launch_pool[device];
    if (pool.device != device) {
        FreeLaunchPool(pool);
        pool.device = device;
    }

    if (pool.nonce_cap < batch) {
        pool.nonce_cap = batch;
    }
    if (pool.ws_bytes < ws_bytes) {
        if (pool.d_workspace) hipFree(pool.d_workspace);
        pool.d_workspace = nullptr;
        if (hipMalloc(&pool.d_workspace, ws_bytes) != hipSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
        pool.ws_bytes = ws_bytes;
    }
    if (pool.digest_cap < batch) {
        if (pool.d_gate_count) hipFree(pool.d_gate_count);
        if (pool.d_passed_nonces) hipFree(pool.d_passed_nonces);
        if (pool.d_passed_sigma) hipFree(pool.d_passed_sigma);
        if (pool.d_passed_seed_a) hipFree(pool.d_passed_seed_a);
        if (pool.d_passed_seed_b) hipFree(pool.d_passed_seed_b);
        if (pool.d_seed_midstate) hipFree(pool.d_seed_midstate);
        if (pool.d_header_midstate) hipFree(pool.d_header_midstate);
        pool.d_gate_count = nullptr;
        pool.d_passed_nonces = nullptr;
        pool.d_passed_sigma = nullptr;
        pool.d_passed_seed_a = nullptr;
        pool.d_passed_seed_b = nullptr;
        pool.d_seed_midstate = nullptr;
        pool.d_header_midstate = nullptr;
        if (hipMalloc(&pool.d_gate_count, sizeof(uint32_t)) != hipSuccess ||
            hipMalloc(&pool.d_passed_nonces, batch * sizeof(uint64_t)) != hipSuccess ||
            hipMalloc(&pool.d_passed_sigma, batch * 32) != hipSuccess ||
            hipMalloc(&pool.d_passed_seed_a, batch * 32) != hipSuccess ||
            hipMalloc(&pool.d_passed_seed_b, batch * 32) != hipSuccess ||
            hipMalloc(&pool.d_seed_midstate, 8 * sizeof(uint32_t)) != hipSuccess ||
            hipMalloc(&pool.d_header_midstate, 8 * sizeof(uint32_t)) != hipSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
        pool.digest_cap = batch;
    }
    return pool.d_workspace && pool.d_gate_count && pool.d_passed_nonces &&
           pool.d_passed_sigma && pool.d_passed_seed_a &&
           pool.d_passed_seed_b && pool.d_seed_midstate &&
           pool.d_header_midstate;
}

void FreeMatrixCache(DeviceMatrixCache& cache)
{
    if (cache.d_A) {
        hipFree(cache.d_A);
        cache.d_A = nullptr;
    }
    if (cache.d_B) {
        hipFree(cache.d_B);
        cache.d_B = nullptr;
    }
}

static Matrix FromSeedHost(const Uint256& seed, uint32_t n)
{
    Matrix out(n, n);
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            out.at(row, col) = field::from_oracle(seed, row * n + col);
        }
    }
    return out;
}

bool EnsureMatricesOnDevice(
    int device,
    const FusedMatMulJob& job,
    uint32_t** d_A,
    uint32_t** d_B)
{
    if (device < 0 || device >= 16) {
        return false;
    }

    // Post-fork jobs derive per-nonce A/B on device; skip CPU FromSeed + H2D.
    if (job.state.height >= 125000) {
        *d_A = nullptr;
        *d_B = nullptr;
        return true;
    }

    DeviceMatrixCache& cache = g_matrix_cache[device];
    const bool same =
        cache.device == device &&
        cache.n == job.n &&
        std::memcmp(cache.seed_a, job.state.seed_a.data, 32) == 0 &&
        std::memcmp(cache.seed_b, job.state.seed_b.data, 32) == 0 &&
        cache.d_A != nullptr &&
        cache.d_B != nullptr;

    if (same) {
        *d_A = cache.d_A;
        *d_B = cache.d_B;
        return true;
    }

    if (cache.device != device) {
        FreeMatrixCache(cache);
        cache.device = device;
    } else {
        FreeMatrixCache(cache);
    }

    const Matrix A = FromSeedHost(job.state.seed_a, job.n);
    const Matrix B = FromSeedHost(job.state.seed_b, job.n);
    const size_t elems = static_cast<size_t>(job.n) * job.n;
    const size_t bytes = elems * sizeof(uint32_t);

    if (hipMalloc(&cache.d_A, bytes) != hipSuccess ||
        hipMalloc(&cache.d_B, bytes) != hipSuccess) {
        FreeMatrixCache(cache);
        return false;
    }
    if (hipMemcpy(cache.d_A, A.data(), bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(cache.d_B, B.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        FreeMatrixCache(cache);
        return false;
    }

    std::memcpy(cache.seed_a, job.state.seed_a.data, 32);
    std::memcpy(cache.seed_b, job.state.seed_b.data, 32);
    cache.n = job.n;
    cache.device = device;

    *d_A = cache.d_A;
    *d_B = cache.d_B;
    return true;
}

HipJobParams MakeHipJobParams(const FusedMatMulJob& job)
{
    HipJobParams p{};
    p.version = job.state.version;
    p.time = job.state.time;
    p.bits = job.state.bits;
    p.n = job.n;
    p.b = job.b;
    p.r = job.r;
    p.epsilon_bits = job.epsilon_bits;
    p.block_height = static_cast<uint32_t>(job.state.height);
    p.fast_v3_scan = EnvEnabled("BTX_MATMUL_FAST_V3_SCAN") ? 1U : 0U;
    p.parent_mtp = job.state.parent_mtp;
    std::memcpy(p.prev_hash, job.state.previous_block_hash.data, 32);
    std::memcpy(p.merkle_root, job.state.merkle_root.data, 32);
    std::memcpy(p.seed_a, job.state.seed_a.data, 32);
    std::memcpy(p.seed_b, job.state.seed_b.data, 32);
    std::memcpy(p.target, job.share_target.data, 32);
    const Uint256 pre_hash = DerivePreHashTarget(job.block_target, job.epsilon_bits);
    std::memcpy(p.pre_hash_target, pre_hash.data, 32);
    return p;
}

bool ProcessPassedNoncesBatched(
    int device,
    const FusedMatMulJob& job,
    size_t batch,
    const uint8_t* d_seed_a,
    const uint8_t* d_seed_b,
    const uint8_t* d_sigma,
    std::vector<Uint256>& digests,
    std::vector<uint8_t>& found,
    MatmulProfile* profile = nullptr)
{
    if (batch == 0) {
        return true;
    }
    if (!EnsureBatchedPool(device, batch, job.n, job.r, job.b)) {
        return false;
    }
    BatchedMatMulPool& pool = g_batched_pool[device];

    if (hipMemcpy(
            pool.d_seed_a, d_seed_a, batch * 32,
            hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemcpy(
            pool.d_seed_b, d_seed_b, batch * 32,
            hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemcpy(
            pool.d_sigma, d_sigma, batch * 32,
            hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemcpy(
            pool.d_block_target, job.block_target.data, 32,
            hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(
            pool.d_share_target, job.share_target.data, 32,
            hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }

    const uint32_t n = job.n;
    const uint32_t r = job.r;
    const uint32_t bsz = job.b;
    const uint32_t blocks_per_axis = n / bsz;
    const uint32_t words_per_nonce = blocks_per_axis * blocks_per_axis;
    const size_t matrix_elements = static_cast<size_t>(n) * n;
    const size_t total_matrix_elements = batch * matrix_elements;
    const uint32_t noise_left_elements = n * r;
    const uint32_t noise_right_elements = r * n;
    const uint32_t compress_elements = bsz * bsz;
    const bool use_512x16x8 = n == 512 && bsz == 16 && r == 8;
    const bool use_fused_b_rhs = use_512x16x8 && !EnvEnabled("BTX_MATMUL_NO_FUSED_B");
    const bool use_fused_a_perturb = use_512x16x8 &&
        !EnvEnabled("BTX_MATMUL_NO_FUSED_A");
    const bool use_precomputed_midstates = use_512x16x8 &&
        (use_fused_a_perturb || !EnvEnabled("BTX_MATMUL_NO_PRECOMP_A_MIDSTATE"));

    {
        GpuStageTimer timer(profile && profile->enabled);
        if (use_precomputed_midstates) {
            if (use_fused_a_perturb ||
                !EnvEnabled("BTX_MATMUL_NO_PRECOMP_A_MIDSTATE")) {
                ComputeSeedMidstates(pool.d_seed_a, batch, pool.d_seed_a_midstate);
            }
            if (use_fused_b_rhs) {
                ComputeSeedMidstates(pool.d_seed_b, batch, pool.d_seed_b_midstate);
            } else {
                GenerateMatrixMidstate512(pool.d_seed_b, batch, pool.d_base_b);
            }
        } else if (n == 512) {
            GenerateMatrixMidstate512(pool.d_seed_a, batch, pool.d_base_a);
            if (!use_fused_b_rhs) {
                GenerateMatrixMidstate512(pool.d_seed_b, batch, pool.d_base_b);
            }
        } else {
            gpasha::GenerateMatrixKernel_launch(
                pool.d_seed_a, static_cast<uint32_t>(batch), n, pool.d_base_a);
            gpasha::GenerateMatrixKernel_launch(
                pool.d_seed_b, static_cast<uint32_t>(batch), n, pool.d_base_b);
        }
        if (profile && profile->enabled) {
            profile->matrix_ms += timer.finish();
        }
    }
    {
        GpuStageTimer timer(profile && profile->enabled);
        gpasha::DeriveNoiseSeedsKernel_launch(
            pool.d_sigma, pool.d_noise_seeds, pool.d_compress_seeds,
            static_cast<uint32_t>(batch));
        if (profile && profile->enabled) {
            profile->noise_seed_ms += timer.finish();
        }
    }
    {
        GpuStageTimer timer(profile && profile->enabled);
        gpasha::GenerateNoiseKernel_launch(
            pool.d_noise_seeds, static_cast<uint32_t>(batch),
            noise_left_elements, 0, pool.d_noise_el);
        gpasha::GenerateNoiseKernel_launch(
            pool.d_noise_seeds, static_cast<uint32_t>(batch),
            noise_right_elements, 1, pool.d_noise_er);
        gpasha::GenerateNoiseKernel_launch(
            pool.d_noise_seeds, static_cast<uint32_t>(batch),
            noise_left_elements, 2, pool.d_noise_fl);
        gpasha::GenerateNoiseKernel_launch(
            pool.d_noise_seeds, static_cast<uint32_t>(batch),
            noise_right_elements, 3, pool.d_noise_fr);
        if (profile && profile->enabled) {
            profile->noise_ms += timer.finish();
        }
    }
    {
        GpuStageTimer timer(profile && profile->enabled);
        gpasha::GenerateCompressKernel_launch(
            pool.d_compress_seeds, static_cast<uint32_t>(batch),
            compress_elements, pool.d_compress);
        if (profile && profile->enabled) {
            profile->compress_ms += timer.finish();
        }
    }

    constexpr uint32_t kThreads = 256;
    const uint32_t matrix_blocks = static_cast<uint32_t>(
        (total_matrix_elements + kThreads - 1) / kThreads);
    {
        GpuStageTimer timer(profile && profile->enabled);
        if (use_512x16x8) {
            {
                GpuStageTimer sub_timer(profile && profile->enabled);
                if (use_fused_a_perturb) {
                    const uint32_t fused_a_threads =
                        EnvEnabled("BTX_MATMUL_FUSED_A_512T") ? 512U : 256U;
                    const uint32_t fused_a_blocks = static_cast<uint32_t>(
                        (total_matrix_elements + fused_a_threads - 1) /
                        fused_a_threads);
                    batch_generate_perturb_matrix_512x8_kernel<<<fused_a_blocks, fused_a_threads>>>(
                        pool.d_seed_a, pool.d_seed_a_midstate,
                        pool.d_noise_el, pool.d_noise_er, total_matrix_elements,
                        pool.d_matrix_a);
                } else {
                    if (use_precomputed_midstates) {
                        GenerateMatrixFromMidstate512(
                            pool.d_seed_a, pool.d_seed_a_midstate, batch,
                            pool.d_base_a);
                    }
                    batch_perturb_matrix_512x8_kernel<<<matrix_blocks, kThreads>>>(
                        pool.d_base_a, pool.d_noise_el, pool.d_noise_er,
                        total_matrix_elements, pool.d_matrix_a);
                }
                if (profile && profile->enabled) {
                    profile->rhs_a_ms += sub_timer.finish();
                }
            }
            const size_t right_rhs_elements = batch * 32U * 16U * 8U;
            const uint32_t right_rhs_blocks = static_cast<uint32_t>(
                (right_rhs_elements + kThreads - 1) / kThreads);
            {
                GpuStageTimer sub_timer(profile && profile->enabled);
                if (EnvEnabled("BTX_MATMUL_OLD_RIGHT_RHS")) {
                    batch_build_right_noise_rhs_512x16x8_kernel<<<right_rhs_blocks, kThreads>>>(
                        pool.d_noise_fr, pool.d_compress, right_rhs_elements,
                        pool.d_noise_er);
                } else {
                    const size_t right_rhs_pairs = batch * 32U * 16U;
                    const uint32_t right_rhs_pair_blocks = static_cast<uint32_t>(
                        (right_rhs_pairs + kThreads - 1) / kThreads);
                    batch_build_right_noise_rhs_grouped_512x16x8_kernel<<<right_rhs_pair_blocks, kThreads>>>(
                        pool.d_noise_fr, pool.d_compress, right_rhs_pairs,
                        pool.d_noise_er);
                }
                if (profile && profile->enabled) {
                    profile->rhs_right_ms += sub_timer.finish();
                }
            }
            {
                GpuStageTimer sub_timer(profile && profile->enabled);
                if (use_fused_b_rhs) {
                    const uint32_t fused_b_tiles = static_cast<uint32_t>(batch) * 32U * 32U;
                    batch_build_factored_rhs_fused_b_512x16x8_kernel<<<fused_b_tiles, kThreads>>>(
                        pool.d_seed_b, pool.d_seed_b_midstate,
                        pool.d_noise_fl, pool.d_noise_er, pool.d_compress,
                        pool.d_matrix_b);
                } else {
                    batch_build_factored_rhs_direct_512x16x8_kernel<<<matrix_blocks, kThreads>>>(
                        pool.d_base_b, pool.d_noise_fl, pool.d_noise_er,
                        pool.d_compress, total_matrix_elements, pool.d_matrix_b);
                }
                if (profile && profile->enabled) {
                    profile->rhs_b_ms += sub_timer.finish();
                }
            }
        } else {
            batch_perturb_matrix_kernel<<<matrix_blocks, kThreads>>>(
                pool.d_base_a, pool.d_noise_el, pool.d_noise_er, n, r,
                matrix_elements, total_matrix_elements, pool.d_matrix_a);
            batch_perturb_matrix_kernel<<<matrix_blocks, kThreads>>>(
                pool.d_base_b, pool.d_noise_fl, pool.d_noise_fr, n, r,
                matrix_elements, total_matrix_elements, pool.d_matrix_b);
        }
        if (profile && profile->enabled) {
            profile->rhs_ms += timer.finish();
        }
    }

    const uint32_t total_words =
        static_cast<uint32_t>(batch) * words_per_nonce;
    const bool use_wmma_words = use_512x16x8 && DeviceIsGfx11(device) &&
        !EnvEnabled("BTX_MATMUL_NO_WMMA");
    if (profile && profile->enabled) {
        if (use_wmma_words) {
            profile->words_path = "wmma";
        } else if (use_512x16x8) {
            profile->words_path = "factored";
        } else {
            profile->words_path = "generic";
        }
    }
    {
        GpuStageTimer timer(profile && profile->enabled);
        if (use_wmma_words) {
            constexpr uint32_t kWmmaGroups = 64;
            const uint32_t total_wmma_tiles = static_cast<uint32_t>(batch) * 4U * kWmmaGroups;
            batch_factored_words_wmma_partial_512x16_kernel<<<total_wmma_tiles, 32>>>(
                pool.d_matrix_a, pool.d_matrix_b,
                pool.d_base_a);
            const uint32_t reduce_blocks = (total_words + kThreads - 1) / kThreads;
            batch_reduce_wmma_partials_512x16_kernel<<<reduce_blocks, kThreads>>>(
                pool.d_base_a, total_words,
                pool.d_words);
        } else if (use_512x16x8) {
            const uint32_t total_factored_tiles = static_cast<uint32_t>(batch) * 16U * 16U;
            batch_factored_words_512x16_kernel<<<total_factored_tiles, kThreads>>>(
                pool.d_matrix_a, pool.d_matrix_b,
                pool.d_words);
        } else {
            batch_compressed_words_kernel<<<total_words, kThreads>>>(
                pool.d_matrix_a, pool.d_matrix_b, pool.d_compress,
                n, bsz, blocks_per_axis, words_per_nonce, matrix_elements,
                pool.d_words);
        }
        if (profile && profile->enabled) {
            profile->words_ms += timer.finish();
        }
    }

    {
        GpuStageTimer timer(profile && profile->enabled);
        gpasha::HashTranscriptKernel_launch(
            pool.d_words, pool.d_sigma, words_per_nonce, n, bsz,
            static_cast<uint32_t>(batch), pool.d_digests);
        gpasha::CompareDigestsKernel_launch(
            pool.d_digests, pool.d_block_target, pool.d_share_target,
            static_cast<uint32_t>(batch), pool.d_results);
        if (profile && profile->enabled) {
            profile->hash_compare_ms += timer.finish();
        }
    }

    hipError_t err = hipGetLastError();
    if (err == hipSuccess) {
        err = hipDeviceSynchronize();
    }
    if (err != hipSuccess) {
        std::fprintf(
            stderr, "batched HIP transcript failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    std::vector<int32_t> results(batch);
    std::vector<uint8_t> digest_bytes(batch * 32);
    if (hipMemcpy(
            results.data(), pool.d_results, batch * sizeof(int32_t),
            hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(
            digest_bytes.data(), pool.d_digests, digest_bytes.size(),
            hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }

    digests.resize(batch);
    found.assign(batch, 0);
    for (size_t i = 0; i < batch; ++i) {
        if (results[i] != 0) {
            found[i] = 1;
            std::memcpy(digests[i].data, digest_bytes.data() + i * 32, 32);
        }
    }
    return true;
}

size_t WorkspaceUint32Count(uint32_t n, uint32_t r, uint32_t bsz, bool use_v2_seeds)
{
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t noise_elems = 2 * (static_cast<size_t>(n) * r + static_cast<size_t>(r) * n);
    const size_t scratch = static_cast<size_t>(bsz) * bsz;
    const bool use_factored = d_use_factored_path(n, bsz);
    const size_t blocks_per_axis = static_cast<size_t>(n / bsz);
    const size_t matrix_elems = use_v2_seeds ? (2 * nn) : (use_factored ? (2 * nn) : 0);
    const size_t factored_rhs_elems =
        use_factored ? (static_cast<size_t>(bsz) * n * blocks_per_axis) : 0;
    const size_t legacy_scratch_elems = use_factored ? 0 : (scratch * 5);
    return matrix_elems + nn + noise_elems + scratch + factored_rhs_elems + legacy_scratch_elems;
}

} // namespace

size_t FusedWorkspaceBytesPerNonce(uint32_t n, uint32_t r, uint32_t bsz, bool use_v2_seeds)
{
    return WorkspaceUint32Count(n, r, bsz, use_v2_seeds) * sizeof(uint32_t);
}

int AutoBatchSizeForDevice(
    int device,
    uint32_t n,
    uint32_t r,
    uint32_t b,
    int32_t block_height,
    int max_cap)
{
    const bool use_v2 = block_height >= 125000;
    const size_t per_nonce = FusedWorkspaceBytesPerNonce(n, r, b, use_v2);
    if (per_nonce == 0) {
        return 256;
    }

    if (hipSetDevice(device) != hipSuccess) {
        return 256;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        return 256;
    }

    constexpr size_t kReserveBytes = 256 * 1024 * 1024;
    if (free_bytes <= kReserveBytes) {
        return 256;
    }

    const size_t usable = static_cast<size_t>((free_bytes - kReserveBytes) * 0.85);
    int batch = static_cast<int>(usable / per_nonce);
    batch = std::max(64, batch);
    if (max_cap > 0) {
        batch = std::min(batch, max_cap);
    }
    return batch;
}

extern "C" bool LaunchMatMulTranscriptBatch(
    int device,
    const FusedMatMulJob& job,
    const std::vector<uint64_t>& nonces,
    std::vector<Uint256>& out_digests,
    std::vector<bool>& out_found,
    std::vector<bool>* out_gate_passed)
{
    if (nonces.empty()) {
        return false;
    }

    if (hipSetDevice(device) != hipSuccess) {
        return false;
    }

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    if (!EnsureMatricesOnDevice(device, job, &d_A, &d_B)) {
        return false;
    }

    HipJobParams h_params = MakeHipJobParams(job);

    const size_t batch = nonces.size();
    const bool use_v2 = job.state.height >= 125000;
    const size_t ws_uint32_per_nonce = WorkspaceUint32Count(job.n, job.r, job.b, use_v2);
    out_digests.resize(batch);
    out_found.assign(batch, false);
    if (out_gate_passed) {
        out_gate_passed->assign(batch, false);
    }

    // The epsilon gate accepts roughly one nonce in 2^epsilon_bits. Allocate
    // one full workspace initially and grow only when a scan actually finds
    // more candidates.
    const size_t per_nonce_ws_bytes =
        ws_uint32_per_nonce * sizeof(uint32_t);
    if (!EnsureLaunchPool(device, batch, per_nonce_ws_bytes)) {
        return false;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];

    for (size_t i = 1; i < batch; ++i) {
        if (nonces[i] != nonces.front() + i) {
            return false;
        }
    }
    if (hipMemset(pool.d_gate_count, 0, sizeof(uint32_t)) != hipSuccess) {
        return false;
    }

    MatmulProfile profile{};
    profile.enabled = EnvEnabled("BTX_MATMUL_PROFILE");
    profile.scan_batch = batch;
    if (profile.enabled) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
            std::snprintf(profile.arch, sizeof(profile.arch), "%s", prop.gcnArchName);
        }
    }
    {
        GpuStageTimer timer(profile.enabled);
        init_scan_midstates_kernel<<<1, 1>>>(
            h_params, pool.d_seed_midstate, pool.d_header_midstate);
        // RDNA3 (gfx11): 512-wide launches improve sigma-gate occupancy vs 256.
        constexpr int kGateThreads = 512;
        const int gate_blocks =
            static_cast<int>((batch + kGateThreads - 1) / kGateThreads);
        sigma_gate_kernel<<<gate_blocks, kGateThreads>>>(
            h_params, nonces.front(), batch,
            pool.d_seed_midstate, pool.d_header_midstate, pool.d_gate_count,
            pool.d_passed_nonces, pool.d_passed_sigma,
            pool.d_passed_seed_a, pool.d_passed_seed_b);
        if (profile.enabled) {
            profile.sigma_ms = timer.finish();
        }
    }

    if (hipGetLastError() != hipSuccess) {
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        return false;
    }
    uint32_t passed = 0;
    if (hipMemcpy(
            &passed, pool.d_gate_count, sizeof(passed),
            hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    profile.passed = passed;

    if (passed == 0) {
        if (profile.enabled) {
            std::fprintf(stderr,
                "matmul_profile scan=%zu passed=0 arch=%s words_path=none sigma_ms=%.3f\n",
                profile.scan_batch, profile.arch, profile.sigma_ms);
        }
        return true;
    }

    std::vector<uint64_t> passed_nonces(passed);
    if (hipMemcpy(
            passed_nonces.data(), pool.d_passed_nonces,
            passed * sizeof(uint64_t), hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }

    std::vector<Uint256> passed_digests;
    std::vector<uint8_t> passed_found;
    if (!ProcessPassedNoncesBatched(
            device, job, passed, pool.d_passed_seed_a,
            pool.d_passed_seed_b, pool.d_passed_sigma,
            passed_digests, passed_found, &profile)) {
        return false;
    }

    if (profile.enabled) {
        std::fprintf(stderr,
            "matmul_profile scan=%zu passed=%zu arch=%s words_path=%s "
            "sigma_ms=%.3f matrix_ms=%.3f "
            "noise_seed_ms=%.3f noise_ms=%.3f compress_ms=%.3f rhs_ms=%.3f "
            "rhs_a_ms=%.3f rhs_right_ms=%.3f rhs_b_ms=%.3f "
            "words_ms=%.3f hash_compare_ms=%.3f\n",
            profile.scan_batch, profile.passed, profile.arch, profile.words_path,
            profile.sigma_ms,
            profile.matrix_ms, profile.noise_seed_ms, profile.noise_ms,
            profile.compress_ms, profile.rhs_ms, profile.rhs_a_ms,
            profile.rhs_right_ms, profile.rhs_b_ms, profile.words_ms,
            profile.hash_compare_ms);
    }

    bool any_hit = false;
    for (size_t i = 0; i < passed; ++i) {
        const size_t original_index =
            static_cast<size_t>(passed_nonces[i] - nonces.front());
        if (original_index >= batch) {
            return false;
        }
        if (out_gate_passed) {
            (*out_gate_passed)[original_index] = true;
        }
        out_found[original_index] = passed_found[i] != 0;
        if (out_found[original_index]) {
            out_digests[original_index] = passed_digests[i];
            any_hit = true;
        }
    }
    return true;
}

extern "C" bool LaunchMatMulTranscriptCompact(
    int device,
    const FusedMatMulJob& job,
    uint64_t nonce_start,
    size_t nonce_count,
    std::vector<MatMulBatchHit>& out_hits,
    uint64_t* out_gate_passes)
{
    out_hits.clear();
    if (nonce_count == 0) {
        return false;
    }

    if (hipSetDevice(device) != hipSuccess) {
        return false;
    }

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    if (!EnsureMatricesOnDevice(device, job, &d_A, &d_B)) {
        return false;
    }

    HipJobParams h_params = MakeHipJobParams(job);
    const bool use_v2 = job.state.height >= 125000;
    const size_t ws_uint32_per_nonce =
        WorkspaceUint32Count(job.n, job.r, job.b, use_v2);
    const size_t per_nonce_ws_bytes =
        ws_uint32_per_nonce * sizeof(uint32_t);
    if (!EnsureLaunchPool(device, nonce_count, per_nonce_ws_bytes)) {
        return false;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];

    if (hipMemset(pool.d_gate_count, 0, sizeof(uint32_t)) != hipSuccess) {
        return false;
    }

    MatmulProfile profile{};
    profile.enabled = EnvEnabled("BTX_MATMUL_PROFILE");
    profile.scan_batch = nonce_count;
    if (profile.enabled) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
            std::snprintf(profile.arch, sizeof(profile.arch), "%s", prop.gcnArchName);
        }
    }

    {
        GpuStageTimer timer(profile.enabled);
        init_scan_midstates_kernel<<<1, 1>>>(
            h_params, pool.d_seed_midstate, pool.d_header_midstate);
        constexpr int kGateThreads = 512;
        const int gate_blocks =
            static_cast<int>((nonce_count + kGateThreads - 1) / kGateThreads);
        sigma_gate_kernel<<<gate_blocks, kGateThreads>>>(
            h_params, nonce_start, nonce_count,
            pool.d_seed_midstate, pool.d_header_midstate, pool.d_gate_count,
            pool.d_passed_nonces, pool.d_passed_sigma,
            pool.d_passed_seed_a, pool.d_passed_seed_b);
        if (profile.enabled) {
            profile.sigma_ms = timer.finish();
        }
    }

    if (hipGetLastError() != hipSuccess) {
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        return false;
    }

    uint32_t passed = 0;
    if (hipMemcpy(
            &passed, pool.d_gate_count, sizeof(passed),
            hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    if (out_gate_passes) {
        *out_gate_passes = passed;
    }
    profile.passed = passed;

    if (passed == 0) {
        if (profile.enabled) {
            std::fprintf(stderr,
                "matmul_profile scan=%zu passed=0 arch=%s words_path=none sigma_ms=%.3f\n",
                profile.scan_batch, profile.arch, profile.sigma_ms);
        }
        return true;
    }

    std::vector<uint64_t> passed_nonces(passed);
    if (hipMemcpy(
            passed_nonces.data(), pool.d_passed_nonces,
            passed * sizeof(uint64_t), hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }

    std::vector<Uint256> passed_digests;
    std::vector<uint8_t> passed_found;
    if (!ProcessPassedNoncesBatched(
            device, job, passed, pool.d_passed_seed_a,
            pool.d_passed_seed_b, pool.d_passed_sigma,
            passed_digests, passed_found, &profile)) {
        return false;
    }

    if (profile.enabled) {
        std::fprintf(stderr,
            "matmul_profile scan=%zu passed=%zu arch=%s words_path=%s "
            "sigma_ms=%.3f matrix_ms=%.3f "
            "noise_seed_ms=%.3f noise_ms=%.3f compress_ms=%.3f rhs_ms=%.3f "
            "rhs_a_ms=%.3f rhs_right_ms=%.3f rhs_b_ms=%.3f "
            "words_ms=%.3f hash_compare_ms=%.3f\n",
            profile.scan_batch, profile.passed, profile.arch, profile.words_path,
            profile.sigma_ms,
            profile.matrix_ms, profile.noise_seed_ms, profile.noise_ms,
            profile.compress_ms, profile.rhs_ms, profile.rhs_a_ms,
            profile.rhs_right_ms, profile.rhs_b_ms, profile.words_ms,
            profile.hash_compare_ms);
    }

    out_hits.reserve(passed);
    for (size_t i = 0; i < passed; ++i) {
        if (passed_found[i] == 0) {
            continue;
        }
        MatMulBatchHit hit{};
        hit.nonce = passed_nonces[i];
        hit.digest = passed_digests[i];
        hit.found = 1;
        out_hits.push_back(hit);
    }
    return true;
}

extern "C" bool HipVerifyAgainstCpu(
    const FusedMatMulJob& job,
    uint64_t nonce)
{
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) <= 0 || device_count == 0) {
        return true;
    }

    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        return false;
    }

    std::vector<uint64_t> nonces = {nonce};
    std::vector<Uint256> digests;
    std::vector<bool> found;
    if (!LaunchMatMulTranscriptBatch(device, job, nonces, digests, found)) {
        return false;
    }

    PowState verify = job.state;
    verify.nonce = nonce;
    PrepareNonceSeeds(verify);
    Uint256 cpu_digest;
    const bool cpu_hit = ComputeDigestForNonce(verify, job.n, job.b, job.r, cpu_digest) &&
                         Uint256LE(cpu_digest, job.share_target);
    const bool gpu_hit = !found.empty() && found[0];

    if (cpu_hit != gpu_hit) {
        return false;
    }
    if (cpu_hit && digests[0] != cpu_digest) {
        return false;
    }
    return true;
}

#endif // __HIPCC__
