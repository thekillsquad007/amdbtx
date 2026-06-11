// HIP implementation for BTX MatMul PoW transcript search.
// One HIP block per nonce: full sigma derivation, noise, blocked matmul,
// running transcript compression, and target check on device.
// Host uploads constant A/B (FromSeed) once per job; CPU VerifySolution
// cross-checks any hits before submission.

#include "solve.h"
#include "matmul_kernel.h"
#include "matrix.h"
#include "field.h"
#include "uint256.h"



#ifdef __HIPCC__

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

__device__ void d_fill_rect(uint32_t* out, uint32_t rows, uint32_t cols, const uint8_t* seed_internal)
{
    const uint32_t total = rows * cols;
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        out[idx] = d_from_oracle(seed_internal, idx);
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

    if (threadIdx.x == 0) {
        uint8_t sigma_canonical[32];
        d_to_canonical(sigma_internal, sigma_canonical);

        Sha256State st;
        d_sha256_init(&st);
        const char* tag = "matmul-compress-v1";
        d_sha256_update(&st, reinterpret_cast<const uint8_t*>(tag), 18);
        d_sha256_update(&st, sigma_canonical, 32);
        uint8_t seedb[32];
        d_sha256_final(&st, seedb);

        for (int i = 0; i < 32; ++i) {
            s_seed_internal[i] = seedb[31 - i];
        }
    }
    __syncthreads();

    const uint32_t len = bsz * bsz;
    for (uint32_t k = threadIdx.x; k < len; k += blockDim.x) {
        compress_v[k] = d_from_oracle(s_seed_internal, k);
    }
}

__device__ bool d_solve_nonce(
    const HipJobParams& job,
    uint32_t* A,
    uint32_t* B,
    uint64_t nonce64,
    uint32_t* C,
    uint32_t* E_L,
    uint32_t* E_R,
    uint32_t* F_L,
    uint32_t* F_R,
    uint32_t* compress_v,
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

    if (threadIdx.x == 0) {
        if (reuse_seed_a && reuse_seed_b) {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = reuse_seed_a[i];
                s_seed_b[i] = reuse_seed_b[i];
            }
        } else if (job.block_height >= 125000) {
            d_matmul_seed_v2(job, nonce64, 0, s_seed_a);
            d_matmul_seed_v2(job, nonce64, 1, s_seed_b);
        } else {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = job.seed_a[i];
                s_seed_b[i] = job.seed_b[i];
            }
        }
        d_derive_sigma(job, nonce64, s_seed_a, s_seed_b, s_sigma);
        s_sigma_pass = job.epsilon_bits == 0 ||
                       d_sigma_below_prehash(s_sigma, job.pre_hash_target);
    }
    __syncthreads();
    if (!s_sigma_pass) {
        return false;
    }

    if (job.block_height >= 125000 && A && B) {
        d_fill_rect(A, job.n, job.n, s_seed_a);
        __syncthreads();
        d_fill_rect(B, job.n, job.n, s_seed_b);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_derive_noise_seed("matmul_noise_EL_v1", s_sigma, s_seed_el);
        d_derive_noise_seed("matmul_noise_ER_v1", s_sigma, s_seed_er);
        d_derive_noise_seed("matmul_noise_FL_v1", s_sigma, s_seed_fl);
        d_derive_noise_seed("matmul_noise_FR_v1", s_sigma, s_seed_fr);
    }
    __syncthreads();

    d_fill_rect(E_L, job.n, job.r, s_seed_el);
    __syncthreads();
    d_fill_rect(E_R, job.r, job.n, s_seed_er);
    __syncthreads();
    d_fill_rect(F_L, job.n, job.r, s_seed_fl);
    __syncthreads();
    d_fill_rect(F_R, job.r, job.n, s_seed_fr);
    __syncthreads();

    const uint32_t bsz = job.b;
    const uint32_t N = job.n / bsz;
    const uint32_t word_count = N * N;

    d_zero_u32(C, word_count);
    __syncthreads();

    d_build_compress_vec(s_sigma, job.b, compress_v);
    __syncthreads();

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
    const size_t v2_base_elems = use_v2 ? (2 * nn) : 0;
    const size_t per_nonce_elems = v2_base_elems + nn + noise_elems + compress_elems + scratch_elems * 5;

    // Per-nonce workspace layout:
    // [A nn][B nn] (V2 only) [C nn][E_L n*r][E_R r*n][F_L n*r][F_R r*n][compress b*b][ablk][bblk][noise][prod][cblk]
    size_t off = idx * per_nonce_elems;

    uint32_t* A_local = use_v2 ? (workspaces + off) : nullptr;
    if (use_v2) off += nn;
    uint32_t* B_local = use_v2 ? (workspaces + off) : nullptr;
    if (use_v2) off += nn;

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
    uint32_t* ablk = workspaces + off;
    off += scratch_elems;
    uint32_t* bblk = workspaces + off;
    off += scratch_elems;
    uint32_t* noise_blk = workspaces + off;
    off += scratch_elems;
    uint32_t* prod = workspaces + off;
    off += scratch_elems;
    uint32_t* cblk = workspaces + off;

    __shared__ uint8_t s_v2_seed_a[32];
    __shared__ uint8_t s_v2_seed_b[32];

    if (use_v2 && threadIdx.x == 0) {
        d_matmul_seed_v2(params, nonces[idx], 0, s_v2_seed_a);
        d_matmul_seed_v2(params, nonces[idx], 1, s_v2_seed_b);
    }
    __syncthreads();

    uint32_t* use_a = use_v2 ? A_local : const_cast<uint32_t*>(A);
    uint32_t* use_b = use_v2 ? B_local : const_cast<uint32_t*>(B);

    uint8_t digest[32];
    const uint8_t* reuse_a = use_v2 ? s_v2_seed_a : nullptr;
    const uint8_t* reuse_b = use_v2 ? s_v2_seed_b : nullptr;
    const bool hit = d_solve_nonce(
        params, use_a, use_b, nonces[idx], C, E_L, E_R, F_L, F_R, compress_v,
        ablk, bblk, noise_blk, prod, cblk, reuse_a, reuse_b, digest);

    if (threadIdx.x == 0) {
        if (hit) {
            out_found[idx] = 1;
            for (int i = 0; i < 32; ++i) {
                out_digests[idx * 32 + i] = digest[i];
            }
        }
    }
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
    uint64_t* d_nonces = nullptr;
    uint32_t* d_workspace = nullptr;
    uint8_t* d_digests = nullptr;
    uint8_t* d_found = nullptr;
    size_t nonce_cap = 0;
    size_t ws_bytes = 0;
    size_t digest_cap = 0;
};

DeviceLaunchPool g_launch_pool[16];

void FreeLaunchPool(DeviceLaunchPool& pool)
{
    if (pool.d_nonces) hipFree(pool.d_nonces);
    if (pool.d_workspace) hipFree(pool.d_workspace);
    if (pool.d_digests) hipFree(pool.d_digests);
    if (pool.d_found) hipFree(pool.d_found);
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
        if (pool.d_nonces) hipFree(pool.d_nonces);
        pool.d_nonces = nullptr;
        if (hipMalloc(&pool.d_nonces, batch * sizeof(uint64_t)) != hipSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
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
    const size_t digest_bytes = batch * 32;
    if (pool.digest_cap < batch) {
        if (pool.d_digests) hipFree(pool.d_digests);
        if (pool.d_found) hipFree(pool.d_found);
        pool.d_digests = nullptr;
        pool.d_found = nullptr;
        if (hipMalloc(&pool.d_digests, digest_bytes) != hipSuccess ||
            hipMalloc(&pool.d_found, batch) != hipSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
        pool.digest_cap = batch;
    }
    return pool.d_nonces && pool.d_workspace && pool.d_digests && pool.d_found;
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
    std::memcpy(p.prev_hash, job.state.previous_block_hash.data, 32);
    std::memcpy(p.merkle_root, job.state.merkle_root.data, 32);
    std::memcpy(p.seed_a, job.state.seed_a.data, 32);
    std::memcpy(p.seed_b, job.state.seed_b.data, 32);
    std::memcpy(p.target, job.share_target.data, 32);
    const Uint256 pre_hash = DerivePreHashTarget(job.block_target, job.epsilon_bits);
    std::memcpy(p.pre_hash_target, pre_hash.data, 32);
    return p;
}

size_t WorkspaceUint32Count(uint32_t n, uint32_t r, uint32_t bsz, bool use_v2_seeds)
{
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t noise_elems = 2 * (static_cast<size_t>(n) * r + static_cast<size_t>(r) * n);
    const size_t scratch = static_cast<size_t>(bsz) * bsz;
    const size_t v2_base = use_v2_seeds ? (2 * nn) : 0;
    return v2_base + nn + noise_elems + scratch + scratch * 5;
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
    std::vector<bool>& out_found)
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
    const size_t ws_bytes = batch * ws_uint32_per_nonce * sizeof(uint32_t);

    out_digests.resize(batch);
    out_found.assign(batch, false);

    if (!EnsureLaunchPool(device, batch, ws_bytes)) {
        return false;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];

    std::vector<uint8_t> h_found(batch, 0);
    std::vector<uint8_t> h_digests(batch * 32);

    if (hipMemset(pool.d_found, 0, batch) != hipSuccess) {
        return false;
    }
    if (hipMemcpy(pool.d_nonces, nonces.data(), batch * sizeof(uint64_t), hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }

    matmul_nonce_kernel<<<static_cast<int>(batch), kBlockThreads>>>(
        h_params, d_A, d_B, pool.d_nonces, pool.d_workspace, pool.d_digests, pool.d_found);

    if (hipGetLastError() != hipSuccess) {
        return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        return false;
    }
    if (hipMemcpy(h_found.data(), pool.d_found, batch, hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }

    bool any_hit = false;
    for (size_t i = 0; i < batch; ++i) {
        out_found[i] = h_found[i] != 0;
        if (out_found[i]) {
            any_hit = true;
        }
    }

    if (any_hit) {
        if (hipMemcpy(h_digests.data(), pool.d_digests, batch * 32, hipMemcpyDeviceToHost) !=
            hipSuccess) {
            return false;
        }
        for (size_t i = 0; i < batch; ++i) {
            if (out_found[i]) {
                std::memcpy(out_digests[i].data, h_digests.data() + i * 32, 32);
            }
        }
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