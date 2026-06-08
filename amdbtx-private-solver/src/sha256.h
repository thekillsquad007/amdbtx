#pragma once
#include <cstdint>
#include <cstring>

// Self-contained SHA-256 implementation (FIPS 180-4).
// CPU only — no OpenSSL dependency.

namespace sha256 {

static constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

struct Ctx {
    uint32_t h[8];
    uint64_t total_len;
    uint8_t buf[64];
    size_t buf_len;
};

inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t Ch(uint32_t e, uint32_t f, uint32_t g) {
    return (e & f) ^ (~e & g);
}

inline uint32_t Maj(uint32_t a, uint32_t b, uint32_t c) {
    return (a & b) ^ (a & c) ^ (b & c);
}

inline uint32_t bsig0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline uint32_t bsig1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline uint32_t ssig0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline uint32_t ssig1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

inline void block_compress(Ctx& ctx, const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t(block[i*4]) << 24)
              | (uint32_t(block[i*4+1]) << 16)
              | (uint32_t(block[i*4+2]) << 8)
              | uint32_t(block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = ssig1(w[i-2]) + w[i-7] + ssig0(w[i-15]) + w[i-16];
    }

    uint32_t a = ctx.h[0];
    uint32_t b = ctx.h[1];
    uint32_t c = ctx.h[2];
    uint32_t d = ctx.h[3];
    uint32_t e = ctx.h[4];
    uint32_t f = ctx.h[5];
    uint32_t g = ctx.h[6];
    uint32_t h = ctx.h[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + bsig1(e) + Ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = bsig0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx.h[0] += a;
    ctx.h[1] += b;
    ctx.h[2] += c;
    ctx.h[3] += d;
    ctx.h[4] += e;
    ctx.h[5] += f;
    ctx.h[6] += g;
    ctx.h[7] += h;
}

inline void init(Ctx& ctx) {
    ctx.h[0] = 0x6a09e667;
    ctx.h[1] = 0xbb67ae85;
    ctx.h[2] = 0x3c6ef372;
    ctx.h[3] = 0xa54ff53a;
    ctx.h[4] = 0x510e527f;
    ctx.h[5] = 0x9b05688c;
    ctx.h[6] = 0x1f83d9ab;
    ctx.h[7] = 0x5be0cd19;
    ctx.total_len = 0;
    ctx.buf_len = 0;
}

inline void update(Ctx& ctx, const uint8_t* data, size_t len) {
    ctx.total_len += len;
    if (ctx.buf_len > 0) {
        size_t fill = 64 - ctx.buf_len;
        if (len < fill) {
            memcpy(ctx.buf + ctx.buf_len, data, len);
            ctx.buf_len += len;
            return;
        }
        memcpy(ctx.buf + ctx.buf_len, data, fill);
        block_compress(ctx, ctx.buf);
        data += fill;
        len -= fill;
        ctx.buf_len = 0;
    }
    while (len >= 64) {
        block_compress(ctx, data);
        data += 64;
        len -= 64;
    }
    if (len > 0) {
        memcpy(ctx.buf, data, len);
        ctx.buf_len = len;
    }
}

inline void final(Ctx& ctx, uint8_t out[32]) {
    uint64_t bit_len = ctx.total_len * 8;
    uint8_t pad = 0x80;
    update(ctx, &pad, 1);
    pad = 0x00;
    while (ctx.buf_len != 56) {
        uint8_t z = 0x00;
        update(ctx, &z, 1);
    }
    uint8_t len_be[8];
    for (int i = 7; i >= 0; i--) {
        len_be[i] = static_cast<uint8_t>(bit_len & 0xFF);
        bit_len >>= 8;
    }
    update(ctx, len_be, 8);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (ctx.h[i] >> 24) & 0xFF;
        out[i*4+1] = (ctx.h[i] >> 16) & 0xFF;
        out[i*4+2] = (ctx.h[i] >> 8) & 0xFF;
        out[i*4+3] = ctx.h[i] & 0xFF;
    }
}

// Single-shot SHA-256 hash.
inline void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    Ctx ctx;
    init(ctx);
    update(ctx, data, len);
    final(ctx, out);
}

// Double SHA-256 (hash of hash).
inline void sha256d(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint8_t first[32];
    sha256(data, len, first);
    sha256(first, 32, out);
}

// HMAC-SHA-256
inline void hmac_sha256(const uint8_t* key, size_t key_len,
                        const uint8_t* data, size_t data_len,
                        uint8_t out[32]) {
    uint8_t k_pad[64];
    uint8_t o_pad[64];
    memset(k_pad, 0, 64);
    memset(o_pad, 0, 64);

    if (key_len > 64) {
        uint8_t k_hash[32];
        sha256(key, key_len, k_hash);
        memcpy(k_pad, k_hash, 32);
    } else {
        memcpy(k_pad, key, key_len);
    }
    memcpy(o_pad, k_pad, 64);

    for (int i = 0; i < 64; i++) {
        k_pad[i] ^= 0x36;
        o_pad[i] ^= 0x5C;
    }

    Ctx inner;
    init(inner);
    update(inner, k_pad, 64);
    update(inner, data, data_len);
    uint8_t inner_hash[32];
    final(inner, inner_hash);

    Ctx outer;
    init(outer);
    update(outer, o_pad, 64);
    update(outer, inner_hash, 32);
    final(outer, out);
}

} // namespace sha256
