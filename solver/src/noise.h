#pragma once
#include "sha256.h"
#include "field.h"
#include <cstdint>
#include <cstring>

// Noise generation from sigma using SHA-256 PRF + rejection sampling.
// Domain separation: "matmul_noise_EL_v1", etc.

struct NoisePair {
    field::Element EL[512 * 8]; // n x r
    field::Element ER[8 * 512]; // r x n
    field::Element FL[512 * 8]; // n x r
    field::Element FR[8 * 512]; // r x n
};

// Fill a buffer of `count` field elements using SHA-256 PRF with rejection sampling.
// tag is the domain separation string (null-terminated).
static void prf_fill(field::Element* out, uint32_t count,
                     const uint8_t sigma[32], const char* tag,
                     size_t tag_len) {
    uint32_t filled = 0;
    uint32_t ctr = 0;
    while (filled < count) {
        // Build message: sigma || LE32(ctr) || tag
        uint8_t msg[32 + 4 + 64];
        size_t msg_len = 0;
        memcpy(msg + msg_len, sigma, 32);
        msg_len += 32;
        msg[msg_len++] = ctr & 0xFF;
        msg[msg_len++] = (ctr >> 8) & 0xFF;
        msg[msg_len++] = (ctr >> 16) & 0xFF;
        msg[msg_len++] = (ctr >> 24) & 0xFF;
        memcpy(msg + msg_len, tag, tag_len);
        msg_len += tag_len;

        // Hash to get 32 bytes of pseudo-random data
        uint8_t h[32];
        sha256::sha256(msg, msg_len, h);

        // Extract up to 8 field elements from 32 bytes (4 bytes each)
        for (int i = 0; i < 8 && filled < count; i++) {
            uint32_t raw = (uint32_t(h[i*4]) << 24)
                         | (uint32_t(h[i*4+1]) << 16)
                         | (uint32_t(h[i*4+2]) << 8)
                         | uint32_t(h[i*4+3]);
            raw &= 0x7FFFFFFF; // mod 2^31
            if (raw < field::Q) { // rejection: skip if >= Q
                out[filled++] = raw;
            }
        }
        ctr++;
    }
}

inline void noise_generate(const uint8_t sigma[32], uint32_t n, uint32_t r, NoisePair& out) {
    prf_fill(out.EL, n * r, sigma, "matmul_noise_EL_v1", 18);
    prf_fill(out.ER, r * n, sigma, "matmul_noise_ER_v1", 18);
    prf_fill(out.FL, n * r, sigma, "matmul_noise_FL_v1", 18);
    prf_fill(out.FR, r * n, sigma, "matmul_noise_FR_v1", 18);
}

inline void derive_compression_vector(const uint8_t sigma[32], uint32_t b, field::Element* out) {
    prf_fill(out, b * b, sigma, "matmul_compress_v1", 18);
}
