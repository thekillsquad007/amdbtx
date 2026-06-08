#pragma once
#include "field.h"
#include "sha256.h"
#include <cstdint>
#include <cstring>

// Transcript compression: block decomposition + inner product → rolling SHA-256d.

// C_prime is the noisy matmul result (n x n), stored row-major.
// compress_vec is the compression vector (b x b).
// out_words receives (n/b)*(n/b)*(n/b) compressed field elements.
inline void transcript_compress(
    const field::Element* C_prime, uint32_t n,
    const field::Element* compress_vec, uint32_t b,
    field::Element* out_words, uint32_t& out_count)
{
    uint32_t nb = n / b;
    out_count = 0;

    for (uint32_t i = 0; i < nb; i++) {
        for (uint32_t j = 0; j < nb; j++) {
            for (uint32_t l = 0; l < nb; l++) {
                // Compute inner product of the b×b sub-block of C'
                // (rows [i*b, (i+1)*b), cols [j*b, (j+1)*b))
                // with the compression vector, but only the rows
                // that correspond to block l.
                //
                // Actually, per the BTX spec: the block decomposition
                // produces a 3D index space (i, j, l). The compression
                // is over the b×b block at position (i, l) in C'
                // crossed with the b×b block at position (l, j) in
                // the transpose — or equivalently, just the b×b sub-block
                // of the intermediate accumulation indexed by (i, j, l).
                //
                // For each (i, j, l): compress the b×b block
                // C'_block[i][j] (the (i, j) block of the full n×n result)
                // using the compression vector.

                field::Element acc = 0;
                for (uint32_t bi = 0; bi < b; bi++) {
                    for (uint32_t bj = 0; bj < b; bj++) {
                        uint32_t row = i * b + bi;
                        uint32_t col = j * b + bj;
                        field::Element cval = C_prime[row * n + col];
                        field::Element cv = compress_vec[bi * b + bj];
                        acc = field::add(acc, field::mul(cval, cv));
                    }
                }
                out_words[out_count++] = acc;
            }
        }
    }
}

// Finalize transcript: hash the compressed words with SHA-256d.
// The words are serialized as LE32 field elements before hashing.
inline void transcript_finalize(
    const field::Element* words, uint32_t count,
    uint8_t out_hash[32])
{
    // Serialize: count * 4 bytes
    uint8_t* buf = new uint8_t[count * 4];
    for (uint32_t i = 0; i < count; i++) {
        buf[i*4]   = words[i] & 0xFF;
        buf[i*4+1] = (words[i] >> 8) & 0xFF;
        buf[i*4+2] = (words[i] >> 16) & 0xFF;
        buf[i*4+3] = (words[i] >> 24) & 0xFF;
    }
    sha256::sha256d(buf, count * 4, out_hash);
    delete[] buf;
}
