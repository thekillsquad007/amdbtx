#include "solve.h"
#include "sha256.h"
#include <chrono>
#include <cstring>
#include <thread>

bool Uint256LE(const Uint256& a, const Uint256& b) {
    for (int i = 0; i < 32; ++i) {
        if (a.data[i] < b.data[i]) return true;
        if (a.data[i] > b.data[i]) return false;
    }
    return true;
}

Uint256 DeriveSigma(const PowState& state) {
    auto prev_bytes = ToCanonicalBytes(state.previous_block_hash);
    auto merk_bytes = ToCanonicalBytes(state.merkle_root);
    auto seed_a_bytes = ToCanonicalBytes(state.seed_a);
    auto seed_b_bytes = ToCanonicalBytes(state.seed_b);

    CSHA256 hasher;
    uint8_t version_le[4];
    WriteLE32(version_le, static_cast<uint32_t>(state.version));
    uint8_t time_le[4];
    WriteLE32(time_le, state.time);
    uint8_t bits_le[4];
    WriteLE32(bits_le, state.bits);
    uint8_t nonce_le[8];
    for (int i = 0; i < 8; ++i) nonce_le[i] = uint8_t(state.nonce >> (i * 8));
    uint8_t dim_le[2];
    dim_le[0] = uint8_t(state.matmul_dim & 0xFF);
    dim_le[1] = uint8_t(state.matmul_dim >> 8);

    hasher.Write(version_le, 4);
    hasher.Write(prev_bytes.data(), 32);
    hasher.Write(merk_bytes.data(), 32);
    hasher.Write(time_le, 4);
    hasher.Write(bits_le, 4);
    hasher.Write(nonce_le, 8);
    hasher.Write(dim_le, 2);
    hasher.Write(seed_a_bytes.data(), 32);
    hasher.Write(seed_b_bytes.data(), 32);

    uint8_t header_hash[32];
    hasher.Finalize(header_hash);
    uint8_t sigma_bytes[32];
    CSHA256().Write(header_hash, 32).Finalize(sigma_bytes);
    return BytesToUint256(sigma_bytes);
}

Uint256 DeriveBlockTarget(uint32_t bits) {
    uint32_t exponent = bits >> 24;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (bits & 0x800000) mantissa |= 0x800000;

    Uint256 target;
    std::memset(target.data, 0, 32);

    if (exponent <= 3) {
        mantissa >>= 8 * (3 - exponent);
        for (int i = 0; i < 4; ++i)
            target.data[i] = uint8_t(mantissa >> (i * 8));
    } else {
        int shift = static_cast<int>(exponent) - 3;
        int byte_pos = shift / 8;
        int bit_off = shift % 8;
        uint8_t mbytes[3];
        mbytes[0] = uint8_t(mantissa >> 16);
        mbytes[1] = uint8_t(mantissa >> 8);
        mbytes[2] = uint8_t(mantissa);
        if (byte_pos < 32) {
            for (int i = 0; i < 3 && byte_pos + i < 32; ++i) {
                target.data[byte_pos + i] |= mbytes[i] << bit_off;
                if (bit_off > 0 && byte_pos + i + 1 < 32)
                    target.data[byte_pos + i + 1] |= mbytes[i] >> (8 - bit_off);
            }
        }
    }
    return target;
}

static Matrix FromSeed(const Uint256& seed, uint32_t n) {
    Matrix out(n, n);
    for (uint32_t row = 0; row < n; ++row)
        for (uint32_t col = 0; col < n; ++col)
            out.at(row, col) = field::from_oracle(seed, row * n + col);
    return out;
}

bool SolveCPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
              const Uint256& block_target, const Uint256& share_target,
              uint64_t& max_tries, double max_seconds,
              uint64_t& tries_used, double& elapsed_s)
{
    auto t0 = std::chrono::steady_clock::now();
    tries_used = 0;

    Matrix A = FromSeed(state.seed_a, n);
    Matrix B = FromSeed(state.seed_b, n);

    while (max_tries > 0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (max_seconds > 0 && elapsed >= max_seconds) {
            elapsed_s = elapsed;
            return false;
        }

        Uint256 sigma = DeriveSigma(state);
        noise::NoisePair np = noise::Generate(sigma, n, r);
        Matrix A_prime = A + np.E_L * np.E_R;
        Matrix B_prime = B + np.F_L * np.F_R;
        Uint256 digest = transcript::ComputeProductCommittedDigestFromPerturbed(A_prime, B_prime, b, sigma);

        --max_tries;
        ++tries_used;

        bool is_block = Uint256LE(digest, block_target);
        if (is_block || Uint256LE(digest, share_target)) {
            state.digest = digest;
            auto t1 = std::chrono::steady_clock::now();
            elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            return true;
        }

        if (state.nonce == UINT64_MAX) break;
        ++state.nonce;
    }

    auto t1 = std::chrono::steady_clock::now();
    elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    return false;
}
