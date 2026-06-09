#include "solve.h"
#include "sha256.h"
#include "uint256.h"
#include <chrono>
#include <cstring>
#include <array>
#include <string>

// CompactSize serialization for HashWriter-compatible SHA-256
static void WriteCompactSize(CSHA256& hasher, uint64_t val) {
    if (val < 253) {
        uint8_t c = uint8_t(val);
        hasher.Write(&c, 1);
    } else if (val < 0x10000) {
        uint8_t buf[3] = {0xFD, uint8_t(val), uint8_t(val >> 8)};
        hasher.Write(buf, 3);
    } else if (val < 0x100000000ULL) {
        uint8_t buf[5] = {0xFE, uint8_t(val), uint8_t(val >> 8), uint8_t(val >> 16), uint8_t(val >> 24)};
        hasher.Write(buf, 5);
    } else {
        uint8_t buf[9] = {0xFF, uint8_t(val), uint8_t(val >> 8), uint8_t(val >> 16), uint8_t(val >> 24),
                          uint8_t(val >> 32), uint8_t(val >> 40), uint8_t(val >> 48), uint8_t(val >> 56)};
        hasher.Write(buf, 9);
    }
}

// BTX v0.32.3 DeterministicMatMulSeedV2:
// SHA-256(CompactSize("BTX_MATMUL_SEED_V2") || "BTX_MATMUL_SEED_V2" ||
//         prev_hash(32 MSB-first) || height(4 LE) || version(4 LE) ||
//         merkle_root(32 MSB-first) || time(4 LE) || bits(4 LE) ||
//         nonce64(8 LE) || matmul_dim(2 LE) || which(1))
// NOTE: The order is: prev_hash, height, version (matching BTX HashWriter serialization
// where strings include CompactSize prefix, then uint's are serialized in order of
// field declaration in the serialization call).
Uint256 DeterministicMatMulSeedV2(
    const Uint256& prev_hash, int32_t height,
    int32_t version, const Uint256& merkle_root,
    uint32_t time, uint32_t bits,
    uint64_t nonce64, uint16_t matmul_dim,
    uint8_t which)
{
    CSHA256 hasher;
    const std::string tag = "BTX_MATMUL_SEED_V2";
    WriteCompactSize(hasher, tag.size());
    hasher.Write(reinterpret_cast<const uint8_t*>(tag.data()), tag.size());
    hasher.Write(prev_hash.data, 32);
    uint8_t h_le[4]; WriteLE32(h_le, static_cast<uint32_t>(height));
    hasher.Write(h_le, 4);
    WriteLE32(h_le, static_cast<uint32_t>(version));
    hasher.Write(h_le, 4);
    hasher.Write(merkle_root.data, 32);
    WriteLE32(h_le, time);
    hasher.Write(h_le, 4);
    WriteLE32(h_le, bits);
    hasher.Write(h_le, 4);
    uint8_t n64_le[8]; WriteLE64(n64_le, nonce64);
    hasher.Write(n64_le, 8);
    uint8_t dim_le[2]; WriteLE16(dim_le, matmul_dim);
    hasher.Write(dim_le, 2);
    hasher.Write(&which, 1);
    uint8_t digest[32]; hasher.Finalize(digest);
    return BytesToUint256(digest);
}

void Uint256ToMsbBytes(const Uint256& v, uint8_t out[32]) {
    for (int i = 0; i < 32; ++i)
        out[i] = v.data[31 - i];
}

static void ShiftLeftMsbBytesInPlace(uint8_t be[32], uint32_t bits) {
    for (uint32_t s = 0; s < bits; ++s) {
        uint8_t overflow = 0;
        for (int i = 0; i < 32; ++i) {
            uint8_t new_overflow = (be[i] & 0x80) ? 1 : 0;
            be[i] = static_cast<uint8_t>((be[i] << 1) | overflow);
            overflow = new_overflow;
        }
        if (overflow) {
            std::memset(be, 0xFF, 32);
            return;
        }
    }
}

Uint256 DerivePreHashTarget(const Uint256& target, uint32_t epsilon_bits) {
    uint8_t be[32];
    Uint256ToMsbBytes(target, be);
    if (epsilon_bits > 0)
        ShiftLeftMsbBytesInPlace(be, epsilon_bits);
    Uint256 out;
    for (int i = 0; i < 32; ++i)
        out.data[31 - i] = be[i];
    return out;
}

bool Uint256LE(const Uint256& a, const Uint256& b) {
    for (int i = 31; i >= 0; --i) {
        if (a.data[i] < b.data[i]) return true;
        if (a.data[i] > b.data[i]) return false;
    }
    return true;
}

Uint256 DeriveSigma(const PowState& state) {
    // Match BTX's ComputeMatMulHeaderHash + DeriveSigma:
    // SHA256d(version(4LE) || prev_hash(32LE) || merkle_root(32LE) ||
    //         time(4LE) || bits(4LE) || nonce(8LE) || dim(2LE) ||
    //         seed_a(32LE) || seed_b(32LE))
    CSHA256 hasher;
    uint8_t version_le[4];
    WriteLE32(version_le, static_cast<uint32_t>(state.version));
    uint8_t time_le[4];
    WriteLE32(time_le, state.time);
    uint8_t bits_le[4];
    WriteLE32(bits_le, state.bits);
    uint8_t nonce_le[8];
    WriteLE64(nonce_le, state.nonce);
    uint8_t dim_le[2];
    WriteLE16(dim_le, state.matmul_dim);

    hasher.Write(version_le, 4);
    hasher.Write(state.previous_block_hash.data, 32);
    hasher.Write(state.merkle_root.data, 32);
    hasher.Write(time_le, 4);
    hasher.Write(bits_le, 4);
    hasher.Write(nonce_le, 8);
    hasher.Write(dim_le, 2);
    hasher.Write(state.seed_a.data, 32);
    hasher.Write(state.seed_b.data, 32);

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

    // BTX SetCompact: LSB-first storage (data[0]=LSB)
    // For n=3 bytes: data[0]=low, data[2]=high. Then <<= 8*(exponent-3).
    Uint256 target;
    std::memset(target.data, 0, 32);

    if (exponent <= 3) {
        mantissa >>= 8 * (3 - exponent);
        for (int i = 0; i < 4; ++i)
            target.data[i] = uint8_t(mantissa >> (i * 8));
    } else {
        int shift = static_cast<int>(exponent) - 3;
        int byte_pos = shift;
        int bit_off = 0;
        uint8_t mbytes[3];
        mbytes[0] = uint8_t(mantissa);       // low byte → data[byte_pos]
        mbytes[1] = uint8_t(mantissa >> 8);   // mid byte → data[byte_pos+1]
        mbytes[2] = uint8_t(mantissa >> 16);  // high byte → data[byte_pos+2]
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

bool ComputeDigestForNonce(PowState& state, uint32_t n, uint32_t b, uint32_t r, Uint256& out_digest) {
    const bool use_v2 = (state.height >= 125000);
    Matrix A_copy(n, n), B_copy(n, n);
    if (use_v2) {
        state.seed_a = DeterministicMatMulSeedV2(
            state.previous_block_hash, state.height,
            state.version, state.merkle_root,
            state.time, state.bits,
            state.nonce, state.matmul_dim, 0);
        state.seed_b = DeterministicMatMulSeedV2(
            state.previous_block_hash, state.height,
            state.version, state.merkle_root,
            state.time, state.bits,
            state.nonce, state.matmul_dim, 1);
        A_copy = FromSeed(state.seed_a, n);
        B_copy = FromSeed(state.seed_b, n);
    } else {
        A_copy = FromSeed(state.seed_a, n);
        B_copy = FromSeed(state.seed_b, n);
    }

    Uint256 sigma = DeriveSigma(state);
    noise::NoisePair np = noise::Generate(sigma, n, r);
    const Matrix E = np.E_L * np.E_R;
    const Matrix F = np.F_L * np.F_R;
    const Matrix A_prime = A_copy + E;
    const Matrix B_prime = B_copy + F;
    out_digest = transcript::ComputeProductCommittedDigestFromPerturbed(A_prime, B_prime, b, sigma);
    return true;
}

bool SolveCPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
               const Uint256& block_target, const Uint256& share_target,
               uint64_t& max_tries, double max_seconds,
               uint64_t& tries_used, double& elapsed_s,
               uint32_t epsilon_bits)
{
    auto t0 = std::chrono::steady_clock::now();
    tries_used = 0;
    const Uint256 pre_hash_target = DerivePreHashTarget(share_target, epsilon_bits);

    while (max_tries > 0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (max_seconds > 0 && elapsed >= max_seconds) {
            elapsed_s = elapsed;
            return false;
        }

        if (epsilon_bits > 0) {
            const Uint256 sigma = DeriveSigma(state);
            if (!Uint256LE(sigma, pre_hash_target)) {
                --max_tries;
                ++tries_used;
                if (state.nonce == UINT64_MAX) break;
                ++state.nonce;
                continue;
            }
        }

        Uint256 digest;
        ComputeDigestForNonce(state, n, b, r, digest);

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
