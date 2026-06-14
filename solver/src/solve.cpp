#include "solve.h"
#include "sha256.h"
#include "uint256.h"
#include <chrono>
#include <cstring>
#include <array>
#include <string>
#include <sstream>
#include <iomanip>

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

Uint256 DeterministicMatMulSeedV3(
    const Uint256& prev_hash, int64_t parent_mtp, int32_t height,
    int32_t version, const Uint256& merkle_root,
    uint32_t time, uint32_t bits,
    uint64_t nonce64, uint16_t matmul_dim,
    uint8_t which)
{
    CSHA256 hasher;
    const std::string tag = "BTX_MATMUL_SEED_V3";
    WriteCompactSize(hasher, tag.size());
    hasher.Write(reinterpret_cast<const uint8_t*>(tag.data()), tag.size());
    hasher.Write(prev_hash.data, 32);
    uint8_t value_le[8];
    WriteLE64(value_le, static_cast<uint64_t>(parent_mtp));
    hasher.Write(value_le, 8);
    WriteLE32(value_le, static_cast<uint32_t>(height));
    hasher.Write(value_le, 4);
    WriteLE32(value_le, static_cast<uint32_t>(version));
    hasher.Write(value_le, 4);
    hasher.Write(merkle_root.data, 32);
    WriteLE32(value_le, time);
    hasher.Write(value_le, 4);
    WriteLE32(value_le, bits);
    hasher.Write(value_le, 4);
    WriteLE64(value_le, nonce64);
    hasher.Write(value_le, 8);
    uint8_t dim_le[2];
    WriteLE16(dim_le, matmul_dim);
    hasher.Write(dim_le, 2);
    hasher.Write(&which, 1);
    uint8_t digest[32];
    hasher.Finalize(digest);
    return BytesToUint256(digest);
}

void Uint256ToMsbBytes(const Uint256& v, uint8_t out[32]) {
    for (int i = 0; i < 32; ++i)
        out[i] = v.data[31 - i];
}

Uint256 DerivePreHashTarget(const Uint256& target, uint32_t epsilon_bits) {
    if (epsilon_bits == 0)
        return target;
    if (epsilon_bits >= 256) {
        Uint256 zero;
        return zero;
    }

    // Match BTX arith_uint256: data[0]=LSB, operator<<= on the full 256-bit integer.
    uint64_t limb[4] = {};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j)
            limb[i] |= uint64_t(target.data[i * 8 + j]) << (8 * j);
    }

    const uint32_t word_shift = epsilon_bits / 64;
    const uint32_t bit_shift = epsilon_bits % 64;
    uint64_t out[4] = {};
    for (int i = 0; i < 4; ++i) {
        const int src = i - static_cast<int>(word_shift);
        if (src < 0)
            continue;
        out[i] = limb[src] << bit_shift;
        if (bit_shift > 0 && src > 0)
            out[i] |= limb[src - 1] >> (64 - bit_shift);
    }

    Uint256 result;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j)
            result.data[i * 8 + j] = uint8_t((out[i] >> (8 * j)) & 0xFF);
    }
    return result;
}

bool Uint256LE(const Uint256& a, const Uint256& b) {
    for (int i = 31; i >= 0; --i) {
        if (a.data[i] < b.data[i]) return true;
        if (a.data[i] > b.data[i]) return false;
    }
    return true;
}

// Sigma from SHA-256 uses BytesToUint256 (data[0]=hash[0]=MSB). Pool targets use
// HexToUint256 (data[31]=MSB). Compare both in MSB-first byte order for the ε gate.
static bool SigmaLE(const Uint256& sigma_sha, const Uint256& target_hex) {
    for (int i = 0; i < 32; ++i) {
        uint8_t s = sigma_sha.data[i];
        uint8_t t = target_hex.data[31 - i];
        if (s < t) return true;
        if (s > t) return false;
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

std::string Uint256ToHex(const Uint256& v) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 31; i >= 0; --i)
        oss << std::setw(2) << static_cast<unsigned>(v.data[i]);
    return oss.str();
}

void PrepareNonceSeeds(PowState& state) {
    if (state.height >= 130500) {
        if (!state.has_parent_mtp) {
            state.seed_a = Uint256{};
            state.seed_b = Uint256{};
            return;
        }
        state.seed_a = DeterministicMatMulSeedV3(
            state.previous_block_hash, state.parent_mtp, state.height,
            state.version, state.merkle_root,
            state.time, state.bits,
            state.nonce, state.matmul_dim, 0);
        state.seed_b = DeterministicMatMulSeedV3(
            state.previous_block_hash, state.parent_mtp, state.height,
            state.version, state.merkle_root,
            state.time, state.bits,
            state.nonce, state.matmul_dim, 1);
    } else if (state.height >= 125000) {
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
    }
}

static void BuildPerturbedMatrices(
    PowState& state, uint32_t n, uint32_t b, uint32_t r,
    Matrix& a_prime, Matrix& b_prime, Uint256& sigma)
{
    Matrix a_copy(n, n), b_copy(n, n);
    PrepareNonceSeeds(state);
    if (state.height >= 125000) {
        a_copy = FromSeed(state.seed_a, n);
        b_copy = FromSeed(state.seed_b, n);
    } else {
        a_copy = FromSeed(state.seed_a, n);
        b_copy = FromSeed(state.seed_b, n);
    }

    sigma = DeriveSigma(state);
    noise::NoisePair np = noise::Generate(sigma, n, r);
    const Matrix e = np.E_L * np.E_R;
    const Matrix f = np.F_L * np.F_R;
    a_prime = a_copy + e;
    b_prime = b_copy + f;
}

uint32_t CountCompressedWordDiffs(
    PowState& state, uint32_t n, uint32_t b, uint32_t r,
    const field::Element* gpu_words, size_t word_count)
{
    Matrix a_prime(n, n), b_prime(n, n);
    Uint256 sigma;
    BuildPerturbedMatrices(state, n, b, r, a_prime, b_prime, sigma);

    const uint32_t blocks_per_axis = n / b;
    const size_t expected = size_t(blocks_per_axis) * blocks_per_axis;
    if (word_count != expected) return static_cast<uint32_t>(expected);

    auto compress_vec = transcript::DeriveCompressionVector(sigma, b);
    uint32_t diffs = 0;
    size_t idx = 0;
    for (uint32_t i = 0; i < blocks_per_axis; ++i) {
        for (uint32_t j = 0; j < blocks_per_axis; ++j) {
            field::Element compressed_acc = 0;
            for (uint32_t ell = 0; ell < blocks_per_axis; ++ell) {
                Matrix product = a_prime.block(i, ell, b) * b_prime.block(ell, j, b);
                compressed_acc = field::add(
                    compressed_acc, transcript::CompressBlock(product, compress_vec));
            }
            if (compressed_acc != gpu_words[idx]) ++diffs;
            ++idx;
        }
    }
    return diffs;
}

bool ComputeDigestForNonce(PowState& state, uint32_t n, uint32_t b, uint32_t r, Uint256& out_digest) {
    Matrix a_prime(n, n), b_prime(n, n);
    Uint256 sigma;
    BuildPerturbedMatrices(state, n, b, r, a_prime, b_prime, sigma);
    out_digest = transcript::ComputeProductCommittedDigestFromPerturbed(a_prime, b_prime, b, sigma);
    return true;
}

bool ComputeProductPayloadForNonce(
    PowState& state, uint32_t n, uint32_t b, uint32_t r,
    std::vector<field::Element>& out_product)
{
    Matrix a_prime(n, n), b_prime(n, n);
    Uint256 sigma;
    BuildPerturbedMatrices(state, n, b, r, a_prime, b_prime, sigma);
    const Matrix product = a_prime * b_prime;
    out_product.assign(product.data(), product.data() + product.size());
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
    const bool use_nonce_seeds = (state.height >= 125000);
    const Uint256 pre_hash_target = DerivePreHashTarget(block_target, epsilon_bits);

    if (state.height >= 130500 && !state.has_parent_mtp) {
        elapsed_s = 0;
        return false;
    }

    while (max_tries > 0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (max_seconds > 0 && elapsed >= max_seconds) {
            elapsed_s = elapsed;
            return false;
        }

        if (use_nonce_seeds) {
            PrepareNonceSeeds(state);
        }
        if (epsilon_bits > 0) {
            const Uint256 sigma = DeriveSigma(state);
            if (!SigmaLE(sigma, pre_hash_target)) {
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
