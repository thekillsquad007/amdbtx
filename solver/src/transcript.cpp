#include "transcript.h"
#include "sha256.h"
#include <cstring>
#include <stdexcept>

namespace transcript {

static Uint256 DeriveCompressionSeed(const Uint256& sigma) {
    auto sigma_bytes = ToCanonicalBytes(sigma);
    CSHA256 hasher;
    hasher.Write(reinterpret_cast<const uint8_t*>(COMPRESS_TAG.data()), COMPRESS_TAG.size());
    hasher.Write(sigma_bytes.data(), sigma_bytes.size());
    uint8_t digest[32]; hasher.Finalize(digest);
    return BytesToUint256(digest);
}

std::vector<field::Element> DeriveCompressionVector(const Uint256& sigma, uint32_t b) {
    Uint256 seed = DeriveCompressionSeed(sigma);
    uint64_t len = uint64_t(b) * b;
    std::vector<field::Element> vec;
    vec.reserve(len);
    for (uint64_t k = 0; k < len; ++k)
        vec.push_back(field::from_oracle(seed, static_cast<uint32_t>(k)));
    return vec;
}

field::Element CompressBlock(const Matrix& block_bb, const std::vector<field::Element>& v) {
    uint64_t len = uint64_t(block_bb.rows()) * block_bb.cols();
    return field::dot(block_bb.data(), v.data(), static_cast<uint32_t>(len));
}

Uint256 HashMatrixWords(const field::Element* words, size_t count) {
    CSHA256 hasher;
    for (size_t i = 0; i < count; ++i) {
        uint8_t buf[4]; WriteLE32(buf, words[i]);
        hasher.Write(buf, 4);
    }
    uint8_t inner[32]; hasher.Finalize(inner);
    uint8_t dbl[32]; CSHA256().Write(inner, 32).Finalize(dbl);
    return BytesToUint256(dbl);
}

Uint256 FinalizeProductCommittedDigest(const Uint256& c_prime_hash, const Uint256& sigma, uint32_t dim, uint32_t b) {
    CSHA256 outer;
    auto sigma_bytes = ToCanonicalBytes(sigma);
    outer.Write(reinterpret_cast<const uint8_t*>(PRODUCT_DIGEST_TAG.data()), PRODUCT_DIGEST_TAG.size());
    outer.Write(sigma_bytes.data(), 32);
    outer.Write(c_prime_hash.data, 32);
    uint8_t dim_buf[4]; WriteLE32(dim_buf, dim); outer.Write(dim_buf, 4);
    uint8_t block_buf[4]; WriteLE32(block_buf, b); outer.Write(block_buf, 4);
    uint8_t inner[32]; outer.Finalize(inner);
    uint8_t result[32]; CSHA256().Write(inner, 32).Finalize(result);
    return BytesToUint256(result);
}

Uint256 ComputeProductCommittedDigestFromPerturbed(const Matrix& A_prime, const Matrix& B_prime, uint32_t b, const Uint256& sigma) {
    uint32_t n = A_prime.rows();
    uint32_t blocks_per_axis = n / b;
    auto compress_vec = DeriveCompressionVector(sigma, b);
    std::vector<field::Element> compressed_blocks;
    compressed_blocks.reserve(static_cast<size_t>(blocks_per_axis) * blocks_per_axis);
    for (uint32_t i = 0; i < blocks_per_axis; ++i) {
        for (uint32_t j = 0; j < blocks_per_axis; ++j) {
            field::Element compressed_acc = 0;
            for (uint32_t ell = 0; ell < blocks_per_axis; ++ell) {
                Matrix product = A_prime.block(i, ell, b) * B_prime.block(ell, j, b);
                compressed_acc = field::add(compressed_acc, CompressBlock(product, compress_vec));
            }
            compressed_blocks.push_back(compressed_acc);
        }
    }
    Uint256 c_prime_hash = HashMatrixWords(compressed_blocks.data(), compressed_blocks.size());
    return FinalizeProductCommittedDigest(c_prime_hash, sigma, n, b);
}

}
