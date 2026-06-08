#include "noise.h"
#include "sha256.h"

namespace noise {

Uint256 DeriveNoiseSeed(std::string_view domain_tag, const Uint256& sigma) {
    auto sigma_bytes = ToCanonicalBytes(sigma);
    CSHA256 hasher;
    hasher.Write(reinterpret_cast<const uint8_t*>(domain_tag.data()), domain_tag.size());
    hasher.Write(sigma_bytes.data(), sigma_bytes.size());
    uint8_t digest[32]; hasher.Finalize(digest);
    return BytesToUint256(digest);
}

static Matrix FromSeedRect(const Uint256& seed, uint32_t rows, uint32_t cols) {
    Matrix out(rows, cols);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t col = 0; col < cols; ++col)
            out.at(row, col) = field::from_oracle(seed, row * cols + col);
    return out;
}

NoisePair Generate(const Uint256& sigma, uint32_t n, uint32_t r) {
    Uint256 tag_el = DeriveNoiseSeed(TAG_EL, sigma);
    Uint256 tag_er = DeriveNoiseSeed(TAG_ER, sigma);
    Uint256 tag_fl = DeriveNoiseSeed(TAG_FL, sigma);
    Uint256 tag_fr = DeriveNoiseSeed(TAG_FR, sigma);
    return {
        FromSeedRect(tag_el, n, r),
        FromSeedRect(tag_er, r, n),
        FromSeedRect(tag_fl, n, r),
        FromSeedRect(tag_fr, r, n),
    };
}

}
