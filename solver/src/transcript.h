#ifndef TRANSCRIPT_H
#define TRANSCRIPT_H

#include "matrix.h"
#include "noise.h"
#include "uint256.h"
#include <vector>
#include <string_view>

namespace transcript {

inline constexpr std::string_view COMPRESS_TAG{"matmul-compress-v1"};
inline constexpr std::string_view PRODUCT_DIGEST_TAG{"matmul-product-digest-v3"};

std::vector<field::Element> DeriveCompressionVector(const Uint256& sigma, uint32_t b);
field::Element CompressBlock(const Matrix& block_bb, const std::vector<field::Element>& v);

Uint256 ComputeProductCommittedDigestFromPerturbed(const Matrix& A_prime, const Matrix& B_prime, uint32_t b, const Uint256& sigma);

Uint256 ComputeTranscriptDigestFromPerturbed(const Matrix& A_prime, const Matrix& B_prime, uint32_t b, const Uint256& sigma);

Uint256 HashMatrixWords(const field::Element* words, size_t count);

Uint256 FinalizeProductCommittedDigest(const Uint256& c_prime_hash, const Uint256& sigma, uint32_t dim, uint32_t b);

}

#endif
