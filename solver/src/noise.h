#ifndef NOISE_H
#define NOISE_H

#include "matrix.h"
#include "uint256.h"
#include <string_view>

namespace noise {

inline constexpr std::string_view TAG_EL{"matmul_noise_EL_v1"};
inline constexpr std::string_view TAG_ER{"matmul_noise_ER_v1"};
inline constexpr std::string_view TAG_FL{"matmul_noise_FL_v1"};
inline constexpr std::string_view TAG_FR{"matmul_noise_FR_v1"};

struct NoisePair {
    Matrix E_L;
    Matrix E_R;
    Matrix F_L;
    Matrix F_R;
};

Uint256 DeriveNoiseSeed(std::string_view domain_tag, const Uint256& sigma);
NoisePair Generate(const Uint256& sigma, uint32_t n, uint32_t r);

}

#endif
