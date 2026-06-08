#ifndef UINT256_H
#define UINT256_H

#include <cstdint>
#include <cstring>
#include <array>

struct Uint256 {
    uint8_t data[32] = {};

    bool operator==(const Uint256& o) const {
        return std::memcmp(data, o.data, 32) == 0;
    }
    bool operator!=(const Uint256& o) const {
        return !(*this == o);
    }
    bool IsNull() const {
        Uint256 zero;
        return *this == zero;
    }
};

inline std::array<uint8_t, 32> ToCanonicalBytes(const Uint256& value) {
    std::array<uint8_t, 32> out;
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = value.data[out.size() - 1 - i];
    return out;
}

inline Uint256 BytesToUint256(const uint8_t* bytes) {
    Uint256 out;
    for (size_t i = 0; i < 32; ++i)
        out.data[i] = bytes[31 - i];
    return out;
}

#endif
