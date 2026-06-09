#include "field.h"
#include "sha256.h"
#include <cassert>

namespace field {

static constexpr uint32_t REDUCE_INTERVAL = 4;

Element dot(const Element* a, const Element* b, uint32_t len) {
    uint64_t acc = 0;
    uint32_t pending = 0;
    for (uint32_t i = 0; i < len; ++i) {
        acc += uint64_t(a[i]) * b[i];
        if (++pending == REDUCE_INTERVAL) {
            acc = reduce64(acc);
            pending = 0;
        }
    }
    return reduce64(acc);
}

Element inv(Element a) {
    assert(a != 0);
    uint32_t exp = MODULUS - 2;
    Element result = 1;
    Element base = a;
    while (exp > 0) {
        if ((exp & 1U) != 0) result = mul(result, base);
        exp >>= 1;
        if (exp > 0) base = mul(base, base);
    }
    return result;
}

// Matches BTX::FromOracle: SHA-256(ToCanonicalBytes(seed) + index [+ retry]) per retry.
Element from_oracle(const Uint256& seed, uint32_t index) {
    auto seed_bytes = ToCanonicalBytes(seed);
    for (uint32_t retry = 0; retry < 256; ++retry) {
        CSHA256 hasher;
        hasher.Write(seed_bytes.data(), seed_bytes.size());
        uint8_t idx_le[4]; WriteLE32(idx_le, index);
        hasher.Write(idx_le, 4);
        if (retry > 0) {
            uint8_t retry_le[4]; WriteLE32(retry_le, retry);
            hasher.Write(retry_le, 4);
        }
        uint8_t hash[32]; hasher.Finalize(hash);
        uint32_t candidate = ReadLE32(hash) & MODULUS;
        if (candidate < MODULUS) return candidate;
    }
    CSHA256 hasher;
    hasher.Write(seed_bytes.data(), seed_bytes.size());
    uint8_t idx_le[4]; WriteLE32(idx_le, index);
    hasher.Write(idx_le, 4);
    const uint8_t fallback_tag[] = "oracle-fallback";
    hasher.Write(fallback_tag, sizeof(fallback_tag) - 1);
    uint8_t hash[32]; hasher.Finalize(hash);
    return ReadLE32(hash) % MODULUS;
}

}