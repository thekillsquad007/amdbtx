#pragma once
#include <cstdint>
#include <cstring>

// Mersenne prime M31: q = 2^31 - 1 = 2147483647
// GPU-friendly uint32 arithmetic with double-Mersenne-fold reduction.

namespace field {

using Element = uint32_t;

constexpr Element Q = 2147483647u; // 2^31 - 1

// Double Mersenne fold — safe for all uint64 inputs.
// Single-fold variant is buggy for x >= 2^62.
inline __host__ __device__ Element reduce64(uint64_t x) {
    x = (x & 0x7FFFFFFF) + (x >> 31);
    x = (x & 0x7FFFFFFF) + (x >> 31);
    if (x >= Q) x -= Q;
    return static_cast<Element>(x);
}

inline __host__ __device__ Element add(Element a, Element b) {
    uint32_t r = a + b;
    if (r >= Q) r -= Q;
    return r;
}

inline __host__ __device__ Element sub(Element a, Element b) {
    uint32_t r = a - b;
    r += (a < b) ? Q : 0;
    return r;
}

inline __host__ __device__ Element mul(Element a, Element b) {
    return reduce64(static_cast<uint64_t>(a) * b);
}

// Dot product of two vectors with periodic reduction.
inline Element dot(const Element* a, const Element* b, uint32_t len) {
    uint64_t acc = 0;
    for (uint32_t i = 0; i < len; i++) {
        acc += static_cast<uint64_t>(a[i]) * b[i];
        if ((i & 0x3F) == 0x3F) {
            acc = reduce64(acc);
        }
    }
    return reduce64(acc);
}

} // namespace field
