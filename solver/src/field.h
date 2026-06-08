#ifndef FIELD_H
#define FIELD_H

#include <cstdint>
#include <cassert>
#include "uint256.h"

namespace field {
using Element = uint32_t;
constexpr Element MODULUS = 0x7FFFFFFFU;

inline Element reduce64(uint64_t x) {
    uint64_t fold1 = (x & uint64_t(MODULUS)) + (x >> 31);
    uint32_t lo = uint32_t(fold1 & MODULUS);
    uint32_t hi = uint32_t(fold1 >> 31);
    uint32_t result = lo + hi;
    uint32_t ge_mask = uint32_t(-int32_t(result >= MODULUS));
    result -= (MODULUS & ge_mask);
    return result;
}

inline Element add(Element a, Element b) {
    uint32_t s = a + b;
    if (s >= MODULUS) s -= MODULUS;
    return s;
}

inline Element sub(Element a, Element b) {
    if (a >= b) return a - b;
    return a + MODULUS - b;
}

inline Element mul(Element a, Element b) {
    return reduce64(uint64_t(a) * b);
}

inline Element neg(Element a) {
    if (a == 0) return 0;
    return MODULUS - a;
}

inline Element from_uint32(uint32_t x) {
    return reduce64(x);
}

Element inv(Element a);
Element dot(const Element* a, const Element* b, uint32_t len);
Element from_oracle(const Uint256& seed, uint32_t index);
}

using Element = field::Element;
constexpr Element MODULUS = field::MODULUS;

#endif
