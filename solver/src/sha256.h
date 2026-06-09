#ifndef SHA256_H
#define SHA256_H

#include <cstdint>
#include <cstddef>

class CSHA256 {
public:
    static constexpr size_t OUTPUT_SIZE = 32;
    CSHA256();
    CSHA256& Write(const uint8_t* data, size_t len);
    void Finalize(uint8_t hash[OUTPUT_SIZE]);
    CSHA256& Reset();
private:
    uint32_t s[8];
    uint8_t buf[64];
    uint64_t bytes;
};

class CHash256 {
public:
    CHash256& Write(const uint8_t* data, size_t len) {
        outer.Write(data, len);
        return *this;
    }
    void Finalize(uint8_t hash[CSHA256::OUTPUT_SIZE]) {
        uint8_t tmp[CSHA256::OUTPUT_SIZE];
        outer.Finalize(tmp);
        CSHA256().Write(tmp, CSHA256::OUTPUT_SIZE).Finalize(hash);
    }
private:
    CSHA256 outer;
};

inline void WriteLE16(uint8_t* ptr, uint16_t val) {
    ptr[0] = uint8_t(val); ptr[1] = uint8_t(val >> 8);
}

inline void WriteLE32(uint8_t* ptr, uint32_t val) {
    ptr[0] = uint8_t(val); ptr[1] = uint8_t(val >> 8);
    ptr[2] = uint8_t(val >> 16); ptr[3] = uint8_t(val >> 24);
}

inline void WriteLE64(uint8_t* ptr, uint64_t val) {
    for (int i = 0; i < 8; ++i) ptr[i] = uint8_t(val >> (i * 8));
}

inline uint32_t ReadLE32(const uint8_t* ptr) {
    return uint32_t(ptr[0]) | (uint32_t(ptr[1]) << 8) |
           (uint32_t(ptr[2]) << 16) | (uint32_t(ptr[3]) << 24);
}

#endif
