#include "sha256.h"

#include <cstring>

static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
static inline uint32_t sigma0(uint32_t x) { return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22); }
static inline uint32_t sigma1(uint32_t x) { return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25); }
static inline uint32_t lsigma0(uint32_t x) { return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }
static inline uint32_t lsigma1(uint32_t x) { return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }

CSHA256::CSHA256() { Reset(); }

CSHA256& CSHA256::Reset() {
    s[0]=0x6a09e667; s[1]=0xbb67ae85; s[2]=0x3c6ef372; s[3]=0xa54ff53a;
    s[4]=0x510e527f; s[5]=0x9b05688c; s[6]=0x1f83d9ab; s[7]=0x5be0cd19;
    bytes = 0;
    std::memset(buf, 0, sizeof(buf));
    return *this;
}

static void sha256_round(uint32_t* s, const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t(block[i*4])<<24) | (uint32_t(block[i*4+1])<<16) |
               (uint32_t(block[i*4+2])<<8) | uint32_t(block[i*4+3]);
    for (int i = 16; i < 64; ++i)
        w[i] = lsigma1(w[i-2]) + w[i-7] + lsigma0(w[i-15]) + w[i-16];

    uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + sigma1(e) + ch(e,f,g) + K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

CSHA256& CSHA256::Write(const uint8_t* data, size_t len) {
    size_t bufsize = bytes % 64;
    bytes += len;
    if (bufsize) {
        size_t need = 64 - bufsize;
        if (len < need) {
            std::memcpy(buf + bufsize, data, len);
            return *this;
        }
        std::memcpy(buf + bufsize, data, need);
        sha256_round(s, buf);
        data += need;
        len -= need;
    }
    while (len >= 64) {
        sha256_round(s, data);
        data += 64;
        len -= 64;
    }
    if (len) std::memcpy(buf, data, len);
    return *this;
}

void CSHA256::Finalize(uint8_t hash[OUTPUT_SIZE]) {
    size_t bufsize = bytes % 64;
    uint64_t totalbits = bytes * 8;
    buf[bufsize++] = 0x80;
    if (bufsize > 56) {
        std::memset(buf + bufsize, 0, 64 - bufsize);
        sha256_round(s, buf);
        bufsize = 0;
    }
    std::memset(buf + bufsize, 0, 56 - bufsize);
    for (int i = 0; i < 8; ++i)
        buf[63-i] = uint8_t(totalbits >> (i*8));
    sha256_round(s, buf);
    for (int i = 0; i < 8; ++i) {
        hash[i*4]   = uint8_t(s[i] >> 24);
        hash[i*4+1] = uint8_t(s[i] >> 16);
        hash[i*4+2] = uint8_t(s[i] >> 8);
        hash[i*4+3] = uint8_t(s[i]);
    }
}
