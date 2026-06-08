#include "gpu_sha256.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include "field.h"

namespace gpasha {

__device__ __forceinline__ uint32_t d_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __constant__ uint32_t d_K[64];

namespace {
const uint32_t h_K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

bool g_K_initialized = false;

void EnsureKInitialized() {
  if (g_K_initialized) return;
  hipMemcpyToSymbol(d_K, h_K, sizeof(h_K));
  g_K_initialized = true;
}
}

__device__ void sha256_single_block(const uint8_t* padded_block, uint8_t* hash_out) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t(padded_block[i*4])<<24) | (uint32_t(padded_block[i*4+1])<<16) |
               (uint32_t(padded_block[i*4+2])<<8) | uint32_t(padded_block[i*4+3]);
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = d_rotr(w[i-15],7) ^ d_rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = d_rotr(w[i-2],17) ^ d_rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a;
    uint32_t e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + (d_rotr(e,6) ^ d_rotr(e,11) ^ d_rotr(e,25)) +
                      ((e & f) ^ (~e & g)) + d_K[i] + w[i];
        uint32_t t2 = (d_rotr(a,2) ^ d_rotr(a,13) ^ d_rotr(a,22)) +
                      ((a & b) ^ (a & c) ^ (b & c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    a+=0x6a09e667; b+=0xbb67ae85; c+=0x3c6ef372; d+=0xa54ff53a;
    e+=0x510e527f; f+=0x9b05688c; g+=0x1f83d9ab; h+=0x5be0cd19;
    hash_out[0]=a>>24; hash_out[1]=a>>16; hash_out[2]=a>>8; hash_out[3]=a;
    hash_out[4]=b>>24; hash_out[5]=b>>16; hash_out[6]=b>>8; hash_out[7]=b;
    hash_out[8]=c>>24; hash_out[9]=c>>16; hash_out[10]=c>>8; hash_out[11]=c;
    hash_out[12]=d>>24; hash_out[13]=d>>16; hash_out[14]=d>>8; hash_out[15]=d;
    hash_out[16]=e>>24; hash_out[17]=e>>16; hash_out[18]=e>>8; hash_out[19]=e;
    hash_out[20]=f>>24; hash_out[21]=f>>16; hash_out[22]=f>>8; hash_out[23]=f;
    hash_out[24]=g>>24; hash_out[25]=g>>16; hash_out[26]=g>>8; hash_out[27]=g;
    hash_out[28]=h>>24; hash_out[29]=h>>16; hash_out[30]=h>>8; hash_out[31]=h;
}

__device__ void d_WriteLE32(uint8_t* ptr, uint32_t val) {
    ptr[0] = uint8_t(val);
    ptr[1] = uint8_t(val >> 8);
    ptr[2] = uint8_t(val >> 16);
    ptr[3] = uint8_t(val >> 24);
}

__device__ uint32_t d_ReadLE32(const uint8_t* ptr) {
    return uint32_t(ptr[0]) | (uint32_t(ptr[1]) << 8) |
           (uint32_t(ptr[2]) << 16) | (uint32_t(ptr[3]) << 24);
}

__device__ void d_sha256_pad(uint8_t* block, uint32_t data_len) {
    for (uint32_t i = data_len; i < 64; ++i) block[i] = 0;
    block[data_len] = 0x80;
    uint64_t bits = uint64_t(data_len) * 8;
    block[62] = uint8_t((bits >> 8) & 0xFF);
    block[63] = uint8_t(bits & 0xFF);
}

__device__ Element d_from_oracle(const uint8_t* seed_be, uint32_t index) {
    for (uint32_t retry = 0; retry < 256; ++retry) {
        uint8_t block[64];
        for (int i = 0; i < 64; ++i) block[i] = 0;
        for (int i = 0; i < 32; ++i) block[i] = seed_be[i];
        d_WriteLE32(block + 32, index);
        uint32_t data_len = 36;
        if (retry > 0) {
            d_WriteLE32(block + 36, retry);
            data_len = 40;
        }
        d_sha256_pad(block, data_len);
        uint8_t hash[32];
        sha256_single_block(block, hash);
        uint32_t candidate = d_ReadLE32(hash) & MODULUS;
        if (candidate < MODULUS) return candidate;
    }
    uint8_t block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = seed_be[i];
    d_WriteLE32(block + 32, index);
    const char fallback_tag[] = "oracle-fallback";
    for (int i = 0; fallback_tag[i]; ++i) block[36 + i] = fallback_tag[i];
    uint32_t data_len = 36 + 15;
    d_sha256_pad(block, data_len);
    uint8_t hash[32];
    sha256_single_block(block, hash);
    return d_ReadLE32(hash) % MODULUS;
}

__device__ void d_DeriveNoiseSeed(const uint8_t* sigma_le, uint32_t tag_idx, uint8_t* seed_out_le) {
    const char* tags[5] = {
        "matmul_noise_EL_v1",
        "matmul_noise_ER_v1",
        "matmul_noise_FL_v1",
        "matmul_noise_FR_v1",
        "matmul-compress-v1"
    };
    uint8_t tag_lens[5] = {18, 18, 18, 18, 18};
    const char* tag = tags[tag_idx];
    uint8_t tag_len = tag_lens[tag_idx];
    uint8_t block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < tag_len; ++i) block[i] = tag[i];
    for (int i = 0; i < 32; ++i) block[tag_len + i] = sigma_le[31 - i];
    uint32_t total_len = tag_len + 32;
    d_sha256_pad(block, total_len);
    uint8_t hash[32];
    sha256_single_block(block, hash);
    for (int i = 0; i < 32; ++i) seed_out_le[i] = hash[31 - i];
}

__global__ void DeriveNoiseSeedsKernel(
    const uint8_t* sigma_batch,
    uint8_t* noise_seeds,
    uint8_t* compress_seeds,
    uint32_t batch_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce_idx = idx / 5;
    uint32_t seed_idx = idx % 5;
    if (nonce_idx >= batch_size) return;
    const uint8_t* sigma = sigma_batch + size_t(nonce_idx) * 32;
    uint8_t seed_out[32];
    d_DeriveNoiseSeed(sigma, seed_idx, seed_out);
    if (seed_idx < 4) {
        uint8_t* out = noise_seeds + (size_t(nonce_idx) * 4 + seed_idx) * 32;
        for (int i = 0; i < 32; ++i) out[i] = seed_out[i];
    } else {
        uint8_t* out = compress_seeds + size_t(nonce_idx) * 32;
        for (int i = 0; i < 32; ++i) out[i] = seed_out[i];
    }
}

__global__ void GenerateNoiseKernel(
    const uint8_t* noise_seeds,
    uint32_t batch_size,
    uint32_t num_elements,
    uint32_t seed_index,
    Element* output)
{
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t nonce_idx = static_cast<uint32_t>(gid / num_elements);
    uint32_t elem_idx = static_cast<uint32_t>(gid % num_elements);
    if (nonce_idx >= batch_size) return;
    const uint8_t* seed_le = noise_seeds + (size_t(nonce_idx) * 4 + seed_index) * 32;
    uint8_t seed_be[32];
    for (int i = 0; i < 32; ++i) seed_be[i] = seed_le[31 - i];
    Element val = d_from_oracle(seed_be, elem_idx);
    output[size_t(nonce_idx) * num_elements + elem_idx] = val;
}

__global__ void GenerateCompressKernel(
    const uint8_t* compress_seeds,
    uint32_t batch_size,
    uint32_t num_elements,
    Element* output)
{
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t nonce_idx = static_cast<uint32_t>(gid / num_elements);
    uint32_t elem_idx = static_cast<uint32_t>(gid % num_elements);
    if (nonce_idx >= batch_size) return;
    const uint8_t* seed_le = compress_seeds + size_t(nonce_idx) * 32;
    uint8_t seed_be[32];
    for (int i = 0; i < 32; ++i) seed_be[i] = seed_le[31 - i];
    Element val = d_from_oracle(seed_be, elem_idx);
    output[size_t(nonce_idx) * num_elements + elem_idx] = val;
}

__global__ void HashTranscriptKernel(
    const Element* compressed_words,
    const uint8_t* sigma_batch,
    uint32_t words_per_nonce,
    uint32_t n,
    uint32_t b,
    uint32_t batch_size,
    uint8_t* digest_batch)
{
    uint32_t nonce_idx = blockIdx.x;
    if (nonce_idx >= batch_size) return;
    const Element* words = compressed_words + size_t(nonce_idx) * words_per_nonce;
    const uint8_t* sigma_le = sigma_batch + size_t(nonce_idx) * 32;

    uint8_t h_block[64];
    for (int i = 0; i < 64; ++i) h_block[i] = 0;
    uint32_t data_len = 0;
    uint8_t running_hash[32];
    bool has_midstate = false;
    for (uint32_t w = 0; w < words_per_nonce; ++w) {
        d_WriteLE32(h_block + data_len, words[w]);
        data_len += 4;
        if (data_len == 64) {
            if (!has_midstate) {
                sha256_single_block(h_block, running_hash);
                has_midstate = true;
            } else {
                uint8_t tmp_block[64];
                for (int i = 0; i < 32; ++i) tmp_block[i] = running_hash[i];
                for (int i = 0; i < 32; ++i) tmp_block[32 + i] = h_block[i];
                sha256_single_block(tmp_block, running_hash);
            }
            for (int i = 0; i < 64; ++i) h_block[i] = 0;
            data_len = 0;
        }
    }
    if (data_len > 0 || !has_midstate) {
        d_sha256_pad(h_block, data_len);
        if (!has_midstate) {
            sha256_single_block(h_block, running_hash);
        } else {
            uint8_t tmp_block[64];
            for (int i = 0; i < 32; ++i) tmp_block[i] = running_hash[i];
            for (int i = 0; i < 32; ++i) tmp_block[32 + i] = h_block[i];
            sha256_single_block(tmp_block, running_hash);
        }
    }

    uint8_t block2[64];
    for (int i = 0; i < 32; ++i) block2[i] = running_hash[i];
    d_sha256_pad(block2, 32);
    uint8_t c_prime_hash_raw[32];
    sha256_single_block(block2, c_prime_hash_raw);

    uint8_t f_block[64];
    for (int i = 0; i < 64; ++i) f_block[i] = 0;
    uint32_t pos = 0;
    const char tag[] = "matmul-product-digest-v3";
    for (int i = 0; tag[i]; ++i) f_block[pos++] = tag[i];
    for (int i = 0; i < 32; ++i) f_block[pos++] = sigma_le[31 - i];
    for (int i = 0; i < 32; ++i) f_block[pos++] = c_prime_hash_raw[i];
    d_WriteLE32(f_block + pos, n); pos += 4;
    d_WriteLE32(f_block + pos, b); pos += 4;
    d_sha256_pad(f_block, pos);
    uint8_t hash2[32];
    sha256_single_block(f_block, hash2);
    uint8_t block3[64];
    for (int i = 0; i < 32; ++i) block3[i] = hash2[i];
    d_sha256_pad(block3, 32);
    uint8_t digest_be[32];
    sha256_single_block(block3, digest_be);

    uint8_t* out = digest_batch + size_t(nonce_idx) * 32;
    for (int i = 0; i < 32; ++i) out[i] = digest_be[31 - i];
}

__global__ void CompareDigestsKernel(
    const uint8_t* digest_batch,
    const uint8_t* block_target,
    const uint8_t* share_target,
    uint32_t batch_size,
    int32_t* results)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const uint8_t* digest = digest_batch + size_t(idx) * 32;
    bool le_block = true, le_share = true;
    for (int i = 0; i < 32; ++i) {
        if (digest[i] < block_target[i]) break;
        if (digest[i] > block_target[i]) { le_block = false; break; }
    }
    for (int i = 0; i < 32; ++i) {
        if (digest[i] < share_target[i]) break;
        if (digest[i] > share_target[i]) { le_share = false; break; }
    }
    results[idx] = 0;
    if (le_block) results[idx] = 2;
    else if (le_share) results[idx] = 1;
}

// Block header structure for sigma derivation.
// All fields are stored in little-endian (matching CPU DeriveSigma).
struct alignas(8) SigmaHeader {
    uint8_t version[4];       // LE32
    uint8_t prev_hash[32];    // canonical bytes (big-endian reversed → LE)
    uint8_t merkle_root[32];  // canonical bytes
    uint8_t time[4];          // LE32
    uint8_t bits[4];          // LE32
    uint8_t nonce_lo[4];      // nonce bits 0-31 (LE)
    uint8_t nonce_hi[4];      // nonce bits 32-63 (LE)
    uint8_t dim[2];           // LE16
    uint8_t seed_a[32];       // canonical bytes
    uint8_t seed_b[32];       // canonical bytes
};
static_assert(sizeof(SigmaHeader) == 150, "SigmaHeader must be 150 bytes");

// Compute SHA-256 mid-state after processing block_count 64-byte blocks.
// s is the state (8 x uint32_t), updated in place.
__device__ void sha256_blocks_midstate(uint32_t s[8], const uint8_t* data, uint32_t block_count) {
    for (uint32_t b = 0; b < block_count; ++b) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(data[b*64 + i*4])<<24) | (uint32_t(data[b*64 + i*4+1])<<16) |
                   (uint32_t(data[b*64 + i*4+2])<<8) | uint32_t(data[b*64 + i*4+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = d_rotr(w[i-15],7) ^ d_rotr(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = d_rotr(w[i-2],17) ^ d_rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = h + (d_rotr(e,6) ^ d_rotr(e,11) ^ d_rotr(e,25)) +
                          ((e & f) ^ (~e & g)) + d_K[i] + w[i];
            uint32_t t2 = (d_rotr(a,2) ^ d_rotr(a,13) ^ d_rotr(a,22)) +
                          ((a & b) ^ (a & c) ^ (b & c));
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
    }
}

// Finalize SHA-256 from mid-state s, processing the final padded block.
__device__ void sha256_finalize_from_midstate(uint32_t s[8], uint8_t hash_out[32]) {
    for (int i = 0; i < 8; ++i) {
        hash_out[i*4]   = uint8_t(s[i] >> 24);
        hash_out[i*4+1] = uint8_t(s[i] >> 16);
        hash_out[i*4+2] = uint8_t(s[i] >> 8);
        hash_out[i*4+3] = uint8_t(s[i]);
    }
}

__global__ void DeriveSigmaGateKernel(
    const SigmaHeader* __restrict__ header_base,
    uint64_t nonce_start,
    const uint8_t* __restrict__ sigma_gate_threshold_be,
    uint32_t epsilon_bits,
    uint64_t batch_size,
    uint32_t* __restrict__ gate_count_out,
    uint64_t* __restrict__ sigma_nonces_out,
    uint8_t* __restrict__ sigma_batch_out)
{
    uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= batch_size) return;

    uint64_t nonce = nonce_start + gid;

    // Build the 150-byte header buffer
    uint8_t hdr[150];
    SigmaHeader base = *header_base;
    for (int i = 0; i < 4; ++i) hdr[i] = base.version[i];
    for (int i = 0; i < 32; ++i) hdr[4 + i] = base.prev_hash[i];
    for (int i = 0; i < 32; ++i) hdr[36 + i] = base.merkle_root[i];
    for (int i = 0; i < 4; ++i) hdr[68 + i] = base.time[i];
    for (int i = 0; i < 4; ++i) hdr[72 + i] = base.bits[i];
    for (int i = 0; i < 4; ++i) hdr[76 + i] = uint8_t(nonce >> (i * 8));
    for (int i = 0; i < 4; ++i) hdr[80 + i] = uint8_t(nonce >> (32 + i * 8));
    hdr[84] = base.dim[0];
    hdr[85] = base.dim[1];
    for (int i = 0; i < 32; ++i) hdr[86 + i] = base.seed_a[i];
    for (int i = 0; i < 32; ++i) hdr[118 + i] = base.seed_b[i];

    // First SHA-256 round: process blocks 0-1 (128 bytes), leaving block 2 for finalization
    uint32_t s[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    sha256_blocks_midstate(s, hdr, 2); // process blocks 0 and 1

    // Block 2: bytes 128-149 (22 bytes) + padding
    uint8_t block2[64];
    for (int i = 0; i < 64; ++i) block2[i] = 0;
    for (int i = 0; i < 22; ++i) block2[i] = hdr[128 + i];
    block2[22] = 0x80;
    uint64_t total_bits = 150 * 8;
    block2[62] = uint8_t((total_bits >> 8) & 0xFF);
    block2[63] = uint8_t(total_bits & 0xFF);

    sha256_blocks_midstate(s, block2, 1);

    uint8_t header_hash[32];
    sha256_finalize_from_midstate(s, header_hash);

    // Second SHA-256: hash the 32-byte header_hash
    uint8_t block3[64];
    for (int i = 0; i < 64; ++i) block3[i] = 0;
    for (int i = 0; i < 32; ++i) block3[i] = header_hash[i];
    block3[32] = 0x80;
    total_bits = 32 * 8;
    block3[62] = uint8_t((total_bits >> 8) & 0xFF);
    block3[63] = uint8_t(total_bits & 0xFF);

    uint32_t s2[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    sha256_blocks_midstate(s2, block3, 1);

    uint8_t sigma_be[32];
    sha256_finalize_from_midstate(s2, sigma_be);

    if (epsilon_bits > 0) {
        for (int i = 0; i < 32; ++i) {
            if (sigma_be[i] < sigma_gate_threshold_be[i]) break;
            if (sigma_be[i] > sigma_gate_threshold_be[i]) return;
        }
    }

    uint32_t idx = atomicAdd(gate_count_out, 1u);
    if (idx >= uint32_t(batch_size)) return;
    sigma_nonces_out[idx] = nonce;
    for (int i = 0; i < 32; ++i)
        sigma_batch_out[idx * 32 + i] = sigma_be[i];
}

void DeriveSigmaGateKernel_launch(
    const SigmaHeader* header,
    uint64_t nonce_start,
    const uint8_t* sigma_gate_threshold_be,
    uint32_t epsilon_bits,
    uint64_t batch_size,
    uint32_t* gate_count,
    uint64_t* sigma_nonces,
    uint8_t* sigma_batch,
    hipStream_t stream)
{
    EnsureKInitialized();
    uint32_t threads = 256;
    uint32_t blocks = (batch_size + threads - 1) / threads;
    hipLaunchKernelGGL(DeriveSigmaGateKernel,
        dim3(blocks), dim3(threads), 0, stream,
        header, nonce_start, sigma_gate_threshold_be,
        epsilon_bits, batch_size,
        gate_count, sigma_nonces, sigma_batch);
}

void DeriveNoiseSeedsKernel_launch(
  const uint8_t* sigma_batch,
  uint8_t* noise_seeds,
  uint8_t* compress_seeds,
  uint32_t batch_size,
  hipStream_t stream)
{
  EnsureKInitialized();
  uint32_t seed_threads = 256;
    uint32_t seed_total = batch_size * 5;
    uint32_t seed_blocks = (seed_total + seed_threads - 1) / seed_threads;
    hipLaunchKernelGGL(DeriveNoiseSeedsKernel,
        dim3(seed_blocks), dim3(seed_threads), 0, stream,
        sigma_batch, noise_seeds, compress_seeds, batch_size);
}

void GenerateNoiseKernel_launch(
    const uint8_t* noise_seeds,
    uint32_t batch_size,
    uint32_t num_elements,
    uint32_t seed_index,
    Element* output,
    hipStream_t stream)
{
    uint32_t total_el = batch_size * num_elements;
    uint32_t gen_blocks = (total_el + 255) / 256;
    hipLaunchKernelGGL(GenerateNoiseKernel,
        dim3(gen_blocks), dim3(256), 0, stream,
        noise_seeds, batch_size, num_elements, seed_index, output);
}

void GenerateCompressKernel_launch(
    const uint8_t* compress_seeds,
    uint32_t batch_size,
    uint32_t num_elements,
    Element* output,
    hipStream_t stream)
{
    uint32_t total_el = batch_size * num_elements;
    uint32_t gen_blocks = (total_el + 255) / 256;
    hipLaunchKernelGGL(GenerateCompressKernel,
        dim3(gen_blocks), dim3(256), 0, stream,
        compress_seeds, batch_size, num_elements, output);
}

void HashTranscriptKernel_launch(
    const Element* compressed_words,
    const uint8_t* sigma_batch,
    uint32_t words_per_nonce,
    uint32_t n,
    uint32_t b,
    uint32_t batch_size,
    uint8_t* digest_batch,
    hipStream_t stream)
{
    hipLaunchKernelGGL(HashTranscriptKernel,
        dim3(batch_size), dim3(1), 0, stream,
        compressed_words, sigma_batch, words_per_nonce, n, b, batch_size, digest_batch);
}

void CompareDigestsKernel_launch(
    const uint8_t* digest_batch,
    const uint8_t* block_target,
    const uint8_t* share_target,
    uint32_t batch_size,
    int32_t* results,
    hipStream_t stream)
{
    uint32_t cmp_blocks = (batch_size + 255) / 256;
    hipLaunchKernelGGL(CompareDigestsKernel,
        dim3(cmp_blocks), dim3(256), 0, stream,
        digest_batch, block_target, share_target, batch_size, results);
}

}
