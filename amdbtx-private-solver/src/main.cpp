// btx-gbt-solve: Standalone HIP/ROCm solver for BTX MatMul PoW.
// Supports single-shot and daemon (stdin/stdout JSON) modes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>

#include "sha256.h"
#include "field.h"
#include "noise.h"
#include "transcript.h"

#include <hip/hip_runtime.h>

// Forward declaration from matmul_kernel.hip
void matmul_launch(
    const uint32_t* d_A, const uint32_t* d_B,
    const uint32_t* d_EL, const uint32_t* d_ER,
    const uint32_t* d_FL, const uint32_t* d_FR,
    const uint32_t* d_compress_vec,
    uint32_t* d_out_words, uint32_t* d_out_counter,
    uint32_t n, uint32_t b, uint32_t r,
    hipStream_t stream);

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t MAT_DIM   = 512;  // n
static constexpr uint32_t BLOCK_B   = 16;   // b
static constexpr uint32_t NOISE_R   = 8;    // r
static constexpr uint32_t EPSILON   = 18;   // sigma gate exponent
static constexpr uint32_t NB        = MAT_DIM / BLOCK_B; // 32
static constexpr uint32_t TOTAL_WORDS = NB * NB * NB;    // 32768

// ============================================================================
// Hex utilities
// ============================================================================

static bool hex_to_bytes(const char* hex, uint8_t* out, size_t out_len) {
    for (size_t i = 0; i < out_len; i++) {
        unsigned int b;
        if (sscanf(hex + i * 2, "%2x", &b) != 1) return false;
        out[i] = static_cast<uint8_t>(b);
    }
    return true;
}

static void bytes_to_hex(const uint8_t* in, size_t len, char* out) {
    for (size_t i = 0; i < len; i++) {
        sprintf(out + i * 2, "%02x", in[i]);
    }
    out[len * 2] = '\0';
}

// ============================================================================
// JSON helpers (minimal, no external dependency)
// ============================================================================

static bool json_get_hex(const char* json, const char* key,
                         uint8_t* out, size_t out_len) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* pos = strstr(json, needle);
    if (!pos) return false;
    pos = strchr(pos + strlen(needle), '"');
    if (!pos) return false;
    pos++;
    const char* end = strchr(pos, '"');
    if (!end) return false;
    size_t hex_len = end - pos;
    char hex_buf[256];
    if (hex_len >= sizeof(hex_buf)) return false;
    memcpy(hex_buf, pos, hex_len);
    hex_buf[hex_len] = '\0';
    return hex_to_bytes(hex_buf, out, out_len);
}

static bool json_get_u64(const char* json, const char* key, uint64_t& out) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* pos = strstr(json, needle);
    if (!pos) return false;
    pos = strchr(pos + strlen(needle), ':');
    if (!pos) return false;
    pos++;
    while (*pos == ' ') pos++;
    out = strtoull(pos, nullptr, 10);
    return true;
}

static bool json_get_f64(const char* json, const char* key, double& out) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* pos = strstr(json, needle);
    if (!pos) return false;
    pos = strchr(pos + strlen(needle), ':');
    if (!pos) return false;
    pos++;
    while (*pos == ' ') pos++;
    out = strtod(pos, nullptr);
    return true;
}

static bool json_get_hex_str(const char* json, const char* key,
                             char* out, size_t out_cap) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* pos = strstr(json, needle);
    if (!pos) return false;
    pos = strchr(pos + strlen(needle), '"');
    if (!pos) return false;
    pos++;
    const char* end = strchr(pos, '"');
    if (!end) return false;
    size_t len = end - pos;
    if (len >= out_cap) return false;
    memcpy(out, pos, len);
    out[len] = '\0';
    return true;
}

// ============================================================================
// Job parsing
// ============================================================================

struct Job {
    uint32_t version;
    uint8_t  prev_hash[32];
    uint8_t  merkle_root[32];
    uint32_t time;
    uint8_t  bits[4];
    uint8_t  seed_a[32];
    uint8_t  seed_b[32];
    uint64_t block_height;
    uint64_t nonce_start;
    uint64_t max_tries;
    double   max_seconds;
    uint8_t  share_target[32];
    bool     has_share_target;
};

static bool parse_job(const char* line, Job& job) {
    memset(&job, 0, sizeof(job));

    uint64_t tmp;
    if (!json_get_u64(line, "version", tmp)) return false;
    job.version = (uint32_t)tmp;

    if (!json_get_hex(line, "prev_hash", job.prev_hash, 32)) return false;
    if (!json_get_hex(line, "merkle_root", job.merkle_root, 32)) return false;

    if (!json_get_u64(line, "time", tmp)) return false;
    job.time = (uint32_t)tmp;

    char bits_hex[16];
    if (!json_get_hex_str(line, "bits", bits_hex, sizeof(bits_hex))) return false;
    if (!hex_to_bytes(bits_hex, job.bits, 4)) return false;

    if (!json_get_hex(line, "seed_a", job.seed_a, 32)) return false;
    if (!json_get_hex(line, "seed_b", job.seed_b, 32)) return false;
    if (!json_get_u64(line, "block_height", job.block_height)) return false;
    if (!json_get_u64(line, "nonce_start", job.nonce_start)) return false;
    if (!json_get_u64(line, "max_tries", job.max_tries)) return false;

    double secs = 5.0;
    json_get_f64(line, "max_seconds", secs);
    job.max_seconds = secs;

    job.has_share_target = json_get_hex(line, "share_target", job.share_target, 32);

    return true;
}

// ============================================================================
// Block header construction and target derivation
// ============================================================================

static void build_header(const Job& job, uint64_t nonce, uint8_t header[80]) {
    header[0]  = job.version & 0xFF;
    header[1]  = (job.version >> 8) & 0xFF;
    header[2]  = (job.version >> 16) & 0xFF;
    header[3]  = (job.version >> 24) & 0xFF;

    memcpy(header + 4, job.prev_hash, 32);
    memcpy(header + 36, job.merkle_root, 32);

    header[68] = job.time & 0xFF;
    header[69] = (job.time >> 8) & 0xFF;
    header[70] = (job.time >> 16) & 0xFF;
    header[71] = (job.time >> 24) & 0xFF;

    memcpy(header + 72, job.bits, 4);

    uint32_t nonce32 = (uint32_t)(nonce & 0xFFFFFFFF);
    header[76] = nonce32 & 0xFF;
    header[77] = (nonce32 >> 8) & 0xFF;
    header[78] = (nonce32 >> 16) & 0xFF;
    header[79] = (nonce32 >> 24) & 0xFF;
}

static void bits_to_target(const uint8_t bits[4], uint8_t target[32]) {
    memset(target, 0, 32);
    uint32_t e = bits[0];
    // bits[1..3] are the mantissa in big-endian order
    uint32_t m = (uint32_t(bits[1]) << 16) | (uint32_t(bits[2]) << 8) | bits[3];
    if (e <= 3) {
        for (uint32_t i = 0; i < e && i < 3; i++) {
            target[32 - 1 - i] = (m >> (8 * i)) & 0xFF;
        }
    } else {
        uint32_t byte_pos = e - 3;
        for (uint32_t i = 0; i < 3; i++) {
            uint32_t idx = 32 - 1 - byte_pos - i;
            if (idx < 32) {
                target[idx] = (m >> (8 * i)) & 0xFF;
            }
        }
    }
}

static void compute_sigma(const uint8_t header[80], uint8_t sigma[32]) {
    sha256::sha256d(header, 80, sigma);
}

static bool hash_below_target(const uint8_t hash[32], const uint8_t target[32]) {
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return false;
}

static bool sigma_gate_pass(const uint8_t sigma[32], const uint8_t target[32], uint32_t epsilon) {
    // Pass if sigma < target * 2^epsilon (left-shift target by epsilon bits).
    // Compare sigma against target << epsilon.
    uint32_t full_bytes = epsilon / 8;
    uint32_t rem_bits   = epsilon % 8;

    for (int i = 0; i < 32; i++) {
        uint32_t t_byte = 0;
        int src_idx = i - (int)full_bytes;
        if (rem_bits == 0) {
            if (src_idx >= 0 && src_idx < 32) {
                t_byte = target[src_idx];
            }
        } else {
            if (src_idx >= 0 && src_idx < 32) {
                t_byte = target[src_idx] << rem_bits;
            }
            if (src_idx - 1 >= 0 && src_idx - 1 < 32) {
                t_byte |= target[src_idx - 1] >> (8 - rem_bits);
            }
        }

        if (sigma[i] < t_byte) return true;
        if (sigma[i] > t_byte) return false;
    }
    return false;
}

// ============================================================================
// Matrix generation and noisy matrix computation
// ============================================================================

static void generate_matrix(field::Element* mat, uint32_t n,
                            const uint8_t seed[32], const char* tag) {
    prf_fill(mat, n * n, seed, tag, strlen(tag));
}

// Compute C = A + B (mod q), element-wise. Both are n*n.
static void mat_add(field::Element* C, const field::Element* A,
                    const field::Element* B, uint32_t n_sq) {
    for (uint32_t i = 0; i < n_sq; i++) {
        C[i] = field::add(A[i], B[i]);
    }
}

// Compute C = L * R where L is (n x r) and R is (r x n), result is (n x n).
// Uses brute-force O(n^2 * r) — fine since n=512, r=8 and this only runs
// when sigma gate passes (rare).
static void matmul_small(field::Element* C,
                         const field::Element* L, const field::Element* R,
                         uint32_t n, uint32_t r) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            uint64_t acc = 0;
            for (uint32_t k = 0; k < r; k++) {
                acc += static_cast<uint64_t>(L[i * r + k]) * R[k * n + j];
                if ((k & 0x1F) == 0x1F) {
                    acc = field::reduce64(acc);
                }
            }
            C[i * n + j] = field::reduce64(acc);
        }
    }
}

// ============================================================================
// GPU memory management
// ============================================================================

struct GpuResources {
    uint32_t* d_A_noisy;
    uint32_t* d_B_noisy;
    uint32_t* d_compress_vec;
    uint32_t* d_out_words;
    uint32_t* d_out_counter;
    hipStream_t stream;
    bool initialized;
};

static bool gpu_init(GpuResources& gpu) {
    gpu.initialized = false;

    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No HIP devices found\n");
        return false;
    }

    hipSetDevice(0);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "Using GPU: %s (%s, %zu MB VRAM)\n",
            prop.name, prop.gcnArchName,
            prop.totalGlobalMem / (1024 * 1024));

    hipStreamCreate(&gpu.stream);

    size_t mat_bytes = MAT_DIM * MAT_DIM * sizeof(uint32_t);
    size_t compress_bytes = BLOCK_B * BLOCK_B * sizeof(uint32_t);
    size_t out_words_bytes = TOTAL_WORDS * sizeof(uint32_t);

    hipMalloc(&gpu.d_A_noisy, mat_bytes);
    hipMalloc(&gpu.d_B_noisy, mat_bytes);
    hipMalloc(&gpu.d_compress_vec, compress_bytes);
    hipMalloc(&gpu.d_out_words, out_words_bytes);
    hipMalloc(&gpu.d_out_counter, sizeof(uint32_t));

    if (!gpu.d_A_noisy || !gpu.d_B_noisy || !gpu.d_compress_vec ||
        !gpu.d_out_words || !gpu.d_out_counter) {
        fprintf(stderr, "GPU memory allocation failed\n");
        return false;
    }

    gpu.initialized = true;
    return true;
}

static void gpu_free(GpuResources& gpu) {
    if (!gpu.initialized) return;
    hipFree(gpu.d_A_noisy);
    hipFree(gpu.d_B_noisy);
    hipFree(gpu.d_compress_vec);
    hipFree(gpu.d_out_words);
    hipFree(gpu.d_out_counter);
    hipStreamDestroy(gpu.stream);
    gpu.initialized = false;
}

// ============================================================================
// Solve one nonce: sigma gate → noise → noisy matrices → GPU → check
// ============================================================================

struct SolveResult {
    bool found;
    bool is_block;
    uint64_t nonce64;
    uint8_t digest[32];
};

static SolveResult solve_nonce(
    const Job& job, uint64_t nonce,
    const field::Element* A, const field::Element* B,
    GpuResources& gpu,
    const uint8_t block_target[32],
    const uint8_t share_target[32], bool has_share,
    // Scratch buffers (reused across nonces)
    NoisePair& noise,
    field::Element* ELxER, field::Element* FLxFR,
    field::Element* A_noisy, field::Element* B_noisy)
{
    SolveResult res;
    res.found = false;
    res.is_block = false;
    res.nonce64 = nonce;

    // Step 1: Build header and compute sigma
    uint8_t header[80];
    build_header(job, nonce, header);
    uint8_t sigma[32];
    compute_sigma(header, sigma);

    // Step 2: Sigma gate — skip if sigma >= block_target * 2^epsilon
    if (!sigma_gate_pass(sigma, block_target, EPSILON)) {
        return res;
    }

    // Step 3: Generate noise matrices from sigma
    noise_generate(sigma, MAT_DIM, NOISE_R, noise);

    // Step 4: Compute EL·ER (n×n) and FL·FR (n×n)
    matmul_small(ELxER, noise.EL, noise.ER, MAT_DIM, NOISE_R);
    matmul_small(FLxFR, noise.FL, noise.FR, MAT_DIM, NOISE_R);

    // Step 5: A_noisy = A + EL·ER, B_noisy = B + FL·FR
    uint32_t n_sq = MAT_DIM * MAT_DIM;
    mat_add(A_noisy, A, ELxER, n_sq);
    mat_add(B_noisy, B, FLxFR, n_sq);

    // Step 6: Derive compression vector
    field::Element compress_vec[BLOCK_B * BLOCK_B];
    derive_compression_vector(sigma, BLOCK_B, compress_vec);

    // Step 7: Upload to GPU
    size_t mat_bytes = n_sq * sizeof(uint32_t);
    size_t compress_bytes = BLOCK_B * BLOCK_B * sizeof(uint32_t);

    hipMemcpyAsync(gpu.d_A_noisy, A_noisy, mat_bytes,
                   hipMemcpyHostToDevice, gpu.stream);
    hipMemcpyAsync(gpu.d_B_noisy, B_noisy, mat_bytes,
                   hipMemcpyHostToDevice, gpu.stream);
    hipMemcpyAsync(gpu.d_compress_vec, compress_vec, compress_bytes,
                   hipMemcpyHostToDevice, gpu.stream);

    // Step 8: Launch kernel
    // Pass nullptr for EL/ER/FL/FR — they're unused (noise is folded into A/B)
    matmul_launch(
        gpu.d_A_noisy, gpu.d_B_noisy,
        nullptr, nullptr, nullptr, nullptr,
        gpu.d_compress_vec,
        gpu.d_out_words, gpu.d_out_counter,
        MAT_DIM, BLOCK_B, NOISE_R,
        gpu.stream
    );

    // Step 9: Copy results back
    std::vector<uint32_t> words(TOTAL_WORDS);
    hipMemcpyAsync(words.data(), gpu.d_out_words,
                   TOTAL_WORDS * sizeof(uint32_t),
                   hipMemcpyDeviceToHost, gpu.stream);
    hipStreamSynchronize(gpu.stream);

    // Step 10: Finalize transcript
    uint8_t digest[32];
    transcript_finalize(words.data(), TOTAL_WORDS, digest);
    memcpy(res.digest, digest, 32);

    // Step 11: Check against targets
    if (has_share && hash_below_target(digest, share_target)) {
        res.found = true;
        res.is_block = false;
    } else if (hash_below_target(digest, block_target)) {
        res.found = true;
        res.is_block = true;
    }

    return res;
}

// ============================================================================
// Single-shot mode
// ============================================================================

static int run_single_shot(int argc, char** argv) {
    Job job;
    memset(&job, 0, sizeof(job));

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--version") == 0 && i + 1 < argc) {
            job.version = (uint32_t)strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--prev-hash") == 0 && i + 1 < argc) {
            hex_to_bytes(argv[++i], job.prev_hash, 32);
        } else if (strcmp(argv[i], "--merkle-root") == 0 && i + 1 < argc) {
            hex_to_bytes(argv[++i], job.merkle_root, 32);
        } else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            job.time = (uint32_t)strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--bits") == 0 && i + 1 < argc) {
            hex_to_bytes(argv[++i], job.bits, 4);
        } else if (strcmp(argv[i], "--seed-a") == 0 && i + 1 < argc) {
            hex_to_bytes(argv[++i], job.seed_a, 32);
        } else if (strcmp(argv[i], "--seed-b") == 0 && i + 1 < argc) {
            hex_to_bytes(argv[++i], job.seed_b, 32);
        } else if (strcmp(argv[i], "--block-height") == 0 && i + 1 < argc) {
            job.block_height = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--nonce-start") == 0 && i + 1 < argc) {
            job.nonce_start = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--max-tries") == 0 && i + 1 < argc) {
            job.max_tries = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--max-seconds") == 0 && i + 1 < argc) {
            job.max_seconds = strtod(argv[++i], nullptr);
        } else if (strcmp(argv[i], "--share-target") == 0 && i + 1 < argc) {
            hex_to_bytes(argv[++i], job.share_target, 32);
            job.has_share_target = true;
        }
    }

    if (job.max_tries == 0) job.max_tries = 2000000;
    if (job.max_seconds <= 0) job.max_seconds = 5.0;

    GpuResources gpu;
    if (!gpu_init(gpu)) return 1;

    printf("Generating matrices A and B from seeds...\n");
    std::vector<field::Element> A(MAT_DIM * MAT_DIM);
    std::vector<field::Element> B(MAT_DIM * MAT_DIM);
    generate_matrix(A.data(), MAT_DIM, job.seed_a, "matmul_matrix_A_v1");
    generate_matrix(B.data(), MAT_DIM, job.seed_b, "matmul_matrix_B_v1");

    uint8_t block_target[32];
    bits_to_target(job.bits, block_target);

    // Scratch buffers for noise computation (reused per nonce)
    NoisePair noise;
    std::vector<field::Element> ELxER(MAT_DIM * MAT_DIM);
    std::vector<field::Element> FLxFR(MAT_DIM * MAT_DIM);
    std::vector<field::Element> A_noisy(MAT_DIM * MAT_DIM);
    std::vector<field::Element> B_noisy(MAT_DIM * MAT_DIM);

    auto start_time = std::chrono::steady_clock::now();
    uint64_t nonce = job.nonce_start;
    uint64_t end_nonce = job.nonce_start + job.max_tries;

    printf("Starting solve: nonce range [%lu, %lu), max %.1f seconds\n",
           (unsigned long)nonce, (unsigned long)end_nonce, job.max_seconds);

    while (nonce < end_nonce) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed >= job.max_seconds) break;

        SolveResult res = solve_nonce(
            job, nonce, A.data(), B.data(), gpu,
            block_target, job.share_target, job.has_share_target,
            noise,
            ELxER.data(), FLxFR.data(),
            A_noisy.data(), B_noisy.data());

        if (res.found) {
            char digest_hex[65];
            bytes_to_hex(res.digest, 32, digest_hex);
            auto end = std::chrono::steady_clock::now();
            double final_elapsed = std::chrono::duration<double>(end - start_time).count();
            printf("{\"found\":true,\"nonce64\":%lu,\"nonce64_end\":%lu,\"digest\":\"%s\",\"elapsed_s\":%.6f,\"tries_used\":%lu,\"is_block\":%s}\n",
                   (unsigned long)nonce, (unsigned long)(nonce + 1),
                   digest_hex, final_elapsed,
                   (unsigned long)(nonce - job.nonce_start + 1),
                   res.is_block ? "true" : "false");
            gpu_free(gpu);
            return 0;
        }

        nonce++;
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start_time).count();
    printf("{\"found\":false,\"tries_used\":%lu,\"elapsed_s\":%.6f}\n",
           (unsigned long)(nonce - job.nonce_start), elapsed);

    gpu_free(gpu);
    return 0;
}

// ============================================================================
// Daemon mode
// ============================================================================

static int run_daemon() {
    GpuResources gpu;
    if (!gpu_init(gpu)) {
        fprintf(stderr, "{\"event\":\"gpu_init_failed\"}\n");
        return 1;
    }

    fprintf(stderr, "{\"event\":\"daemon_ready\"}\n");

    // Matrix cache
    std::vector<field::Element> A(MAT_DIM * MAT_DIM);
    std::vector<field::Element> B(MAT_DIM * MAT_DIM);
    bool matrices_loaded = false;
    uint8_t cached_seed_a[32], cached_seed_b[32];

    // Scratch buffers for noise
    NoisePair noise;
    std::vector<field::Element> ELxER(MAT_DIM * MAT_DIM);
    std::vector<field::Element> FLxFR(MAT_DIM * MAT_DIM);
    std::vector<field::Element> A_noisy(MAT_DIM * MAT_DIM);
    std::vector<field::Element> B_noisy(MAT_DIM * MAT_DIM);

    char line[8192];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        Job job;
        if (!parse_job(line, job)) {
            fprintf(stderr, "Failed to parse job JSON\n");
            continue;
        }

        // Generate or cache matrices
        if (!matrices_loaded ||
            memcmp(cached_seed_a, job.seed_a, 32) != 0 ||
            memcmp(cached_seed_b, job.seed_b, 32) != 0) {
            generate_matrix(A.data(), MAT_DIM, job.seed_a, "matmul_matrix_A_v1");
            generate_matrix(B.data(), MAT_DIM, job.seed_b, "matmul_matrix_B_v1");
            memcpy(cached_seed_a, job.seed_a, 32);
            memcpy(cached_seed_b, job.seed_b, 32);
            matrices_loaded = true;
        }

        uint8_t block_target[32];
        bits_to_target(job.bits, block_target);

        auto start_time = std::chrono::steady_clock::now();
        uint64_t nonce = job.nonce_start;
        uint64_t end_nonce = job.nonce_start + job.max_tries;
        bool found = false;

        while (nonce < end_nonce) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed >= job.max_seconds) break;

            SolveResult res = solve_nonce(
                job, nonce, A.data(), B.data(), gpu,
                block_target, job.share_target, job.has_share_target,
                noise,
                ELxER.data(), FLxFR.data(),
                A_noisy.data(), B_noisy.data());

            if (res.found) {
                auto end = std::chrono::steady_clock::now();
                double final_elapsed = std::chrono::duration<double>(end - start_time).count();
                char digest_hex[65];
                bytes_to_hex(res.digest, 32, digest_hex);
                printf("{\"found\":true,\"nonce64\":%lu,\"nonce64_end\":%lu,\"digest\":\"%s\",\"elapsed_s\":%.6f,\"tries_used\":%lu,\"is_block\":%s}\n",
                       (unsigned long)nonce, (unsigned long)(nonce + 1),
                       digest_hex, final_elapsed,
                       (unsigned long)(nonce - job.nonce_start + 1),
                       res.is_block ? "true" : "false");
                fflush(stdout);
                found = true;
                break;
            }

            nonce++;
        }

        if (!found) {
            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start_time).count();
            printf("{\"found\":false,\"nonce64_end\":%lu,\"elapsed_s\":%.6f,\"tries_used\":%lu,\"is_block\":false}\n",
                   (unsigned long)nonce, elapsed,
                   (unsigned long)(nonce - job.nonce_start));
            fflush(stdout);
        }
    }

    gpu_free(gpu);
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s --daemon | --version V --prev-hash H ... \n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--daemon") == 0) {
        return run_daemon();
    }

    return run_single_shot(argc, argv);
}
