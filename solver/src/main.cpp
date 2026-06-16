#include "solve.h"
#include "sha256.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <iomanip>

static Uint256 HexToUint256(const std::string& hex) {
    Uint256 result;
    std::memset(result.data, 0, 32);
    std::string padded = hex;
    while (padded.size() < 64) padded = padded + "0";
    if (padded.size() > 64) padded = padded.substr(0, 64);
    // Match BTX SetHex: first hex pair → data[31] (MSB), last → data[0] (LSB)
    for (int i = 0; i < 32; ++i) {
        unsigned int byte;
        std::sscanf(padded.c_str() + i * 2, "%02x", &byte);
        result.data[31 - i] = static_cast<uint8_t>(byte);
    }
    return result;
}

static uint32_t ParseBits(const std::string& bits_str) {
    if (bits_str.size() > 2 && bits_str[0] == '0' && (bits_str[1] == 'x' || bits_str[1] == 'X'))
        return static_cast<uint32_t>(std::stoul(bits_str.substr(2), nullptr, 16));
    return static_cast<uint32_t>(std::stoul(bits_str, nullptr, 16));
}

struct DaemonConfig {
    uint32_t matmul_n = 512;
    uint32_t matmul_b = 16;
    uint32_t matmul_r = 8;
    uint32_t epsilon_bits = 18;
    std::string backend = "hip";
    uint32_t solver_threads = 8;
    uint32_t batch_size = 128;
};

static std::string extract_json_string(const std::string& s, const std::string& key) {
    auto pos = s.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    auto colon = s.find(':', pos + key.size() + 2);
    if (colon == std::string::npos) return "";
    auto q1 = s.find('"', colon);
    if (q1 == std::string::npos) return "";
    auto q2 = s.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return s.substr(q1 + 1, q2 - q1 - 1);
}

static std::string extract_json_number(const std::string& s, const std::string& key) {
    auto pos = s.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    auto colon = s.find(':', pos + key.size() + 2);
    if (colon == std::string::npos) return "";
    size_t start = colon + 1;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) ++start;
    size_t end = start;
    if (end < s.size() && s[end] == '-') ++end;
    while (end < s.size() && (std::isdigit(s[end]) || s[end] == '.' || s[end] == 'e' || s[end] == 'E' || s[end] == '+' || s[end] == '-')) ++end;
    if (end == start) return "";
    return s.substr(start, end - start);
}

int main(int argc, char* argv[]) {
    DaemonConfig config;
    bool daemon_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--daemon") daemon_mode = true;
        else if (arg == "--matmul-n" && i+1 < argc) config.matmul_n = std::stoul(argv[++i]);
        else if (arg == "--matmul-b" && i+1 < argc) config.matmul_b = std::stoul(argv[++i]);
        else if (arg == "--matmul-r" && i+1 < argc) config.matmul_r = std::stoul(argv[++i]);
        else if (arg == "--epsilon-bits" && i+1 < argc) config.epsilon_bits = std::stoul(argv[++i]);
        else if (arg == "--backend" && i+1 < argc) config.backend = argv[++i];
        else if (arg == "--solver-threads" && i+1 < argc) config.solver_threads = std::stoul(argv[++i]);
        else if (arg == "--batch-size" && i+1 < argc) config.batch_size = std::stoul(argv[++i]);
        else if (arg == "--share-target") { if (i+1 < argc) ++i; }
        else if (arg == "--version") {
            std::cerr << "btx-gbt-solve-hip 2.1.0 (BTX V3 parent-MTP)" << std::endl;
            return 0;
        } else if (arg == "--help") {
            std::cerr << "Usage: " << argv[0] << " --daemon [--matmul-n N] [--matmul-b B] [--matmul-r R] "
                      << "[--epsilon-bits E] [--backend hip|cpu] [--solver-threads T] [--batch-size S] "
                      << "[--share-target HEX]" << std::endl;
            return 0;
        }
    }

    if (!daemon_mode) {
        std::cerr << "Error: --daemon mode is required" << std::endl;
        return 1;
    }

    if (config.backend == "hip") {
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err == hipSuccess && device_count > 0) {
            hipDeviceProp_t prop{};
            if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
                std::cerr << "HIP GPU detected: " << prop.name << " arch=" << prop.gcnArchName
                          << " memory=" << (prop.totalGlobalMem / (1024*1024)) << "MB" << std::endl;
            } else {
                std::cerr << "HIP GPU enumeration failed, falling back to CPU" << std::endl;
                config.backend = "cpu";
            }
        } else {
            std::cerr << "No HIP GPU found, falling back to CPU" << std::endl;
            config.backend = "cpu";
        }
    }

    std::cerr << "Solver config: n=" << config.matmul_n
              << " b=" << config.matmul_b
              << " r=" << config.matmul_r
              << " epsilon=" << config.epsilon_bits
              << " batch_size=" << config.batch_size
              << " backend=" << config.backend << std::endl;

    std::cerr << "{\"event\":\"daemon_ready\"}" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;

        uint32_t job_n = config.matmul_n;
        uint32_t job_b = config.matmul_b;
        uint32_t job_r = config.matmul_r;

        std::string jn_str = extract_json_number(line, "matmul_n");
        if (!jn_str.empty()) job_n = static_cast<uint32_t>(std::stoul(jn_str));
        std::string jb_str = extract_json_number(line, "matmul_b");
        if (!jb_str.empty()) job_b = static_cast<uint32_t>(std::stoul(jb_str));
        std::string jr_str = extract_json_number(line, "matmul_r");
        if (!jr_str.empty()) job_r = static_cast<uint32_t>(std::stoul(jr_str));

        PowState state;
        std::memset(&state, 0, sizeof(state));
        state.matmul_dim = static_cast<uint16_t>(job_n);

        uint64_t nonce_start = 0;
        uint64_t max_tries = 2000000;
        double max_seconds = 5.0;
        uint32_t max_results = 64;

        Uint256 share_target;
        std::memset(share_target.data, 0xFF, 32);

        state.version = std::stoi(extract_json_number(line, "version"));
        state.previous_block_hash = HexToUint256(extract_json_string(line, "prev_hash"));
        state.merkle_root = HexToUint256(extract_json_string(line, "merkle_root"));
        state.time = static_cast<uint32_t>(std::stoul(extract_json_number(line, "time")));

        std::string bits_str = extract_json_string(line, "bits");
        if (bits_str.empty()) bits_str = extract_json_number(line, "bits");
        state.bits = bits_str.empty() ? 0 : ParseBits(bits_str);

        state.seed_a = HexToUint256(extract_json_string(line, "seed_a"));
        state.seed_b = HexToUint256(extract_json_string(line, "seed_b"));

        std::string ns_str = extract_json_number(line, "nonce_start");
        if (!ns_str.empty()) nonce_start = std::stoull(ns_str);

        std::string mt_str = extract_json_number(line, "max_tries");
        if (!mt_str.empty()) max_tries = std::stoull(mt_str);

        std::string ms_str = extract_json_number(line, "max_seconds");
        if (!ms_str.empty()) max_seconds = std::stod(ms_str);
        std::string mr_str = extract_json_number(line, "max_results");
        if (!mr_str.empty()) max_results = static_cast<uint32_t>(std::stoul(mr_str));

        std::string st_str = extract_json_string(line, "share_target");
        if (!st_str.empty()) share_target = HexToUint256(st_str);

        std::string bh_str = extract_json_number(line, "block_height");
        state.height = bh_str.empty() ? 0 : std::stoi(bh_str);
        std::string parent_mtp_str = extract_json_number(line, "parent_mtp");
        state.has_parent_mtp = !parent_mtp_str.empty();
        if (state.has_parent_mtp) {
            state.parent_mtp = std::stoll(parent_mtp_str);
        }
        state.nonce = nonce_start;

        std::string eb_str = extract_json_number(line, "epsilon_bits");
        uint32_t job_epsilon_bits = config.epsilon_bits;
        if (!eb_str.empty()) job_epsilon_bits = static_cast<uint32_t>(std::stoul(eb_str));
        const bool include_product_payload =
            extract_json_number(line, "include_product_payload") == "1";

        if (state.height >= 130500 && !state.has_parent_mtp) {
            std::cout
                << "{\"found\":false,\"error\":\"parent_mtp is required at block_height >= 130500\","
                << "\"backend\":\"" << config.backend << "\"}" << std::endl;
            continue;
        }
        if (extract_json_number(line, "emit_seeds") == "1") {
            PrepareNonceSeeds(state);
            std::cout
                << "{\"ok\":true,\"block_height\":" << state.height
                << ",\"nonce64\":" << state.nonce
                << ",\"seed_a\":\"" << Uint256ToHex(state.seed_a)
                << "\",\"seed_b\":\"" << Uint256ToHex(state.seed_b)
                << "\"}" << std::endl;
            continue;
        }

        Uint256 block_target = DeriveBlockTarget(state.bits);

        uint64_t tries_used = 0;
        uint64_t gate_passes = 0;
        uint64_t words_hits = 0;
        uint64_t cpu_verify_misses = 0;
        double elapsed_s = 0;
        bool found;
        bool cpu_fallback = false;
        std::string backend_used = config.backend;
        std::vector<PowState> solutions;
        uint64_t scanned_nonce_end = nonce_start;

        if (config.backend == "hip") {
            found = SolveGPU(state, job_n, job_b, job_r,
                             block_target, share_target, max_tries, max_seconds,
                             tries_used, elapsed_s,
                             config.batch_size, job_epsilon_bits, &cpu_fallback,
                             &gate_passes, &words_hits, &cpu_verify_misses,
                             &solutions, max_results, &scanned_nonce_end);
            backend_used = cpu_fallback ? "cpu" : "hip";
        } else {
            found = SolveCPU(state, job_n, job_b, job_r,
                             block_target, share_target,
                             max_tries, max_seconds, tries_used, elapsed_s,
                             job_epsilon_bits);
            backend_used = "cpu";
        }

        bool is_block = found && Uint256LE(state.digest, block_target);
        std::vector<field::Element> product_payload;
        if (is_block && include_product_payload) {
            ComputeProductPayloadForNonce(
                state, job_n, job_b, job_r, product_payload);
        }
        uint64_t nonce64_end = backend_used == "hip"
            ? scanned_nonce_end
            : (found ? state.nonce
                : (state.nonce > nonce_start ? state.nonce - 1 : nonce_start));

        std::cout << "{";
        std::cout << "\"found\":" << (found ? "true" : "false") << ",";
        std::cout << "\"nonce64\":" << state.nonce << ",";
        std::cout << "\"nonce64_end\":" << nonce64_end << ",";
        if (found) {
            std::cout << "\"digest\":\"" << Uint256ToHex(state.digest) << "\",";
            std::cout << "\"ntime\":" << state.time << ",";
            std::cout << "\"is_block\":" << (is_block ? "true" : "false") << ",";
            if (!product_payload.empty()) {
                std::cout << "\"matrix_c\":[";
                for (size_t i = 0; i < product_payload.size(); ++i) {
                    if (i != 0) std::cout << ",";
                    std::cout << product_payload[i];
                }
                std::cout << "],";
            }
        }
        std::cout << "\"elapsed_s\":" << std::fixed << std::setprecision(2) << elapsed_s << ",";
        std::cout << "\"tries_used\":" << tries_used << ",";
        if (gate_passes > 0)
            std::cout << "\"gate_passes\":" << gate_passes << ",";
        if (words_hits > 0)
            std::cout << "\"words_hits\":" << words_hits << ",";
        if (cpu_verify_misses > 0)
            std::cout << "\"cpu_verify_misses\":" << cpu_verify_misses << ",";
        if (!solutions.empty()) {
            std::cout << "\"solutions\":[";
            for (size_t i = 0; i < solutions.size(); ++i) {
                const PowState& solution = solutions[i];
                const bool solution_is_block =
                    Uint256LE(solution.digest, block_target);
                if (i != 0) std::cout << ",";
                std::cout << "{";
                std::cout << "\"nonce64\":" << solution.nonce << ",";
                std::cout << "\"digest\":\"" << Uint256ToHex(solution.digest) << "\",";
                std::cout << "\"ntime\":" << solution.time << ",";
                std::cout << "\"is_block\":"
                          << (solution_is_block ? "true" : "false");
                std::cout << "}";
            }
            std::cout << "],";
        }
        std::cout << "\"backend\":\"" << backend_used << "\"";
        std::cout << "}" << std::endl;
    }

    return 0;
}
