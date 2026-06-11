#include "solve.h"
#include "matmul_kernel.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <vector>

struct GpuContext {
    bool initialized{false};
    int device_id{-1};
    hipStream_t compute_stream{nullptr};
};

static GpuContext g_gpu;

static bool InitGPU()
{
    if (g_gpu.initialized) {
        return true;
    }
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        return false;
    }

    int best_device = 0;
    if (device_count > 1) {
        size_t best_mem = 0;
        for (int i = 0; i < device_count; ++i) {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, i) == hipSuccess &&
                prop.totalGlobalMem > best_mem) {
                best_mem = prop.totalGlobalMem;
                best_device = i;
            }
        }
    }

    err = hipSetDevice(best_device);
    if (err != hipSuccess) {
        return false;
    }
    err = hipStreamCreateWithFlags(&g_gpu.compute_stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
        return false;
    }
    g_gpu.device_id = best_device;
    g_gpu.initialized = true;
    return true;
}

static int GpuDeviceId()
{
    return g_gpu.device_id;
}

static void InvalidateGpu()
{
    g_gpu.initialized = false;
}

bool SolveGPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
              const Uint256& block_target, const Uint256& share_target,
              uint64_t& max_tries, double max_seconds,
              uint64_t& tries_used, double& elapsed_s,
              uint32_t batch_size, uint32_t epsilon_bits,
              bool* cpu_fallback, uint64_t* gate_passes,
              uint64_t* words_hits, uint64_t* cpu_verify_misses)
{
    if (cpu_fallback) {
        *cpu_fallback = false;
    }
    if (gate_passes) {
        *gate_passes = 0;
    }
    if (words_hits) {
        *words_hits = 0;
    }
    if (cpu_verify_misses) {
        *cpu_verify_misses = 0;
    }
    if (!InitGPU()) {
        std::fprintf(stderr, "InitGPU failed, using CPU backend\n");
        if (cpu_fallback) {
            *cpu_fallback = true;
        }
        return SolveCPU(state, n, b, r, block_target, share_target,
                        max_tries, max_seconds, tries_used, elapsed_s, epsilon_bits);
    }

    auto t0 = std::chrono::steady_clock::now();
    tries_used = 0;

    const int device = GpuDeviceId();
    int launch_batch = static_cast<int>(batch_size);
    if (launch_batch <= 0) {
        launch_batch = AutoBatchSizeForDevice(device, n, r, b, state.height, 2048);
    } else {
        const int auto_max = AutoBatchSizeForDevice(device, n, r, b, state.height, 4096);
        launch_batch = std::min(launch_batch, auto_max);
    }
    launch_batch = std::max(1, launch_batch);

    FusedMatMulJob fused_job{};
    fused_job.state = state;
    fused_job.n = n;
    fused_job.b = b;
    fused_job.r = r;
    fused_job.epsilon_bits = epsilon_bits;
    fused_job.share_target = share_target;
    fused_job.block_target = block_target;

    uint64_t current_nonce = state.nonce;
    while (max_tries > 0) {
        auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t0).count();
        if (max_seconds > 0 && elapsed >= max_seconds) {
            break;
        }

        const uint64_t this_batch =
            std::min<uint64_t>(max_tries, static_cast<uint64_t>(launch_batch));
        std::vector<uint64_t> batch_nonces(this_batch);
        for (uint64_t i = 0; i < this_batch; ++i) {
            batch_nonces[i] = current_nonce + i;
        }

        std::vector<Uint256> digests;
        std::vector<bool> found;
        if (!LaunchMatMulTranscriptBatch(device, fused_job, batch_nonces, digests, found)) {
            std::fprintf(stderr, "Fused HIP kernel failed, falling back to CPU\n");
            InvalidateGpu();
            if (cpu_fallback) {
                *cpu_fallback = true;
            }
            state.nonce = current_nonce;
            auto t1 = std::chrono::steady_clock::now();
            elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            return SolveCPU(state, n, b, r, block_target, share_target,
                            max_tries, max_seconds, tries_used, elapsed_s, epsilon_bits);
        }

        tries_used += this_batch;
        max_tries -= this_batch;
        current_nonce += this_batch;
        if (gate_passes) {
            *gate_passes += this_batch;
        }

        uint32_t miss_logs = 0;
        for (size_t bi = 0; bi < batch_nonces.size(); ++bi) {
            if (!found[bi]) {
                continue;
            }
            if (words_hits) {
                ++(*words_hits);
            }

            PowState verify = state;
            verify.nonce = batch_nonces[bi];
            PrepareNonceSeeds(verify);

            Uint256 cpu_digest;
            ComputeDigestForNonce(verify, n, b, r, cpu_digest);
            const bool is_block = Uint256LE(cpu_digest, block_target);
            const bool is_share = !is_block && Uint256LE(cpu_digest, share_target);
            if (!is_block && !is_share) {
                if (cpu_verify_misses) {
                    ++(*cpu_verify_misses);
                }
                if (miss_logs < 2) {
                    std::fprintf(stderr,
                        "digest verify miss nonce=%llu gpu=%s cpu=%s\n",
                        static_cast<unsigned long long>(verify.nonce),
                        Uint256ToHex(digests[bi]).substr(0, 16).c_str(),
                        Uint256ToHex(cpu_digest).substr(0, 16).c_str());
                    ++miss_logs;
                }
                continue;
            }

            state = verify;
            state.digest = cpu_digest;
            auto t1 = std::chrono::steady_clock::now();
            elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            return true;
        }
    }

    state.nonce = current_nonce;
    auto t1 = std::chrono::steady_clock::now();
    elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    return false;
}