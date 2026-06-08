#include "solve.h"
#include "sha256.h"
#include "transcript.h"
#include "gpu_sha256.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <vector>
#include <atomic>

namespace {

__host__ __device__ __forceinline__ Element GPUReduce64(uint64_t x) {
    uint64_t fold1 = (x & uint64_t(MODULUS)) + (x >> 31);
    uint32_t lo = uint32_t(fold1 & MODULUS);
    uint32_t hi = uint32_t(fold1 >> 31);
    uint32_t result = lo + hi;
    uint32_t ge_mask = uint32_t(-int32_t(result >= MODULUS));
    result -= (MODULUS & ge_mask);
    return result;
}

__host__ __device__ __forceinline__ Element GPUFieldAdd(Element a, Element b) {
    uint32_t s = a + b;
    if (s >= MODULUS) s -= MODULUS;
    return s;
}

__host__ __device__ __forceinline__ Element GPUFieldMul(Element a, Element b) {
    return GPUReduce64(uint64_t(a) * b);
}

constexpr uint32_t MAX_BLOCK_THREADS = 256;
constexpr uint32_t WORKSPACE_THREADS = 256;

__device__ __forceinline__ void ReducePartialsInPlace(Element* partials, uint32_t tid) {
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] = GPUFieldAdd(partials[tid], partials[tid + stride]);
        }
        __syncthreads();
    }
}

__global__ void BuildPerturbedMatrixKernel(
    const Element* base_matrix,
    const Element* noise_left,
    const Element* noise_right,
    uint32_t n, uint32_t r,
    size_t total_elements,
    uint32_t matrix_elements,
    uint32_t noise_left_elements,
    uint32_t noise_right_elements,
    Element* output)
{
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;
    uint32_t batch_index = static_cast<uint32_t>(gid / matrix_elements);
    uint32_t local_index = static_cast<uint32_t>(gid % matrix_elements);
    uint32_t row = local_index / n;
    uint32_t col = local_index % n;
    const Element* nl = noise_left + size_t(batch_index) * noise_left_elements;
    const Element* nr = noise_right + size_t(batch_index) * noise_right_elements;
    uint64_t acc = 0;
    uint32_t pending = 0;
    for (uint32_t k = 0; k < r; ++k) {
        acc += uint64_t(nl[row * r + k]) * nr[k * n + col];
        if (++pending == 4) { acc = GPUReduce64(acc); pending = 0; }
    }
    output[gid] = GPUFieldAdd(base_matrix[local_index], GPUReduce64(acc));
}

template <bool PrefixMode>
__global__ void ComputeCompressedWordsFusedKernel(
    const Element* __restrict__ matrix_a_batch,
    const Element* __restrict__ matrix_b_batch,
    const Element* __restrict__ compress_batch,
    uint32_t n, uint32_t block_size,
    uint32_t blocks_per_axis,
    uint32_t pair_count_per_request,
    uint32_t words_per_request,
    uint32_t matrix_elements,
    uint32_t compress_elements,
    Element* __restrict__ output)
{
    __shared__ Element partials[MAX_BLOCK_THREADS];
    uint32_t pair_index = blockIdx.x;
    uint32_t batch_index = pair_index / pair_count_per_request;
    uint32_t local_pair_index = pair_index % pair_count_per_request;
    uint32_t j = local_pair_index % blocks_per_axis;
    uint32_t i = local_pair_index / blocks_per_axis;
    uint32_t tid = threadIdx.x;
    uint32_t active_threads = block_size * block_size;
    const Element* matrix_a = matrix_a_batch + size_t(batch_index) * matrix_elements;
    const Element* matrix_b = matrix_b_batch + size_t(batch_index) * matrix_elements;
    const Element* compress_vec = compress_batch + size_t(batch_index) * compress_elements;
    uint32_t output_offset = batch_index * words_per_request + (PrefixMode ? local_pair_index * blocks_per_axis : local_pair_index);
    bool active = tid < active_threads;
    uint32_t x = active ? (tid / block_size) : 0;
    uint32_t y = active ? (tid % block_size) : 0;
    uint32_t row = i * block_size + x;
    uint32_t col = j * block_size + y;
    size_t row_offset = size_t(row) * n;
    Element compress_coeff = active ? compress_vec[tid] : 0;
    Element running_total = 0;
    for (uint32_t ell = 0; ell < blocks_per_axis; ++ell) {
        Element partial = 0;
        if (active) {
            uint32_t middle_base = ell * block_size;
            uint64_t acc = 0;
            uint32_t pending = 0;
            for (uint32_t k = 0; k < block_size; ++k) {
                acc += uint64_t(matrix_a[row_offset + (middle_base + k)]) * matrix_b[size_t(middle_base + k) * n + col];
                if (++pending == 4) { acc = GPUReduce64(acc); pending = 0; }
            }
            partial = GPUFieldMul(GPUReduce64(acc), compress_coeff);
        }
        partials[tid] = partial;
        __syncthreads();
        ReducePartialsInPlace(partials, tid);
        if (tid == 0) {
            running_total = GPUFieldAdd(running_total, partials[0]);
            if constexpr (PrefixMode) {
                output[output_offset + ell] = running_total;
            }
        }
        __syncthreads();
    }
    if constexpr (!PrefixMode) {
        if (tid == 0) output[output_offset] = running_total;
    }
}

uint32_t ResolveThreadCount(uint32_t block_size) {
    uint32_t threads = 1;
    uint32_t required = std::min<uint32_t>(block_size * block_size, MAX_BLOCK_THREADS);
    while (threads < required) threads <<= 1;
    return std::min<uint32_t>(threads, MAX_BLOCK_THREADS);
}

static bool EnsureBuffer(Element*& ptr, size_t& capacity, size_t required) {
    if (capacity >= required) return true;
    hipFree(ptr);
    hipError_t err = hipMalloc(&ptr, required * sizeof(Element));
    if (err != hipSuccess) { ptr = nullptr; capacity = 0; return false; }
    capacity = required;
    return true;
}

static bool EnsureBufferU8(uint8_t*& ptr, size_t& capacity, size_t required) {
    if (capacity >= required) return true;
    hipFree(ptr);
    hipError_t err = hipMalloc(&ptr, required);
    if (err != hipSuccess) { ptr = nullptr; capacity = 0; return false; }
    capacity = required;
    return true;
}

static bool EnsureBufferI32(int32_t*& ptr, size_t& capacity, size_t required) {
    if (capacity >= required) return true;
    hipFree(ptr);
    hipError_t err = hipMalloc(&ptr, required * sizeof(int32_t));
    if (err != hipSuccess) { ptr = nullptr; capacity = 0; return false; }
    capacity = required;
    return true;
}

struct GpuContext {
  bool initialized{false};
  int device_id{-1};
};
static GpuContext g_gpu;

static bool InitGPU() {
  if (g_gpu.initialized) return true;
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) return false;
  err = hipSetDevice(0);
  if (err != hipSuccess) return false;
  g_gpu.device_id = 0;
  g_gpu.initialized = true;
  return true;
}

static Matrix FromSeedGPU(const Uint256& seed, uint32_t n) {
    Matrix out(n, n);
    for (uint32_t row = 0; row < n; ++row)
        for (uint32_t col = 0; col < n; ++col)
            out.at(row, col) = field::from_oracle(seed, row * n + col);
    return out;
}

static Uint256 LeftShiftUint256(const Uint256& val, uint32_t shift) {
    Uint256 result;
    std::memset(result.data, 0, 32);
    if (shift >= 256) return result;
    uint32_t byte_shift = shift / 8;
    uint32_t bit_shift = shift % 8;
    for (int i = 31; i >= 0; --i) {
        int src = i - static_cast<int>(byte_shift);
        if (src < 0) continue;
        result.data[i] = val.data[src] << bit_shift;
        int prev = src - 1;
        if (prev >= 0 && bit_shift > 0)
            result.data[i] |= val.data[prev] >> (8 - bit_shift);
    }
    return result;
}

struct GateEntry {
    uint64_t nonce;
    Uint256 sigma;
};

}

bool SolveGPU(PowState& state, uint32_t n, uint32_t b, uint32_t r,
              const Uint256& block_target, const Uint256& share_target,
              uint64_t& max_tries, double max_seconds,
              uint64_t& tries_used, double& elapsed_s,
              uint32_t batch_size, uint32_t epsilon_bits)
{
  if (!InitGPU())
    return SolveCPU(state, n, b, r, block_target, share_target,
      max_tries, max_seconds, tries_used, elapsed_s);

  auto t0 = std::chrono::steady_clock::now();
    tries_used = 0;
    if (batch_size < 1) batch_size = 1;
    if (batch_size > 1024) batch_size = 1024;

    uint32_t matrix_elements = n * n;
    uint32_t noise_left_elements = n * r;
    uint32_t noise_right_elements = r * n;
    uint32_t compress_elements = b * b;
    uint32_t blocks_per_axis = n / b;
    uint32_t pair_count_per_request = blocks_per_axis * blocks_per_axis;
    uint32_t words_per_request = pair_count_per_request;
    uint32_t thread_count = ResolveThreadCount(b);

    Uint256 sigma_gate_threshold = LeftShiftUint256(share_target, epsilon_bits);

  Matrix A = FromSeedGPU(state.seed_a, n);
  Matrix B = FromSeedGPU(state.seed_b, n);

    Element* d_base_a{nullptr}; size_t cap_base_a{0};
    Element* d_base_b{nullptr}; size_t cap_base_b{0};
    Element* d_noise_e_l{nullptr}; size_t cap_nl{0};
    Element* d_noise_e_r{nullptr}; size_t cap_nr{0};
    Element* d_noise_f_l{nullptr}; size_t cap_fl{0};
    Element* d_noise_f_r{nullptr}; size_t cap_fr{0};
    Element* d_compress{nullptr}; size_t cap_cv{0};
    Element* d_matrix_a{nullptr}; size_t cap_ma{0};
    Element* d_matrix_b{nullptr}; size_t cap_mb{0};
    Element* d_output{nullptr}; size_t cap_out{0};
    uint8_t* d_sigma_batch{nullptr}; size_t cap_sigma{0};
    uint8_t* d_noise_seeds{nullptr}; size_t cap_nseeds{0};
    uint8_t* d_compress_seeds{nullptr}; size_t cap_cseeds{0};

    auto cleanup = [&]() {
        hipFree(d_base_a); hipFree(d_base_b);
        hipFree(d_noise_e_l); hipFree(d_noise_e_r);
        hipFree(d_noise_f_l); hipFree(d_noise_f_r);
        hipFree(d_compress);
        hipFree(d_matrix_a); hipFree(d_matrix_b);
        hipFree(d_output);
        hipFree(d_sigma_batch); hipFree(d_noise_seeds);
        hipFree(d_compress_seeds);
    };

if (!EnsureBuffer(d_base_a, cap_base_a, matrix_elements) ||
    !EnsureBuffer(d_base_b, cap_base_b, matrix_elements) ||
    !EnsureBuffer(d_matrix_a, cap_ma, matrix_elements) ||
    !EnsureBuffer(d_matrix_b, cap_mb, matrix_elements) ||
    !EnsureBuffer(d_output, cap_out, words_per_request) ||
    !EnsureBufferU8(d_sigma_batch, cap_sigma, size_t(batch_size) * 32) ||
    !EnsureBufferU8(d_noise_seeds, cap_nseeds, size_t(batch_size) * 4 * 32) ||
    !EnsureBufferU8(d_compress_seeds, cap_cseeds, size_t(batch_size) * 32))
{
    cleanup();
    return SolveCPU(state, n, b, r, block_target, share_target,
                    max_tries, max_seconds, tries_used, elapsed_s);
}

  hipMemcpy(d_base_a, A.data(), matrix_elements * sizeof(Element), hipMemcpyHostToDevice);
  hipMemcpy(d_base_b, B.data(), matrix_elements * sizeof(Element), hipMemcpyHostToDevice);

  PowState base_state = state;
uint64_t current_nonce = state.nonce;

  while (max_tries > 0) {
    auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (max_seconds > 0 && elapsed >= max_seconds) break;

    std::vector<GateEntry> gate_entries;
    gate_entries.reserve(batch_size);

    while (gate_entries.size() < batch_size && max_tries > 0) {
            now = std::chrono::steady_clock::now();
            elapsed = std::chrono::duration<double>(now - t0).count();
            if (max_seconds > 0 && elapsed >= max_seconds) break;

            base_state.nonce = current_nonce++;
            --max_tries;
            ++tries_used;

            Uint256 sigma = DeriveSigma(base_state);
            if (epsilon_bits == 0 || Uint256LE(sigma, sigma_gate_threshold)) {
                gate_entries.push_back({base_state.nonce, sigma});
            }
        }

    if (gate_entries.empty()) continue;

    uint32_t actual_batch = static_cast<uint32_t>(gate_entries.size());

        if (!EnsureBufferU8(d_sigma_batch, cap_sigma, size_t(actual_batch) * 32) ||
            !EnsureBufferU8(d_noise_seeds, cap_nseeds, size_t(actual_batch) * 4 * 32) ||
            !EnsureBufferU8(d_compress_seeds, cap_cseeds, size_t(actual_batch) * 32) ||
            !EnsureBuffer(d_noise_e_l, cap_nl, size_t(actual_batch) * noise_left_elements) ||
            !EnsureBuffer(d_noise_e_r, cap_nr, size_t(actual_batch) * noise_right_elements) ||
            !EnsureBuffer(d_noise_f_l, cap_fl, size_t(actual_batch) * noise_left_elements) ||
            !EnsureBuffer(d_noise_f_r, cap_fr, size_t(actual_batch) * noise_right_elements) ||
            !EnsureBuffer(d_compress, cap_cv, size_t(actual_batch) * compress_elements) ||
            !EnsureBuffer(d_matrix_a, cap_ma, size_t(actual_batch) * matrix_elements) ||
            !EnsureBuffer(d_matrix_b, cap_mb, size_t(actual_batch) * matrix_elements) ||
            !EnsureBuffer(d_output, cap_out, size_t(actual_batch) * words_per_request))
        {
            cleanup();
            return SolveCPU(state, n, b, r, block_target, share_target,
                            max_tries, max_seconds, tries_used, elapsed_s);
        }

    std::vector<uint8_t> host_sigma(size_t(actual_batch) * 32);
    for (uint32_t i = 0; i < actual_batch; ++i)
      std::memcpy(host_sigma.data() + size_t(i) * 32, gate_entries[i].sigma.data, 32);
    hipMemcpy(d_sigma_batch, host_sigma.data(), host_sigma.size(), hipMemcpyHostToDevice);

    gpasha::DeriveNoiseSeedsKernel_launch(
      d_sigma_batch, d_noise_seeds, d_compress_seeds, actual_batch);

    gpasha::GenerateNoiseKernel_launch(
      d_noise_seeds, actual_batch, noise_left_elements, 0u, d_noise_e_l);
    gpasha::GenerateNoiseKernel_launch(
      d_noise_seeds, actual_batch, noise_left_elements, 2u, d_noise_f_l);
    gpasha::GenerateNoiseKernel_launch(
      d_noise_seeds, actual_batch, noise_right_elements, 1u, d_noise_e_r);
    gpasha::GenerateNoiseKernel_launch(
      d_noise_seeds, actual_batch, noise_right_elements, 3u, d_noise_f_r);

    gpasha::GenerateCompressKernel_launch(
      d_compress_seeds, actual_batch, compress_elements, d_compress);

    size_t batch_matrix_elements = size_t(actual_batch) * matrix_elements;
        uint32_t build_blocks = (batch_matrix_elements + WORKSPACE_THREADS - 1) / WORKSPACE_THREADS;
        hipLaunchKernelGGL(BuildPerturbedMatrixKernel,
            dim3(build_blocks), dim3(WORKSPACE_THREADS), 0, 0,
            d_base_a, d_noise_e_l, d_noise_e_r, n, r,
            batch_matrix_elements, matrix_elements,
            noise_left_elements, noise_right_elements, d_matrix_a);
    hipLaunchKernelGGL(BuildPerturbedMatrixKernel,
      dim3(build_blocks), dim3(WORKSPACE_THREADS), 0, 0,
      d_base_b, d_noise_f_l, d_noise_f_r, n, r,
      batch_matrix_elements, matrix_elements,
      noise_left_elements, noise_right_elements, d_matrix_b);

    uint32_t total_pairs = actual_batch * pair_count_per_request;
        hipLaunchKernelGGL(ComputeCompressedWordsFusedKernel<false>,
            dim3(total_pairs), dim3(thread_count), 0, 0,
            d_matrix_a, d_matrix_b, d_compress,
            n, b, blocks_per_axis, pair_count_per_request,
            words_per_request, matrix_elements, compress_elements, d_output);

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
            cleanup();
            return SolveCPU(state, n, b, r, block_target, share_target,
                            max_tries, max_seconds, tries_used, elapsed_s);
        }

        size_t total_output_elements = size_t(actual_batch) * words_per_request;
    std::vector<Element> host_output(total_output_elements);
    hipMemcpy(host_output.data(), d_output, total_output_elements * sizeof(Element), hipMemcpyDeviceToHost);

    for (uint32_t bi = 0; bi < actual_batch; ++bi) {
            const Element* batch_words = host_output.data() + size_t(bi) * words_per_request;
            Uint256 c_prime_hash = transcript::HashMatrixWords(batch_words, words_per_request);
            Uint256 digest = transcript::FinalizeProductCommittedDigest(
                c_prime_hash, gate_entries[bi].sigma, n, b);
            bool is_block = Uint256LE(digest, block_target);
            if (is_block || Uint256LE(digest, share_target)) {
                state.nonce = gate_entries[bi].nonce;
                state.digest = digest;
                auto t1 = std::chrono::steady_clock::now();
                elapsed_s = std::chrono::duration<double>(t1 - t0).count();
                cleanup();
                return true;
            }
        }
    }

    state.nonce = current_nonce;
    auto t1 = std::chrono::steady_clock::now();
    elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    cleanup();
    return false;
}
