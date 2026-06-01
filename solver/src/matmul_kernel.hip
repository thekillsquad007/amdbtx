#include <hip/hip_runtime.h>
#include <cstdint>

// M31 field helpers (device-side)
__device__ __forceinline__ uint32_t field_reduce64(uint64_t x) {
    x = (x & 0x7FFFFFFF) + (x >> 31);
    x = (x & 0x7FFFFFFF) + (x >> 31);
    if (x >= 2147483647u) x -= 2147483647u;
    return (uint32_t)x;
}

__device__ __forceinline__ uint32_t field_add(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    if (r >= 2147483647u) r -= 2147483647u;
    return r;
}

__device__ __forceinline__ uint32_t field_mul(uint32_t a, uint32_t b) {
    return field_reduce64((uint64_t)a * b);
}

// Each thread block computes one compressed word for one (i, j, l) triple.
// Block index maps to (i, j, l) via: l = idx % nb, j = (idx/nb) % nb, i = idx/(nb*nb)
// Each block has b*b = 256 threads. Each thread handles one element of the b*b block.
//
// Steps:
//   1. Load A_noisy[i*b+bi, l*b+0..b] and B_noisy[l*b+0..b, j*b+bj] into shared memory
//   2. Thread computes its dot product (b multiplications)
//   3. Multiply by compression vector element
//   4. Block-wide reduction to sum all b*b contributions
//   5. Thread 0 writes compressed word to output buffer
//
// Note: A_noisy and B_noisy are the noise-augmented matrices, already computed on the host.
// The EL/ER/FL/FR parameters are kept for API compatibility but unused in this kernel.

extern "C" __global__ void matmul_kernel(
    const uint32_t* __restrict__ A,       // A_noisy: n x n, row-major
    const uint32_t* __restrict__ B,       // B_noisy: n x n, row-major
    const uint32_t* __restrict__ EL,      // n x r (unused — noise folded into A/B on host)
    const uint32_t* __restrict__ ER,      // r x n (unused)
    const uint32_t* __restrict__ FL,      // n x r (unused)
    const uint32_t* __restrict__ FR,      // r x n (unused)
    const uint32_t* __restrict__ compress_vec, // b x b
    uint32_t* __restrict__ out_words,     // output compressed words
    uint32_t* __restrict__ out_counter,   // atomic counter for output position
    uint32_t n, uint32_t b, uint32_t r,
    uint32_t target_check,
    const uint32_t* __restrict__ share_target)
{
    uint32_t block_idx = blockIdx.x;
    uint32_t nb = n / b;

    // Decode 3D index
    uint32_t l = block_idx % nb;
    uint32_t tmp = block_idx / nb;
    uint32_t j = tmp % nb;
    uint32_t i = tmp / nb;

    uint32_t tid = threadIdx.x;
    uint32_t bi = tid / b;  // row within the b×b block
    uint32_t bj = tid % b;  // col within the b×b block

    uint32_t row_a = i * b + bi;   // row in A_noisy
    uint32_t col_a = l * b + bj;   // col in A_noisy (== row in B_noisy)
    uint32_t row_b = l * b + bi;   // row in B_noisy
    uint32_t col_b = j * b + bj;   // col in B_noisy

    // Shared memory: A tile (b x b) + B tile (b x b) + reduction buffer (b x b)
    extern __shared__ uint32_t smem[];
    uint32_t* sA = smem;
    uint32_t* sB = smem + b * b;
    uint32_t* sReduce = smem + 2 * b * b;

    // Load tiles from device global memory
    sA[bi * b + bj] = A[row_a * n + col_a];
    sB[bi * b + bj] = B[row_b * n + col_b];
    __syncthreads();

    // Each thread computes the dot product of row bi of A-tile and col bj of B-tile
    uint32_t dot_val = 0;
    for (uint32_t t = 0; t < b; t++) {
        dot_val = field_add(dot_val, field_mul(sA[bi * b + t], sB[t * b + bj]));
    }

    // Multiply by compression vector element
    uint32_t compressed = field_mul(dot_val, compress_vec[bi * b + bj]);

    // Write to reduction buffer
    sReduce[bi * b + bj] = compressed;
    __syncthreads();

    // Tree reduction: sum all b*b = 256 elements
    for (uint32_t stride = (b * b) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sReduce[tid] = field_add(sReduce[tid], sReduce[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes the final compressed word
    if (tid == 0) {
        uint32_t pos = atomicAdd(out_counter, 1);
        out_words[pos] = sReduce[0];
    }
}

// Host-side launch wrapper
void matmul_launch(
    const uint32_t* d_A, const uint32_t* d_B,
    const uint32_t* d_EL, const uint32_t* d_ER,
    const uint32_t* d_FL, const uint32_t* d_FR,
    const uint32_t* d_compress_vec,
    uint32_t* d_out_words, uint32_t* d_out_counter,
    uint32_t n, uint32_t b, uint32_t r,
    hipStream_t stream)
{
    uint32_t nb = n / b;
    uint32_t total_blocks = nb * nb * nb;
    uint32_t threads_per_block = b * b; // 256

    size_t smem_size = 3 * b * b * sizeof(uint32_t);

    // Zero the counter
    uint32_t zero = 0;
    hipMemcpy(d_out_counter, &zero, sizeof(uint32_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(
        matmul_kernel,
        dim3(total_blocks),
        dim3(threads_per_block),
        smem_size,
        stream,
        d_A, d_B, d_EL, d_ER, d_FL, d_FR,
        d_compress_vec, d_out_words, d_out_counter,
        n, b, r, 0, nullptr
    );
}
