// =============================================================================
// residual_sample.cu - CUDA kernel for residual distribution sampling
// =============================================================================
//
// CUDA CONCEPTS EXPLAINED:
//
// Shared Memory (__shared__):
//   Fast on-chip memory (~100x faster than global memory) shared by all
//   threads in a block. Limited size (~48 KB). Perfect for intermediate
//   results that threads need to cooperate on (like our reduction sum).
//
// Parallel Reduction:
//   A pattern to compute a single value (sum, min, max) from many values.
//   Instead of one thread summing N values sequentially (O(N) time),
//   we use N/2 threads in parallel for O(log N) steps:
//
//   Step 0: Thread i adds element i and element i + N/2
//   Step 1: Thread i adds its result with result from i + N/4
//   ...and so on until one value remains.
//
//   Example with 8 values [a b c d e f g h]:
//   Step 0: 4 threads -> [a+e  b+f  c+g  d+h]
//   Step 1: 2 threads -> [a+e+c+g  b+f+d+h]
//   Step 2: 1 thread  -> [a+b+c+d+e+f+g+h]
//
// __syncthreads():
//   A barrier that forces ALL threads in a block to reach this point
//   before any thread proceeds. Essential when threads read data that
//   other threads have written to shared memory.
//
// =============================================================================

#include "residual_sample.cuh"
#include <cstdio>

// ---------------------------------------------------------------------------
// residual_sample_kernel: Compute and sample from residual distribution
// ---------------------------------------------------------------------------
__global__ void residual_sample_kernel(
    const float* p_probs,       // [k x vocab_size] target probs
    const float* q_probs,       // [k x vocab_size] draft probs
    int          rejected_idx,  // Which position (row) was rejected
    float        rand_val,      // Random value for inverse CDF sampling
    int          vocab_size,    // Vocabulary size V
    int*         sampled_token  // Output: sampled token ID
) {
    // ---- Shared memory for parallel reduction ----
    // Each thread will store its partial sum here, then we reduce.
    // __shared__ means this memory is allocated once per block, and all
    // threads in the block can read/write to it.
    __shared__ float partial_sums[RESIDUAL_BLOCK_SIZE];

    int tid = threadIdx.x;               // Thread index within this block
    int num_threads = blockDim.x;        // Total threads in block (= RESIDUAL_BLOCK_SIZE)

    // ---- Point to the correct row in the probability matrices ----
    // The rejected position's distributions are at row 'rejected_idx'
    const float* p_row = p_probs + rejected_idx * vocab_size;
    const float* q_row = q_probs + rejected_idx * vocab_size;

    // =====================================================================
    // Phase 1: Compute residuals and local partial sums
    // =====================================================================
    // Each thread processes multiple vocabulary entries in a strided pattern:
    //   Thread 0 handles indices: 0, 256, 512, ...
    //   Thread 1 handles indices: 1, 257, 513, ...
    //   etc.
    //
    // This "strided" access pattern ensures coalesced memory access:
    // adjacent threads read adjacent memory locations, which is much faster
    // on GPUs because the memory controller can batch these into one
    // transaction (a "coalesced" read of 128 bytes at once).

    float local_sum = 0.0f;
    for (int v = tid; v < vocab_size; v += num_threads) {
        float residual = fmaxf(0.0f, p_row[v] - q_row[v]);
        local_sum += residual;
    }

    // Store this thread's partial sum in shared memory
    partial_sums[tid] = local_sum;

    // ---- Barrier: wait for ALL threads to finish Phase 1 ----
    // Without this, some threads might start Phase 2 before others
    // have written their partial sums, leading to incorrect results.
    __syncthreads();

    // =====================================================================
    // Phase 2: Parallel reduction to compute total sum
    // =====================================================================
    // We reduce 256 partial sums down to 1 total sum in log2(256) = 8 steps.
    //
    // Stride starts at 128 and halves each step:
    //   stride=128: threads 0-127 add elements 128-255 to elements 0-127
    //   stride=64:  threads 0-63  add elements 64-127  to elements 0-63
    //   stride=32:  threads 0-31  add elements 32-63   to elements 0-31
    //   ...
    //   stride=1:   thread 0      adds element 1       to element 0
    //
    // After all steps, partial_sums[0] contains the total sum.

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        // Barrier: ensure all additions at this level complete before next level
        __syncthreads();
    }

    // Total normalization constant is now in partial_sums[0]
    float total_sum = partial_sums[0];

    // =====================================================================
    // Phase 3: Inverse CDF sampling (single thread)
    // =====================================================================
    // Only thread 0 performs the sampling. While this isn't parallel,
    // a sequential scan of 50257 floats takes ~50 microseconds on a GPU
    // thread, which is acceptable. A parallel prefix sum would add complexity
    // without meaningful speedup at this scale.

    if (tid == 0) {
        if (total_sum <= 1e-10f) {
            // Edge case: residual is zero everywhere (p ≈ q)
            // Fall back to sampling from target distribution
            float cdf = 0.0f;
            for (int v = 0; v < vocab_size; v++) {
                cdf += p_row[v];
                if (cdf >= rand_val) {
                    *sampled_token = v;
                    return;
                }
            }
            *sampled_token = vocab_size - 1;
            return;
        }

        // Normal case: sample from normalized residual distribution
        // We walk through the vocabulary, accumulating the CDF:
        //   CDF(v) = sum_{i=0}^{v} residual(i) / total_sum
        // The sampled token is the first v where CDF(v) >= rand_val
        //
        // Optimization: instead of dividing each residual by total_sum,
        // we multiply rand_val by total_sum (same comparison, fewer divisions)
        float threshold = rand_val * total_sum;
        float cumsum = 0.0f;

        for (int v = 0; v < vocab_size; v++) {
            float residual = fmaxf(0.0f, p_row[v] - q_row[v]);
            cumsum += residual;
            if (cumsum >= threshold) {
                *sampled_token = v;
                return;
            }
        }

        // Floating-point edge case: accumulated rounding errors might
        // prevent cumsum from reaching threshold. Return last token.
        *sampled_token = vocab_size - 1;
    }
}

// ---------------------------------------------------------------------------
// gpu_residual_sample: Host wrapper using pre-allocated device memory
// ---------------------------------------------------------------------------
int gpu_residual_sample(
    const float* d_p_probs,     // DEVICE pointer to target probs
    const float* d_q_probs,     // DEVICE pointer to draft probs
    int          rejected_idx,  // Position of first rejection
    float        rand_val,      // Random value for sampling
    int          vocab_size     // Vocabulary size
) {
    // Allocate device memory for the output token
    int* d_sampled_token;
    CUDA_CHECK(cudaMalloc(&d_sampled_token, sizeof(int)));

    // Launch kernel: 1 block, RESIDUAL_BLOCK_SIZE threads
    residual_sample_kernel<<<1, RESIDUAL_BLOCK_SIZE>>>(
        d_p_probs,
        d_q_probs,
        rejected_idx,
        rand_val,
        vocab_size,
        d_sampled_token
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    int h_sampled_token;
    CUDA_CHECK(cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_sampled_token));
    return h_sampled_token;
}

// ---------------------------------------------------------------------------
// gpu_full_speculative_decode: Complete GPU pipeline
// ---------------------------------------------------------------------------
void gpu_full_speculative_decode(
    const float* h_p_probs,       // Host: target probs [k x V]
    const float* h_q_probs,       // Host: draft probs [k x V]
    const int*   h_draft_tokens,  // Host: token IDs [k]
    const float* h_rand_vals,     // Host: random values [k]
    float        rand_residual,   // Random value for residual sampling
    int          k,               // Number of draft tokens
    int          vocab_size,      // Vocabulary size
    int*         out_n_accepted,  // Output: accepted count
    int*         out_bonus_token  // Output: bonus token from residual
) {
    // =====================================================================
    // Allocate all device memory upfront
    // =====================================================================
    size_t probs_bytes  = (size_t)k * vocab_size * sizeof(float);
    size_t tokens_bytes = (size_t)k * sizeof(int);
    size_t rand_bytes   = (size_t)k * sizeof(float);

    float* d_p_probs;
    float* d_q_probs;
    int*   d_draft_tokens;
    float* d_rand_vals;
    bool*  d_accepted;
    int*   d_first_rejected;

    CUDA_CHECK(cudaMalloc(&d_p_probs,        probs_bytes));
    CUDA_CHECK(cudaMalloc(&d_q_probs,        probs_bytes));
    CUDA_CHECK(cudaMalloc(&d_draft_tokens,   tokens_bytes));
    CUDA_CHECK(cudaMalloc(&d_rand_vals,      rand_bytes));
    CUDA_CHECK(cudaMalloc(&d_accepted,       (size_t)k * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_first_rejected, sizeof(int)));

    // =====================================================================
    // Upload data to device
    // =====================================================================
    CUDA_CHECK(cudaMemcpy(d_p_probs, h_p_probs, probs_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_probs, h_q_probs, probs_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft_tokens, h_draft_tokens, tokens_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rand_vals, h_rand_vals, rand_bytes,
                          cudaMemcpyHostToDevice));

    int init_val = k;
    CUDA_CHECK(cudaMemcpy(d_first_rejected, &init_val, sizeof(int),
                          cudaMemcpyHostToDevice));

    // =====================================================================
    // Step 1: Parallel verification
    // =====================================================================
    verify_tokens_kernel<<<1, k>>>(
        d_p_probs, d_q_probs, d_draft_tokens, d_rand_vals,
        d_accepted, d_first_rejected, vocab_size, k
    );
    CUDA_CHECK(cudaGetLastError());

    // =====================================================================
    // Step 2: Read first_rejected to decide if residual sampling is needed
    // =====================================================================
    int h_first_rejected;
    CUDA_CHECK(cudaMemcpy(&h_first_rejected, d_first_rejected, sizeof(int),
                          cudaMemcpyDeviceToHost));

    *out_n_accepted = h_first_rejected;  // All before first rejection are accepted
    *out_bonus_token = -1;

    // =====================================================================
    // Step 3: Residual sampling (only if there was a rejection)
    // =====================================================================
    if (h_first_rejected < k) {
        int* d_sampled_token;
        CUDA_CHECK(cudaMalloc(&d_sampled_token, sizeof(int)));

        residual_sample_kernel<<<1, RESIDUAL_BLOCK_SIZE>>>(
            d_p_probs, d_q_probs,
            h_first_rejected,
            rand_residual,
            vocab_size,
            d_sampled_token
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(out_bonus_token, d_sampled_token, sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_sampled_token));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // =====================================================================
    // Cleanup
    // =====================================================================
    CUDA_CHECK(cudaFree(d_p_probs));
    CUDA_CHECK(cudaFree(d_q_probs));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_rand_vals));
    CUDA_CHECK(cudaFree(d_accepted));
    CUDA_CHECK(cudaFree(d_first_rejected));
}
