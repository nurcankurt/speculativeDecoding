// =============================================================================
// verification.cu - CUDA kernel implementation for parallel token verification
// =============================================================================
//
// CUDA CONCEPTS EXPLAINED FOR BEGINNERS:
//
// GPU Execution Model:
//   - A GPU runs thousands of lightweight threads simultaneously
//   - Threads are organized into "blocks" (groups), blocks form a "grid"
//   - Each thread has a unique ID: threadIdx.x (within block) + blockIdx.x * blockDim.x
//
// Memory Hierarchy:
//   - Global memory: Large, slow, accessible by all threads (like RAM)
//   - Shared memory: Small, fast, shared within a block (like L1 cache)
//   - Registers: Fastest, private to each thread
//
// __global__: Keyword marking a function as a "kernel" - runs on GPU, called from CPU
// __device__: Keyword marking a function that runs on GPU, called from GPU code only
//
// atomicMin: Ensures thread-safe comparison-and-update when multiple threads
//            write to the same memory location simultaneously
//
// <<<blocks, threads>>>: Kernel launch syntax specifying grid dimensions
//
// =============================================================================

#include "verification.cuh"
#include <cstdio>

// ---------------------------------------------------------------------------
// verify_tokens_kernel: Each thread verifies one draft token in parallel
// ---------------------------------------------------------------------------
__global__ void verify_tokens_kernel(
    const float* p_probs,       // Target model probs [k x vocab_size], device memory
    const float* q_probs,       // Draft model probs [k x vocab_size], device memory
    const int*   draft_tokens,  // Proposed token IDs [k], device memory
    const float* rand_vals,     // Random values for acceptance [k], device memory
    bool*        accepted,      // Output: per-token acceptance [k], device memory
    int*         first_rejected,// Output: min rejected index [1], device memory
    int          vocab_size,    // Vocabulary size (e.g., 50257)
    int          k              // Number of draft tokens
) {
    // ---- Compute this thread's token index ----
    // threadIdx.x: Thread index within the block (0 to blockDim.x - 1)
    // blockIdx.x:  Block index within the grid
    // blockDim.x:  Number of threads per block
    // For our launch config <<<1, k>>>, blockIdx.x is always 0,
    // so tid simply equals threadIdx.x
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Guard: Don't process threads beyond k
    // This is crucial when blockDim.x might be larger than k
    if (tid >= k) return;

    // ---- Look up the draft token for this position ----
    int token_id = draft_tokens[tid];

    // ---- Compute acceptance ratio ----
    // p_val: probability the TARGET model assigns to this token at position tid
    // q_val: probability the DRAFT model assigns to this token at position tid
    //
    // Memory access pattern: p_probs is stored as a flat 1D array in row-major order
    // Row tid starts at offset tid * vocab_size
    // The specific token's probability is at offset tid * vocab_size + token_id
    float p_val = p_probs[tid * vocab_size + token_id];
    float q_val = q_probs[tid * vocab_size + token_id];

    // Compute ratio = P_target / P_draft
    // Handle edge case: if draft probability is ~0, set a large ratio
    // (target likely wouldn't have sampled this token either, but be safe)
    float ratio;
    if (q_val < 1e-10f) {
        ratio = (p_val < 1e-10f) ? 1.0f : 1e10f;
    } else {
        ratio = p_val / q_val;
    }

    // ---- Acceptance test ----
    // Accept if: rand_vals[tid] < min(1.0, ratio)
    //
    // When ratio >= 1.0 (target likes this token MORE than draft):
    //   -> threshold = 1.0, always accept
    // When ratio < 1.0 (target likes this token LESS than draft):
    //   -> threshold = ratio, accept with probability = ratio
    float threshold = fminf(1.0f, ratio);

    if (rand_vals[tid] < threshold) {
        accepted[tid] = true;   // Token accepted
    } else {
        accepted[tid] = false;  // Token rejected

        // ---- Record this rejection using atomicMin ----
        // Multiple threads may reject simultaneously. We need the MINIMUM
        // rejected index because all tokens after the first rejection are
        // also invalid (they were generated assuming the rejected token).
        //
        // atomicMin atomically computes: *first_rejected = min(*first_rejected, tid)
        // This is thread-safe even when multiple threads call it concurrently.
        atomicMin(first_rejected, tid);
    }
}

// ---------------------------------------------------------------------------
// gpu_verify_tokens: Host-side wrapper that manages the full GPU pipeline
// ---------------------------------------------------------------------------
int gpu_verify_tokens(
    const float* h_p_probs,       // Host: target probs [k x vocab_size]
    const float* h_q_probs,       // Host: draft probs [k x vocab_size]
    const int*   h_draft_tokens,  // Host: token IDs [k]
    const float* h_rand_vals,     // Host: random values [k]
    int          k,               // Number of draft tokens
    int          vocab_size,      // Vocabulary size
    bool*        h_accepted       // Host output: acceptance array [k]
) {
    // =====================================================================
    // Step 1: Calculate memory sizes
    // =====================================================================
    size_t probs_bytes  = (size_t)k * vocab_size * sizeof(float);  // [k x V] floats
    size_t tokens_bytes = (size_t)k * sizeof(int);                 // [k] ints
    size_t rand_bytes   = (size_t)k * sizeof(float);               // [k] floats
    size_t accept_bytes = (size_t)k * sizeof(bool);                // [k] bools

    // =====================================================================
    // Step 2: Allocate GPU (device) memory
    // =====================================================================
    // cudaMalloc allocates memory on the GPU's global memory
    // The first argument receives a pointer to the allocated device memory
    // IMPORTANT: Device pointers cannot be dereferenced on the CPU!
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
    CUDA_CHECK(cudaMalloc(&d_accepted,       accept_bytes));
    CUDA_CHECK(cudaMalloc(&d_first_rejected, sizeof(int)));

    // =====================================================================
    // Step 3: Copy input data from host (CPU) to device (GPU)
    // =====================================================================
    // cudaMemcpy directions:
    //   cudaMemcpyHostToDevice   - CPU -> GPU
    //   cudaMemcpyDeviceToHost   - GPU -> CPU
    //   cudaMemcpyDeviceToDevice - GPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_p_probs, h_p_probs, probs_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_probs, h_q_probs, probs_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft_tokens, h_draft_tokens, tokens_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rand_vals, h_rand_vals, rand_bytes,
                          cudaMemcpyHostToDevice));

    // Initialize first_rejected to k (meaning "no rejection found yet")
    int init_val = k;
    CUDA_CHECK(cudaMemcpy(d_first_rejected, &init_val, sizeof(int),
                          cudaMemcpyHostToDevice));

    // =====================================================================
    // Step 4: Launch the kernel
    // =====================================================================
    // Kernel launch syntax: kernel<<<num_blocks, threads_per_block>>>(args...)
    //
    // For small k (1-20), we use a single block with k threads.
    // This is simple but doesn't fully utilize the GPU (GPUs have thousands
    // of cores). The verification step is intentionally lightweight - the
    // real parallelism opportunity is in residual sampling over the vocabulary.
    int threads_per_block = k;
    int num_blocks = 1;

    verify_tokens_kernel<<<num_blocks, threads_per_block>>>(
        d_p_probs,
        d_q_probs,
        d_draft_tokens,
        d_rand_vals,
        d_accepted,
        d_first_rejected,
        vocab_size,
        k
    );

    // ---- Check for kernel launch errors ----
    // cudaGetLastError returns any error from the most recent kernel launch.
    // Kernel launches are asynchronous, so errors may not appear immediately.
    CUDA_CHECK(cudaGetLastError());

    // cudaDeviceSynchronize waits for ALL GPU operations to complete.
    // This ensures any execution errors are caught here.
    CUDA_CHECK(cudaDeviceSynchronize());

    // =====================================================================
    // Step 5: Copy results back to host
    // =====================================================================
    CUDA_CHECK(cudaMemcpy(h_accepted, d_accepted, accept_bytes,
                          cudaMemcpyDeviceToHost));

    int h_first_rejected;
    CUDA_CHECK(cudaMemcpy(&h_first_rejected, d_first_rejected, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // =====================================================================
    // Step 6: Free device memory
    // =====================================================================
    // Always free allocated GPU memory to prevent leaks!
    // GPU memory is limited (typically 4-80 GB) and precious.
    CUDA_CHECK(cudaFree(d_p_probs));
    CUDA_CHECK(cudaFree(d_q_probs));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_rand_vals));
    CUDA_CHECK(cudaFree(d_accepted));
    CUDA_CHECK(cudaFree(d_first_rejected));

    return h_first_rejected;  // k if all accepted, else index of first rejection
}
