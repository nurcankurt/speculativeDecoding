// =============================================================================
// residual_sample.cuh - CUDA kernel declarations for residual sampling
// =============================================================================
// After verification rejects a token, we need to sample a replacement from
// the "residual" distribution:
//
//   residual(v) = max(0, p(v) - q(v))    for each vocab token v
//
// This captures where the target model (p) assigns MORE probability than
// the draft model (q). Sampling from this distribution corrects the output
// to match the target model's distribution exactly.
//
// CUDA Concepts Used:
//   __shared__ memory  - Fast on-chip memory shared within a thread block
//   Parallel reduction - Pattern for computing sum/min/max across threads
//   __syncthreads()    - Barrier synchronization within a block
// =============================================================================

#ifndef RESIDUAL_SAMPLE_CUH
#define RESIDUAL_SAMPLE_CUH

#include <cuda_runtime.h>
#include "verification.cuh"  // For CUDA_CHECK macro

// Block size for the residual sampling kernel
// 256 threads is a good default: not too few (underutilized), not too many
// (register pressure). Each thread processes ceil(50257/256) ≈ 197 vocab entries.
#define RESIDUAL_BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// residual_sample_kernel: Sample a token from the residual distribution
// ---------------------------------------------------------------------------
// Launch configuration: <<<1, RESIDUAL_BLOCK_SIZE>>>
//
// This kernel performs three phases:
//   Phase 1: Compute residual[v] = max(0, p[v] - q[v]) and partial sums
//   Phase 2: Parallel reduction to get total normalization constant
//   Phase 3: Thread 0 performs inverse CDF sampling
//
// Parameters:
//   p_probs       - [k x vocab_size] target model probs (device ptr)
//   q_probs       - [k x vocab_size] draft model probs (device ptr)
//   rejected_idx  - Index of the rejected position (which row to use)
//   rand_val      - Random value in [0,1) for sampling
//   vocab_size    - Vocabulary size V
//   sampled_token - [1] output: the sampled token ID (device ptr)
__global__ void residual_sample_kernel(
    const float* p_probs,
    const float* q_probs,
    int          rejected_idx,
    float        rand_val,
    int          vocab_size,
    int*         sampled_token
);

// ---------------------------------------------------------------------------
// gpu_residual_sample: Host wrapper for residual sampling
// ---------------------------------------------------------------------------
// Launches the residual sampling kernel and returns the sampled token.
// Uses pre-allocated device memory for p_probs and q_probs (passed as
// device pointers) to avoid redundant transfers when called after
// verification.
//
// Parameters:
//   d_p_probs     - [k x vocab_size] target probs (DEVICE pointer)
//   d_q_probs     - [k x vocab_size] draft probs (DEVICE pointer)
//   rejected_idx  - Position of first rejected token
//   rand_val      - Random value for sampling
//   vocab_size    - Vocabulary size
//
// Returns:
//   Sampled token ID from residual distribution
int gpu_residual_sample(
    const float* d_p_probs,
    const float* d_q_probs,
    int          rejected_idx,
    float        rand_val,
    int          vocab_size
);

// ---------------------------------------------------------------------------
// gpu_full_speculative_decode: Combined verification + residual sampling
// ---------------------------------------------------------------------------
// Performs the complete speculative decoding pipeline on the GPU:
//   1. Upload data to device
//   2. Run parallel verification kernel
//   3. If any token rejected, run residual sampling kernel
//   4. Download results to host
//
// Parameters:
//   h_p_probs       - [k x V] target probs (host pointer)
//   h_q_probs       - [k x V] draft probs (host pointer)
//   h_draft_tokens  - [k] token IDs (host pointer)
//   h_rand_vals     - [k] random values for verification (host pointer)
//   rand_residual   - Random value for residual sampling
//   k               - Number of draft tokens
//   vocab_size      - Vocabulary size
//   out_n_accepted  - Output: number of accepted tokens
//   out_bonus_token - Output: token from residual sampling (-1 if all accepted)
void gpu_full_speculative_decode(
    const float* h_p_probs,
    const float* h_q_probs,
    const int*   h_draft_tokens,
    const float* h_rand_vals,
    float        rand_residual,
    int          k,
    int          vocab_size,
    int*         out_n_accepted,
    int*         out_bonus_token
);

#endif // RESIDUAL_SAMPLE_CUH
