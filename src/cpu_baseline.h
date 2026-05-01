// =============================================================================
// cpu_baseline.h - CPU sequential implementation of speculative decoding
// =============================================================================
// This serves as the reference/baseline implementation for benchmarking.
// The CPU version processes tokens sequentially, which is how speculative
// decoding naturally works: check token 0, then 1, etc., stopping at the
// first rejection.
// =============================================================================

#ifndef CPU_BASELINE_H
#define CPU_BASELINE_H

// ---------------------------------------------------------------------------
// CpuResult: Holds the output of CPU speculative decoding verification
// ---------------------------------------------------------------------------
struct CpuResult {
    int n_accepted;          // Number of consecutively accepted draft tokens
    int first_rejected_idx;  // Index of the first rejected token (-1 if all accepted)
    int bonus_token;         // Token sampled from residual distribution (at rejection point)
};

// ---------------------------------------------------------------------------
// cpu_speculative_decode: Sequential CPU verification + residual sampling
// ---------------------------------------------------------------------------
// Parameters:
//   p_probs       - Target model probabilities, row-major [k x vocab_size]
//                   p_probs[i * vocab_size + v] = P_target(token v | position i)
//   q_probs       - Draft model probabilities, row-major [k x vocab_size]
//                   q_probs[i * vocab_size + v] = P_draft(token v | position i)
//   draft_tokens  - Token IDs proposed by the draft model [k]
//   rand_vals     - Pre-generated uniform random values for acceptance [k]
//                   Using the same random values as GPU for fair comparison
//   rand_residual - Random value for residual sampling [1]
//   k             - Number of draft tokens to verify
//   vocab_size    - Vocabulary size (50257 for GPT-2)
//
// Returns:
//   CpuResult with acceptance count, rejection index, and bonus token
//
// Algorithm:
//   For i = 0, 1, ..., k-1:
//     ratio = p[i][draft_tokens[i]] / q[i][draft_tokens[i]]
//     if rand_vals[i] < min(1, ratio):
//       accept token i
//     else:
//       reject token i, stop checking further tokens
//       sample bonus token from residual distribution (p - q)+ / ||...||
CpuResult cpu_speculative_decode(
    const float* p_probs,
    const float* q_probs,
    const int*   draft_tokens,
    const float* rand_vals,
    float        rand_residual,
    int          k,
    int          vocab_size
);

#endif // CPU_BASELINE_H
