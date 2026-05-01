// =============================================================================
// cpu_baseline.cpp - CPU sequential speculative decoding implementation
// =============================================================================
// This is the straightforward sequential algorithm:
//   1. Check each draft token in order
//   2. Accept if random value < min(1, p/q ratio)
//   3. On first rejection, sample from residual distribution
//
// This serves as ground truth and benchmark baseline for the GPU version.
// =============================================================================

#include "cpu_baseline.h"
#include <cmath>
#include <algorithm>

CpuResult cpu_speculative_decode(
    const float* p_probs,       // [k x vocab_size] target model probabilities
    const float* q_probs,       // [k x vocab_size] draft model probabilities
    const int*   draft_tokens,  // [k] proposed token IDs
    const float* rand_vals,     // [k] random values for acceptance test
    float        rand_residual, // random value for residual sampling
    int          k,             // number of draft tokens
    int          vocab_size     // vocabulary size (50257)
) {
    CpuResult result;
    result.n_accepted = 0;
    result.first_rejected_idx = -1;
    result.bonus_token = -1;

    // =========================================================================
    // Phase 1: Sequential Verification
    // =========================================================================
    // For each draft token position i, compute the acceptance ratio:
    //   ratio = p(token_i) / q(token_i)
    // where p is the target model probability and q is the draft model probability.
    //
    // Accept if: uniform_random < min(1, ratio)
    //
    // Intuition: If the target model assigns higher probability than the draft
    // model (ratio >= 1), always accept. If lower (ratio < 1), accept with
    // probability proportional to the ratio.
    //
    // IMPORTANT: We must check sequentially because once a token is rejected,
    // all subsequent tokens are also rejected (they were generated conditioned
    // on the rejected token).

    for (int i = 0; i < k; i++) {
        int token_id = draft_tokens[i];

        // Look up probabilities for this specific token
        // Memory layout: row i starts at offset i * vocab_size
        float p_val = p_probs[i * vocab_size + token_id];  // target prob
        float q_val = q_probs[i * vocab_size + token_id];  // draft prob

        // Avoid division by zero: if draft assigns zero probability,
        // the token would never have been sampled, but handle gracefully
        float ratio;
        if (q_val < 1e-10f) {
            ratio = (p_val < 1e-10f) ? 1.0f : 1e10f;
        } else {
            ratio = p_val / q_val;
        }

        // Acceptance test: accept if random value < min(1, ratio)
        float threshold = std::fmin(1.0f, ratio);

        if (rand_vals[i] < threshold) {
            // Token accepted - continue to next token
            result.n_accepted++;
        } else {
            // Token rejected - stop here, record rejection position
            result.first_rejected_idx = i;
            break;
        }
    }

    // If all tokens were accepted, no residual sampling needed
    // (in practice, one would sample an additional token from the target
    // model at position k, but that's outside this verification step)
    if (result.first_rejected_idx == -1) {
        return result;
    }

    // =========================================================================
    // Phase 2: Residual Sampling
    // =========================================================================
    // At the first rejected position, we sample a new token from the
    // "residual" distribution:
    //
    //   residual(v) = max(0, p(v) - q(v))
    //
    // This distribution captures where the target model assigns MORE
    // probability than the draft model. By sampling from this, we
    // correct for the draft model's bias.
    //
    // The sampling uses inverse CDF (cumulative distribution function):
    //   1. Compute unnormalized residual for all vocab entries
    //   2. Sum for normalization constant
    //   3. Walk through vocab, accumulating CDF until it exceeds random threshold

    int rej = result.first_rejected_idx;
    const float* p_row = p_probs + rej * vocab_size;  // target probs at rejection
    const float* q_row = q_probs + rej * vocab_size;  // draft probs at rejection

    // --- Step 1 & 2: Compute residuals and normalization constant ---
    double norm_sum = 0.0;  // Use double for numerical stability
    for (int v = 0; v < vocab_size; v++) {
        float residual = std::fmax(0.0f, p_row[v] - q_row[v]);
        norm_sum += residual;
    }

    // --- Step 3: Inverse CDF sampling ---
    if (norm_sum <= 1e-10) {
        // Edge case: residual is zero everywhere (p == q exactly)
        // Fall back to sampling from target distribution directly
        double cdf = 0.0;
        for (int v = 0; v < vocab_size; v++) {
            cdf += p_row[v];
            if (cdf >= rand_residual) {
                result.bonus_token = v;
                return result;
            }
        }
        result.bonus_token = vocab_size - 1;
    } else {
        // Normal case: sample from normalized residual distribution
        double cdf = 0.0;
        double threshold = rand_residual * norm_sum;  // Pre-multiply to avoid
                                                      // division each iteration
        for (int v = 0; v < vocab_size; v++) {
            float residual = std::fmax(0.0f, p_row[v] - q_row[v]);
            cdf += residual;
            if (cdf >= threshold) {
                result.bonus_token = v;
                return result;
            }
        }
        // Floating point edge case: return last token
        result.bonus_token = vocab_size - 1;
    }

    return result;
}
