// =============================================================================
// verification.cuh - CUDA kernel declarations for parallel token verification
// =============================================================================
// CUDA Concepts Used:
//   __global__    - Marks a function as a GPU kernel (callable from host)
//   atomicMin     - Thread-safe minimum operation across parallel threads
//   cudaMemcpy    - Transfer data between CPU (host) and GPU (device) memory
//   cudaMalloc    - Allocate memory on the GPU
//   cudaFree      - Free GPU memory
// =============================================================================

#ifndef VERIFICATION_CUH
#define VERIFICATION_CUH

#include <cuda_runtime.h>  // CUDA runtime API (cudaMalloc, cudaMemcpy, etc.)
#include <cstdio>

// ---------------------------------------------------------------------------
// CUDA_CHECK: Error checking macro for all CUDA API calls
// ---------------------------------------------------------------------------
// Every CUDA function returns a cudaError_t status code. This macro checks
// if the operation succeeded and prints a helpful error message if not.
// ALWAYS wrap CUDA calls with this macro during development!
#define CUDA_CHECK(call) do {                                                \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA Error at %s:%d - %s (code %d)\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err), (int)err);      \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while(0)

// ---------------------------------------------------------------------------
// GpuVerificationResult: Output of GPU-based verification
// ---------------------------------------------------------------------------
struct GpuVerificationResult {
    int  n_accepted;          // Number of accepted tokens (0 to k)
    int  first_rejected_idx;  // Index of first rejection (-1 if all accepted)
    int  bonus_token;         // Token from residual sampling (-1 if none)
    bool* d_accepted;         // Device array of per-token acceptance flags
};

// ---------------------------------------------------------------------------
// verify_tokens_kernel: Parallel verification of draft tokens
// ---------------------------------------------------------------------------
// Launch configuration:  <<<1, k>>>   (1 block, k threads)
//
// Each thread i independently checks whether draft token i should be accepted:
//   ratio = p[i][token_i] / q[i][token_i]
//   accepted[i] = (rand_vals[i] < min(1.0, ratio))
//
// Thread-safe minimum via atomicMin finds the first rejected position.
//
// Parameters:
//   p_probs        - [k x vocab_size] target model probabilities (device ptr)
//   q_probs        - [k x vocab_size] draft model probabilities (device ptr)
//   draft_tokens   - [k] token IDs proposed by draft model (device ptr)
//   rand_vals      - [k] uniform random values in [0,1) for acceptance (device ptr)
//   accepted       - [k] output: per-token acceptance flags (device ptr)
//   first_rejected - [1] output: index of first rejected token (device ptr)
//                    initialized to k (meaning "none rejected")
//   vocab_size     - vocabulary size V (e.g., 50257)
//   k              - number of draft tokens
__global__ void verify_tokens_kernel(
    const float* p_probs,
    const float* q_probs,
    const int*   draft_tokens,
    const float* rand_vals,
    bool*        accepted,
    int*         first_rejected,
    int          vocab_size,
    int          k
);

// ---------------------------------------------------------------------------
// gpu_verify_tokens: Host wrapper for launching the verification kernel
// ---------------------------------------------------------------------------
// Manages device memory allocation, kernel launch, and result retrieval.
// All GPU memory is allocated and freed within this function.
//
// Parameters:
//   p_probs       - [k x vocab_size] target probs (host pointer)
//   q_probs       - [k x vocab_size] draft probs (host pointer)
//   draft_tokens  - [k] token IDs (host pointer)
//   rand_vals     - [k] random values (host pointer)
//   k             - number of draft tokens
//   vocab_size    - vocabulary size
//   h_accepted    - [k] output: per-token acceptance (host pointer, pre-allocated)
//
// Returns:
//   Index of first rejected token, or k if all accepted
int gpu_verify_tokens(
    const float* p_probs,
    const float* q_probs,
    const int*   draft_tokens,
    const float* rand_vals,
    int          k,
    int          vocab_size,
    bool*        h_accepted
);

#endif // VERIFICATION_CUH
