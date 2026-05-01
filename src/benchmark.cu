// =============================================================================
// benchmark.cu - Benchmark GPU vs CPU speculative decoding across k values
// =============================================================================
//
// This file orchestrates the performance comparison:
//   1. Generates synthetic probability distributions (valid softmax outputs)
//   2. Times the GPU pipeline using cudaEvent
//   3. Times the CPU baseline using std::chrono
//   4. Outputs a formatted results table
//
// CUDA TIMING EXPLAINED:
//
// cudaEventRecord(event, stream):
//   Places a timestamp marker in the GPU's command queue (stream).
//   The event is "recorded" when the GPU reaches this point during execution.
//
// cudaEventSynchronize(event):
//   Blocks the CPU until the GPU has processed all commands up to and
//   including this event. After this call, we know the event has been recorded.
//
// cudaEventElapsedTime(&ms, start, stop):
//   Computes the time in milliseconds between two recorded events.
//   This gives the actual GPU execution time, independent of CPU activity.
//
// =============================================================================

#include "benchmark.cuh"
#include "verification.cuh"
#include "residual_sample.cuh"
#include "cpu_baseline.h"

#include <cuda_runtime.h>
#include <curand.h>         // cuRAND host API for generating random numbers on GPU

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// generate_synthetic_probs: Create fake but valid probability distributions
// ---------------------------------------------------------------------------
// Generates random probability distributions that sum to 1 for each row.
// This simulates what GPT-2 softmax outputs would look like.
//
// We generate on CPU for simplicity and reproducibility, then copy to GPU
// during the timed section (or exclude transfer from timing).
static void generate_synthetic_probs(
    std::vector<float>& probs,   // Output: [k x vocab_size] probabilities
    int k,                       // Number of draft positions
    int vocab_size,              // Vocabulary size
    unsigned int seed            // Random seed for reproducibility
) {
    probs.resize((size_t)k * vocab_size);
    std::srand(seed);

    for (int i = 0; i < k; i++) {
        float row_sum = 0.0f;
        // Generate random positive values
        for (int v = 0; v < vocab_size; v++) {
            // Use exponential-like distribution for more realistic probs
            // (most tokens have low prob, few have high prob)
            float raw = (float)std::rand() / RAND_MAX;
            raw = raw * raw * raw;  // Cube for skewness (mimics real LM distributions)
            probs[i * vocab_size + v] = raw + 1e-8f;  // Small epsilon to avoid zeros
            row_sum += probs[i * vocab_size + v];
        }
        // Normalize to sum to 1 (like softmax output)
        for (int v = 0; v < vocab_size; v++) {
            probs[i * vocab_size + v] /= row_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// generate_draft_tokens: Create random token IDs for draft model output
// ---------------------------------------------------------------------------
static void generate_draft_tokens(
    std::vector<int>& tokens,
    int k,
    int vocab_size,
    unsigned int seed
) {
    tokens.resize(k);
    std::srand(seed);
    for (int i = 0; i < k; i++) {
        tokens[i] = std::rand() % vocab_size;
    }
}

// ---------------------------------------------------------------------------
// BenchmarkResult: Stores timing for one configuration
// ---------------------------------------------------------------------------
struct BenchmarkResult {
    int    k;
    float  gpu_time_ms;   // Average GPU kernel time
    float  cpu_time_ms;   // Average CPU time
    float  speedup;       // cpu_time / gpu_time
    int    gpu_accepted;  // Tokens accepted by GPU
    int    cpu_accepted;  // Tokens accepted by CPU
};

// ---------------------------------------------------------------------------
// benchmark_single_k: Run benchmark for a specific k value
// ---------------------------------------------------------------------------
static BenchmarkResult benchmark_single_k(
    int k,
    int vocab_size,
    int num_trials
) {
    BenchmarkResult result;
    result.k = k;

    // ---- Generate synthetic test data ----
    std::vector<float> p_probs, q_probs;
    std::vector<int>   draft_tokens;

    // Use different seeds so p and q distributions differ (realistic scenario)
    generate_synthetic_probs(p_probs, k, vocab_size, 42);
    generate_synthetic_probs(q_probs, k, vocab_size, 123);
    generate_draft_tokens(draft_tokens, k, vocab_size, 456);

    // ---- Generate random values (same for GPU and CPU for fairness) ----
    std::vector<float> rand_vals(k);
    std::srand(789);
    for (int i = 0; i < k; i++) {
        rand_vals[i] = (float)std::rand() / RAND_MAX;
    }
    float rand_residual = (float)std::rand() / RAND_MAX;

    // =====================================================================
    // GPU Benchmark
    // =====================================================================
    // We pre-allocate device memory OUTSIDE the timing loop to measure
    // only the kernel execution time, not memory allocation overhead.

    size_t probs_bytes  = (size_t)k * vocab_size * sizeof(float);
    size_t tokens_bytes = (size_t)k * sizeof(int);
    size_t rand_bytes   = (size_t)k * sizeof(float);

    float* d_p_probs;
    float* d_q_probs;
    int*   d_draft_tokens;
    float* d_rand_vals;
    bool*  d_accepted;
    int*   d_first_rejected;
    int*   d_sampled_token;

    CUDA_CHECK(cudaMalloc(&d_p_probs,        probs_bytes));
    CUDA_CHECK(cudaMalloc(&d_q_probs,        probs_bytes));
    CUDA_CHECK(cudaMalloc(&d_draft_tokens,   tokens_bytes));
    CUDA_CHECK(cudaMalloc(&d_rand_vals,      rand_bytes));
    CUDA_CHECK(cudaMalloc(&d_accepted,       (size_t)k * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_first_rejected, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sampled_token,  sizeof(int)));

    // Upload data once (not timed - we measure kernel time only)
    CUDA_CHECK(cudaMemcpy(d_p_probs, p_probs.data(), probs_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_probs, q_probs.data(), probs_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft_tokens, draft_tokens.data(), tokens_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rand_vals, rand_vals.data(), rand_bytes,
                          cudaMemcpyHostToDevice));

    // ---- Create cudaEvents for GPU timing ----
    // cudaEvents are GPU-side timestamps. They provide sub-microsecond
    // accuracy for measuring GPU kernel execution time.
    cudaEvent_t gpu_start, gpu_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    // ---- Warmup run ----
    // The first kernel launch on a GPU has extra overhead: driver initialization,
    // JIT compilation of PTX code, context setup, etc. We do a warmup run
    // so these one-time costs don't pollute our measurements.
    {
        int init_val = k;
        CUDA_CHECK(cudaMemcpy(d_first_rejected, &init_val, sizeof(int),
                              cudaMemcpyHostToDevice));
        verify_tokens_kernel<<<1, k>>>(
            d_p_probs, d_q_probs, d_draft_tokens, d_rand_vals,
            d_accepted, d_first_rejected, vocab_size, k
        );
        residual_sample_kernel<<<1, RESIDUAL_BLOCK_SIZE>>>(
            d_p_probs, d_q_probs, 0, rand_residual, vocab_size, d_sampled_token
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Timed GPU runs ----
    float total_gpu_ms = 0.0f;
    int last_gpu_accepted = 0;

    for (int trial = 0; trial < num_trials; trial++) {
        // Reset first_rejected for each trial
        int init_val = k;
        CUDA_CHECK(cudaMemcpy(d_first_rejected, &init_val, sizeof(int),
                              cudaMemcpyHostToDevice));

        // ---- Record start event ----
        // This places a timestamp in the GPU command stream
        CUDA_CHECK(cudaEventRecord(gpu_start));

        // ---- Launch verification kernel ----
        verify_tokens_kernel<<<1, k>>>(
            d_p_probs, d_q_probs, d_draft_tokens, d_rand_vals,
            d_accepted, d_first_rejected, vocab_size, k
        );

        // We need first_rejected on the host to decide whether to launch
        // the residual kernel. This requires a sync + device-to-host copy.
        // In a production system, you'd use CUDA streams/graphs to avoid this.
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_first_rejected;
        CUDA_CHECK(cudaMemcpy(&h_first_rejected, d_first_rejected, sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (h_first_rejected < k) {
            // Launch residual sampling kernel
            residual_sample_kernel<<<1, RESIDUAL_BLOCK_SIZE>>>(
                d_p_probs, d_q_probs,
                h_first_rejected,
                rand_residual,
                vocab_size,
                d_sampled_token
            );
        }

        // ---- Record stop event ----
        CUDA_CHECK(cudaEventRecord(gpu_stop));

        // ---- Wait for GPU to finish ----
        CUDA_CHECK(cudaEventSynchronize(gpu_stop));

        // ---- Compute elapsed time ----
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, gpu_start, gpu_stop));
        total_gpu_ms += ms;
        last_gpu_accepted = h_first_rejected;
    }

    result.gpu_time_ms = total_gpu_ms / num_trials;
    result.gpu_accepted = std::min(last_gpu_accepted, k);

    // Cleanup GPU timing events
    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));

    // Free device memory
    CUDA_CHECK(cudaFree(d_p_probs));
    CUDA_CHECK(cudaFree(d_q_probs));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_rand_vals));
    CUDA_CHECK(cudaFree(d_accepted));
    CUDA_CHECK(cudaFree(d_first_rejected));
    CUDA_CHECK(cudaFree(d_sampled_token));

    // =====================================================================
    // CPU Benchmark
    // =====================================================================
    // Using std::chrono::high_resolution_clock for CPU timing.
    // This is the standard C++ way to measure elapsed time with high precision.

    // ---- Warmup ----
    {
        CpuResult dummy = cpu_speculative_decode(
            p_probs.data(), q_probs.data(), draft_tokens.data(),
            rand_vals.data(), rand_residual, k, vocab_size
        );
        (void)dummy;  // Suppress unused variable warning
    }

    // ---- Timed CPU runs ----
    auto cpu_start = std::chrono::high_resolution_clock::now();

    CpuResult cpu_result;
    for (int trial = 0; trial < num_trials; trial++) {
        cpu_result = cpu_speculative_decode(
            p_probs.data(), q_probs.data(), draft_tokens.data(),
            rand_vals.data(), rand_residual, k, vocab_size
        );
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        cpu_end - cpu_start
    );

    result.cpu_time_ms = (float)cpu_duration.count() / (num_trials * 1e6f);
    result.cpu_accepted = cpu_result.n_accepted;

    // ---- Compute speedup ----
    // speedup > 1 means GPU is faster, < 1 means CPU is faster
    if (result.gpu_time_ms > 0.0f) {
        result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    } else {
        result.speedup = 0.0f;
    }

    return result;
}

// ---------------------------------------------------------------------------
// print_separator: Print a horizontal rule for the table
// ---------------------------------------------------------------------------
static void print_separator(int total_width) {
    for (int i = 0; i < total_width; i++) printf("-");
    printf("\n");
}

// ---------------------------------------------------------------------------
// run_benchmarks: Execute full benchmark suite and print results
// ---------------------------------------------------------------------------
void run_benchmarks(int vocab_size, int num_trials) {
    // Print GPU device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n");
    printf("==========================================================\n");
    printf("  SPECULATIVE DECODING BENCHMARK\n");
    printf("==========================================================\n");
    printf("  GPU Device:       %s\n", prop.name);
    printf("  Compute Cap:      %d.%d\n", prop.major, prop.minor);
    printf("  SM Count:         %d\n", prop.multiProcessorCount);
    printf("  Global Memory:    %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Vocab Size:       %d\n", vocab_size);
    printf("  Trials per k:     %d\n", num_trials);
    printf("==========================================================\n\n");

    // ---- Define k values to test ----
    int k_values[] = {1, 3, 5, 10, 20};
    int num_k = sizeof(k_values) / sizeof(k_values[0]);

    // ---- Run benchmarks ----
    std::vector<BenchmarkResult> results;
    for (int i = 0; i < num_k; i++) {
        int k = k_values[i];
        printf("  Benchmarking k=%2d ...", k);
        fflush(stdout);

        BenchmarkResult r = benchmark_single_k(k, vocab_size, num_trials);
        results.push_back(r);

        printf(" done (GPU: %.4f ms, CPU: %.4f ms)\n",
               r.gpu_time_ms, r.cpu_time_ms);
    }

    // =====================================================================
    // Print Results Table
    // =====================================================================
    printf("\n");
    int col_w = 78;
    print_separator(col_w);
    printf("| %-8s | %4s | %12s | %12s | %9s | %8s | %8s |\n",
           "Method", "k", "Time (ms)", "Kernel (ms)", "Speedup",
           "GPU Acc.", "CPU Acc.");
    print_separator(col_w);

    for (const auto& r : results) {
        // GPU row
        printf("| %-8s | %4d | %12.6f | %12.6f | %9s | %8d | %8s |\n",
               "GPU", r.k, r.gpu_time_ms, r.gpu_time_ms, "-",
               r.gpu_accepted, "-");
        // CPU row
        char speedup_str[32];
        snprintf(speedup_str, sizeof(speedup_str), "%.2fx", r.speedup);
        printf("| %-8s | %4d | %12.6f | %12s | %9s | %8s | %8d |\n",
               "CPU", r.k, r.cpu_time_ms, "-", speedup_str,
               "-", r.cpu_accepted);
        print_separator(col_w);
    }

    // =====================================================================
    // Summary
    // =====================================================================
    printf("\n");
    printf("  SUMMARY TABLE (simplified):\n\n");
    printf("  +--------+------+------------+----------+\n");
    printf("  | Method | k    | Time (ms)  | Speedup  |\n");
    printf("  +--------+------+------------+----------+\n");

    for (const auto& r : results) {
        printf("  | GPU    | %4d | %10.6f | %8s |\n",
               r.k, r.gpu_time_ms, "-");
        printf("  | CPU    | %4d | %10.6f | %7.2fx |\n",
               r.k, r.cpu_time_ms, r.speedup);
        printf("  +--------+------+------------+----------+\n");
    }

    printf("\n");
    printf("  NOTE: Speedup = CPU_time / GPU_time\n");
    printf("  - Speedup > 1.0 means GPU is faster than CPU\n");
    printf("  - Speedup < 1.0 means CPU is faster than GPU\n");
    printf("  - For small k, CPU may be faster due to GPU launch overhead\n");
    printf("  - GPU advantage grows with larger vocabulary sizes\n");
    printf("\n");
}

// =============================================================================
// =============================================================================
//   BATCHED BENCHMARKS - This is where the GPU SHINES!
// =============================================================================
// =============================================================================
//
// The single-sequence benchmark above uses only k threads (1-20).
// A GPU with 16 SMs and 1024 threads/SM can run 16,384 threads at once.
// With k=5 and 1 sequence, we use 5 threads — 0.03% utilization!
//
// In real inference servers, you process MANY sequences simultaneously
// (batch serving). With batch_size=512 and k=5, we have:
//   - Verification: 512 * 5 = 2,560 threads
//   - Residual sampling: 512 blocks * 256 threads = 131,072 threads
// Now the GPU is properly utilized and will outperform the CPU.
// =============================================================================

// ---------------------------------------------------------------------------
// batched_verify_kernel: Verify tokens for B sequences in parallel
// ---------------------------------------------------------------------------
// Each thread handles one token from one sequence.
// Total threads needed: batch_size * k
//
// Memory layout (all flattened):
//   p_probs[seq_idx * k * V + tok_idx * V + v]  = P_target for sequence seq_idx,
//                                                  position tok_idx, vocab v
__global__ void batched_verify_kernel(
    const float* p_probs,         // [B*k, V] target probs (all sequences concatenated)
    const float* q_probs,         // [B*k, V] draft probs
    const int*   draft_tokens,    // [B*k] token IDs
    const float* rand_vals,       // [B*k] random values
    int*         first_rejected,  // [B] output: first rejection per sequence
    int          vocab_size,
    int          k,
    int          total_tokens     // = batch_size * k
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_tokens) return;

    // Which sequence and which position within that sequence
    int seq_idx = tid / k;
    int tok_idx = tid % k;

    int token_id = draft_tokens[tid];
    float p_val = p_probs[tid * vocab_size + token_id];
    float q_val = q_probs[tid * vocab_size + token_id];

    float ratio = (q_val < 1e-10f)
                  ? ((p_val < 1e-10f) ? 1.0f : 1e10f)
                  : (p_val / q_val);
    float threshold = fminf(1.0f, ratio);

    if (rand_vals[tid] >= threshold) {
        // Rejected: atomicMin to find earliest rejection in this sequence
        atomicMin(&first_rejected[seq_idx], tok_idx);
    }
}

// ---------------------------------------------------------------------------
// batched_residual_kernel: Residual sampling for B sequences in parallel
// ---------------------------------------------------------------------------
// Launch config: <<<batch_size, 256>>>
// Each BLOCK handles one sequence's residual sampling.
// blockIdx.x = sequence index
__global__ void batched_residual_kernel(
    const float* p_probs,          // [B*k, V]
    const float* q_probs,          // [B*k, V]
    const int*   first_rejected,   // [B] rejection positions
    const float* rand_residuals,   // [B] random values for sampling
    int*         bonus_tokens,     // [B] output: sampled tokens
    int          vocab_size,
    int          k
) {
    __shared__ float partial_sums[256];

    int batch_idx = blockIdx.x;     // One block per sequence
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int rej_idx = first_rejected[batch_idx];

    // If no rejection in this sequence, no bonus token needed
    if (rej_idx >= k) {
        if (tid == 0) bonus_tokens[batch_idx] = -1;
        return;
    }

    // Row index in the flattened [B*k, V] array
    int row = batch_idx * k + rej_idx;
    const float* p_row = p_probs + row * vocab_size;
    const float* q_row = q_probs + row * vocab_size;

    // Phase 1: Compute residuals and partial sums (strided access)
    float local_sum = 0.0f;
    for (int v = tid; v < vocab_size; v += num_threads) {
        local_sum += fmaxf(0.0f, p_row[v] - q_row[v]);
    }

    partial_sums[tid] = local_sum;
    __syncthreads();

    // Phase 2: Parallel reduction
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    float total = partial_sums[0];

    // Phase 3: Thread 0 does CDF sampling
    if (tid == 0) {
        if (total <= 1e-10f) {
            bonus_tokens[batch_idx] = 0;
            return;
        }
        float thresh = rand_residuals[batch_idx] * total;
        float cumsum = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            cumsum += fmaxf(0.0f, p_row[v] - q_row[v]);
            if (cumsum >= thresh) {
                bonus_tokens[batch_idx] = v;
                return;
            }
        }
        bonus_tokens[batch_idx] = vocab_size - 1;
    }
}

// ---------------------------------------------------------------------------
// run_batched_benchmarks: Benchmark batched GPU vs CPU
// ---------------------------------------------------------------------------
void run_batched_benchmarks(int vocab_size, int num_trials) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n");
    printf("==========================================================\n");
    printf("  BATCHED SPECULATIVE DECODING BENCHMARK\n");
    printf("==========================================================\n");
    printf("  GPU Device:       %s\n", prop.name);
    printf("  SM Count:         %d\n", prop.multiProcessorCount);
    printf("  Fixed k:          5 (draft tokens per sequence)\n");
    printf("  Vocab Size:       %d\n", vocab_size);
    printf("  Trials:           %d\n", num_trials);
    printf("  Batch sizes:      1, 16, 64, 256, 512\n");
    printf("==========================================================\n\n");

    const int k = 5;
    int batch_sizes[] = {1, 16, 64, 256, 512};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    // Header
    printf("  +--------+-------+--------------+--------------+----------+\n");
    printf("  | Method | Batch | Time (ms)    | Seq/ms       | Speedup  |\n");
    printf("  +--------+-------+--------------+--------------+----------+\n");

    for (int bi = 0; bi < num_batches; bi++) {
        int B = batch_sizes[bi];
        int total_tokens = B * k;
        size_t total_rows = (size_t)B * k;

        printf("  Benchmarking batch=%d ...", B);
        fflush(stdout);

        // ---- Generate synthetic data for B sequences ----
        // Flatten: [B*k, V] for probs, [B*k] for tokens
        std::vector<float> p_probs(total_rows * vocab_size);
        std::vector<float> q_probs(total_rows * vocab_size);
        std::vector<int>   draft_tokens(total_tokens);
        std::vector<float> rand_vals(total_tokens);
        std::vector<float> rand_residuals(B);

        std::srand(42);
        // Generate normalized probability rows
        for (size_t row = 0; row < total_rows; row++) {
            float row_sum = 0.0f;
            for (int v = 0; v < vocab_size; v++) {
                float raw = (float)std::rand() / RAND_MAX;
                raw = raw * raw * raw + 1e-8f;
                p_probs[row * vocab_size + v] = raw;
                row_sum += raw;
            }
            for (int v = 0; v < vocab_size; v++)
                p_probs[row * vocab_size + v] /= row_sum;
        }

        std::srand(123);
        for (size_t row = 0; row < total_rows; row++) {
            float row_sum = 0.0f;
            for (int v = 0; v < vocab_size; v++) {
                float raw = (float)std::rand() / RAND_MAX;
                raw = raw * raw * raw + 1e-8f;
                q_probs[row * vocab_size + v] = raw;
                row_sum += raw;
            }
            for (int v = 0; v < vocab_size; v++)
                q_probs[row * vocab_size + v] /= row_sum;
        }

        std::srand(456);
        for (int i = 0; i < total_tokens; i++)
            draft_tokens[i] = std::rand() % vocab_size;

        std::srand(789);
        for (int i = 0; i < total_tokens; i++)
            rand_vals[i] = (float)std::rand() / RAND_MAX;
        for (int i = 0; i < B; i++)
            rand_residuals[i] = (float)std::rand() / RAND_MAX;

        // =================================================================
        // GPU: Batched kernels
        // =================================================================
        size_t probs_bytes     = total_rows * vocab_size * sizeof(float);
        size_t tokens_bytes    = total_tokens * sizeof(int);
        size_t rand_bytes      = total_tokens * sizeof(float);
        size_t rej_bytes       = B * sizeof(int);
        size_t rand_res_bytes  = B * sizeof(float);
        size_t bonus_bytes     = B * sizeof(int);

        float* d_p; float* d_q; int* d_tokens; float* d_rand;
        int* d_rej; float* d_rand_res; int* d_bonus;

        CUDA_CHECK(cudaMalloc(&d_p,        probs_bytes));
        CUDA_CHECK(cudaMalloc(&d_q,        probs_bytes));
        CUDA_CHECK(cudaMalloc(&d_tokens,   tokens_bytes));
        CUDA_CHECK(cudaMalloc(&d_rand,     rand_bytes));
        CUDA_CHECK(cudaMalloc(&d_rej,      rej_bytes));
        CUDA_CHECK(cudaMalloc(&d_rand_res, rand_res_bytes));
        CUDA_CHECK(cudaMalloc(&d_bonus,    bonus_bytes));

        CUDA_CHECK(cudaMemcpy(d_p, p_probs.data(), probs_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q, q_probs.data(), probs_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tokens, draft_tokens.data(), tokens_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rand, rand_vals.data(), rand_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rand_res, rand_residuals.data(), rand_res_bytes,
                              cudaMemcpyHostToDevice));

        // Kernel launch config for verification
        int threads = 256;
        int blocks_verify = (total_tokens + threads - 1) / threads;

        // Initialize first_rejected to k for all sequences
        std::vector<int> init_rej(B, k);

        cudaEvent_t gpu_start, gpu_stop;
        CUDA_CHECK(cudaEventCreate(&gpu_start));
        CUDA_CHECK(cudaEventCreate(&gpu_stop));

        // Warmup
        CUDA_CHECK(cudaMemcpy(d_rej, init_rej.data(), rej_bytes,
                              cudaMemcpyHostToDevice));
        batched_verify_kernel<<<blocks_verify, threads>>>(
            d_p, d_q, d_tokens, d_rand, d_rej, vocab_size, k, total_tokens);
        batched_residual_kernel<<<B, 256>>>(
            d_p, d_q, d_rej, d_rand_res, d_bonus, vocab_size, k);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        float total_gpu_ms = 0.0f;
        for (int t = 0; t < num_trials; t++) {
            CUDA_CHECK(cudaMemcpy(d_rej, init_rej.data(), rej_bytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(gpu_start));

            batched_verify_kernel<<<blocks_verify, threads>>>(
                d_p, d_q, d_tokens, d_rand, d_rej, vocab_size, k, total_tokens);
            batched_residual_kernel<<<B, 256>>>(
                d_p, d_q, d_rej, d_rand_res, d_bonus, vocab_size, k);

            CUDA_CHECK(cudaEventRecord(gpu_stop));
            CUDA_CHECK(cudaEventSynchronize(gpu_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, gpu_start, gpu_stop));
            total_gpu_ms += ms;
        }
        float avg_gpu_ms = total_gpu_ms / num_trials;

        CUDA_CHECK(cudaEventDestroy(gpu_start));
        CUDA_CHECK(cudaEventDestroy(gpu_stop));
        CUDA_CHECK(cudaFree(d_p));
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_tokens));
        CUDA_CHECK(cudaFree(d_rand));
        CUDA_CHECK(cudaFree(d_rej));
        CUDA_CHECK(cudaFree(d_rand_res));
        CUDA_CHECK(cudaFree(d_bonus));

        // =================================================================
        // CPU: Process B sequences sequentially
        // =================================================================
        auto cpu_start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < num_trials; t++) {
            for (int b = 0; b < B; b++) {
                const float* p_row = p_probs.data() + (size_t)b * k * vocab_size;
                const float* q_row = q_probs.data() + (size_t)b * k * vocab_size;
                const int*   tok   = draft_tokens.data() + b * k;
                const float* rv    = rand_vals.data() + b * k;

                cpu_speculative_decode(p_row, q_row, tok, rv,
                                       rand_residuals[b], k, vocab_size);
            }
        }

        auto cpu_end = std::chrono::high_resolution_clock::now();
        float avg_cpu_ms = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(
            cpu_end - cpu_start).count() / (num_trials * 1e6f);

        float speedup = (avg_gpu_ms > 0.0f) ? avg_cpu_ms / avg_gpu_ms : 0.0f;
        float gpu_seq_per_ms = (avg_gpu_ms > 0.0f) ? B / avg_gpu_ms : 0.0f;
        float cpu_seq_per_ms = (avg_cpu_ms > 0.0f) ? B / avg_cpu_ms : 0.0f;

        printf(" done\n");

        // Print results
        printf("  | GPU    | %5d | %12.4f | %10.1f   | %8s |\n",
               B, avg_gpu_ms, gpu_seq_per_ms, "-");
        char sp[32];
        snprintf(sp, sizeof(sp), "%.2fx", speedup);
        printf("  | CPU    | %5d | %12.4f | %10.1f   | %8s |\n",
               B, avg_cpu_ms, cpu_seq_per_ms, sp);
        printf("  +--------+-------+--------------+--------------+----------+\n");
    }

    printf("\n");
    printf("  Speedup > 1.0  =>  GPU is FASTER\n");
    printf("  Seq/ms = throughput (sequences processed per millisecond)\n");
    printf("\n");
}

