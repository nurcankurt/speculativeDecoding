// =============================================================================
// stream_pipeline.cu - CUDA Streams: Overlapping Transfer and Compute
// =============================================================================
//
// LECTURE 6: CUDA Streams and Concurrency
//
// PROBLEM:
//   Without streams, the GPU pipeline looks like this (sequential):
//     [Transfer batch_0 to GPU] -> [Run kernel on batch_0] ->
//     [Transfer batch_1 to GPU] -> [Run kernel on batch_1] -> ...
//   The GPU sits idle during transfers, and the PCIe bus sits idle during kernels.
//
// SOLUTION - Two streams + double buffering:
//   Stream 0: [Transfer batch_0] -> [Kernel batch_0] -> [Transfer batch_2] -> ...
//   Stream 1:          [Transfer batch_1] -> [Kernel batch_1] -> ...
//
//   Timeline:
//     |--Transfer_0--|--Transfer_2--|
//                  |--Kernel_0--|--Kernel_2--|
//          |--Transfer_1--|--Transfer_3--|
//                       |--Kernel_1--|--Kernel_3--|
//
//   The overlapping sections = FREE speedup (hardware runs them in parallel)
//
// KEY REQUIREMENT:
//   cudaMemcpyAsync ONLY overlaps if the host buffer is PINNED (page-locked).
//   With pageable memory, CUDA silently falls back to synchronous transfer.
//   This is why pinned memory (Lecture 4) and streams (Lecture 6) go together.
//
// =============================================================================

#include "stream_pipeline.cuh"
#include "benchmark.cuh"      // For batched_verify_kernel declaration
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Reuse batched_verify_kernel from benchmark.cu (already compiled in)
// We declare it here so stream_pipeline.cu can call it.
// ---------------------------------------------------------------------------
extern __global__ void batched_verify_kernel(
    const float* p_probs,
    const float* q_probs,
    const int*   draft_tokens,
    const float* rand_vals,
    int*         first_rejected,
    int          vocab_size,
    int          k,
    int          total_tokens
);

// ---------------------------------------------------------------------------
// fill_synthetic_batch: Generate dummy data into PINNED host buffers
// ---------------------------------------------------------------------------
// In a real system this would be: copy HuggingFace tensor outputs here.
// We use synthetic data so the benchmark can run standalone.
static void fill_synthetic_batch(
    float* h_p, float* h_q, int* h_tokens, float* h_rand,
    int total_tokens, int vocab_size, unsigned seed)
{
    std::srand(seed);
    for (int row = 0; row < total_tokens; row++) {
        float sum_p = 0.0f, sum_q = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float rp = (float)std::rand()/RAND_MAX; rp = rp*rp*rp + 1e-8f;
            float rq = (float)std::rand()/RAND_MAX; rq = rq*rq*rq + 1e-8f;
            h_p[row * vocab_size + v] = rp;
            h_q[row * vocab_size + v] = rq;
            sum_p += rp; sum_q += rq;
        }
        for (int v = 0; v < vocab_size; v++) {
            h_p[row * vocab_size + v] /= sum_p;
            h_q[row * vocab_size + v] /= sum_q;
        }
        h_tokens[row] = std::rand() % vocab_size;
        h_rand[row]   = (float)std::rand() / RAND_MAX;
    }
}

// ---------------------------------------------------------------------------
// run_stream_benchmark
// ---------------------------------------------------------------------------
void run_stream_benchmark(int vocab_size, int num_batches) {
    printf("\n");
    printf("==========================================================\n");
    printf("  CUDA STREAM PIPELINE BENCHMARK\n");
    printf("  (Lecture 6: Streams and Concurrency)\n");
    printf("==========================================================\n\n");
    printf("  WHAT THIS MEASURES:\n");
    printf("  Without streams: Transfer->Kernel->Transfer->Kernel (sequential)\n");
    printf("  With streams:    Transfer and Kernel run CONCURRENTLY\n");
    printf("  Requirement:     Pinned memory (cudaMallocHost) for async transfer\n\n");

    const int k           = 5;
    const int batch_size  = 64;    // sequences per batch
    int total_tokens      = batch_size * k;

    size_t probs_bytes  = (size_t)total_tokens * vocab_size * sizeof(float);
    size_t tokens_bytes = (size_t)total_tokens * sizeof(int);
    size_t rand_bytes   = (size_t)total_tokens * sizeof(float);
    size_t rej_bytes    = (size_t)batch_size   * sizeof(int);

    // ---- Allocate PINNED host buffers (double buffer: slot 0 and 1) ----
    // CRITICAL: cudaMemcpyAsync requires pinned memory for true async behavior.
    // With malloc(), the async call silently becomes synchronous.
    float* h_p[2];    float* h_q[2];
    int*   h_tok[2];  float* h_rand[2];
    int*   h_rej[2];

    for (int s = 0; s < 2; s++) {
        CUDA_CHECK(cudaMallocHost(&h_p[s],    probs_bytes));
        CUDA_CHECK(cudaMallocHost(&h_q[s],    probs_bytes));
        CUDA_CHECK(cudaMallocHost(&h_tok[s],  tokens_bytes));
        CUDA_CHECK(cudaMallocHost(&h_rand[s], rand_bytes));
        CUDA_CHECK(cudaMallocHost(&h_rej[s],  rej_bytes));
    }

    // ---- Allocate device buffers (double buffer) ----
    float* d_p[2];    float* d_q[2];
    int*   d_tok[2];  float* d_rand[2];
    int*   d_rej[2];

    for (int s = 0; s < 2; s++) {
        CUDA_CHECK(cudaMalloc(&d_p[s],    probs_bytes));
        CUDA_CHECK(cudaMalloc(&d_q[s],    probs_bytes));
        CUDA_CHECK(cudaMalloc(&d_tok[s],  tokens_bytes));
        CUDA_CHECK(cudaMalloc(&d_rand[s], rand_bytes));
        CUDA_CHECK(cudaMalloc(&d_rej[s],  rej_bytes));
    }

    // ---- Create streams ----
    // cudaStreamNonBlocking: this stream does NOT wait on the default stream.
    // Without this flag, streams would still serialize at the default stream.
    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));

    // ---- Create events for inter-stream synchronization ----
    // cudaEventDisableTiming: faster event, used only for sync not timing
    cudaEvent_t done[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming));

    // Kernel launch config
    int threads = 256;
    int blocks  = (total_tokens + threads - 1) / threads;

    std::vector<int> init_rej(batch_size, k);  // k = "no rejection"

    // =========================================================================
    // SEQUENTIAL BASELINE (no streams, blocking transfers)
    // =========================================================================
    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));

    // Warmup
    fill_synthetic_batch(h_p[0], h_q[0], h_tok[0], h_rand[0],
                         total_tokens, vocab_size, 42);
    CUDA_CHECK(cudaMemcpy(d_p[0], h_p[0], probs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rej[0], init_rej.data(), rej_bytes, cudaMemcpyHostToDevice));
    batched_verify_kernel<<<blocks, threads>>>(
        d_p[0], d_q[0], d_tok[0], d_rand[0], d_rej[0],
        vocab_size, k, total_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t_start));
    for (int batch = 0; batch < num_batches; batch++) {
        // Synchronous: CPU waits for each transfer to complete before next op
        CUDA_CHECK(cudaMemcpy(d_p[0],   h_p[0],   probs_bytes,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q[0],   h_q[0],   probs_bytes,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tok[0], h_tok[0], tokens_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rand[0],h_rand[0],rand_bytes,   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rej[0], init_rej.data(), rej_bytes, cudaMemcpyHostToDevice));

        batched_verify_kernel<<<blocks, threads>>>(
            d_p[0], d_q[0], d_tok[0], d_rand[0], d_rej[0],
            vocab_size, k, total_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());  // Block until kernel done
    }
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));
    float seq_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&seq_ms, t_start, t_stop));
    float seq_per_batch = seq_ms / num_batches;

    // =========================================================================
    // STREAMED PIPELINE (double buffering + async transfers)
    // =========================================================================
    // Pipeline pattern:
    //   Iteration 0: fill slot 0, async-upload slot 0, launch kernel slot 0
    //   Iteration 1: fill slot 1, async-upload slot 1, launch kernel slot 1,
    //                (slot 0 kernel may still be running - OVERLAP!)
    //   Iteration 2: wait for slot 0 to be free, fill slot 0, ...

    // Pre-fill both slots before starting
    fill_synthetic_batch(h_p[0], h_q[0], h_tok[0], h_rand[0],
                         total_tokens, vocab_size, 42);
    fill_synthetic_batch(h_p[1], h_q[1], h_tok[1], h_rand[1],
                         total_tokens, vocab_size, 99);

    CUDA_CHECK(cudaEventRecord(t_start));

    for (int batch = 0; batch < num_batches; batch++) {
        int s = batch % 2;  // Ping-pong between slot 0 and 1

        // Wait until this slot is free (previous use of this slot finished)
        // On first two batches, no waiting needed.
        if (batch >= 2) {
            // cudaStreamWaitEvent: stream[s] will wait until done[s] fires
            // done[s] was recorded at the end of the last time we used slot s
            CUDA_CHECK(cudaStreamWaitEvent(stream[s], done[s], 0));
        }

        // Async transfer: returns immediately, GPU executes later
        CUDA_CHECK(cudaMemcpyAsync(d_p[s],   h_p[s],   probs_bytes,
                                   cudaMemcpyHostToDevice, stream[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_q[s],   h_q[s],   probs_bytes,
                                   cudaMemcpyHostToDevice, stream[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_tok[s], h_tok[s], tokens_bytes,
                                   cudaMemcpyHostToDevice, stream[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_rand[s],h_rand[s],rand_bytes,
                                   cudaMemcpyHostToDevice, stream[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_rej[s], init_rej.data(), rej_bytes,
                                   cudaMemcpyHostToDevice, stream[s]));

        // Kernel: executes in stream[s], AFTER the transfers above
        // But CONCURRENTLY with transfers/kernels in the OTHER stream
        batched_verify_kernel<<<blocks, threads, 0, stream[s]>>>(
            d_p[s], d_q[s], d_tok[s], d_rand[s], d_rej[s],
            vocab_size, k, total_tokens);

        // Record event: signals that this slot's work has been queued
        CUDA_CHECK(cudaEventRecord(done[s], stream[s]));

        // CPU: prepare next batch's data while GPU processes current one
        // This is the key: CPU work and GPU work overlap in time
        int next_s = (s + 1) % 2;
        fill_synthetic_batch(h_p[next_s], h_q[next_s], h_tok[next_s],
                             h_rand[next_s], total_tokens, vocab_size,
                             batch * 1234 + 77);
    }

    // Drain both streams
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[1]));
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));
    float str_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&str_ms, t_start, t_stop));
    float str_per_batch = str_ms / num_batches;

    // =========================================================================
    // Print Results
    // =========================================================================
    float speedup = seq_per_batch / str_per_batch;

    printf("  Config: batch_size=%d, k=%d, vocab=%d, num_batches=%d\n\n",
           batch_size, k, vocab_size, num_batches);
    printf("  +--------------------+------------+\n");
    printf("  | Method             | Time/batch |\n");
    printf("  +--------------------+------------+\n");
    printf("  | Sequential (no streams) | %7.4f ms |\n", seq_per_batch);
    printf("  | Streamed (2 streams)    | %7.4f ms |\n", str_per_batch);
    printf("  +--------------------+------------+\n");
    printf("  | Stream Speedup          |   %.2fx    |\n", speedup);
    printf("  +--------------------+------------+\n\n");

    printf("  EXPLANATION:\n");
    printf("  Sequential: GPU idle during transfers, PCIe idle during kernels\n");
    printf("  Streamed:   Transfer(batch N+1) overlaps with Kernel(batch N)\n");
    printf("  Pinned memory is REQUIRED: async transfers need page-locked buffers\n\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));
    for (int s = 0; s < 2; s++) {
        CUDA_CHECK(cudaStreamDestroy(stream[s]));
        CUDA_CHECK(cudaEventDestroy(done[s]));
        CUDA_CHECK(cudaFreeHost(h_p[s]));
        CUDA_CHECK(cudaFreeHost(h_q[s]));
        CUDA_CHECK(cudaFreeHost(h_tok[s]));
        CUDA_CHECK(cudaFreeHost(h_rand[s]));
        CUDA_CHECK(cudaFreeHost(h_rej[s]));
        CUDA_CHECK(cudaFree(d_p[s]));
        CUDA_CHECK(cudaFree(d_q[s]));
        CUDA_CHECK(cudaFree(d_tok[s]));
        CUDA_CHECK(cudaFree(d_rand[s]));
        CUDA_CHECK(cudaFree(d_rej[s]));
    }
}