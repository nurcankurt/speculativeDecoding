// =============================================================================
// benchmark.cuh - Benchmark framework declarations
// =============================================================================
// Uses cudaEvent-based timing for GPU kernels and <chrono> for CPU timing.
//
// CUDA TIMING CONCEPTS:
//
// cudaEvent:
//   GPU-side timestamps that accurately measure kernel execution time.
//   Unlike CPU timers, cudaEvents account for the asynchronous nature of
//   GPU execution and measure actual GPU time, not wall-clock time.
//
//   Usage pattern:
//     cudaEventRecord(start)     // Insert timestamp in GPU command stream
//     kernel<<<...>>>(...)       // Queue kernel
//     cudaEventRecord(stop)      // Insert another timestamp
//     cudaEventSynchronize(stop) // Wait for stop event to complete
//     cudaEventElapsedTime(&ms, start, stop)  // Compute elapsed time
//
// Why not use CPU timers for GPU?
//   Kernel launches are ASYNCHRONOUS - the CPU returns immediately after
//   queuing the kernel, before the GPU finishes executing it. A CPU timer
//   would measure queue overhead, not actual computation time.
// =============================================================================

#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

// ---------------------------------------------------------------------------
// run_benchmarks: Execute benchmarks across multiple k values
// ---------------------------------------------------------------------------
// Tests k = {1, 3, 5, 10, 20} and outputs a formatted comparison table:
//
//   method | k | time_ms | speedup
//   -------|---|---------|--------
//   GPU    | 1 | 0.042   | 1.00x
//   CPU    | 1 | 0.015   | 0.36x
//   ...
//
// Parameters:
//   vocab_size - Vocabulary size (50257 for GPT-2)
//   num_trials - Number of trials to average for stable timing (default: 100)
void run_benchmarks(int vocab_size = 50257, int num_trials = 100);

// ---------------------------------------------------------------------------
// run_batched_benchmarks: Benchmark with BATCH processing (many sequences)
// ---------------------------------------------------------------------------
// This is where the GPU truly shines! Instead of verifying 1 sequence,
// we verify B sequences in parallel. The GPU launches B*k threads for
// verification and B blocks for residual sampling.
//
// Tests batch_sizes = {1, 16, 64, 256, 512} with fixed k=5.
void run_batched_benchmarks(int vocab_size = 50257, int num_trials = 50);

#endif // BENCHMARK_CUH
