#ifndef STREAM_PIPELINE_CUH
#define STREAM_PIPELINE_CUH

#include <cuda_runtime.h>
#include "verification.cuh"

// run_stream_benchmark: Sequential vs Streamed pipeline karsilastirmasi
// Lecture 6: CUDA Streams and Concurrency
void run_stream_benchmark(int vocab_size = 50257, int num_trials = 30);

#endif