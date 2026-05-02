
// =============================================================================
// pinned_benchmark.cu - Pinned vs Pageable Memory Benchmark (Lecture 4)
// =============================================================================
#include "verification.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float measure_pageable(int k, int vocab_size, int num_trials) {
    size_t bytes = (size_t)k * vocab_size * sizeof(float);
    float* h_pageable = (float*)malloc(bytes);
    float* d_gpu;
    CUDA_CHECK(cudaMalloc(&d_gpu, bytes));
    for (size_t i = 0; i < (size_t)k * vocab_size; i++)
        h_pageable[i] = (float)(i % 100) / 100.0f;

    CUDA_CHECK(cudaMemcpy(d_gpu, h_pageable, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < num_trials; t++)
        CUDA_CHECK(cudaMemcpy(d_gpu, h_pageable, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_pageable);
    CUDA_CHECK(cudaFree(d_gpu));
    return ms / num_trials;
}

static float measure_pinned(int k, int vocab_size, int num_trials) {
    size_t bytes = (size_t)k * vocab_size * sizeof(float);
    float* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));  // PAGE-LOCKED memory
    float* d_gpu;
    CUDA_CHECK(cudaMalloc(&d_gpu, bytes));
    for (size_t i = 0; i < (size_t)k * vocab_size; i++)
        h_pinned[i] = (float)(i % 100) / 100.0f;

    CUDA_CHECK(cudaMemcpy(d_gpu, h_pinned, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < num_trials; t++)
        CUDA_CHECK(cudaMemcpy(d_gpu, h_pinned, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_pinned));  // cudaFreeHost ile serbest bırak!
    CUDA_CHECK(cudaFree(d_gpu));
    return ms / num_trials;
}

void run_pinned_benchmark(int num_trials) {
    printf("\\n");
    printf("==========================================================\\n");
    printf("  PINNED vs PAGEABLE MEMORY BENCHMARK (Lecture 4)\\n");
    printf("==========================================================\\n\\n");

    int k_values[] = {1, 5, 10, 20, 50};
    int vocab_size = 50257;

    printf("  +------+----------+---------------+---------------+----------+\\n");
    printf("  |   k  | Size(MB) | Pageable (ms) | Pinned   (ms) | Speedup  |\\n");
    printf("  +------+----------+---------------+---------------+----------+\\n");

    for (int i = 0; i < 5; i++) {
        int k = k_values[i];
        size_t bytes = (size_t)k * vocab_size * sizeof(float);
        float mb = bytes / (1024.0f * 1024.0f);
        float pageable_ms = measure_pageable(k, vocab_size, num_trials);
        float pinned_ms   = measure_pinned(k, vocab_size, num_trials);
        float speedup = pageable_ms / pinned_ms;
        printf("  | %4d | %8.3f | %13.6f | %13.6f | %8.2fx |\\n",
               k, mb, pageable_ms, pinned_ms, speedup);
    }
    printf("  +------+----------+---------------+---------------+----------+\\n");
    printf("\\n  SONUC: Pinned memory, buyuk transferlerde pageable'dan hizlidir.\\n");
    printf("  Neden: OS sayfayi RAM'de sabit tutar, GPU direkt okur (ara kopya yok)\\n\\n");
}
