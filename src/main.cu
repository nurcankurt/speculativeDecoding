// =============================================================================
// main.cu - Entry point for the Speculative Decoding CUDA Pipeline
// =============================================================================
//
// SPECULATIVE DECODING OVERVIEW:
//
// Speculative decoding accelerates autoregressive language model inference by
// using a small, fast "draft" model to propose k tokens, then verifying them
// in parallel with the large, accurate "target" model.
//
// Algorithm:
//   1. Draft model generates k candidate tokens quickly
//   2. Target model scores all k tokens in ONE forward pass (not k passes!)
//   3. Verification: compare draft and target probabilities for each token
//      - Accept token i if: random() < min(1, P_target/P_draft)
//      - Reject: stop at first rejection, discard all subsequent tokens
//   4. Residual sampling: at the rejection point, sample a correction token
//      from the distribution max(0, P_target - P_draft) / Z
//
// This program implements step 3 (verification) and step 4 (residual sampling)
// as CUDA kernels, benchmarked against a CPU baseline.
//
// USAGE:
//   ./speculative_decoding                    # Run benchmarks with synthetic data
//   ./speculative_decoding --benchmark        # Same as above
//   ./speculative_decoding --data-dir <path>  # Use .npy files from <path>
//   ./speculative_decoding --help             # Show usage
//
// EXPECTED .NPY FILES (when using --data-dir):
//   <path>/p_probs.npy      - Target model probabilities [k, 50257] float32
//   <path>/q_probs.npy      - Draft model probabilities  [k, 50257] float32
//   <path>/draft_tokens.npy - Draft token IDs             [k]        int32/int64
//
// BUILD:
//   mkdir build && cd build && cmake .. && cmake --build . --config Release
//   OR: nvcc -arch=sm_75 src/*.cu src/*.cpp -o speculative_decoding -lcurand
//
// =============================================================================

#include "verification.cuh"
#include "residual_sample.cuh"
#include "benchmark.cuh"
#include "pinned_benchmark.cuh"
#include "cpu_baseline.h"
#include "npy_loader.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// print_usage: Display command-line help
// ---------------------------------------------------------------------------
static void print_usage(const char* prog_name) {
    printf("\n");
    printf("Speculative Decoding - CUDA Pipeline\n");
    printf("=====================================\n\n");
    printf("Usage:\n");
    printf("  %s [OPTIONS]\n\n", prog_name);
    printf("Options:\n");
    printf("  --benchmark          Run benchmarks with synthetic data (default)\n");
    printf("  --data-dir <path>    Load .npy files from directory:\n");
    printf("                         <path>/p_probs.npy      [k, V] float32\n");
    printf("                         <path>/q_probs.npy      [k, V] float32\n");
    printf("                         <path>/draft_tokens.npy [k]    int32/int64\n");
    printf("  --vocab-size <V>     Override vocabulary size (default: 50257)\n");
    printf("  --trials <N>         Number of benchmark trials (default: 100)\n");
    printf("  --help               Show this help message\n");
    printf("\n");
}

// ---------------------------------------------------------------------------
// print_cuda_device_info: Display GPU hardware information
// ---------------------------------------------------------------------------
static void print_cuda_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable GPU detected!\n");
        fprintf(stderr, "Make sure you have an NVIDIA GPU and drivers installed.\n");
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n");
    printf("==========================================================\n");
    printf("  CUDA DEVICE INFORMATION\n");
    printf("==========================================================\n");
    printf("  Device:           %s\n", prop.name);
    printf("  Compute Cap:      %d.%d\n", prop.major, prop.minor);
    printf("  SMs:              %d\n", prop.multiProcessorCount);
    printf("  Max Threads/SM:   %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max Threads/Block:%d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size:        %d\n", prop.warpSize);
    printf("  Global Memory:    %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Mem/Block: %.1f KB\n",
           prop.sharedMemPerBlock / 1024.0);
    printf("==========================================================\n");
}

// ---------------------------------------------------------------------------
// run_with_npy_data: Load .npy files and run speculative decoding
// ---------------------------------------------------------------------------
static void run_with_npy_data(const std::string& data_dir, int vocab_size) {
    printf("\n--- Loading data from: %s ---\n\n", data_dir.c_str());

    // Construct file paths
    std::string p_path = data_dir + "/p_probs.npy";
    std::string q_path = data_dir + "/q_probs.npy";
    std::string t_path = data_dir + "/draft_tokens.npy";

    // Load .npy files
    NpyArray p_arr = load_npy(p_path);
    NpyArray q_arr = load_npy(q_path);
    NpyArray t_arr = load_npy(t_path);

    // Validate shapes
    if (!p_arr.is_float || !q_arr.is_float) {
        fprintf(stderr, "ERROR: p_probs and q_probs must be float arrays\n");
        exit(EXIT_FAILURE);
    }
    if (t_arr.is_float) {
        fprintf(stderr, "ERROR: draft_tokens must be an integer array\n");
        exit(EXIT_FAILURE);
    }
    if (p_arr.ndim() != 2 || q_arr.ndim() != 2) {
        fprintf(stderr, "ERROR: p_probs and q_probs must be 2D arrays [k, V]\n");
        exit(EXIT_FAILURE);
    }
    if (t_arr.ndim() != 1) {
        fprintf(stderr, "ERROR: draft_tokens must be a 1D array [k]\n");
        exit(EXIT_FAILURE);
    }

    int k = (int)p_arr.shape[0];
    int V = (int)p_arr.shape[1];

    if (V != vocab_size) {
        printf("NOTE: Detected vocab_size=%d from data (overriding default)\n", V);
        vocab_size = V;
    }
    if ((int)q_arr.shape[0] != k || (int)q_arr.shape[1] != vocab_size) {
        fprintf(stderr, "ERROR: Shape mismatch: p_probs and q_probs must have "
                        "same shape\n");
        exit(EXIT_FAILURE);
    }
    if ((int)t_arr.shape[0] != k) {
        fprintf(stderr, "ERROR: draft_tokens length (%d) must match k (%d)\n",
                (int)t_arr.shape[0], k);
        exit(EXIT_FAILURE);
    }

    printf("\n  k (draft tokens): %d\n", k);
    printf("  Vocab size:       %d\n", vocab_size);

    // Generate random values for acceptance testing
    std::vector<float> rand_vals(k);
    std::srand(42);
    for (int i = 0; i < k; i++) {
        rand_vals[i] = (float)std::rand() / RAND_MAX;
    }
    float rand_residual = (float)std::rand() / RAND_MAX;

    // =====================================================================
    // Run CPU Baseline
    // =====================================================================
    printf("\n--- CPU Baseline ---\n");
    CpuResult cpu_result = cpu_speculative_decode(
        p_arr.data_float.data(),
        q_arr.data_float.data(),
        t_arr.data_int.data(),
        rand_vals.data(),
        rand_residual,
        k,
        vocab_size
    );

    printf("  Accepted:    %d / %d tokens\n", cpu_result.n_accepted, k);
    if (cpu_result.first_rejected_idx >= 0) {
        printf("  Rejected at: position %d\n", cpu_result.first_rejected_idx);
        printf("  Bonus token: %d\n", cpu_result.bonus_token);
    } else {
        printf("  All tokens accepted!\n");
    }

    // =====================================================================
    // Run GPU Pipeline
    // =====================================================================
    printf("\n--- GPU Pipeline ---\n");
    int gpu_n_accepted = 0;
    int gpu_bonus_token = -1;

    gpu_full_speculative_decode(
        p_arr.data_float.data(),
        q_arr.data_float.data(),
        t_arr.data_int.data(),
        rand_vals.data(),
        rand_residual,
        k,
        vocab_size,
        &gpu_n_accepted,
        &gpu_bonus_token
    );

    printf("  Accepted:    %d / %d tokens\n", gpu_n_accepted, k);
    if (gpu_n_accepted < k) {
        printf("  Rejected at: position %d\n", gpu_n_accepted);
        printf("  Bonus token: %d\n", gpu_bonus_token);
    } else {
        printf("  All tokens accepted!\n");
    }

    // =====================================================================
    // Verify Agreement
    // =====================================================================
    printf("\n--- Verification ---\n");
    bool match = (cpu_result.n_accepted == gpu_n_accepted);
    if (cpu_result.first_rejected_idx >= 0 && gpu_n_accepted < k) {
        match = match && (cpu_result.bonus_token == gpu_bonus_token);
    }

    if (match) {
        printf("  [PASS] CPU and GPU results MATCH!\n");
    } else {
        printf("  [FAIL] CPU and GPU results DIFFER!\n");
        printf("         CPU accepted=%d, GPU accepted=%d\n",
               cpu_result.n_accepted, gpu_n_accepted);
        printf("         CPU bonus=%d, GPU bonus=%d\n",
               cpu_result.bonus_token, gpu_bonus_token);
    }
    printf("\n");
}

// =============================================================================
// main: Parse arguments and dispatch to appropriate mode
// =============================================================================
int main(int argc, char* argv[]) {
    // ---- Parse command-line arguments ----
    bool do_benchmark = true;
    std::string data_dir;
    int vocab_size = 50257;   // GPT-2 vocabulary size
    int num_trials = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            do_benchmark = true;
        } else if (strcmp(argv[i], "--data-dir") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "ERROR: --data-dir requires a path argument\n");
                return 1;
            }
            data_dir = argv[++i];
            do_benchmark = false;
        } else if (strcmp(argv[i], "--vocab-size") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "ERROR: --vocab-size requires a number\n");
                return 1;
            }
            vocab_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trials") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "ERROR: --trials requires a number\n");
                return 1;
            }
            num_trials = atoi(argv[++i]);
        } else {
            fprintf(stderr, "ERROR: Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // ---- Print GPU info ----
    print_cuda_device_info();

    // ---- Dispatch ----
    if (!data_dir.empty()) {
        // Mode 1: Load .npy files and run verification
        run_with_npy_data(data_dir, vocab_size);
    }

    if (do_benchmark) {
        // Mode 2: Run benchmarks with synthetic data
        run_benchmarks(vocab_size, num_trials);

        // Mode 3: Run batched benchmarks (GPU advantage!)
        run_batched_benchmarks(vocab_size, num_trials > 50 ? 50 : num_trials);

        // Lecture 4: Pinned Memory Benchmark
        run_pinned_benchmark(num_trials > 50 ? 50 : num_trials);
    }

    printf("Done.\n");
    return 0;
}
