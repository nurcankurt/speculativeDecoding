# CUDA Profile Results - Speculative Decoding Pipeline

**GPU:** NVIDIA GeForce RTX 3060 Laptop GPU  
**Tool:** Nsight Systems 2025.6.3 (nvprof yerine)  
**Command:** `./speculative_decoding --benchmark --trials 5`

---

## 1. CUDA API Summary (Top Operations)

| Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Name |
|----------|-----------------|-----------|----------|------|
| 53,0 | 489,863,972 | 217 | 2,257,437 | cudaMemcpy |
| 23,0 | 214,327,675 | 62 | 3,456,898 | cudaEventSynchronize |
| 8,0 | 81,395,623 | 90 | 904,395 | cudaMalloc |
| 5,0 | 48,697,823 | 51 | 954,859 | cudaDeviceSynchronize |
| 4,0 | 44,523,718 | 15 | 2,968,247 | cudaHostAlloc |
| 2,0 | 22,497,628 | 90 | 249,973 | cudaFree |
| 1,0 | 14,919,270 | 15 | 994,618 | cudaFreeHost |
| 0,0 | 2,403,026 | 121 | 19,859 | cudaLaunchKernel |
| 0,0 | 1,100,871 | 129 | 8,533 | cudaEventRecord |
| 0,0 | 469,423 | 25 | 18,776 | cudaMemcpyAsync |

**Observation:** cudaMemcpy dominates (53%) - indicates data transfer bottleneck.

---

## 2. CUDA GPU Kernel Summary

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Kernel Name |
|----------|-----------------|-----------|----------|-------------|
| 73,0 | 191,065,571 | 30 | 6,368,852 | `batched_residual_kernel` |
| 26,0 | 68,584,278 | 20 | 3,429,213 | `residual_sample_kernel` |
| 0,0 | 137,278 | 41 | 3,348 | `batched_verify_kernel` |
| 0,0 | 67,648 | 30 | 2,254 | `verify_tokens_kernel` |

**Observation:** Residual sampling kernels dominate (99% of kernel time).

---

## 3. CUDA GPU Memory Transfer Summary

| Time (%) | Total Time (ns) | Count | Operation |
|----------|-----------------|-------|-----------|
| 100,0 | 545,109,738 | 217 | CUDA memcpy Host-to-Device |
| 0,0 | 29,631 | 25 | CUDA memcpy Device-to-Host |

**Total Data Transferred:** 3,280.864 MB (Host-to-Device)

---

## 4. Stream Benchmark Results

**Command:** `./speculative_decoding --benchmark --trials 5`

| Method | Time/batch (ms) |
|--------|-----------------|
| Sequential (no streams) | 27.49 |
| Streamed (2 streams) | 134.38 |
| Speedup | 0.20x |

**Note:** Stream performance is lower than sequential - indicates stream overhead dominates for small batch sizes or incorrect stream usage pattern.

---

## 5. Key Findings

### Performance Bottlenecks:
1. **Data Transfer (53%)** - cudaMemcpy is the dominant operation
2. **Memory Allocation (8% + 4%)** - cudaMalloc and cudaHostAlloc could be optimized with pre-allocation
3. **Synchronization Overhead (23%)** - cudaEventSynchronize indicates frequent CPU-GPU sync points

### Kernel Efficiency:
- `batched_residual_kernel` and `residual_sample_kernel` account for 99% of compute time
- Verification kernels are highly optimized (<1% of time)

### Recommendations:
1. Use persistent pinned memory buffers to reduce allocation overhead
2. Batch multiple operations to amortize transfer costs
3. Reduce synchronization frequency where possible
4. Investigate stream pipeline implementation for overlap opportunities

---

## Files Generated:
- `report1.nsys-rep` - Full profile data (can be opened in Nsight Systems GUI)
- `report1.sqlite` - SQLite database with profile data
- `report_kernel.nsys-rep` - Kernel-focused profile
- `report_stream.nsys-rep` - Stream benchmark profile
