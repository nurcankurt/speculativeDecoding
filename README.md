# Speculative Decoding - CUDA Pipeline

Speculative decoding accelerates autoregressive language model inference by using a small, fast "draft" model to propose tokens, then verifying them in parallel with the large, accurate "target" model.

## Requirements

### Hardware
- **NVIDIA GPU** with Compute Capability >= 7.5 (Turing or newer recommended)
- Minimum 4GB VRAM (6GB+ recommended)

### Software
- **CUDA Toolkit** 12.x or 13.x
- **MSVC** (Microsoft Visual C++) - "Desktop development with C++" workload
- **Python 3.8+** (optional, for HuggingFace data generation)

---

## Installation

### 1. Install CUDA Toolkit

Download from: https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvcc --version
```

### 2. Install Visual Studio Build Tools

1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. During installation, select **"Desktop development with C++"** workload
3. This installs MSVC compiler (`cl.exe`) required by nvcc

### 3. Install Python Dependencies (Optional)

For generating real model data with HuggingFace:

```bash
pip install transformers torch numpy
```

---

## Building

### Option A: Direct nvcc compilation (Recommended for Windows)

Open **PowerShell** or **Developer Command Prompt for VS**:

```powershell
# Set MSVC compiler path (adjust VS version if needed)
$env:PATH = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\HostX64\x64;' + $env:PATH

# Compile
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe'
& $nvcc src/main.cu src/verification.cu src/residual_sample.cu src/cpu_baseline.cpp src/benchmark.cu src/npy_loader.cpp src/pinned_benchmark.cu src/stream_pipeline.cu -o speculative_decoding.exe -arch=sm_86 -lcurand -std=c++14 -O2
```

**Note:** Adjust `-arch=sm_86` to match your GPU:
- RTX 30xx (Ampere): `sm_86`
- RTX 40xx (Ada Lovelace): `sm_89`
- RTX 20xx (Turing): `sm_75`
- GTX 10xx (Pascal): `sm_61`

### Option B: CMake (Linux / WSL)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Running

### 1. Synthetic Benchmark

Run with synthetic data (no dependencies):

```powershell
.\speculative_decoding.exe --benchmark --trials 50
```

Options:
- `--trials <N>` - Number of benchmark iterations (default: 100)
- `--vocab-size <V>` - Override vocabulary size (default: 50257)

### 2. Real Model Data (HuggingFace)

Generate real GPT-2 / DistilGPT-2 logits:

```powershell
# Generate data
python src/scripts/generate_data.py

# Run with generated data
.\speculative_decoding.exe --data-dir data
```

The script will:
1. Download GPT-2 and DistilGPT-2 models (~500MB first run)
2. Generate draft tokens with DistilGPT-2
3. Evaluate all tokens with GPT-2 in a single forward pass
4. Save probabilities as `.npy` files in `data/` directory

### 3. Custom Prompt

Edit `src/scripts/generate_data.py`:

```python
K      = 5                                    # Number of draft tokens
PROMPT = "Your custom prompt here"           # Your prompt
```

Then regenerate:
```powershell
python src/scripts/generate_data.py
.\speculative_decoding.exe --data-dir data
```

---

## Project Structure

```
speculativeDecoding/
├── src/
│   ├── main.cu              # Entry point, argument parsing
│   ├── verification.cu(h)   # Token verification kernels
│   ├── residual_sample.cu(h)# Residual sampling kernel
│   ├── benchmark.cu(h)      # Synthetic benchmarks
│   ├── pinned_benchmark.cu  # Pinned vs pageable memory benchmark
│   ├── stream_pipeline.cu(h)# CUDA streams benchmark
│   ├── cpu_baseline.cpp(h)  # CPU reference implementation
│   └── npy_loader.cpp(h)    # NumPy .npy file loader
├── scripts/
│   └── generate_data.py     # HuggingFace data generator
├── CMakeLists.txt           # CMake build configuration
└── README.md                # This file
```

---

## Expected Output

### Benchmark Mode
```
==========================================================
  CUDA DEVICE INFORMATION
==========================================================
  Device:           NVIDIA GeForce RTX 3060 Laptop GPU
  Compute Cap:      8.6
  SM Count:         30
  Global Memory:    6.0 GB
==========================================================

  Benchmarking k= 1 ... done (GPU: 0.0246 ms, CPU: 0.0000 ms)
  Benchmarking k= 5 ... done (GPU: 4.3237 ms, CPU: 0.5859 ms)
  ...
```

### Data Mode
```
--- CPU Baseline ---
  Accepted:    3 / 5 tokens
  Rejected at: position 3
  Bonus token: 503

--- GPU Pipeline ---
  Accepted:    3 / 5 tokens
  Rejected at: position 3
  Bonus token: 503

--- Verification ---
  [PASS] CPU and GPU results MATCH!
```

---

## Profiling

### Using Nsight Systems (CUDA 13.x+)

`nvprof` is deprecated in CUDA 13.x. Use Nsight Systems instead:

```powershell
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems\target-windows-x64\nsys.exe'
& $nsys profile --trace=cuda --stats=true -o profile .\speculative_decoding.exe --benchmark --trials 5
& $nsys stats --report cuda_gpu_kern_sum profile.nsys-rep
```

### Using nvprof (CUDA 12.x and older)

```powershell
nvprof --print-gpu-trace .\speculative_decoding.exe --benchmark --trials 5
nvprof --metrics achieved_occupancy,shared_memory_utilization .\speculative_decoding.exe --benchmark --trials 5
```

---

## Troubleshooting

### "Cannot find compiler 'cl.exe' in PATH"

The MSVC compiler is not in your PATH. Solutions:

1. **Use Developer Command Prompt for VS** (recommended)
2. **Add to PATH manually:**
   ```powershell
   $env:PATH = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\HostX64\x64;' + $env:PATH
   ```
   (Adjust path for your VS version)

### "No CUDA-capable GPU detected"

- Ensure NVIDIA drivers are installed
- Check GPU is properly connected
- Try restarting your computer

### "CUDA error: out of memory"

Reduce batch size or vocabulary size:
```powershell
.\speculative_decoding.exe --benchmark --trials 50 --vocab-size 32000
```

### Python: "ModuleNotFoundError: No module named 'transformers'"

Install dependencies:
```bash
pip install transformers torch numpy
```

---

## References
- [Speculative Decoding Paper](https://arxiv.org/abs/2301.10810)
