# ExLlamaV2 + Gemma 3 27B on Dual RTX 3090 (Alternative to vLLM)

**Updated:** 2026-03-15

## Important: Known Issues with Gemma 3

**ExLlamaV2 has unresolved bugs with Gemma 3 models.** See [research/10-exllamav2-multi-gpu-research.md](../research/10-exllamav2-multi-gpu-research.md) for details.

| Issue | Status |
|-------|--------|
| Looping/nonsense after 2-3 paragraphs | **Open** (GitHub #777) |
| Q8 cache degradation | **Open** (GitHub #751) |
| No EXL2 for 1B/4B/12B | Only 27B available |

**Recommendation:** Stay with vLLM for Gemma 3. Our measured 67 tok/s with vLLM is competitive with ExLlamaV2.

---

## Why Consider ExLlamaV2?

| Feature | vLLM | ExLlamaV2 |
|---------|------|-----------|
| **Quantization** | W4A16, FP8, AWQ | EXL2, GPTQ (highly optimized) |
| **Tensor Parallel** | Yes (mature) | Yes (experimental) |
| **Batching** | PagedAttention | Yes |
| **Speed (single request)** | Good | Often faster (for other models) |
| **API** | OpenAI-compatible | OpenAI-compatible (via TabbyAPI) |
| **Gemma 3 Support** | **Excellent** | **Buggy** |

## Installation

### Option 1: ExLlamaV2 with TabbyAPI (Recommended for Server)

```bash
# Create new environment
python -m venv ~/exllama-env
source ~/exllama-env/bin/activate

# Install ExLlamaV2
pip install exllamav2

# Clone TabbyAPI
git clone https://github.com/theroyallab/tabbyAPI
cd tabbyAPI

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Direct ExLlamaV2

```bash
source ~/exllama-env/bin/activate
pip install exllamav2
```

## Model Conversion

ExLlamaV2 works best with EXL2 quantized models. You can:

1. **Use existing EXL2 models** from HuggingFace:
   - Search for "gemma-3-27b EXL2" on HuggingFace
   - Or convert yourself

2. **Convert existing model**:
```bash
# Download the converter
git clone https://github.com/turboderp/exllamav2
cd exllamav2

# Convert (example for 4-bit)
python convert.py \
    -i google/gemma-3-27b-it \
    -o ./gemma-3-27b-exl2-4bpw \
    -b 4.0 \
    -hb 6 \
    --rope_scale 1.0 \
    --rope_alpha 1.0
```

## Running with Tensor Parallel (Dual GPU)

### TabbyAPI Configuration

Create `config.yml`:
```yaml
model:
  model_dir: /path/to/gemma-3-27b-exl2
  max_seq_len: 8192
  tensor_parallel: true
  gpu_split:
    - 24  # GPU 0 VRAM
    - 24  # GPU 1 VRAM

network:
  host: 0.0.0.0
  port: 8000

sampling:
  default_max_tokens: 512
```

### Launch

```bash
cd tabbyAPI
python main.py --config config.yml
```

## Benchmark Comparison Script

```python
#!/usr/bin/env python3
"""Compare vLLM vs ExLlamaV2 performance."""

import time
import requests

def benchmark_endpoint(url, prompt, max_tokens=256, runs=3):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        resp = requests.post(f"{url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        })
        elapsed = time.perf_counter() - start
        tokens = resp.json()["usage"]["completion_tokens"]
        times.append(tokens / elapsed)
    return sum(times) / len(times)

# Test both
vllm_speed = benchmark_endpoint("http://localhost:8000", "Explain quantum computing.")
exllama_speed = benchmark_endpoint("http://localhost:8001", "Explain quantum computing.")

print(f"vLLM: {vllm_speed:.1f} tok/s")
print(f"ExLlamaV2: {exllama_speed:.1f} tok/s")
```

## Measured Performance (Updated 2026-03-15)

### Our vLLM Results (Optimized)

| Model | Config | Single tok/s | Batch(4) tok/s |
|-------|--------|--------------|----------------|
| 27B W4A16 | vLLM TP=2, CUDA graphs | **67.5** | **244** |
| 12B W4A16 | vLLM TP=2 | **115** | **394** |
| 4B W4A16 | vLLM TP=1 | **176** | **581** |
| 1B W8A8 | vLLM TP=1 | **263** | **887** |

### Expected ExLlamaV2 Performance (Estimated)

| Backend | Quantization | Single Request | Notes |
|---------|--------------|----------------|-------|
| ExLlamaV2 TP | EXL2 4bpw | ~50-70 tok/s | Comparable to vLLM |
| ExLlamaV2 gpu_split | EXL2 4bpw | ~25-40 tok/s | Layer split only |

**Note:** ExLlamaV2 may not outperform our optimized vLLM setup due to:
1. vLLM's CUDA graph optimization (`FULL_DECODE_ONLY`)
2. Marlin kernels already excellent for W4A16
3. ExLlamaV2 TP is still experimental

## NVLink with ExLlamaV2

ExLlamaV2 also benefits from NVLink when using tensor parallel:

```bash
# Verify NVLink
nvidia-smi topo -m
# Should show NV3 between GPU0 and GPU1

# Set NCCL to prefer NVLink
export NCCL_P2P_LEVEL=NVL
```

## Pros/Cons

### ExLlamaV2 Pros
- Faster for quantized models (optimized kernels)
- Lower VRAM overhead
- EXL2 format is highly efficient
- Good for interactive use

### ExLlamaV2 Cons
- Smaller ecosystem than vLLM
- May need model conversion
- Less continuous batching optimization
- Less documentation

## Recommendation

### For Gemma 3: Use vLLM

| Model | Use | Reason |
|-------|-----|--------|
| **1B** | vLLM W8A8 | No EXL2 exists |
| **4B** | vLLM W4A16 | No EXL2 exists |
| **12B** | vLLM W4A16 | No EXL2 exists |
| **27B** | vLLM W4A16 | Known EXL2 bugs |

### For Other Models: Consider ExLlamaV2

ExLlamaV2 excels with Llama 3, Mistral, and Qwen models where:
1. No known bugs exist
2. Full range of EXL2 quantizations available
3. Tensor parallelism is more stable

### When to Revisit ExLlamaV2 for Gemma 3

1. Once GitHub #777 (looping bug) is fixed
2. Once turboderp provides 1B/4B/12B EXL2 quantizations
3. Once ExLlamaV3 (EXL3 format) matures
