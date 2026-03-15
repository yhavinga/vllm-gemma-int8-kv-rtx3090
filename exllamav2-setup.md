# ExLlamaV2 + Gemma 3 27B on Dual RTX 3090 (Alternative to vLLM)

ExLlamaV2 is optimized for quantized models on consumer GPUs and can be 20-40% faster than vLLM for certain workloads.

## Why Try ExLlamaV2?

| Feature | vLLM | ExLlamaV2 |
|---------|------|-----------|
| **Quantization** | W4A16, FP8, AWQ | EXL2, GPTQ (highly optimized) |
| **Tensor Parallel** | Yes | Yes (via TabbyAPI) |
| **Batching** | PagedAttention | Yes |
| **Speed (single request)** | Good | Often faster |
| **API** | OpenAI-compatible | OpenAI-compatible (via TabbyAPI) |

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

## Expected Performance

Based on community benchmarks for dual RTX 3090 with NVLink:

| Backend | Quantization | Single Request | Batch (4) |
|---------|--------------|----------------|-----------|
| vLLM (current) | W4A16 | ~21 tok/s | ~50-80 tok/s |
| ExLlamaV2 | EXL2 4bpw | ~25-35 tok/s | ~60-100 tok/s |

The improvement comes from ExLlamaV2's highly optimized quantization kernels.

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

1. **Keep your current vLLM setup** as baseline (~21 tok/s)
2. **Try ExLlamaV2** as an experiment
3. **Compare** with the benchmark script
4. Use whichever gives better results for your use case
