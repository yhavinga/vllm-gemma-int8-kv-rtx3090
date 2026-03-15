# ExLlamaV2/V3 Multi-GPU Research for Gemma 3

**Date:** 2026-03-15
**Hardware:** 2x RTX 3090 (48GB VRAM total, NVLink NV4)
**Goal:** Evaluate ExLlamaV2/V3 as alternative to vLLM for Gemma 3 (1B, 4B, 12B, 27B)

---

## Executive Summary

### CRITICAL FINDING: Gemma 3 + ExLlamaV2 Has Known Issues

| Model Size | EXL2 Available | EXL3 Available | Recommended |
|------------|----------------|----------------|-------------|
| **1B** | **NO** | **NO** | Use vLLM W8A8 |
| **4B** | **NO** | **NO** | Use vLLM W4A16 |
| **12B** | **YES** (turboderp) | Derivative only | **TEST** |
| **27B** | **YES** (turboderp) | **YES** (turboderp) | **TEST** |

### Key Finding: Stick with vLLM for Gemma 3

**Reasons:**
1. **No EXL2/EXL3 quantizations** for 1B, 4B, 12B base Gemma 3 models
2. **Known bugs** with Gemma 3 27B EXL2: looping/nonsense after 2-3 paragraphs
3. **Long context issues**: Gemma 3's 1024-token sliding window + Q8 cache = problems
4. **vLLM already optimized**: Our 67 tok/s (27B) beats typical ExLlamaV2 performance

---

## ExLlamaV2/V3 Multi-GPU Architecture

### Two Multi-GPU Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **gpu_split** | Layers distributed across GPUs (pipeline) | Models too large for 1 GPU |
| **tensor_parallel** | Computation split within layers (parallel) | Speed optimization |

### Key Difference

```
gpu_split (Pipeline):        tensor_parallel (True TP):
[GPU0: Layers 0-19]          [GPU0: Half of each layer]
         ↓                            ↓↑
[GPU1: Layers 20-39]          [GPU1: Half of each layer]
                              (GPUs work simultaneously)
```

### NVLink Impact

From [Himesh's benchmarks](http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html):

| GPUs | NVLink | Output (t/s) | Improvement |
|------|--------|--------------|-------------|
| 2 | Yes | 715 | +48% |
| 2 | No | 483 | baseline |
| 4 | Yes | 535 | +9% |
| 4 | No | 490 | baseline |

**Key insight:** NVLink provides **~50% speedup** for 2-GPU tensor parallelism!

---

## Available Gemma 3 EXL2/EXL3 Models

### Official turboderp Models

| Model | Format | Bitrates | Source |
|-------|--------|----------|--------|
| [gemma-3-27b-it-exl3](https://huggingface.co/turboderp/gemma-3-27b-it-exl3) | EXL3 | 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0 bpw | turboderp |
| [gemma-3-27b-it-exl2](https://huggingface.co/turboderp/gemma-3-27b-it-exl2) | EXL2 | 3.0, 4.0, 5.0, 6.0 bpw | turboderp |
| [gemma-3-12b-it-exl2](https://huggingface.co/turboderp/gemma-3-12b-it-exl2) | EXL2 | 3.0, 4.0, 5.0, 6.0 bpw | turboderp |

### What's Missing

| Model | EXL2 | EXL3 | Alternative |
|-------|------|------|-------------|
| gemma-3-1b-it | **NONE** | **NONE** | RedHatAI W8A8 (vLLM) |
| gemma-3-4b-it | **NONE** | **NONE** | RedHatAI W4A16 (vLLM) |

### Community Derivatives

| Model | Format | Notes |
|-------|--------|-------|
| [Tiger-Gemma-12B-v3-EXL3](https://huggingface.co/ArtusDev/TheDrummer_Tiger-Gemma-12B-v3-EXL3) | EXL3 | Fine-tuned variant |
| [gemma-3-27b-it-abliterated-exl2](https://huggingface.co/Kooten/gemma-3-27b-it-abliterated-exl2) | EXL2 | Uncensored variant |

---

## Known Issues with Gemma 3 + ExLlamaV2

### Bug #1: Looping/Nonsense Generation

From [GitHub Issue #777](https://github.com/turboderp-org/exllamav2/issues/777):

**Symptoms:**
- Generates 2-3 coherent paragraphs
- Then enters repetitive nonsense loops
- Affects all EXL2 bitrates (4, 5, 6 bpw)

**Root Causes:**
1. **Config mismatch**: ExLlamaV2 uses different defaults for some parameters
2. **Activation overflow**: Gemma 3 activations exceed float16 max (65,500)
3. **Context issues**: Problems at 7K+ tokens with quantized cache

**Workarounds:**
1. Copy `text_config` and `vision_config` from abliterated variants
2. Keep sequences under 8K tokens
3. Use FP16 cache with 3-4GB VRAM reserve
4. Switch to llama.cpp (proper sliding window attention)

**Status:** Open as of June 2025 - no official fix.

### Bug #2: Q8 Cache Degradation

From [GitHub Issue #751](https://github.com/turboderp-org/exllamav2/issues/751):

> "Gemma3 exllama models seem to perform really poorly with Q8 quantized cache? They just loop and repeat"

### Bug #3: Long Context Sliding Window

Gemma 3's 1024-token sliding window interacts poorly with ExLlamaV2's attention implementation. llama.cpp recently fixed this - ExLlamaV2 has not.

---

## ExLlamaV2 vs ExLlamaV3

### Key Differences

| Feature | ExLlamaV2 | ExLlamaV3 |
|---------|-----------|-----------|
| Format | EXL2 | EXL3 |
| Quantization | GPTQ-based | QTIP-based (SOTA) |
| Tensor Parallelism | Experimental | Improved |
| File structure | Renames tensors | Preserves structure |
| Gemma 3 support | Yes (buggy) | Yes |

### EXL3 Quality Claims

From [HuggingFace discussion](https://huggingface.co/turboderp/gemma-3-27b-it-exl3/discussions/1):

> "Looking at the Kl divergence graphs, you've effectively made a quantization format where 6bpw has become virtually indistinguishable from 8-bit."

> "The 4bpw version is around a 5_K_S equivalent, and the 3.5bpw is on par with a 4_K_M model despite being noticeably smaller."

---

## Tensor Parallelism Performance (ExLlamaV2)

From [GitHub Issue #571](https://github.com/turboderp-org/exllamav2/issues/571):

### Test: Llama 3.1 405B (6-bit EXL2) on 8x Ada6000

| Metric | Non-TP | TP | Improvement |
|--------|--------|----|----|
| Prompt throughput (4K) | 245 t/s | 356 t/s | **~1.5x** |
| Inference throughput (4K) | 2.9 t/s | 8.6 t/s | **~3x** |

### Bottlenecks Identified

1. **CPU overhead (primary)**: Multiple CUDA streams = latency
   - flash-attn Python call: ~15 μs × 4 calls/layer × 80 layers = 5ms per token
   - At 20 tok/s, 10% of CPU time is just Python call overhead

2. **Not NVLink limited**: Testing showed PCIe x8→x16 had negligible impact
   - System DRAM bandwidth is the bottleneck
   - Uses pinned buffers, not GPU-to-GPU P2P

---

## TabbyAPI Configuration for Multi-GPU

### Installation

```bash
# Create environment
python -m venv ~/exllama-env
source ~/exllama-env/bin/activate

# Install ExLlamaV2
pip install exllamav2

# Clone TabbyAPI
git clone https://github.com/theroyallab/tabbyAPI
cd tabbyAPI
pip install -r requirements.txt
```

### Configuration Options

From [TabbyAPI Wiki](https://github.com/theroyallab/tabbyAPI/wiki/02.-Server-options):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tensor_parallel` | Bool | false | Enable tensor parallelism |
| `gpu_split_auto` | Bool | true | Auto-distribute across GPUs |
| `gpu_split` | List[Float] | [] | Manual GB allocation per GPU |
| `autosplit_reserve` | List[Int] | [96] | MB to reserve per GPU |

### Dual RTX 3090 Config Example

```yaml
# config.yml
model:
  model_dir: /models/turboderp/gemma-3-27b-it-exl2
  max_seq_len: 8192

  # Multi-GPU settings
  tensor_parallel: true
  gpu_split: [20.0, 20.0]  # Leave ~4GB per GPU

  # Cache settings (avoid Q8 for Gemma 3!)
  cache_mode: FP16

network:
  host: 0.0.0.0
  port: 8001

sampling:
  default_max_tokens: 512
```

### Launch

```bash
cd tabbyAPI
python main.py --config config.yml
```

---

## VRAM Requirements (EXL2)

### Gemma 3 27B EXL2

| Bitrate | Model Size | Cache (8K) | Total | Fits Single 24GB? |
|---------|------------|------------|-------|-------------------|
| 3.0 bpw | ~10 GB | ~2.5 GB | ~12.5 GB | Yes |
| 4.0 bpw | ~13.5 GB | ~2.5 GB | ~16 GB | Yes |
| 5.0 bpw | ~17 GB | ~2.5 GB | ~19.5 GB | Tight |
| 6.0 bpw | ~20 GB | ~2.5 GB | ~22.5 GB | No |

For dual RTX 3090 with tensor_parallel:
- Use 4.0-5.0 bpw for best quality/VRAM balance
- Leave 3-4 GB free per GPU to avoid Gemma 3 bugs

---

## Performance Comparison: vLLM vs ExLlamaV2

### MEASURED RESULTS (2026-03-15)

| Backend | Mode | Single tok/s | Notes |
|---------|------|--------------|-------|
| **vLLM W4A16** | **TP=2** | **67.5** | Tensor parallelism works |
| ExLlamaV2 EXL2 4bpw | gpu_split | **42** | **TP NOT SUPPORTED** |

**vLLM is 60% faster than ExLlamaV2 for Gemma 3 27B!**

### Critical Discovery: No Tensor Parallelism for Gemma 3

```
AssertionError: Tensor-parallel is NOT supported for Gemma3ForConditionalGeneration
```

ExLlamaV2 only supports `gpu_split` (layer distribution) for Gemma 3, not true tensor parallelism. This means:
- GPU 0 processes layers 0-19, then passes to GPU 1
- GPU 1 processes layers 20-39
- **Sequential, not parallel** - one GPU waits while the other works

### Why vLLM Wins

1. **True tensor parallelism**: Both GPUs work simultaneously on each layer
2. **NVLink utilization**: vLLM uses NVLink for inter-GPU communication
3. **CUDA graphs**: Our optimized config (`FULL_DECODE_ONLY`) is proven
4. **Marlin kernels**: Already excellent for W4A16 quantization

---

## Recommendation: Stay with vLLM

### For Each Model Size

| Model | Recommendation | Reason |
|-------|----------------|--------|
| **1B** | **vLLM W8A8** | No EXL2/EXL3 exists |
| **4B** | **vLLM W4A16** | No EXL2/EXL3 exists |
| **12B** | **vLLM W4A16** | No EXL2/EXL3 exists |
| **27B** | **vLLM W4A16** | Known Gemma 3 bugs in ExLlamaV2 |

### Why vLLM Wins for This Use Case

1. **Quantized Models Available**: RedHatAI provides all sizes
2. **Proven Performance**: 67 tok/s measured, 6x over baseline
3. **Stable**: No looping/nonsense bugs
4. **Long Context**: Works with 128K tokens
5. **CUDA Graphs**: Our config is optimized

### When to Consider ExLlamaV2

1. **Future**: Once Gemma 3 bugs are fixed
2. **Other Models**: ExLlamaV2 excels with Llama, Mistral, Qwen
3. **Lower VRAM**: EXL2 3bpw can fit 27B on single GPU
4. **EXL3**: The newer format may have fewer bugs

---

## If You Want to Test ExLlamaV2 Anyway

### Prerequisites

```bash
# Create separate environment (don't mix with vLLM!)
python -m venv ~/exllama-env
source ~/exllama-env/bin/activate

# Install
pip install exllamav2>=0.2.9
pip install tabbyapi
```

### Download 27B EXL2 (4bpw)

```bash
huggingface-cli download turboderp/gemma-3-27b-it-exl2 \
    --revision 4.0bpw \
    --local-dir ~/models/gemma-3-27b-it-exl2-4bpw
```

### Test Script (Single Request)

```python
#!/usr/bin/env python3
"""Quick ExLlamaV2 test for Gemma 3."""

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
import time

# Config
model_dir = "/home/yeb/models/gemma-3-27b-it-exl2-4bpw"

config = ExLlamaV2Config(model_dir)
config.max_seq_len = 8192

# Multi-GPU with tensor parallelism
config.tensor_parallel = True

model = ExLlamaV2(config)
model.load()

cache = ExLlamaV2Cache(model, max_seq_len=8192)
tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2DynamicGenerator(
    model=model,
    cache=cache,
    tokenizer=tokenizer,
)

# Test
prompt = "Explain quantum computing in simple terms."
start = time.perf_counter()
output = generator.generate(prompt, max_new_tokens=256)
elapsed = time.perf_counter() - start

print(f"Generated {len(output)} chars in {elapsed:.2f}s")
print(output)
```

### Watch for These Issues

1. **Looping text** after 2-3 paragraphs → config bug
2. **Degraded output** at long context → reduce to <8K
3. **Memory errors** with Q8 cache → use FP16
4. **Slow compared to vLLM** → expected, not a bug

---

## Conclusion

### Current State (March 2026)

| Factor | vLLM | ExLlamaV2 |
|--------|------|-----------|
| Gemma 3 support | **Excellent** | Buggy |
| Multi-GPU TP | **Proven** | Experimental |
| All sizes available | **Yes** | 27B only |
| Performance | **67 tok/s** | ~50-70 (estimated) |
| Stability | **High** | Issues reported |

### Verdict

**Stay with vLLM for Gemma 3 on dual RTX 3090 with NVLink.**

ExLlamaV2/V3 are excellent for other models (Llama 3, Mistral, Qwen) but the Gemma 3 implementation has unresolved bugs. The lack of quantized models for 1B/4B/12B sizes makes it impractical for smaller Gemma 3 variants.

Consider revisiting ExLlamaV3 (EXL3 format) once:
1. Gemma 3 bugs are fixed
2. Official turboderp quants exist for 1B/4B/12B
3. TabbyAPI tensor parallelism is stable

---

## References

### Primary Sources
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [ExLlamaV3 GitHub](https://github.com/turboderp-org/exllamav3)
- [TabbyAPI GitHub](https://github.com/theroyallab/tabbyAPI)
- [TabbyAPI Wiki](https://github.com/theroyallab/tabbyAPI/wiki/02.-Server-options)

### Models
- [turboderp/gemma-3-27b-it-exl2](https://huggingface.co/turboderp/gemma-3-27b-it-exl2)
- [turboderp/gemma-3-27b-it-exl3](https://huggingface.co/turboderp/gemma-3-27b-it-exl3)
- [turboderp EXL3 Collection](https://huggingface.co/collections/turboderp/exl3-models)

### Bug Reports
- [Gemma 3 Looping Bug #777](https://github.com/turboderp-org/exllamav2/issues/777)
- [Gemma 3 QAT Request #751](https://github.com/turboderp-org/exllamav2/issues/751)
- [Tensor Parallelism Discussion #571](https://github.com/turboderp-org/exllamav2/issues/571)

### Benchmarks
- [vLLM NVLink Benchmarks](http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html)
- [Multi-GPU Comparison Article](https://www.ahmadosman.com/blog/do-not-use-llama-cpp-or-ollama-on-multi-gpus-setups-use-vllm-or-exllamav2/)
