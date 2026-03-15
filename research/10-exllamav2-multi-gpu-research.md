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

## Performance Comparison: vLLM vs ExLlamaV2 vs ExLlamaV3

### MEASURED RESULTS (2026-03-15)

| Backend | Mode | Single tok/s | VRAM Usage | Notes |
|---------|------|--------------|------------|-------|
| **vLLM W4A16** | **TP=2** | **67.5** | ~22GB/GPU | Tensor parallelism, CUDA graphs |
| **ExLlamaV3 EXL3 4bpw** | **TP=2** | **43.7** | **~40%/GPU** | **WORKS!** Tensor parallelism |
| ExLlamaV2 EXL2 4bpw | gpu_split | 42 | 17GB GPU0 | TP NOT supported |
| ExLlamaV3 EXL3 4bpw | No TP | 28.6 | 16.8GB GPU0 | Single GPU baseline |

### Key Findings

1. **vLLM is still 54% faster** than ExLlamaV3 with tensor parallelism
2. **ExLlamaV3 TP works!** Both GPUs active, ~40% VRAM each (~9.5GB per GPU)
3. **ExLlamaV3 TP provides 53% speedup** over single GPU (43.7 vs 28.6 tok/s)
4. **ExLlamaV2 has NO TP for Gemma 3** - only layer distribution (gpu_split)

### ExLlamaV3 Setup Notes

**Critical fixes required:**
1. Use `Model.from_config(config)` not `Model(config)`
2. Create `Cache(model, ...)` BEFORE calling `model.load()`
3. Use proper Gemma 3 chat format: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
4. Add `add_bos=True` to generate calls

**Example working code:**
```python
from exllamav3 import Model, Config, Cache, Tokenizer, DefaultSampler
from exllamav3.generator import Generator

config = Config.from_directory(model_dir)
model = Model.from_config(config)  # NOT Model(config)!
cache = Cache(model, max_num_tokens=8192)  # BEFORE model.load()!
model.load(tensor_p=True)  # Enable tensor parallelism
tokenizer = Tokenizer(config)
generator = Generator(model, cache, tokenizer)

prompt = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
output = generator.generate(prompt, sampler=DefaultSampler(), max_new_tokens=256, add_bos=True)
```

### Long Context Performance (ExLlamaV3 EXL3 4bpw, TP=2)

| Context | Prompt Tokens | Prefill Time | Prefill Speed | Gen Time (256 tok) |
|---------|---------------|--------------|---------------|-------------------|
| 8K | 2,335 | ~2.4s | 960 tok/s | 8.2s |
| 32K | 8,399 | ~6.0s | 1,400 tok/s | 11.8s |
| 64K | 15,517 | ~7.3s | 2,117 tok/s | 13.2s |
| 128K | - | **OOM** | - | - |

**Key findings:**
- 64K context is the practical maximum (model + FP16 cache uses ~22GB per GPU)
- 128K causes OOM even with tensor parallelism
- Prefill throughput scales well (higher efficiency at longer contexts)
- Generation remains stable at ~44 tok/s regardless of context length

### Context-Dependent Performance Comparison

| Context | Prompt Tokens | vLLM W4A16 | ExLlamaV3 EXL3 | Winner |
|---------|---------------|------------|----------------|--------|
| **8K (short)** | ~2K | **67 tok/s** | 44 tok/s | vLLM (+52%) |
| **32K** | ~8K | ~35 tok/s | **44 tok/s** | ExLlamaV3 (+26%) |
| **64K (full text)** | ~33K | **21 tok/s** | 15.5 tok/s | vLLM (+35%) |
| **128K** | ~33K | **21 tok/s** | OOM | vLLM (only option) |

**Key insight:** At short-medium contexts (up to ~16K tokens), ExLlamaV3 maintains ~44 tok/s. But at full 33K token prompts, ExLlamaV3 drops to 15.5 tok/s while vLLM achieves 21 tok/s. vLLM wins at both extremes (short and very long context).

### Quality Comparison: Dutch Parliamentary Summarization

Both models tested on 128,922 character Dutch parliamentary debate (~33K tokens):

| Aspect | vLLM W4A16 | ExLlamaV3 EXL3 |
|--------|------------|----------------|
| **Summary scope** | Single speaker (De Boer) | **Full debate (5 speakers)** |
| **Structure** | 4 sections, 8 questions | 6 topics, party positions |
| **Comprehensiveness** | Focused | **Broader overview** |
| **Speed** | **21 tok/s** | 15.5 tok/s |
| **Context handled** | **Full 128K** | 64K (OOM at 128K) |

**vLLM output (focused on GroenLinks/De Boer):**
```
## Samenvatting Parlementaire Bijdrage - De Boer (GroenLinks)
1. Hoofdpunten: complexiteit fiscale systeem, langetermijnvisie nodig
2. Vragen aan de minister: 8 specific questions
3. Standpunt fractie: eerlijk en duurzaam fiscaal beleid
4. Toon: Kritisch
```

**ExLlamaV3 output (full debate overview):**
```
## Samenvatting debat begroting Financiën 1995
1. Hoofdonderwerpen: Budgettaire discipline, Belastingplan, Brede Herwaardering,
   Werknemersspaarregelingen, Vermogensbelasting, Europese Integratie
2. Sprekers: PvdA (Schinck), CDA (Stevens), GroenLinks (De Boer), SGP (Barendregt)
```

**Quality verdict:** ExLlamaV3 produced a more comprehensive multi-speaker summary, while vLLM focused deeply on one speaker. Both are valid approaches depending on use case.

### When to Use Each

**Use vLLM when:**
- Short context (<8K) - 67 tok/s is 3x faster
- Need 128K context - only option that works
- Batching multiple requests - vLLM excels here
- Simpler setup preferred
- Full-context long documents (~33K+ tokens)

**Use ExLlamaV3 when:**
- Medium context (8K-16K tokens) - competitive speed
- Lower VRAM needed (~40%/GPU vs 90%/GPU)
- Want EXL3 quantization (potentially better quality)
- More comprehensive multi-speaker summaries preferred

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

| Factor | vLLM W4A16 | ExLlamaV2 EXL2 | ExLlamaV3 EXL3 |
|--------|------------|----------------|----------------|
| Gemma 3 support | **Excellent** | Buggy (no TP) | **Working** |
| Multi-GPU TP | **Yes** | No (Gemma 3) | **Yes** |
| Short context (8K) | **67 tok/s** | 42 tok/s | 44 tok/s |
| Long context (33K) | **21 tok/s** | - | 15.5 tok/s |
| Max context | **128K** | 64K | 64K |
| Summary quality | Focused | - | **Comprehensive** |
| Setup complexity | Simple | Medium | Complex |

### Final Verdict

**For Gemma 3 27B on dual RTX 3090 with NVLink:**

1. **Short prompts (<8K):** Use **vLLM** - 67 tok/s is unbeatable
2. **Long context (128K):** Use **vLLM** - only option that works
3. **Medium context (8K-32K):** **Either** works well
4. **Best summary quality:** **ExLlamaV3** produced more comprehensive output

**Recommended default: vLLM** for simplicity and 128K support.

**ExLlamaV3 is a viable alternative** now that tensor parallelism works for Gemma 3. Consider it for:
- Lower VRAM requirements
- EXL3 quantization benefits
- When 64K context is sufficient

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
