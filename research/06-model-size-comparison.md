# Gemma 3 Model Size Comparison Research

**Date:** 2026-03-15
**Hardware:** 2x RTX 3090 (48GB VRAM total, NVLink NV4)
**vLLM:** 0.17.1

---

## Executive Summary: MEASURED RESULTS

### Performance by Model Size (All Measured on RTX 3090)

| Model | Quant | GPU Config | Single tok/s | Batch(4) tok/s |
|-------|-------|------------|--------------|----------------|
| **1B** | W8A8 | 1x GPU | **263** | **887** |
| **1B** | W8A8 | 2x GPU (TP=2) | **298** | **948** |
| **4B** | W4A16 | 1x GPU | **176** | **581** |
| **4B** | W4A16 | 2x GPU (TP=2) | **212** | **675** |
| **12B** | W4A16 | 1x GPU | **83** | **301** |
| **12B** | W4A16 | 2x GPU (TP=2) | **115** | **394** |
| **27B** | W4A16 | 2x GPU (TP=2) | **67.5** | **244** |

**Tensor Parallelism Speedups:**
- 1B TP=2: **+13%** faster than TP=1. See [09-1b-tensor-parallel-comparison.md](09-1b-tensor-parallel-comparison.md).
- 4B TP=2: **+20%** faster than TP=1. See [08-4b-tensor-parallel-comparison.md](08-4b-tensor-parallel-comparison.md).
- 12B TP=2: **+39%** faster than TP=1. See [07-12b-tensor-parallel-comparison.md](07-12b-tensor-parallel-comparison.md).

**Key Insight:** Larger models benefit more from TP=2 (39% > 20% > 13%).

### Recommendation: RedHatAI W4A16 for Most Use Cases

| Model Size | Best Choice | Reason |
|------------|-------------|--------|
| **1B** | RedHatAI W8A8 | Only quantized option for vLLM, blazing fast |
| **4B** | RedHatAI W4A16 | Best quality/speed balance for single GPU |
| **12B** | RedHatAI W4A16 (TP=2) | **115 tok/s with dual GPU**, or 83 tok/s single GPU |
| **27B** | RedHatAI W4A16 | Proven 67 tok/s, requires TP=2 with NVLink |

**Key Finding:** RedHatAI models are recommended for better vLLM integration and compressed-tensors format.

---

## Available Quantized Models

### RedHatAI (Recommended)

| Size | W8A8 | W4A16 | Notes |
|------|------|-------|-------|
| 1B | ✅ `gemma-3-1b-it-quantized.w8a8` | ❌ | Only option |
| 4B | ✅ `gemma-3-4b-it-quantized.w8a8` | ✅ `gemma-3-4b-it-quantized.w4a16` | W4A16 preferred |
| 12B | ✅ `gemma-3-12b-it-quantized.w8a8` | ✅ `gemma-3-12b-it-quantized.w4a16` | W4A16 fits single GPU |
| 27B | ❌ | ✅ `gemma-3-27b-it-quantized.w4a16` | Requires TP=2 |

**Quantization Method:** compressed-tensors (Marlin kernels)

### ISTA-DASLab

| Size | GPTQ-4b-128g | Notes |
|------|--------------|-------|
| 1B | ❌ | Not available |
| 4B | ✅ `gemma-3-4b-it-GPTQ-4b-128g` | |
| 12B | ✅ `gemma-3-12b-it-GPTQ-4b-128g` | |
| 27B | ✅ `gemma-3-27b-it-GPTQ-4b-128g` | |

**Quantization Method:** GPTQ 4-bit, group size 128

### Google QAT (GGUF only - not vLLM compatible)

| Size | Model |
|------|-------|
| 1B | `google/gemma-3-1b-it-qat-q4_0-gguf` |
| 4B | `google/gemma-3-4b-it-qat-q4_0-gguf` |
| 12B | `google/gemma-3-12b-it-qat-q4_0-gguf` |
| 27B | `google/gemma-3-27b-it-qat-q4_0-gguf` |

**Note:** GGUF requires llama.cpp/Ollama. Not compatible with vLLM tensor parallelism.

---

## VRAM Requirements

| Model | W8A8 | W4A16/GPTQ | Recommended GPU Config |
|-------|------|------------|------------------------|
| 1B | ~1.3 GB | N/A | Single GPU |
| 4B | ~5 GB | ~2.6 GB | Single GPU |
| 12B | ~14 GB | ~7 GB | Single GPU (W4A16), TP=2 (W8A8) |
| 27B | N/A | ~13.5 GB | TP=2 required |

---

## Performance Benchmarks (RTX 3090) - MEASURED

All results measured 2026-03-15 with vLLM 0.17.1.

### Complete Results Table

| Model | Quant | TP | Context | Single (medium) | Single (long) | Batch(4) |
|-------|-------|----|---------|-----------------|--------------:|----------|
| **1B** | W8A8 | 1 | 4K | **268.8 tok/s** | 250.3 tok/s | **898.5 tok/s** |
| **4B** | W4A16 | 1 | 4K | **177.4 tok/s** | 172.4 tok/s | **606.3 tok/s** |
| **12B** | W4A16 | 1 | 4K | **83.0 tok/s** | 80.3 tok/s | **301.3 tok/s** |
| **27B** | W4A16 | 2 | 8K | **67.5 tok/s** | 65.5 tok/s | **244.3 tok/s** |

### 27B Baseline Comparison

| Config | Single tok/s | Batch(4) tok/s | Notes |
|--------|-------------|----------------|-------|
| **CUDA Graphs (production)** | **67.5** | **244.3** | Current optimal |
| torch.compile, no cudagraph | 20 | 71 | 3x slower |
| Baseline (enforce-eager) | 11 | 40 | 6x slower |

### Scaling Analysis

| Metric | 1B→4B | 4B→12B | 12B→27B |
|--------|-------|--------|---------|
| Single speed ratio | 0.66x | 0.47x | 0.81x |
| Batch speed ratio | 0.67x | 0.50x | 0.81x |
| Model size ratio | 4x | 3x | 2.25x |

**Insight:** The 12B→27B scaling is more efficient (0.81x) because 27B uses tensor parallelism across two GPUs, effectively doubling memory bandwidth.

---

## Optimal Launch Configurations

### 1B Model (Single GPU)

```bash
export CUDA_VISIBLE_DEVICES=0

vllm serve RedHatAI/gemma-3-1b-it-quantized.w8a8 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

**Attention Backend:** FLASH_ATTN (auto-selected)

### 4B Model (Single GPU)

```bash
export CUDA_VISIBLE_DEVICES=0

vllm serve RedHatAI/gemma-3-4b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

### 12B Model (Single GPU)

```bash
export CUDA_VISIBLE_DEVICES=0

vllm serve RedHatAI/gemma-3-12b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

### 27B Model (Dual GPU with NVLink)

```bash
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

---

## RedHatAI vs ISTA-DASLab Comparison

### Technical Differences

| Aspect | RedHatAI W4A16 | ISTA-DASLab GPTQ |
|--------|----------------|------------------|
| Format | compressed-tensors | GPTQ |
| Kernel | Marlin | Marlin (via GPTQModel) |
| Group size | N/A (channelwise) | 128 |
| Vision tower | Quantized | Preserved (FP16) |
| vLLM support | Native | Native |

### Quality Assessment (27B, Dutch Summarization Test)

Both models produced equivalent quality summaries in our testing:
- Correct identification of main points
- Proper parliamentary language style
- Accurate question extraction

**Verdict:** No meaningful quality difference for practical use.

### Compatibility Notes

1. **SGLang:** ISTA-DASLab GPTQ works better with SGLang due to preserved vision tower
2. **vLLM:** Both work equally well
3. **Tensor Parallelism:** Both support TP=2 for 27B

---

## Why RedHatAI is Recommended

1. **Official Support:** Maintained by Red Hat / vLLM team
2. **Consistent Format:** Uses newer compressed-tensors format
3. **All Sizes:** Provides W8A8 for 1B (only option)
4. **Documentation:** Better documented model cards
5. **Performance:** Identical to GPTQ (Marlin kernel used for both)

---

## Attention Backend Selection (vLLM 0.17.1)

vLLM auto-selects the optimal attention backend:

| Model Type | Backend | Notes |
|------------|---------|-------|
| Gemma3ForCausalLM (1B) | FLASH_ATTN | Text-only model |
| Gemma3ForConditionalGeneration (4B+) | TRITON_ATTN | Multimodal model |

**Important:** FlashInfer is NOT used for Gemma 3 on RTX 3090. The flashinfer package is still required as a dependency but doesn't affect performance.

---

## Environment Requirements

### Fixed Issue: SGLang Contamination

If you installed SGLang in the same venv, it may have corrupted flashinfer:

```bash
# Check versions
pip show flashinfer-cubin flashinfer-python | grep Version

# Fix if mismatched
pip install flashinfer-cubin==0.6.4 --force-reinstall
```

### Package Versions (Known Good)

```
vllm==0.17.1
torch==2.10.0+cu128
flashinfer-python==0.6.4
flashinfer-cubin==0.6.4
```

---

## Use Case Recommendations

| Use Case | Model | Rationale |
|----------|-------|-----------|
| **Edge/Mobile backend** | 1B W8A8 | Tiny, fast, fits anywhere |
| **Low-latency chat** | 4B W4A16 | Good quality, ~100+ tok/s |
| **Balanced quality/speed** | 12B W4A16 | Excellent quality, single GPU |
| **Maximum quality** | 27B W4A16 | Best output, dual GPU |
| **Batch processing** | 27B W4A16 | 244 tok/s aggregate |

---

## Scripts

- `scripts/launch-server.sh` - 27B with 8K context (TP=2)
- `scripts/launch-server-32k.sh` - 27B with 32K context
- `scripts/launch-server-128k.sh` - 27B with 128K context
- `scripts/benchmark.py` - Performance measurement
- `scripts/benchmark_all_sizes.py` - Multi-model comparison

---

## ExLlamaV2/V3 Alternative Evaluation

**See full research:** [10-exllamav2-multi-gpu-research.md](10-exllamav2-multi-gpu-research.md)

### Summary: vLLM is Better for Gemma 3

| Model | EXL2/EXL3 Available | Recommendation |
|-------|---------------------|----------------|
| **1B** | **NO** | Use vLLM W8A8 |
| **4B** | **NO** | Use vLLM W4A16 |
| **12B** | **NO** | Use vLLM W4A16 |
| **27B** | Yes (buggy) | Use vLLM W4A16 |

### Key Issues with ExLlamaV2 + Gemma 3

1. **No quantizations for 1B/4B/12B** - turboderp only provides 27B
2. **Known bugs** - Looping/nonsense generation after 2-3 paragraphs
3. **Long context issues** - Gemma 3's sliding window not properly implemented
4. **Q8 cache problems** - Must use FP16 cache, losing VRAM savings

### When ExLlamaV2 Makes Sense

- **Other models** (Llama 3, Mistral, Qwen) - excellent support
- **Extreme VRAM constraints** - EXL2 3bpw can fit 27B on single GPU
- **Future** - Once Gemma 3 bugs are fixed

---

## References

- [RedHatAI Gemma 3 Models](https://huggingface.co/RedHatAI)
- [ISTA-DASLab GPTQ Collection](https://huggingface.co/collections/ISTA-DASLab/gemma3-gptq)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Gemma 3 Technical Report](https://ai.google.dev/gemma)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [turboderp gemma-3-27b-it-exl2](https://huggingface.co/turboderp/gemma-3-27b-it-exl2)
