# Gemma 3 1B: TP=1 vs TP=2 Comparison

**Date:** 2026-03-15
**Hardware:** 2x RTX 3090 (48GB VRAM total, NVLink NV4)
**vLLM:** 0.17.1
**Model:** RedHatAI/gemma-3-1b-it-quantized.w8a8

---

## Executive Summary

**Key Finding: TP=2 provides only 13% speedup for 1B - smallest gain of all sizes**

| Metric | TP=1 (1 GPU) | TP=2 (2 GPUs) | Improvement |
|--------|--------------|---------------|-------------|
| **Single Request** | 263 tok/s | **298 tok/s** | **+13%** |
| Batch (4) | 887 tok/s | 948 tok/s | +7% |
| Batch (8) | 1929 tok/s | 2060 tok/s | +7% |
| Batch (16) | 3649 tok/s | 3747 tok/s | +3% |
| Batch (32) | 5932 tok/s | 6044 tok/s | +2% |
| KV Cache | 714K tokens | 735K tokens | **+3%** |

**Recommendation:** Use TP=1 for 1B unless you need the extra 13% single-request speed and both GPUs are available.

---

## TP Speedup Comparison: All Model Sizes

| Model | TP Speedup | TP=1 (tok/s) | TP=2 (tok/s) |
|-------|------------|--------------|--------------|
| **1B** | **+13%** | 263 | 298 |
| **4B** | **+20%** | 176 | 212 |
| **12B** | **+39%** | 83 | 115 |

**Insight:** Larger models benefit more from TP=2. The 1B model is so fast that communication overhead becomes more significant relative to compute time.

---

## Why 1B Benefits Less from TP=2

### 1. Already Very Fast
- 263 tok/s = 3.8 ms per token
- Communication overhead (52 μs) = 1.4% of token time
- Less room for improvement

### 2. Fewer Layers
- 1B: 26 layers
- 4B: 34 layers
- 12B: 40 layers

Fewer layers = less compute to parallelize.

### 3. Smaller Hidden Dimension
- 1B: 1152 hidden dim
- 4B: 2560 hidden dim
- 12B: 3584 hidden dim

Smaller dimensions = faster operations, less benefit from splitting.

### 4. KV Cache Already Huge
- TP=1: 714K tokens (211x concurrency at 8K context)
- TP=2: 735K tokens (+3%)

The 1B model is so small that KV cache is already abundant. TP=2 doesn't add much.

---

## Measured Results

### Single Request Performance

| Config | Tokens | Time | Throughput |
|--------|--------|------|------------|
| TP=1 | 256 | 0.97s | **263.0 tok/s** |
| TP=2 | 256 | 0.86s | **298.2 tok/s** |

### Batch Throughput Comparison

| Batch Size | TP=1 | TP=2 | Improvement |
|------------|------|------|-------------|
| 4 | 887 | 948 | +7% |
| 8 | 1929 | 2060 | +7% |
| 16 | 3649 | 3747 | +3% |
| 32 | 5932 | 6044 | +2% |

**Note:** At higher batch sizes, TP=2's advantage shrinks. The single-GPU's memory bandwidth can handle the parallel batches efficiently.

---

## Attention Backend Difference

Unlike 4B/12B which use TRITON_ATTN (multimodal models), 1B uses:
- **FLASH_ATTN** - optimized for text-only models

This may partially explain the different scaling characteristics.

---

## Configurations

### TP=1 Configuration (Recommended for 1B)

```bash
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

vllm serve RedHatAI/gemma-3-1b-it-quantized.w8a8 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

### TP=2 Configuration

```bash
CUDA_VISIBLE_DEVICES=0,1
CUDA_FORCE_P2P_ACCESS=1
VLLM_SKIP_P2P_CHECK=1
NCCL_P2P_LEVEL=NVL
NCCL_BUFF_SIZE=16777216
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

vllm serve RedHatAI/gemma-3-1b-it-quantized.w8a8 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

---

## Recommendation: When to Use Each Config

### Use TP=1 (Recommended for 1B)
- Run another model on GPU1 (e.g., 12B or 27B)
- Already fast enough at 263 tok/s
- Simpler deployment
- No inter-GPU communication overhead

### Use TP=2
- Need maximum single-request latency (298 tok/s)
- Both GPUs dedicated to 1B workload
- Extremely high throughput requirements (6000+ tok/s)

---

## Full Comparison Table

| Model | Config | Single tok/s | Batch(8) | KV Cache | TP Speedup |
|-------|--------|--------------|----------|----------|------------|
| 1B | TP=1 | 263 | 1929 | 714K | - |
| 1B | TP=2 | 298 | 2060 | 735K | **+13%** |
| 4B | TP=1 | 176 | 1170 | 121K | - |
| 4B | TP=2 | 212 | 1355 | 267K | **+20%** |
| 12B | TP=1 | 83 | 765* | 32K | - |
| 12B | TP=2 | 115 | 771* | 86K | **+39%** |
| 27B | TP=2 | 67 | 244 | - | N/A |

*Batch(8) varies due to warmup; values approximate.

**Conclusion:** The smaller the model, the less benefit from tensor parallelism. For 1B, stick with TP=1 unless you specifically need the extra 13% speed.
