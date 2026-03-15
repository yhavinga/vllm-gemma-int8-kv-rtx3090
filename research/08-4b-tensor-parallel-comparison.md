# Gemma 3 4B: TP=1 vs TP=2 Comparison

**Date:** 2026-03-15
**Hardware:** 2x RTX 3090 (48GB VRAM total, NVLink NV4)
**vLLM:** 0.17.1
**Model:** RedHatAI/gemma-3-4b-it-quantized.w4a16

---

## Executive Summary

**Key Finding: TP=2 is 20% faster than TP=1 for single requests**

| Metric | TP=1 (1 GPU) | TP=2 (2 GPUs) | Improvement |
|--------|--------------|---------------|-------------|
| **Single Request** | 176 tok/s | **212 tok/s** | **+20%** |
| Batch (4) | 581 tok/s | 675 tok/s | +16% |
| Batch (8) | 1170 tok/s | 1355 tok/s | +16% |
| Batch (16) | 2089 tok/s | 2425 tok/s | +16% |
| Batch (32) | N/A | **3646 tok/s** | - |
| KV Cache | 121K tokens | **267K tokens** | **2.2x** |
| Max Concurrency | 31.7x | **70x** | **2.2x** |

**Recommendation:** TP=2 provides consistent 16-20% speedup and 2.2x KV cache, but trade-off is using both GPUs.

---

## Comparison with 12B TP Results

| Model | TP Speedup | Notes |
|-------|------------|-------|
| **4B** | +20% | Moderate improvement |
| **12B** | +39% | Larger improvement |
| **27B** | N/A | Requires TP=2 to fit |

**Insight:** Larger models benefit more from TP=2 because:
1. More layers = more compute to parallelize
2. Higher memory bandwidth requirements
3. Communication overhead is relatively smaller

---

## Measured Results

### Single Request Performance

| Config | Tokens | Time | Throughput |
|--------|--------|------|------------|
| TP=1 | 256 | 1.45s | **176.2 tok/s** |
| TP=2 | 256 | 1.20s | **212.3 tok/s** |

TP=2 is **20% faster** for single requests.

### Batch Throughput (Aggregate tok/s)

| Batch Size | TP=1 | TP=2 | Improvement |
|------------|------|------|-------------|
| 4 | 581 | 675 | +16% |
| 8 | 1170 | 1355 | +16% |
| 12 | 1623 | 1870 | +15% |
| 16 | 2089 | 2425 | +16% |
| 24 | - | 2930 | - |
| 32 | - | 3646 | - |

TP=2 consistently **16% faster** for batched workloads.

### KV Cache Capacity

| Metric | TP=1 | TP=2 | Ratio |
|--------|------|------|-------|
| KV Cache Size | 120,784 tokens | 267,376 tokens | **2.2x** |
| Max Concurrency (8K context) | 31.65x | 70.05x | **2.2x** |

---

## When to Use TP=2 for 4B

**Use TP=2 when:**
- You need maximum single-request latency (212 vs 176 tok/s)
- You need high concurrent request capacity (70x vs 32x)
- Both GPUs are available and not needed for other tasks

**Use TP=1 when:**
- You want to run another model on the second GPU (e.g., 27B on GPU1)
- Single GPU performance (176 tok/s) is sufficient
- You want simpler deployment

---

## Configurations

### TP=1 Configuration

```bash
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

vllm serve RedHatAI/gemma-3-4b-it-quantized.w4a16 \
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

vllm serve RedHatAI/gemma-3-4b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

---

## Theoretical Analysis

### Model Specs

```
Gemma 3 4B W4A16:
- Model size: ~2.6 GB (4-bit weights)
- Hidden dim: 2560
- Layers: 34
```

### Communication Overhead

```
Per-token all-reduce: 34 layers × 2 × 2560 × 2 bytes = 347 KB
At 176 tok/s: 347 KB × 176 = 61 MB/s
NVLink capacity: 112.5 GB/s
Utilization: 0.05%
```

Communication overhead is negligible.

### Why TP=2 Helps Less Than 12B

1. **4B already very fast** - less room for improvement
2. **Fewer layers** (34 vs 40) - less compute to parallelize
3. **Smaller hidden dim** (2560 vs 3584) - each operation faster
4. **Fixed overhead** - CUDA graph capture, sync overhead is constant

---

## Summary Table: All Models

| Model | Config | Single tok/s | Batch(4) | KV Cache |
|-------|--------|--------------|----------|----------|
| 4B | TP=1 | 176 | 581 | 121K |
| 4B | TP=2 | **212** (+20%) | 675 | 267K |
| 12B | TP=1 | 83 | 299 | 32K |
| 12B | TP=2 | **115** (+39%) | 394 | 86K |
| 27B | TP=2 | 67 | 244 | - |

**Key Insight:** For smaller models, TP=2 provides moderate speedup (+20%). For larger models, the speedup is more significant (+39%).
