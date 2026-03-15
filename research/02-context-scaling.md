# Context Scaling Research Log

**Date:** 2026-03-15
**Goal:** Optimize vLLM for 32K and 128K context on dual RTX 3090 (2×24GB)
**Result:** ALL CONTEXT SIZES WORK WITH CUDA GRAPHS - NO PERFORMANCE LOSS!

## Executive Summary

| Context | Single tok/s | Batch tok/s | GPU Mem Util | Status |
|---------|--------------|-------------|--------------|--------|
| 8K      | 67           | 244         | 0.90         | OPTIMAL |
| 32K     | 67           | 243         | 0.90         | OPTIMAL |
| 128K    | 67           | 241         | 0.95         | OPTIMAL |

**Key Finding:** W4A16 quantization + CUDA graphs + NVLink allows full 128K context with NO performance degradation!

---

## Memory Analysis

### Gemma 3 27B Architecture
- Decoder layers: 46
- Attention heads: 64
- KV heads: 8 (GQA - grouped query attention)
- Head dimension: 128
- Quantization: W4A16 (4-bit weights, 16-bit activations)

### Theoretical KV Cache per Token
```
KV cache per token per layer:
  K: 8 heads × 128 dim × 2 bytes (bf16) = 2,048 bytes
  V: 8 heads × 128 dim × 2 bytes (bf16) = 2,048 bytes
  Total: 4,096 bytes per token per layer

Total 46 layers: 4,096 × 46 = 188,416 bytes ≈ 184 KB per token
```

### Theoretical Memory per Context Size
| Context | KV Cache (total) | KV Cache (per GPU, TP=2) |
|---------|------------------|--------------------------|
| 8K      | 1.5 GB           | ~0.75 GB                 |
| 32K     | 6.0 GB           | ~3.0 GB                  |
| 128K    | 24.0 GB          | ~12.0 GB                 |

### Available Memory Budget (per GPU)
```
Total VRAM: 24 GB
Model weights (W4A16, TP=2): ~6.75 GB
CUDA graphs (FULL_DECODE_ONLY): ~2-3 GB estimate
Remaining for KV cache: ~14-15 GB
```

**Why 128K works:** With TP=2, KV cache is split across GPUs. 12 GB per GPU + 6.75 GB model + 2 GB graphs = ~21 GB. With 0.95 utilization (22.8 GB), it fits!

---

## Test Results

### 8K Context (Baseline)
**Config:** CUDA graphs, gpu_memory_utilization=0.90
```
Single request: 67.3 tok/s
Batch (4 concurrent): 244 tok/s
```

### 32K Context
**Config:** CUDA graphs, gpu_memory_utilization=0.90
```
Single request: 67.4 tok/s
Batch (4 concurrent): 243 tok/s
```
**Observation:** Identical to 8K - no performance loss!

### 128K Context
**Config:** CUDA graphs, gpu_memory_utilization=0.95
```
Single request: 67.1 tok/s
Batch (4 concurrent): 241 tok/s
```
**Observation:** Nearly identical to 8K - minimal performance loss!

---

## Optimal Configurations

### 8K Context (launch-optimized.sh)
```bash
--max-model-len 8192
--gpu-memory-utilization 0.90
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

### 32K Context (launch-optimized-32k.sh)
```bash
--max-model-len 32768
--gpu-memory-utilization 0.90
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

### 128K Context (launch-optimized-128k.sh)
```bash
--max-model-len 131072
--gpu-memory-utilization 0.95  # Higher to fit KV cache
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

---

## Key Insights

1. **W4A16 quantization is key** - Reduces model from ~54GB to ~13.5GB, leaving room for KV cache
2. **CUDA graphs work at all context sizes** - FULL_DECODE_ONLY mode uses minimal extra memory
3. **NVLink is critical** - Enables efficient tensor parallel without bottleneck
4. **--disable-custom-all-reduce required** - RTX 3090 custom_all_reduce crashes in CUDA graph capture
5. **Limited capture sizes [1,2,4,8,16,32]** - Reduces graph memory overhead
6. **0.95 memory utilization for 128K** - Needed to fit larger KV cache

---

## Performance Summary

**All context sizes achieve ~67 tok/s single request, ~240+ tok/s batch throughput!**

This is a 6x improvement over baseline (enforce-eager at 11 tok/s).

The dual RTX 3090 with NVLink can efficiently run Gemma 3 27B at its full 128K context length without sacrificing speed.
