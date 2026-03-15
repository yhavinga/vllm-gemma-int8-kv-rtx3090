# Long Context Optimization Research

**Date:** 2026-03-15
**Hardware:** 2x RTX 3090 (NVLink), 48GB VRAM total
**Model:** Gemma 3 27B (W4A16 quantized)
**vLLM Version:** 0.17.1

## Executive Summary

| Context Size | Speed | Bottleneck | CUDA Graphs |
|--------------|-------|------------|-------------|
| < 4K tokens | **68 tok/s** | Compute | Working |
| > 4K tokens | **24 tok/s** | Memory bandwidth | Eager fallback |

The 3x speed difference is due to Gemma 3's **cascade attention** triggering eager mode fallback for long sequences.

---

## Root Cause Analysis

### Gemma 3 Architecture

Gemma 3 27B uses a hybrid attention pattern:
- **52 sliding window attention layers** (window size ~4096)
- **10 full attention layers** (attend to all tokens)

When sequence length exceeds the sliding window:
1. Cascade attention kicks in (combining sliding + full attention)
2. vLLM falls back from CUDA graphs to eager mode
3. Performance drops from ~68 tok/s to ~24 tok/s

### vLLM Warning

```
WARNING: No piecewise cudagraph for executing cascade attention.
Will fall back to eager execution if a batch runs into cascade attentions.
```

---

## CUDA Graph Modes Tested

| Mode | Cascade Support | Result |
|------|-----------------|--------|
| `FULL_DECODE_ONLY` | Falls to NONE | 24 tok/s (eager) |
| `FULL_AND_PIECEWISE` | Falls to PIECEWISE | OOM on RTX 3090 |
| `PIECEWISE` | Should work | Crashed on RTX 3090 + TP=2 |
| `--disable-cascade-attn` | Disabled | 24 tok/s (no improvement) |

### Key Finding

Disabling cascade attention removes the warning but doesn't improve speed because the bottleneck is **memory bandwidth**, not CUDA graph overhead.

---

## Memory Bandwidth Analysis

### KV Cache Size Calculation

For 28K input tokens on Gemma 3 27B:
- Layers: 46
- KV heads: 16 (GQA)
- Head dim: 256
- Per token: 46 × 16 × 256 × 2 (K+V) × 2 bytes = ~1.2 MB
- Total for 28K tokens: ~33 GB (split across 2 GPUs = ~16.5 GB each)

### Theoretical Maximum Speed

```
RTX 3090 memory bandwidth: ~936 GB/s per GPU
KV cache read per decode step: ~16.5 GB
Time per decode: 16.5 GB / 936 GB/s = ~17.6 ms
Theoretical max: ~57 tok/s
```

### Actual vs Theoretical

- Theoretical: ~57 tok/s
- Actual: ~24 tok/s
- Efficiency: ~42%

The gap is due to:
- Tensor parallel NVLink synchronization overhead
- Attention computation
- Other memory reads (model weights, activations)
- Non-overlapped memory operations

---

## FP8 KV Cache: Why Not on RTX 3090

### Hardware Requirements

| Architecture | SM Version | FP8 Support | Example GPUs |
|--------------|------------|-------------|--------------|
| Ampere | 8.6 | **No** | RTX 3090, A100 |
| Ada Lovelace | 8.9 | **Yes** | RTX 4090, L40 |
| Hopper | 9.0 | **Yes** | H100, H200 |

### Technical Reason

FP8 (E4M3/E5M2 formats) requires native tensor core instructions that don't exist on Ampere GPUs. Attempting to use `--kv-cache-dtype fp8` results in:

```
ValueError: type fp8e4nv not supported in this architecture.
```

### Potential Alternatives

| Method | Status | Would Help? |
|--------|--------|-------------|
| FP8 KV cache | Not available on Ampere | 2x KV capacity |
| INT8 KV cache | Not implemented in vLLM | 2x KV capacity |
| Sliding window only | Loses full attention | Quality tradeoff |

INT8 tensor cores exist on Ampere, but vLLM hasn't implemented INT8 KV cache yet. See [GitHub Issue #33480](https://github.com/vllm-project/vllm/issues/33480).

---

## Piecewise CUDA Graphs Deep Dive

### How It Works

vLLM has 5 cudagraph modes:

```
NONE              - No CUDA graphs (eager)
PIECEWISE         - Piecewise graphs only (attention outside graph)
FULL              - Full graphs for all batches
FULL_DECODE_ONLY  - Full for decode, NONE for prefill/mixed
FULL_AND_PIECEWISE - Full for decode, piecewise for prefill/mixed
```

### Cascade Attention Compatibility

From [PR #20059](https://github.com/vllm-project/vllm/pull/20059):

> "While cascade attention is not cudagraph compatible, it is now compatible with all possible cudagraph mode configurations. If a batch uses cascade attention, it always gets dispatched to PIECEWISE mode if available (otherwise NONE)."

### The Problem on RTX 3090

- `FULL_DECODE_ONLY` has no piecewise fallback → falls to NONE (eager)
- `FULL_AND_PIECEWISE` requires more memory → OOM
- `PIECEWISE` alone crashes with TP=2 on RTX 3090

---

## Optimization Attempts

### What Was Tried

1. **CUDA graphs (FULL_DECODE_ONLY)** - Works for short context, eager for long
2. **FULL_AND_PIECEWISE mode** - OOM during warmup
3. **PIECEWISE mode** - Crashed with TP=2
4. **--disable-cascade-attn** - No speed improvement
5. **max_num_seqs=1** - No improvement
6. **Various NCCL tuning** - Already optimized

### What Would Help (But Can't Do)

1. **FP8 KV cache** - Hardware doesn't support
2. **INT8 KV cache** - Not implemented in vLLM
3. **FlashInfer backend** - Doesn't support Gemma 3's interleaved attention

---

## Recommendations

### For Maximum Speed

1. **Stay under 4K context** when possible → 68 tok/s
2. **Use batching** for throughput → 244 tok/s with 4 concurrent requests
3. **Accept 24 tok/s** for long context → near-optimal for hardware

### For Future Upgrades

1. **RTX 4090/5090** - FP8 KV cache support, ~2x long-context speed
2. **Wait for vLLM updates** - INT8 KV cache may come
3. **Try SGLang** - Different engine, may handle Gemma 3 differently

---

## Launch Script (Optimized)

```bash
#!/bin/bash
# Optimized for RTX 3090 + Gemma 3 27B

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216

vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]}' \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

---

## References

- [vLLM CUDA Graphs Documentation](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [PR #20059 - Piecewise Cudagraph Support](https://github.com/vllm-project/vllm/pull/20059)
- [Issue #23261 - Graph Splitting and Attention Fusion](https://github.com/vllm-project/vllm/issues/23261)
- [FP8 KV Cache Requirements](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [Gemma 3 FlashInfer Issue](https://github.com/vllm-project/vllm/issues/20865)
- [INT8 KV Cache Feature Request](https://github.com/vllm-project/vllm/issues/33480)
- [RTX 3090 Benchmarks](http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html)
