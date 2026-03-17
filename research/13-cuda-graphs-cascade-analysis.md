# CUDA Graphs and Cascade Attention Analysis

## Summary

Investigated whether "piecewise CUDA graphs for cascade attention" could solve Gemma 3's
long-context performance cliff. **Conclusion: The hypothesis is wrong.** The performance
cliff is caused by memory bandwidth, not CUDA graph issues.

## The Misunderstanding

The assumption was that cascade attention disables CUDA graphs, causing the ~4K token
performance cliff. This is incorrect for two reasons:

1. **Cascade attention is disabled for sliding window models** (flash_attn.py:992-993)
2. **Triton backend doesn't support cascade attention at all** (returns `False` unconditionally)

```python
# vllm/v1/attention/backends/flash_attn.py
def use_cascade_attention(...):
    if use_alibi or use_sliding_window or use_local_attention:
        return False  # Gemma 3 hits this - sliding window
```

## What Cascade Attention Actually Is

vLLM's cascade attention is a **prefix sharing optimization** for batched inference:
- When multiple requests share a common prefix, compute attention once
- Split into: prefix kernel (shared, non-causal) + suffix kernel (per-request, causal)
- Merge results with softmax LSE combination

This is **NOT** the same as Gemma 3's hybrid attention architecture.

## Gemma 3's Actual Architecture

| Component | Layers | Window | Complexity |
|-----------|--------|--------|------------|
| Sliding attention | 52 | 1024 tokens | O(n) |
| Global attention | 10 | Full context | O(n²) |

The 5:1 local-to-global ratio means 10 layers must read the ENTIRE KV cache at every
decode step, regardless of CUDA graphs.

## The Real Bottleneck: Memory Bandwidth

RTX 3090: 936 GB/s memory bandwidth

**Decode memory reads per token:**

| Context | Sliding KV (52 layers) | Global KV (10 layers) | Model Weights | Total |
|---------|------------------------|----------------------|---------------|-------|
| 1K | 0.17 GB | 0.03 GB | 14 GB | 14.2 GB |
| 4K | 0.17 GB | 0.13 GB | 14 GB | 14.3 GB |
| 8K | 0.17 GB | 0.26 GB | 14 GB | 14.4 GB |
| 16K | 0.17 GB | 0.52 GB | 14 GB | 14.7 GB |
| 32K | 0.17 GB | 1.05 GB | 14 GB | 15.2 GB |

*Sliding window layers always read 1024 tokens max, regardless of context length.*

**Theoretical max throughput:**
- 936 GB/s ÷ 14.2 GB = 66 tok/s at 1K context
- 936 GB/s ÷ 15.2 GB = 62 tok/s at 32K context

**Observed (INT8 KV):**
- 61 tok/s at short context
- 9.6 tok/s at 32K context

The gap between theoretical and observed at long context comes from:
1. Attention compute (not just memory reads)
2. Tensor parallelism synchronization overhead
3. Non-overlapped execution

## Why Piecewise CUDA Graphs for Cascade Wouldn't Help

Even if we implemented it, it wouldn't help Gemma 3 because:

1. **Cascade attention is already disabled** for sliding window models
2. **The bottleneck is architectural** - 10 global layers read full context
3. **INT8 KV cache already addresses bandwidth** - 2x memory savings

### What Implementation Would Require (For Other Models)

**Option 1: Pre-capture graphs for prefix length buckets**
```
6 prefix buckets × 8 decode buckets × 62 layers = 2,976 graphs
```
Memory explosion, impractical.

**Option 2: CUDA 12.3+ graph conditionals**
- Requires CUDA 12.3+, PyTorch integration
- Significant engineering effort
- Not in vLLM roadmap

**Option 3: Fused cascade kernel**
- Rewrite FlashAttention internals
- Massive undertaking

## What Would Actually Help Gemma 3

### Already Done
- [x] INT8 KV cache (+87% at long context)
- [x] FULL_DECODE_ONLY CUDA graphs
- [x] Speculative decoding (tried, no improvement - draft overhead exceeded gains)

### Investigated Optimizations

#### 1. Global Layer KV Cache Fusion - NOT FEASIBLE

Each of the 10 global layers has independent K/V projections (different learned weights).
There's no shared data to fuse. The only theoretical optimization would be kernel-level
multi-layer attention, which would actually be WORSE for memory bandwidth (need to keep
10x more KV data resident in registers/shared memory).

#### 2. Layer-specific INT4 KV Quantization - NOT WORTH IT

Analyzed using INT4 for global layers (read full context) while keeping INT8 for sliding
layers (read only 1024 tokens).

| Context | INT8 All | INT4 Global | KV Reduction | Throughput Gain |
|---------|----------|-------------|--------------|-----------------|
| 4K | 0.72 GB | 0.56 GB | 22% | 1.1% |
| 8K | 1.03 GB | 0.72 GB | 30% | 2.1% |
| 16K | 1.66 GB | 1.03 GB | 38% | 4.2% |
| 32K | 2.91 GB | 1.66 GB | 43% | **8.0%** |

**Why not worth it:**
- Maximum 8% speedup at 32K context
- High implementation complexity (new kernels, per-layer config, bit packing)
- Quality risk: INT4 has 16x less precision than INT8 (16 vs 255 levels)
- Model weights (14GB) still dominate bandwidth at all context lengths

INT8 already captures most of the memory bandwidth benefit with acceptable quality.

## Conclusion

The "cascade attention + CUDA graphs" framing was a red herring. Gemma 3's performance
characteristics are fundamentally determined by:

1. **Memory bandwidth** - 10 global layers reading full context
2. **Architectural choice** - 5:1 local:global ratio trades compute for memory

Further optimization should focus on reducing memory bandwidth for global attention
layers, not CUDA graph capture strategies.

## References

- [vLLM cascade attention](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)
- [Gemma 3 architecture](https://huggingface.co/docs/transformers/en/model_doc/gemma3)
- [vLLM issue #14881](https://github.com/vllm-project/vllm/issues/14881) - Cascade + sliding window
