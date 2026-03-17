# INT8 KV Cache Research for RTX 3090

## Summary

INT8 KV cache quantization is **working** for Gemma 3 27B on RTX 3090.

**Key Results:**
- Phase 1 POC: Quality validated (cosine sim > 0.999)
- Phase 2 vLLM integration: **COMPLETE** - coherent generation confirmed
- 2x memory savings for KV cache
- No noticeable quality degradation in practical use

## Phase 2 vLLM Integration - COMPLETE

### What Was Done

Modified vLLM 0.17.1 to support INT8 KV cache:

1. **`vllm/config/cache.py`**: Added `"int8"` to CacheDType literal
2. **`vllm/v1/attention/backend.py`**: Extended `is_quantized_kv_cache()` to include int8
3. **`vllm/v1/attention/backends/triton_attn.py`**: Added int8 to `supported_kv_cache_dtypes`
4. **`vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`**: Added INT8 quantization to write path
5. **`vllm/v1/attention/ops/triton_unified_attention.py`**: Added INT8 dequantization to read path
6. **`vllm/model_executor/layers/attention/attention.py`**:
   - Auto-detect INT8 dtype and use range=127 (instead of FP8's 200/100)
   - Fixed warmup issue: use default scale=0.157 when warmup inputs are zeros

### How to Use

```bash
vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 \
    --kv-cache-dtype int8 \
    --calculate-kv-scales \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
```

### Verified Output Quality

| Prompt | INT8 Response |
|--------|---------------|
| "What is 2+2?" | "4" |
| "Capital of France?" | "Paris." |
| "Python string reverse" | Correct code with docstring |

## Phase 1 POC Results

### Quality Metrics (Attention Output)

| Sequence Length | Cosine Similarity | Max Error | MSE |
|-----------------|-------------------|-----------|-----|
| 256 tokens | 0.999942 | 0.001465 | 1.22e-7 |
| 1024 tokens | 0.999933 | 0.000793 | 3.53e-8 |
| 4096 tokens | 0.999931 | 0.000366 | 9.07e-9 |

**All metrics exceed targets** (cosine sim > 0.999, max error < 0.01).

### Memory Savings

For 32K context (Gemma 3 27B: 46 layers, 16 KV heads, 256 head size):

| Cache Type | Size | Savings |
|------------|------|---------|
| BF16 | 23.00 GB | - |
| INT8 | 11.50 GB | **50%** |
| Scales overhead | 5.75 KB | negligible |

**Implication:** Can run 64K context in same VRAM budget as 32K BF16.

## Technical Approach

### Quantization Strategy

Per-tensor symmetric INT8:
- `scale = absmax / 127`
- Default scale = 20.0 / 127 ≈ 0.157 (when warmup inputs are zeros)
- Quantize: `int8_val = round(bf16_val / scale).clamp(-128, 127)`
- Dequantize: `bf16_val = int8_val * scale`

### Why INT8 over FP8 Emulation

| Factor | INT8 | Software FP8 |
|--------|------|--------------|
| Hardware | Native INT8 tensor cores on Ampere | No FP8, requires emulation |
| Overhead | Low | High (extra ops) |
| Complexity | Moderate | Complex |
| Quality | Proven sufficient | Potentially better range |

INT8 wins on practicality - Ampere has hardware INT8 support.

## Triton Kernel Implementation

### Write Path (triton_reshape_and_cache_flash.py)

```python
elif INT8_KV_CACHE:
    # INT8 quantization: scale, round, clamp to [-128, 127]
    key_scaled = key_load.to(tl.float32) / tl.load(k_scale)
    key_rounded = tl.extra.cuda.libdevice.round(key_scaled)
    key_tile = tl.maximum(tl.minimum(key_rounded, 127.0), -128.0).to(tl.int8)
```

### Read Path (triton_unified_attention.py)

```python
elif INT8_KV_CACHE:
    # INT8 dequantization: multiply by scale
    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
```

## Files Modified

| File | Change |
|------|--------|
| `vllm/config/cache.py` | Add "int8" to CacheDType |
| `vllm/v1/attention/backend.py` | Extend is_quantized_kv_cache() |
| `vllm/v1/attention/backends/triton_attn.py` | Add int8 to supported dtypes |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | INT8 quantization in write path |
| `vllm/v1/attention/ops/triton_unified_attention.py` | INT8 dequantization in read path |
| `vllm/model_executor/layers/attention/attention.py` | INT8 range detection + warmup fix |

## POC Script

See `scripts/int8_kv_cache_poc.py` for the standalone POC:
- `_per_head_quant_int8_kernel`: Triton kernel for quantization
- `_per_head_dequant_int8_kernel`: Triton kernel for dequantization
- Quality tests comparing INT8 vs BF16 attention output

## Performance Results

### Complete Benchmark Matrix

Tested with FULL_DECODE_ONLY CUDA graphs, INT8 KV cache, W4A16 model weights.

| Prompt Tokens | INT8 tok/s | Notes |
|---------------|------------|-------|
| ~350 | **60.6** | Short prompt, CUDA graphs active |
| ~1,300 | 38.0 | Below cascade threshold |
| ~2,300 | 40.1 | Below cascade threshold |
| ~4,300 | 28.1 | **Cascade attention kicks in** |
| ~8,400 | 18.2 | Full cascade mode |
| ~32,000 | 9.6 | Very long context (64K max-model-len) |

### The Cascade Attention Cliff

Gemma 3 uses hybrid attention: sliding window (4K) for local context + full
attention for global context. When sequence length exceeds ~4K tokens:

1. vLLM switches to "cascade attention" mode
2. CUDA graphs are disabled ("No piecewise cudagraph for cascade attention")
3. Performance drops ~50% and degrades further with longer context

**This is the fundamental bottleneck** - not INT8 vs BF16, not CUDA graph mode.
The ~4K token threshold is architectural to Gemma 3.

### INT8 vs BF16 (Long Context)

| Context | BF16 | INT8 | Change |
|---------|------|------|--------|
| ~7K tokens | 24 tok/s | **45 tok/s** | **+87%** |
| ~12K tokens | 24 tok/s | **40 tok/s** | **+67%** |

INT8 still wins big for long context because it halves KV cache memory bandwidth.

### Maximum Context Limits

| max-model-len | CUDA Graph Mode | Works? | Notes |
|---------------|-----------------|--------|-------|
| 32K | FULL_DECODE_ONLY | Yes | Recommended for most use |
| 64K | FULL_DECODE_ONLY | Yes | Tested, ~10 tok/s at 32K prompts |
| 128K | FULL_DECODE_ONLY | **OOM** | Too much KV cache for graphs |
| 128K | enforce-eager | Yes | ~8 tok/s, no CUDA graphs |

**Recommendation:** Use 64K max-model-len with FULL_DECODE_ONLY for best balance
of context length and performance.

### Optimal INT8 Configuration

For balanced performance (up to 64K context):
```bash
vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 --disable-custom-all-reduce \
    --kv-cache-dtype int8 --calculate-kv-scales \
    --max-model-len 65536 --gpu-memory-utilization 0.90 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", \
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

For maximum context (128K, slower):
```bash
vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 --disable-custom-all-reduce \
    --kv-cache-dtype int8 --calculate-kv-scales \
    --max-model-len 131072 --gpu-memory-utilization 0.90 \
    --enforce-eager
```

## Current Limitations

1. **Fixed default scale**: Uses default scale of 0.157 (absmax=20) during warmup
   - Works well for typical transformers
   - May need calibration for models with different K/V ranges

2. **~12% throughput loss vs BF16**: Due to quantization/dequantization overhead

## Next Steps

1. ~~Phase 1: Standalone POC~~ ✅ **DONE**
2. ~~Phase 2: vLLM integration~~ ✅ **DONE**
3. ~~Phase 3: Performance optimization~~ ✅ **DONE** (58-59 tok/s achieved)
4. Future: Fuse dequantization with attention kernel for potential speedup

## Key Learnings

1. **Warmup inputs are zeros**: The `--calculate-kv-scales` feature computes scales during warmup when K/V are zeros. Fixed by defaulting to scale=0.157 when absmax < 0.01.

2. **INT8 range is 127, not 200**: vLLM's FP8 defaults (K_SCALE_CONSTANT=200, V_SCALE_CONSTANT=100) don't work for INT8. Fixed by auto-detecting kv_cache_dtype and using 127.

3. **Triton kernel changes need constexpr**: Adding INT8_KV_CACHE requires updating ALL kernel call sites with the new constexpr parameter.

## References

- vLLM FP8 KV cache implementation (same pattern, different dtype)
- vLLM INT8 weight quantization utils (`int8_utils.py`)
- Triton 3.6.0 documentation for kernel authoring
