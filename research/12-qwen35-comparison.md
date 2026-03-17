# Qwen 3.5 27B vs Gemma 3 27B Comparison

## Summary

Tested Qwen 3.5 27B as potential alternative to Gemma 3 27B for long-context inference.
**Result: Gemma 3 + INT8 KV cache is faster at all context lengths.**

## Why We Tested Qwen 3.5

Gemma 3 has a ~4K token performance cliff due to cascade attention (hybrid sliding
window + full attention). Qwen 3.5 uses DeltaNet (linear attention) which theoretically
avoids this cliff.

| Architecture | Gemma 3 27B | Qwen 3.5 27B |
|--------------|-------------|--------------|
| Attention | Sliding window (4K) + Full | DeltaNet (linear) + Full |
| Layer ratio | 52 sliding : 10 full | 3 DeltaNet : 1 full |
| Max context | 128K | 262K (1M with YaRN) |

## Performance Results

Tested on 2x RTX 3090 with tensor parallelism.

| Prompt Tokens | Gemma 3 (INT8 KV) | Qwen 3.5 (BF16 KV) |
|---------------|-------------------|---------------------|
| ~700 | 40 tok/s | 38 tok/s |
| ~1,300 | 38 tok/s | 32 tok/s |
| ~2,300 | 40 tok/s | 25 tok/s |
| ~4,300 | 28 tok/s | 17 tok/s |
| ~8,300 | 18 tok/s | 10 tok/s |
| ~16,400 | ~12 tok/s | 6 tok/s |

**Gemma 3 with INT8 KV cache is 1.5-2x faster than Qwen 3.5 at all context lengths.**

## Why Qwen 3.5 Is Slower

1. **DeltaNet overhead**: Linear attention still has O(n·d²) complexity per layer.
   The fixed-size state matrix updates add compute that isn't offset by memory savings.

2. **More total compute**: 64 transformer blocks vs Gemma's 62, plus DeltaNet's
   gating mechanisms add FLOPS.

3. **No INT8 KV cache**: We couldn't apply our INT8 optimization to Qwen 3.5's
   hybrid architecture (DeltaNet state + attention KV cache).

4. **Reasoning model overhead**: Qwen 3.5 outputs thinking process by default,
   adding latency and token overhead.

## Dutch Language Quality

Both models handle Dutch well:

| Test | Gemma 3 | Qwen 3.5 |
|------|---------|----------|
| Translation | Accurate | Accurate (verbose) |
| Comprehension | Good | Good |
| Knowledge | Correct | Correct |

Qwen 3.5 always outputs reasoning steps, making responses longer.

## Test Configuration

```bash
# Qwen 3.5 27B
vllm serve "Qwen/Qwen3.5-27B-GPTQ-Int4" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

# Uses GPTQ-Marlin kernels, FlashAttention v2
# Model size: 13.85 GB VRAM
```

## Conclusion

Qwen 3.5's DeltaNet architecture doesn't provide performance benefits on RTX 3090.
The theoretical advantage of linear attention is offset by:
- Additional compute from gating mechanisms
- Inability to apply INT8 KV cache optimization
- Reasoning model overhead

**Recommendation: Stay with Gemma 3 27B + INT8 KV cache for best performance.**
