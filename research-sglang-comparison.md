# SGLang vs vLLM Comparison for Gemma 3 27B

**Date:** 2026-03-15
**Hardware:** 2x RTX 3090 (NVLink), 48GB VRAM total
**Test:** Long context inference speed comparison

---

## Executive Summary

| Engine | Model | Context | Decode Speed | Winner |
|--------|-------|---------|--------------|--------|
| vLLM 0.17.1 | RedHatAI W4A16 | ~28K tokens | **24 tok/s** | ✅ |
| SGLang 0.5.9 | ISTA-DASLab GPTQ | ~27K tokens | 12.7 tok/s | |
| SGLang 0.5.9 | ISTA-DASLab GPTQ | ~15K tokens | 16 tok/s | |

**Conclusion:** vLLM is ~2x faster than SGLang for long context Gemma 3 on RTX 3090 + TP=2.

---

## Test Configuration

### vLLM Setup
```bash
vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

### SGLang Setup
```bash
python -m sglang.launch_server \
  --model-path "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g" \
  --tp 2 \
  --context-length 32768 \
  --mem-fraction-static 0.80 \
  --cuda-graph-max-bs 8
```

---

## SGLang Issues Encountered

### 1. RedHatAI W4A16 Model Incompatibility

The RedHatAI W4A16 model failed to load in SGLang:

```
AssertionError: assert input_size_per_partition % group_size == 0
```

**Cause:** Vision tower quantization incompatible with SGLang's TP=2 weight sharding.

**Workaround:** Used ISTA-DASLab GPTQ model which keeps vision tower in original precision.

### 2. OOM During CUDA Graph Capture

Initial settings caused OOM:
```
torch.AcceleratorError: CUDA error: out of memory
```

**Fix:** Reduced `--mem-fraction-static` from 0.88 to 0.80 and `--cuda-graph-max-bs` from 24 to 8.

### 3. Hybrid SWA Memory Disabled

SGLang does not yet support hybrid sliding window memory for Gemma 3:
```
WARNING: Disable hybrid SWA memory for Gemma3ForConditionalGeneration as it is not yet supported.
```

This may contribute to slower performance compared to vLLM.

### 4. Dependency Conflicts with vLLM

Installing SGLang alongside vLLM caused version conflicts:
- `flashinfer-python`: vLLM wants 0.6.4, SGLang installed 0.6.3
- `torch`: vLLM wants 2.10.0, SGLang installed 2.9.1
- `transformers`, `xgrammar`, `outlines_core`: version mismatches

**Recommendation:** Use separate virtual environments for vLLM and SGLang.

---

## Why SGLang is Slower on This Setup

### 1. RTX 3090 Consumer GPU Optimization

SGLang benchmarks showing 29% faster than vLLM are typically on H100/A100 enterprise GPUs. Consumer GPU tensor parallel may not be as optimized.

### 2. Different Quantization Backends

| Engine | Quantization | Kernel |
|--------|--------------|--------|
| vLLM | W4A16 | Marlin (highly optimized) |
| SGLang | GPTQ 4-bit | GPTQModel/Marlin |

Marlin kernels in vLLM may be more optimized for RTX 3090.

### 3. Hybrid SWA Memory Not Supported

SGLang explicitly disables hybrid sliding window memory for Gemma 3, which could impact memory bandwidth efficiency.

### 4. FlashInfer vs FLASH_ATTN

Different attention backends may have different performance characteristics on Ampere GPUs.

---

## Memory Usage Comparison

| Engine | Model Memory | KV Cache | Available |
|--------|--------------|----------|-----------|
| vLLM | ~10 GB/GPU | ~12 GB/GPU | ~2 GB |
| SGLang | ~9.5 GB/GPU | ~10.9 GB/GPU | ~2.5 GB |

SGLang uses slightly less memory but allocates fewer KV cache tokens (46K vs vLLM's configuration).

---

## Recommendations

### For RTX 3090 + Gemma 3 27B + Long Context

1. **Use vLLM** - 2x faster decode speed (24 vs 12.7 tok/s)
2. **Use RedHatAI W4A16 model** - Better compatibility with vLLM
3. **Accept 24 tok/s** - Near-optimal for hardware

### When to Consider SGLang

- H100/A100 enterprise GPUs
- Workloads with shared prefixes (RadixAttention benefit)
- Multi-turn conversations with prefix caching
- Non-Gemma 3 models with simpler attention patterns

---

## References

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang vs vLLM Benchmarks](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [ISTA-DASLab GPTQ Model](https://huggingface.co/ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g)
- [SGLang Gemma 3 Support PR #4424](https://github.com/sgl-project/sglang/pull/4424)
