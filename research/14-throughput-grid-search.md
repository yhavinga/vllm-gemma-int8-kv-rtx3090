# Gemma 3 1B/4B Throughput Grid Search

**Date:** 2026-03-17
**Hardware:** 2x RTX 3090 (48GB total VRAM, NVLink NV4)
**vLLM:** 0.17.1
**Quantization:** W8A8 (1B), W4A16 (4B)

---

## Executive Summary

Maximum batched throughput benchmarks for synthetic data generation workloads.
Inspired by HuggingFace's Synthetic Data Playbook approach.

### Key Findings

| Model | Best Config | Max Throughput | Context Range |
|-------|-------------|----------------|---------------|
| **1B W8A8** | **DP=2** (data parallel) | **12,513 tok/s** | 4K-32K |
| **4B W4A16** | **DP=2** (data parallel) | **7,317 tok/s** | 4K-32K |
| **4B W4A16** | TP=2 (for 128K) | **4,741 tok/s** | 64K-128K |

**Critical Insight: Data Parallel beats Tensor Parallel for both models!**
- 1B: DP=2 is **51% faster** than TP=1, **71% faster** than TP=2
- 4B: DP=2 is **79% faster** than TP=1, **30% faster** than TP=2

Use TP=2 only when you need long context (64K+) for 4B model.

---

## Complete Results

### Gemma 3 1B W8A8

#### DP=2 (Data Parallel, RECOMMENDED)

| Context | Throughput | Optimal Batch | vs TP=1 |
|---------|------------|---------------|---------|
| **4K** | **12,234 tok/s** | 256 | +52% |
| **8K** | **12,513 tok/s** | 256 | +54% |
| **16K** | **12,161 tok/s** | 256 | +51% |
| **32K** | **11,975 tok/s** | 256 | +51% |

### Gemma 3 4B W4A16

#### DP=2 (Data Parallel, RECOMMENDED for <64K)

| Context | Throughput | Optimal Batch | vs TP=2 |
|---------|------------|---------------|---------|
| **4K** | **7,298 tok/s** | 256 | +30% |
| **8K** | **7,317 tok/s** | 256 | +31% |
| **16K** | **7,296 tok/s** | 256 | +31% |
| **32K** | **7,182 tok/s** | 256 | +34% |

---

## Alternative Configurations (for reference)

### Gemma 3 1B W8A8

#### TP=1 (Single GPU, Recommended)

| Context | Throughput | Optimal Batch | Notes |
|---------|------------|---------------|-------|
| **4K** | **8,069 tok/s** | 256 | |
| **8K** | **8,109 tok/s** | 256 | Peak performance |
| **16K** | **8,075 tok/s** | 256 | |
| **32K** | **7,949 tok/s** | 256 | Max context for 1B |

#### TP=2 (Dual GPU)

| Context | Throughput | Optimal Batch | vs TP=1 |
|---------|------------|---------------|---------|
| 4K | 7,148 tok/s | 64 | -11% |
| 8K | 7,179 tok/s | 64 | -11% |
| 16K | 7,091 tok/s | 64 | -12% |
| 32K | 6,848 tok/s | 64 | -14% |

**Verdict:** Use TP=1 for 1B. NVLink communication overhead reduces throughput.

---

### Gemma 3 4B W4A16

#### TP=2 (Dual GPU, Recommended)

| Context | Throughput | Optimal Batch | vs 4K |
|---------|------------|---------------|-------|
| **4K** | **5,612 tok/s** | 256 | baseline |
| **8K** | **5,600 tok/s** | 256 | -0.2% |
| **16K** | **5,572 tok/s** | 256 | -0.7% |
| **32K** | **5,371 tok/s** | 256 | -4.3% |
| **64K** | **5,216 tok/s** | 256 | -7.1% |
| **128K** | **4,741 tok/s** | 256 | -15.5% |

#### TP=1 (Single GPU)

| Context | Throughput | Optimal Batch | vs TP=2 |
|---------|------------|---------------|---------|
| 4K | 4,066 tok/s | 256 | -28% |
| 8K | 4,067 tok/s | 256 | -27% |
| 16K | 4,037 tok/s | 256 | -28% |
| 32K | 4,014 tok/s | 256 | -25% |

**Verdict:** Use TP=2 for 4B. Dual GPU provides 38% throughput improvement.

---

## Batch Size Scaling

### 1B W8A8 @ 4K Context

| Batch | TP=1 (tok/s) | TP=2 (tok/s) |
|-------|--------------|--------------|
| 1 | 227 | 253 |
| 2 | 406 | 437 |
| 4 | 771 | 821 |
| 8 | 1,645 | 950 |
| 16 | 3,083 | 3,123 |
| 32 | 4,561 | 5,287 |
| 64 | 6,017 | **7,054** |
| 128 | 6,462 | 5,723 |
| **256** | **8,175** | - |

### 4B W4A16 @ 4K Context

| Batch | TP=1 (tok/s) | TP=2 (tok/s) |
|-------|--------------|--------------|
| 1 | 172 | 206 |
| 2 | 225 | 260 |
| 4 | 565 | 659 |
| 8 | 1,070 | 702 |
| 16 | 1,783 | 2,135 |
| 32 | 2,498 | 3,193 |
| 64 | 3,397 | 3,778 |
| 128 | 3,820 | 5,035 |
| **256** | **4,064** | **5,602** |

---

## Optimal Configuration

### For Synthetic Data Generation (1B)

```bash
# Single GPU - 8,000+ tok/s
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

vllm serve RedHatAI/gemma-3-1b-it-quantized.w8a8 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256]}' \
    --port 8000
```

### For Higher Quality Generation (4B)

```bash
# Dual GPU with NVLink - 5,600 tok/s
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216

vllm serve RedHatAI/gemma-3-4b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256]}' \
    --port 8000
```

---

## Context Length Impact

### Throughput Degradation

| Context | 1B (TP=1) | 4B (TP=2) |
|---------|-----------|-----------|
| 4K | 100% | 100% |
| 8K | 100% | 100% |
| 16K | 100% | 99% |
| 32K | 98% | 96% |
| 64K | - | 93% |
| 128K | - | 84% |

**Key Insight:** Context length has minimal impact on throughput until 64K+.
Even at 128K, 4B maintains 84% of short-context throughput.

---

## Model Comparison

### Tokens per Second per Billion Parameters

| Model | Throughput | Normalized (tok/s/B) |
|-------|------------|----------------------|
| 1B | 8,109 | **8,109** |
| 4B | 5,612 | **1,403** |

The 1B model is 5.8x more efficient per parameter for throughput.

### Quality vs Speed Trade-off

| Use Case | Model | Throughput | Reason |
|----------|-------|------------|--------|
| Bulk generation, simple tasks | 1B | 8,109 tok/s | Maximum speed |
| Quality-sensitive generation | 4B | 5,612 tok/s | Better reasoning |
| Long-context synthesis | 4B @ 128K | 4,741 tok/s | Full context support |

---

## Comparison with 27B Model

From [06-model-size-comparison.md](06-model-size-comparison.md):

| Model | Quant | TP | Single tok/s | Batch(4) tok/s | Max Batch tok/s |
|-------|-------|----|--------------|-----------------| --------------- |
| 1B | W8A8 | 1 | 263 | 887 | **8,109** |
| 4B | W4A16 | 1 | 176 | 581 | 4,066 |
| 4B | W4A16 | 2 | 212 | 675 | **5,612** |
| 27B | W4A16 | 2 | 67 | 244 | ~400* |

*Estimated based on batch scaling patterns.

---

## Methodology

### Test Configuration

- **Output tokens per request:** 128
- **Input tokens:** ~512 (standard prompt)
- **Batch sizes tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256
- **Runs per batch size:** 2 (averaged)
- **CUDA Graphs:** FULL_DECODE_ONLY mode
- **Memory utilization:** 0.80-0.85

### Benchmark Script

See `scripts/throughput_grid_search.py` for the full implementation.

```bash
# Run grid search
python scripts/throughput_grid_search.py --models 1b-w8a8 4b-w4a16 --tp 2

# TP=1 comparison
python scripts/throughput_grid_search.py --models 1b-w8a8 4b-w4a16 --tp 1
```

---

## Recommendations for Synthetic Data Generation

### Small-Scale Generation (< 1M tokens/day)
- Use 4B W4A16 with TP=2
- Better quality, 5,600 tok/s = 483M tokens/day

### Large-Scale Generation (> 1M tokens/day)
- Use 1B W8A8 with TP=1
- Maximum speed, 8,109 tok/s = 700M tokens/day

### Mixed Workload
- Run both models on separate GPUs
- 1B on GPU 0, 4B on GPU 1
- Route requests based on quality requirements

---

## References

- [HuggingFace Synthetic Data Playbook](https://huggingfacefw-finephrase.hf.space/the-synthetic-data-playbook-generating-trillions-of-the-finest-tokens.pdf)
- [06-model-size-comparison.md](06-model-size-comparison.md) - Single/batch comparison
- [09-1b-tensor-parallel-comparison.md](09-1b-tensor-parallel-comparison.md) - 1B TP analysis
- [08-4b-tensor-parallel-comparison.md](08-4b-tensor-parallel-comparison.md) - 4B TP analysis
