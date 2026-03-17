# Gemma 3 Throughput Grid Search Results

**Date:** 2026-03-17
**Hardware:** 2x RTX 3090 (48GB VRAM, NVLink NV4)
**vLLM:** 0.17.1
**Metric:** Max batched throughput (tok/s)
**Output tokens:** 128 per request

---

## Summary: Best Configuration per Model

| Model | Best Config | Max Throughput | Optimal Batch |
|-------|-------------|----------------|---------------|
| **1B W8A8** | DP=2 | **12,513 tok/s** | 256 |
| **4B W4A16** | DP=2 | **7,317 tok/s** | 256 |

---

## Gemma 3 1B W8A8

### Throughput (tok/s)

| Context | TP=1 | TP=2 | DP=2 | Best |
|---------|------|------|------|------|
| **4K** | 8,069 | 7,148 | **12,234** | DP=2 |
| **8K** | 8,109 | 7,179 | **12,513** | DP=2 |
| **16K** | 8,075 | 7,091 | **12,161** | DP=2 |
| **32K** | 7,949 | 6,848 | **11,975** | DP=2 |
| 64K | - | - | - | N/A (max 32K) |
| 128K | - | - | - | N/A (max 32K) |

### Optimal Batch Size

| Context | TP=1 | TP=2 | DP=2 |
|---------|------|------|------|
| 4K | 256 | 64 | 256 |
| 8K | 256 | 64 | 256 |
| 16K | 256 | 64 | 256 |
| 32K | 256 | 64 | 256 |

### Speedup vs TP=1 Baseline

| Context | TP=2 | DP=2 |
|---------|------|------|
| 4K | -11% | **+52%** |
| 8K | -11% | **+54%** |
| 16K | -12% | **+51%** |
| 32K | -14% | **+51%** |

**Key Finding:** 1B model is too small for tensor parallelism - the NVLink communication overhead hurts performance. Data parallel achieves near-linear 2x scaling.

---

## Gemma 3 4B W4A16

### Throughput (tok/s)

| Context | TP=1 | TP=2 | DP=2 | Best |
|---------|------|------|------|------|
| **4K** | 4,066 | 5,612 | **7,298** | DP=2 |
| **8K** | 4,067 | 5,600 | **7,317** | DP=2 |
| **16K** | 4,037 | 5,572 | **7,296** | DP=2 |
| **32K** | 4,014 | 5,371 | **7,182** | DP=2 |
| **64K** | - | **5,216** | - | TP=2 |
| **128K** | - | **4,741** | - | TP=2 |

### Optimal Batch Size

| Context | TP=1 | TP=2 | DP=2 |
|---------|------|------|------|
| 4K | 256 | 256 | 256 |
| 8K | 256 | 256 | 256 |
| 16K | 256 | 256 | 256 |
| 32K | 256 | 256 | 256 |
| 64K | - | 256 | - |
| 128K | - | 256 | - |

### Speedup vs TP=1 Baseline

| Context | TP=2 | DP=2 |
|---------|------|------|
| 4K | +38% | **+79%** |
| 8K | +38% | **+80%** |
| 16K | +38% | **+81%** |
| 32K | +34% | **+79%** |

**Key Finding:** DP=2 beats TP=2 by 30% for contexts ≤32K. Use TP=2 only for 64K+ context.

---

## Configuration Comparison

### 1B W8A8 - All Configurations

```
Throughput (tok/s) by Context Length

Config   |    4K    |    8K    |   16K    |   32K    |   64K    |  128K
---------|----------|----------|----------|----------|----------|----------
TP=1     |   8,069  |   8,109  |   8,075  |   7,949  |    -     |    -
TP=2     |   7,148  |   7,179  |   7,091  |   6,848  |    -     |    -
DP=2     |  12,234  |  12,513  |  12,161  |  11,975  |    -     |    -
```

### 4B W4A16 - All Configurations

```
Throughput (tok/s) by Context Length

Config   |    4K    |    8K    |   16K    |   32K    |   64K    |  128K
---------|----------|----------|----------|----------|----------|----------
TP=1     |   4,066  |   4,067  |   4,037  |   4,014  |    -     |    -
TP=2     |   5,612  |   5,600  |   5,572  |   5,371  |   5,216  |   4,741
DP=2     |   7,298  |   7,317  |   7,296  |   7,182  |    -     |    -
```

---

## Batch Size Scaling (4K Context)

### 1B W8A8

| Batch | TP=1 | TP=2 | DP=2 |
|-------|------|------|------|
| 1 | 227 | 253 | 183 |
| 2 | 406 | 437 | 447 |
| 4 | 771 | 821 | 769 |
| 8 | 1,645 | 950 | 1,503 |
| 16 | 3,083 | 3,123 | 3,023 |
| 32 | 4,561 | 5,287 | 5,635 |
| 64 | 6,017 | **7,054** | 8,167 |
| 128 | 6,462 | 5,723 | 10,606 |
| 256 | **8,175** | - | **11,955** |

### 4B W4A16

| Batch | TP=1 | TP=2 | DP=2 |
|-------|------|------|------|
| 1 | 172 | 206 | 115 |
| 2 | 225 | 260 | 320 |
| 4 | 565 | 659 | 434 |
| 8 | 1,070 | 702 | 1,108 |
| 16 | 1,783 | 2,135 | 2,068 |
| 32 | 2,498 | 3,193 | 3,417 |
| 64 | 3,397 | 3,778 | 4,720 |
| 128 | 3,820 | 5,035 | 6,396 |
| 256 | **4,064** | **5,602** | **7,282** |

---

## Recommendations

### For Maximum Throughput (Synthetic Data Generation)

```bash
# 1B model - 12,500 tok/s = 1.08B tokens/day
vllm serve RedHatAI/gemma-3-1b-it-quantized.w8a8 \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --max-model-len 32768

# 4B model - 7,300 tok/s = 630M tokens/day
vllm serve RedHatAI/gemma-3-4b-it-quantized.w4a16 \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --max-model-len 32768
```

### For Long Context (64K-128K)

```bash
# 4B model with TP=2 for long context
vllm serve RedHatAI/gemma-3-4b-it-quantized.w4a16 \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 131072
```

---

## Context Length Limits

| Model | Max Context | Reason |
|-------|-------------|--------|
| 1B W8A8 | 32K | max_position_embeddings=32768 |
| 4B W4A16 | 128K | Sliding window attention |

---

## Raw Data Files

- `throughput-grid-20260317-174312.json` - TP=2 results
- `throughput-grid-20260317-175603.json` - TP=1 results (4B)
- `throughput-grid-20260317-190716.json` - DP=2 results
