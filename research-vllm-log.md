# vLLM CUDA Graph Research Log

**Date:** 2026-03-15
**Goal:** Enable CUDA graphs on Gemma 3 27B with dual RTX 3090 (2×24GB) without OOM

## Baseline (No CUDA Graphs)

**Configuration:** torch.compile + NCCL tuning, `-cc.cudagraph_mode=none`

| Metric | Result |
|--------|--------|
| Single request | ~20 tok/s |
| Batch (4 concurrent) | ~71 tok/s |

---

## Test 1: FULL_DECODE_ONLY Mode

**Configuration:**
```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

**Result:** FAILED

**Error:** CUDA error during custom_all_reduce:
```
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'
```

Graph capture progressed to ~94% (33/35 graphs) then crashed.

**Root cause:** custom_all_reduce kernel incompatible with CUDA graph capture on RTX 3090.

---

## Test 2: FULL_DECODE_ONLY + Limited Sizes + Disable Custom All Reduce

**Configuration:**
```bash
--disable-custom-all-reduce \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
```

**Result:** SUCCESS!

| Metric | Result | vs Baseline |
|--------|--------|-------------|
| Single request (medium) | **67.3 tok/s** | **+237%** |
| Single request (long) | **65.4 tok/s** | **+218%** |
| Batch (4 concurrent) | **244 tok/s** | **+244%** |

**Startup time:** ~120 seconds (graph capture takes time)

---

## Test 3 & 4: Not Needed

Test 2 worked so well that further tests were unnecessary.

---

## Final Summary

| Config | Status | Single tok/s | Batch tok/s | Notes |
|--------|--------|--------------|-------------|-------|
| cudagraph_mode=none (baseline) | OK | 20 | 71 | No CUDA graphs |
| FULL_DECODE_ONLY | FAIL | - | - | custom_all_reduce crash |
| **FULL_DECODE_ONLY + disable-custom-all-reduce** | **OK** | **67** | **244** | **BEST CONFIG** |

---

## Winning Configuration

```bash
source /home/yeb/Developer/gemma/venv/bin/activate

# Environment variables
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Server command
vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

## Key Findings

1. **`--disable-custom-all-reduce` is required** for CUDA graphs to work on RTX 3090 with tensor parallel
2. **FULL_DECODE_ONLY mode** uses less memory than default FULL_AND_PIECEWISE
3. **Limited capture sizes [1,2,4,8,16,32]** further reduces memory and startup time
4. **3.4x speedup** achieved over no-cudagraph baseline
5. **NVLink + NCCL tuning** still important even with custom_all_reduce disabled

## Performance Comparison

| Stage | Single Request | Batch (4) |
|-------|----------------|-----------|
| Baseline (enforce-eager) | 11 tok/s | 40 tok/s |
| torch.compile, no cudagraph | 20 tok/s | 71 tok/s |
| **CUDA graphs (final)** | **67 tok/s** | **244 tok/s** |
| **Total improvement** | **+509%** | **+510%** |
