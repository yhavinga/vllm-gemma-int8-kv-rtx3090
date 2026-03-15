# vLLM + Gemma 3 27B Benchmark Results

**Last Updated:** 2026-03-15
**System:** Dual RTX 3090 (48GB total VRAM) with NVLink (NV4)

---

## Quick Start (Best Configuration)

```bash
source /home/yeb/Developer/gemma/venv/bin/activate

# NCCL/P2P optimizations for NVLink
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

**Result: ~67 tokens/sec single request, ~244 tok/s batch (4 concurrent)**

---

## Context Length Scaling

All context sizes work with CUDA graphs and achieve similar performance:

| Context | Launch Script | GPU Mem Util | Single tok/s | Batch tok/s |
|---------|---------------|--------------|--------------|-------------|
| 8K      | `launch-optimized.sh` | 0.90 | 67 | 244 |
| 32K     | `launch-optimized-32k.sh` | 0.90 | 67 | 243 |
| 128K    | `launch-optimized-128k.sh` | 0.95 | 67 | 241 |

**Key insight:** W4A16 quantization enables full 128K context on dual RTX 3090 with no speed loss!

See `research-context-scaling.md` for detailed memory analysis.

---

## System Configuration

| Component | Value |
|-----------|-------|
| **GPUs** | 2× RTX 3090 (24GB each) |
| **NVLink** | NV4 (4-way bonded, ~112 GB/s bidirectional) |
| **NVIDIA Driver** | 535.288.01 |
| **CUDA** | 12.8 |
| **Python** | 3.10 |
| **PyTorch** | 2.10.0+cu128 |
| **vLLM** | 0.17.1 (pip install) |

## Model Configuration

| Setting | Value |
|---------|-------|
| **Model** | `RedHatAI/gemma-3-27b-it-quantized.w4a16` |
| **Quantization** | W4A16 (4-bit weights, 16-bit activations) |
| **Model Size** | ~13.5 GB (quantized) |
| **Memory per GPU** | ~8.2 GB |
| **Tensor Parallel** | 2 GPUs |
| **Max Context** | 8192 tokens |

---

## Performance Results

### Benchmark: Summarization Task (787 input → ~270 output tokens)

| Configuration | Speed | Time | Notes |
|---------------|-------|------|-------|
| PyTorch 2.10 + `--enforce-eager` | **11 tok/s** | 25.6s | Baseline (vLLM 0.17.0 default) |
| PyTorch 2.11 + torch.compile | **21 tok/s** | 12.7s | **+91% improvement** |

### Multiple Run Consistency (PyTorch 2.11)

| Run | Tokens | Time | Speed |
|-----|--------|------|-------|
| 1 | 271 | 12.7s | 21.3 tok/s |
| 2 | 268 | 12.7s | 21.0 tok/s |
| 3 | 281 | 13.3s | 21.1 tok/s |

**Average: ~21.1 tokens/sec**

---

## Optimization Attempts

### What Works ✅

| Optimization | Status | Impact |
|--------------|--------|--------|
| **Tensor Parallel = 2** | ✅ Required | Splits 27B model across GPUs |
| **W4A16 Quantization** | ✅ Working | Fits in 48GB VRAM |
| **torch.compile (PyTorch 2.11)** | ✅ Working | **+91% speedup** |
| **NVLink** | ✅ Auto-detected | ~3.5x faster GPU communication |
| **Prefix Caching** | ✅ Enabled by default | Faster repeated prompts |
| **Chunked Prefill** | ✅ Enabled by default | Better latency |

### What Doesn't Work ❌

| Optimization | Status | Reason |
|--------------|--------|--------|
| **CUDA Graphs** | ❌ OOM | Insufficient VRAM during graph capture |
| **FP8 KV Cache** | ❌ Not supported | Requires SM 8.9+ (Ada/Hopper GPUs) |
| **Speculative Decoding** | ❌ Not supported | Gemma 3 is multimodal (all 4B+ variants) |

### Why Speculative Decoding Fails

Gemma 3 model variants:
- **1B, 270M**: Text-only (`gemma3_text`)
- **4B, 12B, 27B**: Multimodal (`image-text-to-text`)

vLLM error:
```
NotImplementedError: Speculative Decoding with draft models does not support multimodal models yet
```

Since the target model (27B) is multimodal, speculative decoding cannot be used regardless of draft model choice.

---

## NVLink Status

```
        GPU0    GPU1
GPU0     X      NV4
GPU1    NV4      X
```

Each link provides 14.062 GB/s × 4 links = ~56 GB/s per direction (~112 GB/s bidirectional).

**Verification:**
```bash
nvidia-smi topo -m        # Shows NV4 connection
nvidia-smi nvlink --status  # Shows link status per GPU
```

NCCL automatically uses NVLink when available:
```
backend=nccl
vLLM is using nccl==2.27.5
```

---

## Installation Guide

### Simple Install (Recommended - March 2026)

vLLM 0.17.1 with PyTorch 2.10 works out of the box. No source build needed.

```bash
python -m venv venv
source venv/bin/activate
pip install vllm requests
```

### Verify

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# PyTorch: 2.10.0+cu128

python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
# vLLM: 0.17.1
```

### Historical Note: PyTorch 2.11 Source Build

Previous testing required PyTorch 2.11 RC and building vLLM from source.
This is no longer necessary with vLLM 0.17.1.

---

## Server Command Options

### Optimized (PyTorch 2.11 + torch.compile) - RECOMMENDED

```bash
vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    -cc.cudagraph_mode=none \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

### Fallback (PyTorch 2.10 + enforce-eager)

```bash
vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

### Flag Explanations

| Flag | Purpose |
|------|---------|
| `--trust-remote-code` | Required for Gemma 3 custom code |
| `--tensor-parallel-size 2` | Split model across both GPUs |
| `-cc.cudagraph_mode=none` | Disable CUDA graphs (OOM prevention) |
| `--enforce-eager` | Disable torch.compile (PyTorch 2.10 fallback) |
| `--max-model-len 8192` | Context length (can increase with headroom) |
| `--gpu-memory-utilization 0.90` | Use 90% of VRAM |

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "RedHatAI/gemma-3-27b-it-quantized.w4a16",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }'
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during startup | Reduce `--max-model-len` or `--gpu-memory-utilization` |
| OOM with CUDA graphs | Add `-cc.cudagraph_mode=none` |
| torch.compile crash (PyTorch 2.10) | Add `--enforce-eager` |
| ABI error after PyTorch upgrade | Rebuild vLLM from source |
| FP8 KV cache error | Not supported on RTX 3090 (SM 8.6) |
| Speculative decoding error | Not supported for multimodal models |
| Slow first request | Normal - model warmup/compilation |
| NVLink not detected | Check `nvidia-smi topo -m`, reseat bridge |

---

## Summary

| Metric | Value |
|--------|-------|
| **Best Single Request** | **~67 tokens/sec** |
| **Best Batch (4 concurrent)** | **~244 tokens/sec aggregate** |
| **Configuration** | vLLM 0.17.1 + CUDA graphs + NCCL tuning |
| **Optimal Parallelism** | Tensor Parallel = 2 |
| **Memory Usage** | ~8.2 GB per GPU |
| **VRAM Headroom** | ~15 GB per GPU |

### Key Findings

1. **CUDA Graphs work with `--disable-custom-all-reduce`** - custom_all_reduce crashes on RTX 3090
2. **FULL_DECODE_ONLY mode** uses less memory than default, enables CUDA graphs on 24GB GPUs
3. **Limited capture sizes [1,2,4,8,16,32]** reduces memory and startup time
4. **NCCL/P2P Environment Variables** still important even with custom_all_reduce disabled
5. **Total speedup: 6x** from baseline (11 tok/s → 67 tok/s single, 40 → 244 tok/s batch)
6. **NVLink NV4** provides ~112 GB/s bidirectional bandwidth
7. **FP8 KV Cache** requires Ada/Hopper GPUs (SM 8.9+) → not available on RTX 3090
8. **Speculative Decoding** not supported for multimodal models → all Gemma 3 4B+ are multimodal

---

## Files

- `/home/yeb/Developer/gemma/venv/` - Python virtual environment with vLLM 0.17.1 + PyTorch 2.10
- `/home/yeb/Developer/gemma/launch-optimized.sh` - Launch script with all optimizations
- `/home/yeb/Developer/gemma/benchmark.py` - Benchmark script for measuring performance
- `~/.cache/huggingface/` - Downloaded model weights

---

## Experiment Results (2026-03-15)

### Test Setup
- vLLM 0.17.1 from pip (no source build needed)
- PyTorch 2.10.0+cu128
- Local venv: `/home/yeb/Developer/gemma/venv`

### Baseline: enforce-eager (no torch.compile)

```bash
vllm serve ... --enforce-eager
```

| Metric | Result |
|--------|--------|
| Single request (medium prompt) | **11 tok/s** |
| Batch (4 concurrent) | **40 tok/s aggregate** |

### Optimized: torch.compile + NCCL Tuning

```bash
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

vllm serve ... -cc.cudagraph_mode=none  # No --enforce-eager
```

| Metric | Result | Improvement |
|--------|--------|-------------|
| Single request (medium prompt) | **20 tok/s** | **+82%** |
| Batch (4 concurrent) | **71 tok/s aggregate** | **+77%** |

### Detailed Benchmark Results

**Single Request Performance:**
| Prompt Type | Speed |
|-------------|-------|
| Short | 14-19 tok/s (varies due to warmup) |
| Medium (256 tokens) | 20.0-20.4 tok/s |
| Long | 20.4-20.5 tok/s |

**Batch Throughput (4 concurrent):**
| Run | Aggregate |
|-----|-----------|
| 1 | 61.9 tok/s |
| 2 | 79.9 tok/s |

### What Works ✅

| Optimization | Impact |
|--------------|--------|
| `NCCL_P2P_LEVEL=NVL` | Forces NVLink transport |
| `CUDA_FORCE_P2P_ACCESS=1` | Enables P2P on RTX 3090 |
| `VLLM_SKIP_P2P_CHECK=1` | Bypasses vLLM P2P validation |
| `NCCL_BUFF_SIZE=16777216` | 16MB buffers for NVLink bandwidth |
| torch.compile (no --enforce-eager) | **+82% single request speed** |
| `-cc.cudagraph_mode=none` | Prevents OOM during graph capture |

### What Doesn't Work ❌

| Optimization | Reason |
|--------------|--------|
| CUDA Graphs (default mode) | OOM during graph capture |
| torch.compile without cudagraph_mode=none | Crashes/OOM |

### Helper Scripts
- `launch-optimized.sh` - Launch with all optimizations
- `benchmark.py` - Measure tokens/sec and compare configs
- `exllamav2-setup.md` - Guide for ExLlamaV2 alternative (not yet tested)
