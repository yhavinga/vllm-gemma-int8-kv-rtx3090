# Gemma 3 27B on Dual RTX 3090

Optimized vLLM configuration for running Gemma 3 27B IT on consumer hardware.

## Performance

| Metric | BF16 KV | INT8 KV |
|--------|---------|---------|
| Short context (<4K) | 67 tok/s | 61 tok/s |
| Long context (7K) | 24 tok/s | **45 tok/s** |
| Batch (4 concurrent) | 244 tok/s | - |
| Max context | 32K | **128K** |

INT8 KV cache enables 2x longer context and +87% faster long-context inference on RTX 3090.

## Quick Start

```bash
./scripts/launch-server.sh          # BF16 KV cache, 8K context
./scripts/launch-server-int8.sh     # INT8 KV cache, 64K context (recommended for long context)
```

Server starts on `http://localhost:8000`. First startup takes ~2 minutes for CUDA graph capture.

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "RedHatAI/gemma-3-27b-it-quantized.w4a16",
         "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Requirements

- 2x RTX 3090 (48GB total VRAM)
- NVLink bridge (NV4 recommended)
- Python 3.10+, vLLM 0.17.1

## Documentation

- [Setup Guide](docs/setup-guide.md) - Hardware requirements, installation, troubleshooting
- [Quantization Methods](docs/quantization-methods.md) - Model comparison (W4A16 vs GPTQ)
- [ExLlamaV2 Alternative](docs/exllamav2-alternative.md) - Alternative inference engine

## Research

1. [CUDA Graph Optimization](research/01-cuda-graph-optimization.md) - 3x speedup discovery
2. [Context Scaling](research/02-context-scaling.md) - 8K to 128K with no speed loss
3. [Long Context Bottleneck](research/03-long-context-bottleneck.md) - Memory bandwidth analysis
4. [Quality Testing](research/04-dutch-quality-test.md) - Dutch summarization comparison
5. [SGLang Comparison](research/05-sglang-comparison.md) - vLLM wins on RTX 3090
11. [INT8 KV Cache](research/11-int8-kv-cache.md) - 2x memory savings, +87% long context speed
12. [Qwen 3.5 Comparison](research/12-qwen35-comparison.md) - DeltaNet vs Gemma 3 hybrid attention
13. [CUDA Graphs & Cascade](research/13-cuda-graphs-cascade-analysis.md) - Why cascade attention is a red herring

## Scripts

| Script | Context | Description |
|--------|---------|-------------|
| `scripts/launch-server.sh` | 8K | Default, fastest startup |
| `scripts/launch-server-32k.sh` | 32K | Medium context |
| `scripts/launch-server-128k.sh` | 128K | Full context (Gemma 3 max) |
| `scripts/launch-server-int8.sh` | 64K | **INT8 KV cache** - best for long context |
| `scripts/benchmark.py` | - | Performance measurement |
| `scripts/quality_compare.py` | - | Quality testing with Dutch text |

## Key Findings

- **CUDA Graphs require `--disable-custom-all-reduce`** on RTX 3090
- **FULL_DECODE_ONLY mode** enables graphs on 24GB GPUs
- **W4A16 quantization** fits full 128K context in 48GB VRAM
- **NVLink NV4** provides ~112 GB/s bidirectional bandwidth
- **INT8 KV cache** halves KV memory, +87% speed for >4K tokens (requires vLLM patch)

## INT8 KV Cache (RTX 3090 / Ampere)

RTX 3090 lacks FP8 hardware. We implemented INT8 KV cache quantization via custom Triton kernels.

**Setup:**
```bash
# Apply patch to vLLM 0.17.1
patch -p1 -d $(python -c "import vllm; print(vllm.__path__[0])") < patches/vllm-int8-kv-cache.patch

# Launch with INT8
./scripts/launch-server-int8.sh
```

See [research/11-int8-kv-cache.md](research/11-int8-kv-cache.md) for implementation details and benchmarks.
