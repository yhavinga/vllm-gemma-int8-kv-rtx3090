# Gemma 3 27B on Dual RTX 3090

Optimized vLLM configuration for running Gemma 3 27B IT on consumer hardware.

## Performance

| Metric | Result |
|--------|--------|
| Single request | **67 tok/s** |
| Batch (4 concurrent) | **244 tok/s** |
| Context support | Up to 128K tokens |
| Total improvement | 6x over baseline |

## Quick Start

```bash
./scripts/launch-server.sh
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

## Scripts

| Script | Context | Description |
|--------|---------|-------------|
| `scripts/launch-server.sh` | 8K | Default, fastest startup |
| `scripts/launch-server-32k.sh` | 32K | Medium context |
| `scripts/launch-server-128k.sh` | 128K | Full context (Gemma 3 max) |
| `scripts/benchmark.py` | - | Performance measurement |
| `scripts/quality_compare.py` | - | Quality testing with Dutch text |

## Key Findings

- **CUDA Graphs require `--disable-custom-all-reduce`** on RTX 3090
- **FULL_DECODE_ONLY mode** enables graphs on 24GB GPUs
- **W4A16 quantization** fits full 128K context in 48GB VRAM
- **NVLink NV4** provides ~112 GB/s bidirectional bandwidth
