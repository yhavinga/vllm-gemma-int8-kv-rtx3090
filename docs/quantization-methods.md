# Quantization Methods Comparison

**Date:** 2026-03-15
**Goal:** Compare different quantized Gemma 3 27B models on dual RTX 3090

## Models Tested

### 1. RedHatAI/gemma-3-27b-it-quantized.w4a16 (BASELINE)
- **Format:** compressed-tensors (W4A16)
- **Status:** WORKS
- **Performance:** 67 tok/s single, 244 tok/s batch

### 2. ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g
- **Format:** GPTQ 4-bit, 128 group size
- **Status:** WORKS
- **Performance:** 67 tok/s single, 244 tok/s batch

### 3. pytorch/gemma-3-27b-it-AWQ-INT4
- **Format:** TorchAO AWQ INT4
- **Status:** FAILED
- **Error:** `Int4WeightOnlyConfig.__init__() got an unexpected keyword argument 'layout'`
- **Reason:** Compatibility issue between model config and torchao 0.16.0

## Models NOT Compatible with vLLM

### GGUF Format (llama.cpp only)
- `google/gemma-3-27b-it-qat-q4_0-gguf` - Google's QAT model
- `bartowski/google_gemma-3-27b-it-GGUF` - Various quants
- `unsloth/gemma-3-27b-it-GGUF` - Dynamic 2.0 quants

**Note:** GGUF requires llama.cpp, ollama, or LM Studio. vLLM does not support GGUF.

## Performance Summary

| Model | Format | Status | Single tok/s | Batch tok/s |
|-------|--------|--------|--------------|-------------|
| RedHatAI W4A16 | compressed-tensors | OK | 67 | 244 |
| ISTA-DASLab GPTQ | GPTQ 4b-128g | OK | 67 | 244 |
| pytorch AWQ | TorchAO AWQ | FAIL | - | - |

## Conclusion

**Both RedHatAI W4A16 and ISTA-DASLab GPTQ achieve identical performance.**

The RedHatAI model is recommended as the default because:
1. Officially supported by Red Hat / vLLM team
2. Uses newer compressed-tensors format
3. Actively maintained

## Google QAT Model (GGUF)

Google's Quantization-Aware Training (QAT) model (`google/gemma-3-27b-it-qat-q4_0-gguf`) is trained specifically for 4-bit inference and may have better quality than post-training quantization. However, it requires llama.cpp or similar GGUF-compatible runtime.

To test with llama.cpp:
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# Download model
huggingface-cli download google/gemma-3-27b-it-qat-q4_0-gguf --local-dir ./models

# Run inference
./llama-server -m ./models/gemma-3-27b-it-qat-q4_0.gguf -ngl 99 -c 8192
```

Note: llama.cpp doesn't support tensor parallelism in the same way as vLLM, so multi-GPU performance may differ.
