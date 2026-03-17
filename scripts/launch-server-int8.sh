#!/bin/bash
# Launch vLLM server with INT8 KV cache for RTX 3090
#
# Performance characteristics (Gemma 3 27B, 2x RTX 3090):
#   <4K tokens:  60 tok/s (CUDA graphs active)
#   >4K tokens:  28→10 tok/s (cascade attention, degrades with length)
#
# The 4K threshold is Gemma 3's sliding window size - beyond it, cascade
# attention forces eager mode fallback regardless of CUDA graph settings.

cd /home/yeb/Developer/gemma
source venv/bin/activate

vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --kv-cache-dtype int8 \
    --calculate-kv-scales \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90
