#!/bin/bash
# Launch vLLM server with INT8 KV cache for RTX 3090

cd /home/yeb/Developer/gemma
source venv/bin/activate

# INT8 KV cache configuration
# - Uses half the memory of BF16 for KV cache
# - Enables longer contexts or larger batches
# - calculate-kv-scales computes scales dynamically

vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --kv-cache-dtype int8 \
    --calculate-kv-scales \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
