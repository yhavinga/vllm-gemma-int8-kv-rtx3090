#!/bin/bash
# Launch vLLM with INT8 KV cache using PER-LAYER scales
#
# This fixes the precision loss problem where a single global scale is used
# for all 62 layers. With per-layer scales:
# - Layer 42 (v_absmax=884) gets v_scale=6.96 → full precision for large values
# - Layer 59 (v_absmax=2.6)  gets v_scale=0.02 → full precision for small values
#
# Expected performance: ~60 tok/s with CUDA graphs (vs ~10 tok/s without)

set -e

# Find the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set per-layer scales file
export VLLM_KV_SCALES_FILE="${PROJECT_ROOT}/scales/gemma3_27b_per_layer.json"

# Verify scales file exists
if [ ! -f "$VLLM_KV_SCALES_FILE" ]; then
    echo "ERROR: Per-layer scales file not found: $VLLM_KV_SCALES_FILE"
    echo "Run: python scripts/create_per_layer_scales.py"
    exit 1
fi

echo "============================================================"
echo "vLLM INT8 KV Cache with PER-LAYER Scales"
echo "============================================================"
echo "Scales file: $VLLM_KV_SCALES_FILE"
echo ""

# Activate venv if available
if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# Launch vLLM with INT8 KV cache and CUDA graphs
vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --kv-cache-dtype int8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    "$@"
