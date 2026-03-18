#!/bin/bash
# Recommended production config: INT8-K + FP8-V emulation with per-layer scales
#
# Why this config:
# - K uses INT8: linear quant error → linear softmax error, uniform spacing fine
# - V uses FP8-E4M3 emulated in INT8: logarithmic spacing for heavy-tailed distributions
# - Per-layer scales: 340x variation in absmax across 62 layers needs individual calibration
#
# Performance: 67 tok/s with CUDA graphs, 50% memory reduction vs BF16

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# V uses FP8-E4M3 encoding stored as INT8 bytes
export VLLM_INT8_V_FP8_EMUL=1

# Per-layer calibrated scales (k_scale, v_scale for each of 62 layers)
export VLLM_KV_SCALES_FILE="${PROJECT_ROOT}/scales/gemma3_27b_per_layer.json"

if [ ! -f "$VLLM_KV_SCALES_FILE" ]; then
    echo "ERROR: Scales file not found: $VLLM_KV_SCALES_FILE"
    exit 1
fi

# Activate venv
if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

exec vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --kv-cache-dtype int8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    "$@"
