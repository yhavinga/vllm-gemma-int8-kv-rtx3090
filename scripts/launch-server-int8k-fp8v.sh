#!/bin/bash
# Launch vLLM with INT8-K + FP8-V emulation + per-layer scales
#
# This uses:
# - INT8 for K (keys) with per-layer k_scale
# - FP8-E4M3 emulation for V (values) stored as INT8 bits, with per-layer v_scale
#
# FP8-E4M3 has better dynamic range (±448) than INT8 (±127), which helps
# with the heavy-tailed V distributions seen in some layers.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Enable V-FP8 emulation
export VLLM_INT8_V_FP8_EMUL=1

# Set per-layer scales file
export VLLM_KV_SCALES_FILE="${PROJECT_ROOT}/scales/gemma3_27b_per_layer.json"

# Verify scales file exists
if [ ! -f "$VLLM_KV_SCALES_FILE" ]; then
    echo "ERROR: Per-layer scales file not found: $VLLM_KV_SCALES_FILE"
    exit 1
fi

echo "============================================================"
echo "vLLM: INT8-K + FP8-V Emulation + Per-Layer Scales"
echo "============================================================"
echo "VLLM_INT8_V_FP8_EMUL: $VLLM_INT8_V_FP8_EMUL"
echo "Scales file: $VLLM_KV_SCALES_FILE"
echo ""

# Activate venv if available
if [ -d "${PROJECT_ROOT}/venv" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# Launch vLLM
vllm serve "RedHatAI/gemma-3-27b-it-quantized.w4a16" \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --kv-cache-dtype int8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    "$@"
