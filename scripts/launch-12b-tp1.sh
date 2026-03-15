#!/bin/bash
# 12B TP=1 (Single GPU) Configuration
# For comparison testing against TP=2

set -euo pipefail

MODEL="RedHatAI/gemma-3-12b-it-quantized.w4a16"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Single GPU
export CUDA_VISIBLE_DEVICES=0

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

echo "=== Launching 12B TP=1 (Single GPU) ==="
echo "Model: ${MODEL}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Port: ${PORT}"
echo "Max Context: ${MAX_MODEL_LEN}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../venv/bin/activate"

# Note: No CUDA graphs initially, for fair baseline comparison
# Add --compilation-config for CUDA graphs version
exec vllm serve "${MODEL}" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --port "${PORT}" \
    "$@"
