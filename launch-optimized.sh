#!/bin/bash
# Optimized vLLM + Gemma 3 27B launcher for Dual RTX 3090 with NVLink
# Tested 2026-03-15: ~67 tok/s single, ~244 tok/s batch (4 concurrent)
# Total improvement: 6x over baseline (11 tok/s enforce-eager)

set -euo pipefail

# === Configuration ===
MODEL="RedHatAI/gemma-3-27b-it-quantized.w4a16"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# === GPU Selection ===
export CUDA_VISIBLE_DEVICES=0,1

# === P2P Optimizations for RTX 3090 ===
export CUDA_FORCE_P2P_ACCESS=1
export VLLM_SKIP_P2P_CHECK=1

# === NCCL Optimizations for NVLink ===
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFF_SIZE=16777216

# === Memory Optimizations ===
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

echo "=== Launching vLLM with CUDA Graphs ==="
echo "Model: ${MODEL}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (Tensor Parallel = 2)"
echo "NVLink: Enabled (NCCL_P2P_LEVEL=NVL)"
echo "CUDA Graphs: FULL_DECODE_ONLY with limited capture sizes"
echo "Port: ${PORT}"
echo "Max Context: ${MAX_MODEL_LEN}"
echo "GPU Memory Utilization: ${GPU_MEM_UTIL}"
echo ""
echo "Note: Startup takes ~2 minutes for CUDA graph capture"
echo ""

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"

# Launch vLLM server with CUDA graphs
# --disable-custom-all-reduce is REQUIRED for CUDA graphs on RTX 3090
exec vllm serve "${MODEL}" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --port "${PORT}" \
    "$@"
