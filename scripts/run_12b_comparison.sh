#!/bin/bash
# Master script for 12B TP=1 vs TP=2 comparison
# Runs all configurations and collects results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
PORT=8000
WAIT_TIME=180  # Wait for server startup + CUDA graph capture

echo "========================================"
echo "12B TENSOR PARALLEL COMPARISON TEST"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Activate venv
source "${SCRIPT_DIR}/../venv/bin/activate"

# Function to wait for server
wait_for_server() {
    local max_wait=$1
    local elapsed=0
    echo "Waiting for server to be ready (max ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "Server ready after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  ...waiting (${elapsed}s)"
    done
    echo "Server did not start within ${max_wait}s"
    return 1
}

# Function to stop server
stop_server() {
    echo "Stopping vLLM server..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    # Make sure GPU memory is freed
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill 2>/dev/null || true
    sleep 3
}

# ========================================
# TEST 1: 12B TP=1 (Single GPU, no CUDA graphs)
# ========================================
echo ""
echo "========================================"
echo "TEST 1: 12B TP=1 (Single GPU, baseline)"
echo "========================================"

stop_server

echo "Starting server..."
"${SCRIPT_DIR}/launch-12b-tp1.sh" &
SERVER_PID=$!

if wait_for_server 120; then
    echo ""
    echo "Running benchmark..."
    python "${SCRIPT_DIR}/benchmark_12b_tp_comparison.py" \
        --config-name "12B-TP1-baseline" \
        --runs 3 \
        --skip-context
else
    echo "FAILED: Server did not start"
fi

stop_server

# ========================================
# TEST 2: 12B TP=1 with CUDA graphs
# ========================================
echo ""
echo "========================================"
echo "TEST 2: 12B TP=1 (Single GPU + CUDA graphs)"
echo "========================================"

echo "Starting server with CUDA graphs..."
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
vllm serve RedHatAI/gemma-3-12b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port ${PORT} &
SERVER_PID=$!

if wait_for_server ${WAIT_TIME}; then
    echo ""
    echo "Running benchmark..."
    python "${SCRIPT_DIR}/benchmark_12b_tp_comparison.py" \
        --config-name "12B-TP1-cudagraph" \
        --runs 3 \
        --skip-context
else
    echo "FAILED: Server did not start"
fi

stop_server

# ========================================
# TEST 3: 12B TP=2 (Dual GPU, no CUDA graphs)
# ========================================
echo ""
echo "========================================"
echo "TEST 3: 12B TP=2 (Dual GPU, baseline)"
echo "========================================"

echo "Starting server..."
CUDA_VISIBLE_DEVICES=0,1 \
CUDA_FORCE_P2P_ACCESS=1 \
VLLM_SKIP_P2P_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
NCCL_BUFF_SIZE=16777216 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
vllm serve RedHatAI/gemma-3-12b-it-quantized.w4a16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port ${PORT} &
SERVER_PID=$!

if wait_for_server 120; then
    echo ""
    echo "Running benchmark..."
    python "${SCRIPT_DIR}/benchmark_12b_tp_comparison.py" \
        --config-name "12B-TP2-baseline" \
        --runs 3 \
        --skip-context
else
    echo "FAILED: Server did not start"
fi

stop_server

# ========================================
# TEST 4: 12B TP=2 with CUDA graphs
# ========================================
echo ""
echo "========================================"
echo "TEST 4: 12B TP=2 (Dual GPU + CUDA graphs)"
echo "========================================"

echo "Starting server with CUDA graphs..."
"${SCRIPT_DIR}/launch-12b-tp2.sh" &
SERVER_PID=$!

if wait_for_server ${WAIT_TIME}; then
    echo ""
    echo "Running benchmark..."
    python "${SCRIPT_DIR}/benchmark_12b_tp_comparison.py" \
        --config-name "12B-TP2-cudagraph" \
        --runs 3 \
        --skip-context
else
    echo "FAILED: Server did not start"
fi

stop_server

# ========================================
# SUMMARY
# ========================================
echo ""
echo "========================================"
echo "ALL TESTS COMPLETE"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}"/benchmark-12b-*.json 2>/dev/null || echo "No result files found"
