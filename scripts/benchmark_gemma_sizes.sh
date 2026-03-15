#!/bin/bash
# Comprehensive Gemma 3 benchmark for all model sizes
# Tests RedHatAI vs ISTA-DASLab quantizations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_DIR/results"
PORT=8000

source "$REPO_DIR/venv/bin/activate"

# Environment for all tests
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

benchmark_model() {
    local name="$1"
    local model="$2"
    local tp="$3"
    local max_len="$4"
    local extra_args="${5:-}"

    echo ""
    echo "============================================================"
    echo "BENCHMARKING: $name"
    echo "Model: $model"
    echo "TP: $tp, Max Context: $max_len"
    echo "============================================================"

    # Set GPU visibility
    if [ "$tp" -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
        export CUDA_FORCE_P2P_ACCESS=1
        export VLLM_SKIP_P2P_CHECK=1
        export NCCL_P2P_LEVEL=NVL
        export NCCL_BUFF_SIZE=16777216
    else
        export CUDA_VISIBLE_DEVICES=0
        unset CUDA_FORCE_P2P_ACCESS 2>/dev/null || true
        unset VLLM_SKIP_P2P_CHECK 2>/dev/null || true
    fi

    # Build command
    local cmd="python -m vllm.entrypoints.openai.api_server"
    cmd+=" --model $model"
    cmd+=" --trust-remote-code"
    cmd+=" --tensor-parallel-size $tp"
    cmd+=" --max-model-len $max_len"
    cmd+=" --gpu-memory-utilization 0.90"
    cmd+=" --port $PORT"

    if [ "$tp" -eq 2 ]; then
        cmd+=" --disable-custom-all-reduce"
        cmd+=' --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
    fi

    if [ -n "$extra_args" ]; then
        cmd+=" $extra_args"
    fi

    # Start server
    echo "Starting server..."
    $cmd > /tmp/vllm_benchmark.log 2>&1 &
    local pid=$!

    # Wait for server
    echo -n "Waiting for server..."
    local waited=0
    while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -gt 180 ]; then
            echo " TIMEOUT"
            kill $pid 2>/dev/null || true
            echo "Server log:"
            tail -50 /tmp/vllm_benchmark.log
            return 1
        fi
        echo -n "."
    done
    echo " ready (${waited}s)"

    # Run benchmark
    echo "Running benchmark..."
    python "$SCRIPT_DIR/benchmark.py" \
        --url "http://localhost:$PORT" \
        --model "$model" \
        --runs 3 \
        --batch-size 4

    # Stop server
    echo "Stopping server..."
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 3

    return 0
}

echo "============================================================"
echo "GEMMA 3 COMPREHENSIVE BENCHMARK"
echo "============================================================"
echo "Date: $(date)"
echo "Hardware: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo ""

# Kill any existing vLLM servers
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# 1B Models (single GPU)
echo ""
echo "=== 1B MODELS ==="
benchmark_model "1B-RedHat-W8A8" "RedHatAI/gemma-3-1b-it-quantized.w8a8" 1 4096 || true

# 4B Models (single GPU)
echo ""
echo "=== 4B MODELS ==="
benchmark_model "4B-RedHat-W8A8" "RedHatAI/gemma-3-4b-it-quantized.w8a8" 1 4096 || true
benchmark_model "4B-RedHat-W4A16" "RedHatAI/gemma-3-4b-it-quantized.w4a16" 1 4096 || true
benchmark_model "4B-ISTA-GPTQ" "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g" 1 4096 || true

# 12B Models (single GPU for W4A16/GPTQ, TP=2 for W8A8)
echo ""
echo "=== 12B MODELS ==="
benchmark_model "12B-RedHat-W4A16" "RedHatAI/gemma-3-12b-it-quantized.w4a16" 1 4096 || true
benchmark_model "12B-ISTA-GPTQ" "ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g" 1 4096 || true
benchmark_model "12B-RedHat-W8A8-TP2" "RedHatAI/gemma-3-12b-it-quantized.w8a8" 2 4096 || true

# 27B reference (TP=2)
echo ""
echo "=== 27B REFERENCE ==="
benchmark_model "27B-RedHat-W4A16-TP2" "RedHatAI/gemma-3-27b-it-quantized.w4a16" 2 8192 || true

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
echo "============================================================"
