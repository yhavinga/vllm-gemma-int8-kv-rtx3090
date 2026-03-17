#!/usr/bin/env python3
"""
Throughput Grid Search for Gemma 3 1B and 4B models.

Tests max batched throughput across:
- Models: 1B (W8A8), 4B (W4A16, W8A8)
- Context lengths: 4K, 8K, 16K, 32K, 64K, 128K
- Batch sizes: auto-scaled to find optimal throughput

Inspired by HuggingFace's Synthetic Data Playbook approach:
- Fill GPU memory with batched requests
- Measure sustained generation throughput
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

# Model configurations
# Note: max_context based on model's max_position_embeddings from HF config
MODELS = {
    "1b-w8a8": {
        "name": "RedHatAI/gemma-3-1b-it-quantized.w8a8",
        "size": "1B",
        "quant": "W8A8",
        "model_vram_gb": 1.5,
        "max_context": 32_768,  # Gemma 3 1B: max_position_embeddings=32768
    },
    "4b-w4a16": {
        "name": "RedHatAI/gemma-3-4b-it-quantized.w4a16",
        "size": "4B",
        "quant": "W4A16",
        "model_vram_gb": 3.0,
        "max_context": 131_072,  # Gemma 3 4B+ supports 128K via sliding window
    },
    "4b-w8a8": {
        "name": "RedHatAI/gemma-3-4b-it-quantized.w8a8",
        "size": "4B",
        "quant": "W8A8",
        "model_vram_gb": 5.0,
        "max_context": 131_072,
    },
}

# Context lengths to test
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

# Batch sizes to probe (will auto-scale based on OOM)
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Tokens to generate per request
OUTPUT_TOKENS = 128

# Input prompt that creates ~1K tokens input context
BASE_PROMPT = """You are a helpful AI assistant. Please provide a detailed and comprehensive response.

Context: The following is a technical discussion about machine learning optimization techniques,
specifically focusing on inference optimization for large language models. The discussion covers
quantization methods including W4A16 (4-bit weights, 16-bit activations) and W8A8 (8-bit weights
and activations), as well as batching strategies for maximizing throughput.

Key topics:
1. Memory bandwidth vs compute bound operations
2. KV cache memory consumption at various context lengths
3. Optimal batch sizes for different GPU memory configurations
4. Trade-offs between latency and throughput in production deployments

Please analyze and summarize the key optimization strategies for achieving maximum throughput
in large language model inference. Focus on practical recommendations.

---

"""


def create_prompt_with_padding(target_tokens: int = 512) -> str:
    """Create a prompt with approximately target_tokens input tokens."""
    # Rough estimate: 1 token ~= 4 characters
    base_len = len(BASE_PROMPT)
    target_chars = target_tokens * 4

    if target_chars <= base_len:
        return BASE_PROMPT

    # Add padding text to reach target
    padding = "\n[Context padding for benchmark]\n" * ((target_chars - base_len) // 35)
    return BASE_PROMPT + padding


def wait_for_server(url: str, timeout: int = 600) -> bool:
    """Wait for vLLM server to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def completions_request(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Send completion request and return timing info."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start = time.perf_counter()
    response = requests.post(
        f"{url}/v1/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    data = response.json()

    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", max_tokens)
    prompt_tokens = usage.get("prompt_tokens", 0)

    return {
        "elapsed": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
    }


def measure_batch_throughput(
    url: str,
    model: str,
    batch_size: int,
    prompt: str,
    max_tokens: int = OUTPUT_TOKENS,
    runs: int = 2,
) -> Optional[dict]:
    """Measure aggregate throughput for a batch of concurrent requests."""
    all_results = []

    for run_idx in range(runs):
        start = time.perf_counter()
        results = []

        try:
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(completions_request, url, model, prompt, max_tokens)
                    for _ in range(batch_size)
                ]
                for future in as_completed(futures):
                    results.append(future.result())
        except Exception as e:
            return None

        elapsed = time.perf_counter() - start
        total_tokens = sum(r["completion_tokens"] for r in results)
        throughput = total_tokens / elapsed

        all_results.append({
            "elapsed": elapsed,
            "total_tokens": total_tokens,
            "throughput": throughput,
        })

    throughputs = [r["throughput"] for r in all_results]
    return {
        "batch_size": batch_size,
        "avg_throughput": sum(throughputs) / len(throughputs),
        "max_throughput": max(throughputs),
        "min_throughput": min(throughputs),
        "avg_total_tokens": sum(r["total_tokens"] for r in all_results) / len(all_results),
    }


def find_optimal_batch_size(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int = OUTPUT_TOKENS,
    verbose: bool = True,
) -> dict:
    """Find the batch size that maximizes throughput."""
    best_result = None
    best_throughput = 0

    for batch_size in BATCH_SIZES:
        if verbose:
            print(f"      Testing batch={batch_size}...", end=" ", flush=True)

        result = measure_batch_throughput(url, model, batch_size, prompt, max_tokens, runs=2)

        if result is None:
            if verbose:
                print("FAILED (likely OOM)")
            break

        if verbose:
            print(f"{result['avg_throughput']:.1f} tok/s")

        if result["avg_throughput"] > best_throughput:
            best_throughput = result["avg_throughput"]
            best_result = result
        elif result["avg_throughput"] < best_throughput * 0.95:
            # Throughput dropping, we've passed optimal
            break

    return best_result


def launch_server(
    model: str,
    max_model_len: int,
    port: int = 8000,
    tp: int = 2,  # Tensor parallel size - use both GPUs by default
    gpu_mem_util: float = 0.85,
) -> subprocess.Popen:
    """Launch vLLM server with specified configuration."""
    env = os.environ.copy()

    if tp == 2:
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # NVLink optimizations for dual GPU
        env["CUDA_FORCE_P2P_ACCESS"] = "1"
        env["VLLM_SKIP_P2P_CHECK"] = "1"
        env["NCCL_P2P_LEVEL"] = "NVL"
        env["NCCL_BUFF_SIZE"] = "16777216"
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    # Build compilation config as proper JSON
    compilation_config = json.dumps({
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
    })

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--port", str(port),
        "--no-enable-log-requests",
        # Use FULL_DECODE_ONLY to reduce memory during CUDA graph capture
        "--compilation-config", compilation_config,
    ]

    if tp == 2:
        # Required for CUDA graphs on RTX 3090 with TP=2
        cmd.append("--disable-custom-all-reduce")

    # Write server logs to file instead of PIPE (avoids buffer blocking)
    log_file = open(f"/tmp/vllm-server-{port}.log", "w")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Allows clean termination
    )


def stop_server(proc: subprocess.Popen):
    """Stop vLLM server gracefully."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_grid_search(
    models: list[str],
    context_lengths: list[int],
    url: str = "http://localhost:8000",
    no_launch: bool = False,
    tp: int = 2,  # Tensor parallel size
    verbose: bool = True,
) -> dict:
    """Run full grid search across models and context lengths."""
    results = {}

    for model_key in models:
        model_info = MODELS[model_key]
        model_name = model_info["name"]
        results[model_key] = {"model": model_info, "configs": {}}

        print(f"\n{'='*70}")
        print(f"MODEL: {model_key} ({model_info['size']} {model_info['quant']})")
        print(f"{'='*70}")

        for ctx_len in context_lengths:
            if ctx_len > model_info["max_context"]:
                print(f"\n  Context {ctx_len//1024}K: SKIPPED (exceeds max {model_info['max_context']//1024}K)")
                continue

            ctx_key = f"{ctx_len//1024}k"
            print(f"\n  Context {ctx_len//1024}K:")

            server_proc = None

            try:
                if not no_launch:
                    print(f"    Launching server (TP={tp})...", end=" ", flush=True)
                    # Use conservative memory utilization to avoid OOM during CUDA graph capture
                    base_mem = 0.80
                    if ctx_len >= 65536:
                        base_mem = 0.85
                    elif ctx_len >= 32768:
                        base_mem = 0.82
                    server_proc = launch_server(model_name, ctx_len, tp=tp, gpu_mem_util=base_mem)

                    if not wait_for_server(url, timeout=600):
                        print("TIMEOUT")
                        results[model_key]["configs"][ctx_key] = {"error": "server_timeout"}
                        continue
                    print("READY")

                # Warmup
                print(f"    Warmup...", end=" ", flush=True)
                prompt = create_prompt_with_padding(512)
                try:
                    completions_request(url, model_name, prompt, 32)
                    print("done")
                except Exception as e:
                    print(f"FAILED: {e}")
                    results[model_key]["configs"][ctx_key] = {"error": str(e)}
                    continue

                # Find optimal batch size
                print(f"    Finding optimal batch size:")
                result = find_optimal_batch_size(url, model_name, prompt, verbose=verbose)

                if result:
                    results[model_key]["configs"][ctx_key] = {
                        "context_len": ctx_len,
                        "optimal_batch_size": result["batch_size"],
                        "max_throughput": result["max_throughput"],
                        "avg_throughput": result["avg_throughput"],
                    }
                    print(f"    => OPTIMAL: batch={result['batch_size']}, "
                          f"{result['max_throughput']:.1f} tok/s")
                else:
                    results[model_key]["configs"][ctx_key] = {"error": "no_valid_result"}
                    print(f"    => FAILED: No valid result")

            except KeyboardInterrupt:
                print("\nInterrupted!")
                raise
            except Exception as e:
                print(f"    ERROR: {e}")
                results[model_key]["configs"][ctx_key] = {"error": str(e)}
            finally:
                if server_proc:
                    print(f"    Stopping server...")
                    stop_server(server_proc)
                    time.sleep(5)  # Allow GPU memory to clear

    return results


def print_summary_table(results: dict):
    """Print results as a formatted table."""
    print(f"\n{'='*90}")
    print("THROUGHPUT GRID SEARCH RESULTS (tok/s)")
    print(f"{'='*90}")

    # Header
    ctx_headers = [f"{ctx//1024}K" for ctx in CONTEXT_LENGTHS]
    header = f"{'Model':<15} {'Quant':<8} " + " ".join(f"{h:>10}" for h in ctx_headers)
    print(header)
    print("-" * 90)

    for model_key, data in results.items():
        model_info = data["model"]
        row = f"{model_info['size']:<15} {model_info['quant']:<8} "

        for ctx_len in CONTEXT_LENGTHS:
            ctx_key = f"{ctx_len//1024}k"
            config = data["configs"].get(ctx_key, {})

            if "max_throughput" in config:
                row += f"{config['max_throughput']:>10.1f}"
            elif "error" in config:
                row += f"{'ERR':>10}"
            else:
                row += f"{'-':>10}"

        print(row)

    # Batch size table
    print(f"\n{'='*90}")
    print("OPTIMAL BATCH SIZES")
    print(f"{'='*90}")

    header = f"{'Model':<15} {'Quant':<8} " + " ".join(f"{h:>10}" for h in ctx_headers)
    print(header)
    print("-" * 90)

    for model_key, data in results.items():
        model_info = data["model"]
        row = f"{model_info['size']:<15} {model_info['quant']:<8} "

        for ctx_len in CONTEXT_LENGTHS:
            ctx_key = f"{ctx_len//1024}k"
            config = data["configs"].get(ctx_key, {})

            if "optimal_batch_size" in config:
                row += f"{config['optimal_batch_size']:>10}"
            else:
                row += f"{'-':>10}"

        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Throughput grid search for Gemma 3 1B/4B models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full grid search (all models, all contexts)
  python throughput_grid_search.py

  # Test specific models
  python throughput_grid_search.py --models 1b-w8a8 4b-w4a16

  # Test specific context lengths (in K)
  python throughput_grid_search.py --contexts 4 8 16

  # Use existing server (don't launch)
  python throughput_grid_search.py --no-launch --url http://localhost:8000
        """
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Models to test (default: all)"
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=None,
        help="Context lengths in K (default: 4 8 16 32 64 128)"
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--no-launch", action="store_true", help="Use existing server")
    parser.add_argument("--tp", type=int, default=2, choices=[1, 2], help="Tensor parallel size (default: 2 for dual GPU)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    # Resolve models
    if "all" in args.models:
        models = list(MODELS.keys())
    else:
        models = args.models

    # Resolve context lengths
    if args.contexts:
        context_lengths = [c * 1024 for c in args.contexts]
    else:
        context_lengths = CONTEXT_LENGTHS

    print("=" * 70)
    print("GEMMA 3 THROUGHPUT GRID SEARCH")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Models: {', '.join(models)}")
    print(f"Contexts: {', '.join(f'{c//1024}K' for c in context_lengths)}")
    print(f"Tensor Parallel: {args.tp} GPU{'s' if args.tp > 1 else ''}")
    print(f"Output tokens per request: {OUTPUT_TOKENS}")

    try:
        results = run_grid_search(
            models=models,
            context_lengths=context_lengths,
            url=args.url,
            no_launch=args.no_launch,
            tp=args.tp,
            verbose=not args.quiet,
        )

        print_summary_table(results)

        # Save results
        output_file = args.output or f"throughput-grid-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "models": models,
                    "context_lengths": context_lengths,
                    "output_tokens": OUTPUT_TOKENS,
                },
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
