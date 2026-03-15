#!/usr/bin/env python3
"""
Comprehensive benchmark: 12B TP=1 vs TP=2

Tests:
1. Single request latency at various output lengths
2. Batch throughput at various concurrency levels
3. Long context performance
4. First token latency (TTFT)
5. Memory efficiency

Run this AFTER starting the vLLM server with the appropriate config.
"""

import argparse
import time
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

DEFAULT_URL = "http://localhost:8000"


def get_gpu_memory():
    """Get current GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        gpus = []
        for line in lines:
            used, total = map(int, line.split(','))
            gpus.append({"used_mb": used, "total_mb": total, "pct": used/total*100})
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def chat_completion(url: str, model: str, prompt: str, max_tokens: int = 256,
                    measure_ttft: bool = False) -> dict:
    """Send a chat completion request and return timing info."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": measure_ttft,
    }

    start = time.perf_counter()

    if measure_ttft:
        # Streaming to measure time-to-first-token
        response = requests.post(
            f"{url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
            stream=True,
        )
        response.raise_for_status()

        ttft = None
        completion_tokens = 0
        for line in response.iter_lines():
            if line:
                if ttft is None:
                    ttft = time.perf_counter() - start
                completion_tokens += 1  # Approximate

        elapsed = time.perf_counter() - start
        return {
            "elapsed": elapsed,
            "ttft": ttft,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
        }
    else:
        response = requests.post(
            f"{url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()

        completion_tokens = data["usage"]["completion_tokens"]
        prompt_tokens = data["usage"]["prompt_tokens"]

        return {
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": completion_tokens / elapsed,
        }


def run_latency_test(url: str, model: str, output_lengths: list, runs: int = 3) -> dict:
    """Test latency at different output lengths."""
    prompt = "Write a detailed explanation of quantum computing, covering superposition, entanglement, and quantum gates."
    results = {}

    for length in output_lengths:
        print(f"\n  Output length: {length} tokens...")
        length_results = []

        for i in range(runs):
            result = chat_completion(url, model, prompt, max_tokens=length)
            length_results.append(result)
            print(f"    Run {i+1}: {result['completion_tokens']} tokens in {result['elapsed']:.2f}s "
                  f"({result['tokens_per_sec']:.1f} tok/s)")

        speeds = [r["tokens_per_sec"] for r in length_results]
        results[length] = {
            "runs": runs,
            "avg_tokens_per_sec": statistics.mean(speeds),
            "min_tokens_per_sec": min(speeds),
            "max_tokens_per_sec": max(speeds),
            "stddev": statistics.stdev(speeds) if len(speeds) > 1 else 0,
            "avg_latency_ms": statistics.mean([r["elapsed"] * 1000 for r in length_results]),
        }

    return results


def run_batch_test(url: str, model: str, batch_sizes: list, runs: int = 2) -> dict:
    """Test batch throughput at different concurrency levels."""
    prompt = "Explain the concept of machine learning in simple terms."
    results = {}

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}...")
        batch_results = []

        for i in range(runs):
            start = time.perf_counter()
            request_results = []

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(chat_completion, url, model, prompt, 128)
                    for _ in range(batch_size)
                ]
                for future in as_completed(futures):
                    try:
                        request_results.append(future.result())
                    except Exception as e:
                        print(f"    Request failed: {e}")

            elapsed = time.perf_counter() - start
            total_tokens = sum(r["completion_tokens"] for r in request_results)
            aggregate_throughput = total_tokens / elapsed if elapsed > 0 else 0

            print(f"    Run {i+1}: {total_tokens} tokens in {elapsed:.2f}s "
                  f"({aggregate_throughput:.1f} tok/s aggregate)")

            batch_results.append({
                "elapsed": elapsed,
                "total_tokens": total_tokens,
                "aggregate_throughput": aggregate_throughput,
                "successful_requests": len(request_results),
            })

        throughputs = [r["aggregate_throughput"] for r in batch_results]
        results[batch_size] = {
            "runs": runs,
            "avg_aggregate_throughput": statistics.mean(throughputs),
            "max_aggregate_throughput": max(throughputs),
            "per_request_throughput": statistics.mean(throughputs) / batch_size,
        }

    return results


def run_context_test(url: str, model: str, context_lengths: list, runs: int = 2) -> dict:
    """Test performance at different context lengths."""
    base_text = """The history of artificial intelligence (AI) began in antiquity, with myths,
stories and rumors of artificial beings endowed with intelligence or consciousness by master
craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the
process of human thinking as the mechanical manipulation of symbols. This work culminated in
the invention of the programmable digital computer in the 1940s, a machine based on the
abstract essence of mathematical reasoning. """ * 50  # ~200 tokens, repeat for length

    results = {}

    for ctx_len in context_lengths:
        # Approximate tokens: 1 token ≈ 4 chars
        chars_needed = ctx_len * 4
        prompt = base_text[:chars_needed] + "\n\nSummarize the above text in 2 sentences."

        print(f"\n  Context length: ~{ctx_len} tokens...")
        ctx_results = []

        for i in range(runs):
            try:
                result = chat_completion(url, model, prompt, max_tokens=100)
                ctx_results.append(result)
                print(f"    Run {i+1}: {result['prompt_tokens']} prompt + {result['completion_tokens']} completion "
                      f"({result['tokens_per_sec']:.1f} tok/s)")
            except Exception as e:
                print(f"    Run {i+1}: FAILED - {e}")

        if ctx_results:
            speeds = [r["tokens_per_sec"] for r in ctx_results]
            results[ctx_len] = {
                "runs": len(ctx_results),
                "avg_tokens_per_sec": statistics.mean(speeds),
                "avg_prompt_tokens": statistics.mean([r["prompt_tokens"] for r in ctx_results]),
                "avg_latency_ms": statistics.mean([r["elapsed"] * 1000 for r in ctx_results]),
            }
        else:
            results[ctx_len] = {"error": "all runs failed"}

    return results


def get_model_from_server(url: str) -> str:
    """Get the model name from the server."""
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()["data"]
        return models[0]["id"] if models else "unknown"
    except Exception as e:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Comprehensive 12B TP comparison benchmark")
    parser.add_argument("--url", default=DEFAULT_URL, help="vLLM server URL")
    parser.add_argument("--config-name", required=True, help="Config name (e.g., '12B-TP1' or '12B-TP2')")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test")
    parser.add_argument("--skip-context", action="store_true", help="Skip context length tests")
    args = parser.parse_args()

    print("=" * 70)
    print(f"COMPREHENSIVE BENCHMARK: {args.config_name}")
    print("=" * 70)
    print(f"Server: {args.url}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Check server health and get model
    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        health.raise_for_status()
        model = get_model_from_server(args.url)
        print(f"Model: {model}")
        print("Server: HEALTHY")
    except Exception as e:
        print(f"Server: UNREACHABLE ({e})")
        return 1

    # Get initial GPU memory
    print(f"\nGPU Memory (idle):")
    for i, gpu in enumerate(get_gpu_memory()):
        print(f"  GPU {i}: {gpu.get('used_mb', '?')} / {gpu.get('total_mb', '?')} MB ({gpu.get('pct', '?'):.1f}%)")

    results = {
        "config_name": args.config_name,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "gpu_memory_idle": get_gpu_memory(),
    }

    # Test 1: Latency at various output lengths
    print("\n" + "=" * 70)
    print("TEST 1: LATENCY VS OUTPUT LENGTH")
    print("=" * 70)
    results["latency_test"] = run_latency_test(
        args.url, model,
        output_lengths=[32, 64, 128, 256, 512],
        runs=args.runs
    )

    # Test 2: Batch throughput
    print("\n" + "=" * 70)
    print("TEST 2: BATCH THROUGHPUT")
    print("=" * 70)
    results["batch_test"] = run_batch_test(
        args.url, model,
        batch_sizes=[1, 2, 4, 8],
        runs=2
    )

    # Test 3: Context length scaling (if not skipped)
    if not args.skip_context:
        print("\n" + "=" * 70)
        print("TEST 3: CONTEXT LENGTH SCALING")
        print("=" * 70)
        results["context_test"] = run_context_test(
            args.url, model,
            context_lengths=[512, 1024, 2048, 4096],
            runs=2
        )

    # Get GPU memory after tests
    results["gpu_memory_after"] = get_gpu_memory()
    print(f"\nGPU Memory (after tests):")
    for i, gpu in enumerate(results["gpu_memory_after"]):
        print(f"  GPU {i}: {gpu.get('used_mb', '?')} / {gpu.get('total_mb', '?')} MB ({gpu.get('pct', '?'):.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nLatency (tok/s at different output lengths):")
    for length, data in results["latency_test"].items():
        print(f"  {length:4d} tokens: {data['avg_tokens_per_sec']:.1f} tok/s (±{data['stddev']:.1f})")

    print(f"\nBatch Throughput:")
    for batch_size, data in results["batch_test"].items():
        print(f"  Batch {batch_size}: {data['avg_aggregate_throughput']:.1f} tok/s aggregate "
              f"({data['per_request_throughput']:.1f} per request)")

    if "context_test" in results:
        print(f"\nContext Scaling (tok/s):")
        for ctx_len, data in results["context_test"].items():
            if "error" not in data:
                print(f"  {ctx_len:5d} tokens: {data['avg_tokens_per_sec']:.1f} tok/s")

    # Save results
    output_file = f"results/benchmark-{args.config_name.lower().replace(' ', '-')}-{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
