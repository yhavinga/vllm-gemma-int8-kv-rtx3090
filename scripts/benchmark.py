#!/usr/bin/env python3
"""
Benchmark script for vLLM Gemma 3 27B inference.
Measures tokens/sec for single requests and batch throughput.
"""

import argparse
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Default configuration
DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "RedHatAI/gemma-3-27b-it-quantized.w4a16"

# Test prompts of varying lengths
PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain quantum computing in simple terms.",
    "long": """Summarize the following text:
The history of artificial intelligence (AI) began in antiquity, with myths, stories
and rumors of artificial beings endowed with intelligence or consciousness by master
craftsmen. The seeds of modern AI were planted by philosophers who attempted to
describe the process of human thinking as the mechanical manipulation of symbols.
This work culminated in the invention of the programmable digital computer in the
1940s, a machine based on the abstract essence of mathematical reasoning. This device
and the ideas behind it inspired a handful of scientists to begin seriously discussing
the possibility of building an electronic brain. The field of AI research was founded
at a workshop held on the campus of Dartmouth College, USA during the summer of 1956.
Those who attended would become the leaders of AI research for decades. Many of them
predicted that a machine as intelligent as a human being would exist in no more than
a generation, and they were given millions of dollars to make this vision come true.""",
}


def chat_completion(url: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send a chat completion request and return timing info."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    start = time.perf_counter()
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


def run_single_benchmark(url: str, model: str, prompt_key: str, runs: int = 3) -> dict:
    """Run multiple single-request benchmarks and return stats."""
    prompt = PROMPTS[prompt_key]
    results = []

    print(f"\n  Running {runs} iterations with '{prompt_key}' prompt...")

    for i in range(runs):
        result = chat_completion(url, model, prompt)
        results.append(result)
        print(f"    Run {i+1}: {result['completion_tokens']} tokens in {result['elapsed']:.2f}s "
              f"({result['tokens_per_sec']:.1f} tok/s)")

    speeds = [r["tokens_per_sec"] for r in results]
    return {
        "prompt": prompt_key,
        "runs": runs,
        "avg_tokens_per_sec": statistics.mean(speeds),
        "min_tokens_per_sec": min(speeds),
        "max_tokens_per_sec": max(speeds),
        "stddev": statistics.stdev(speeds) if len(speeds) > 1 else 0,
        "details": results,
    }


def run_batch_benchmark(url: str, model: str, batch_size: int = 4, runs: int = 2) -> dict:
    """Run concurrent requests to measure batch throughput."""
    prompt = PROMPTS["medium"]
    all_results = []

    print(f"\n  Running {runs} batch iterations with {batch_size} concurrent requests...")

    for i in range(runs):
        start = time.perf_counter()
        results = []

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(chat_completion, url, model, prompt, 128)
                for _ in range(batch_size)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.perf_counter() - start
        total_tokens = sum(r["completion_tokens"] for r in results)
        aggregate_throughput = total_tokens / elapsed

        print(f"    Batch {i+1}: {total_tokens} total tokens in {elapsed:.2f}s "
              f"({aggregate_throughput:.1f} tok/s aggregate)")

        all_results.append({
            "elapsed": elapsed,
            "total_tokens": total_tokens,
            "aggregate_throughput": aggregate_throughput,
        })

    throughputs = [r["aggregate_throughput"] for r in all_results]
    return {
        "batch_size": batch_size,
        "runs": runs,
        "avg_aggregate_throughput": statistics.mean(throughputs),
        "max_aggregate_throughput": max(throughputs),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM Gemma 3 27B")
    parser.add_argument("--url", default=DEFAULT_URL, help="vLLM server URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test")
    parser.add_argument("--batch-size", type=int, default=4, help="Concurrent requests for batch test")
    parser.add_argument("--skip-batch", action="store_true", help="Skip batch benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Gemma 3 27B Benchmark")
    print("=" * 60)
    print(f"Server: {args.url}")
    print(f"Model: {args.model}")

    # Check server health
    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        health.raise_for_status()
        print("Server: HEALTHY")
    except Exception as e:
        print(f"Server: UNREACHABLE ({e})")
        return 1

    results = {"single": {}, "batch": None}

    # Single request benchmarks
    print("\n" + "=" * 60)
    print("SINGLE REQUEST BENCHMARKS")
    print("=" * 60)

    for prompt_key in ["short", "medium", "long"]:
        result = run_single_benchmark(args.url, args.model, prompt_key, args.runs)
        results["single"][prompt_key] = result

    # Batch benchmark
    if not args.skip_batch:
        print("\n" + "=" * 60)
        print("BATCH THROUGHPUT BENCHMARK")
        print("=" * 60)
        results["batch"] = run_batch_benchmark(
            args.url, args.model, args.batch_size, runs=2
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nSingle Request Performance:")
    for prompt_key, data in results["single"].items():
        print(f"  {prompt_key:8s}: {data['avg_tokens_per_sec']:.1f} tok/s "
              f"(min: {data['min_tokens_per_sec']:.1f}, max: {data['max_tokens_per_sec']:.1f})")

    if results["batch"]:
        print(f"\nBatch Throughput ({results['batch']['batch_size']} concurrent):")
        print(f"  Aggregate: {results['batch']['avg_aggregate_throughput']:.1f} tok/s")
        print(f"  Max: {results['batch']['max_aggregate_throughput']:.1f} tok/s")

    # Save results
    output_file = f"benchmark-{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
