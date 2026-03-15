#!/usr/bin/env python3
"""
Comprehensive benchmark for Gemma 3 models (1B, 4B, 12B, 27B).
Tests RedHatAI vs ISTA-DASLab quantizations with various configurations.
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
from typing import Optional

import requests

# Model configurations to test
MODELS = {
    # 1B - Only RedHatAI W8A8 available for vLLM
    "1b-redhat-w8a8": {
        "name": "RedHatAI/gemma-3-1b-it-quantized.w8a8",
        "size": "1B",
        "quant": "W8A8",
        "provider": "RedHatAI",
        "vram_gb": 1.5,  # Estimated
    },
    # 4B options
    "4b-redhat-w8a8": {
        "name": "RedHatAI/gemma-3-4b-it-quantized.w8a8",
        "size": "4B",
        "quant": "W8A8",
        "provider": "RedHatAI",
        "vram_gb": 5.0,
    },
    "4b-redhat-w4a16": {
        "name": "RedHatAI/gemma-3-4b-it-quantized.w4a16",
        "size": "4B",
        "quant": "W4A16",
        "provider": "RedHatAI",
        "vram_gb": 3.0,
    },
    "4b-ista-gptq": {
        "name": "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        "size": "4B",
        "quant": "GPTQ-4b",
        "provider": "ISTA-DASLab",
        "vram_gb": 2.6,
    },
    # 12B options
    "12b-redhat-w8a8": {
        "name": "RedHatAI/gemma-3-12b-it-quantized.w8a8",
        "size": "12B",
        "quant": "W8A8",
        "provider": "RedHatAI",
        "vram_gb": 14.0,
    },
    "12b-redhat-w4a16": {
        "name": "RedHatAI/gemma-3-12b-it-quantized.w4a16",
        "size": "12B",
        "quant": "W4A16",
        "provider": "RedHatAI",
        "vram_gb": 7.0,
    },
    "12b-ista-gptq": {
        "name": "ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g",
        "size": "12B",
        "quant": "GPTQ-4b",
        "provider": "ISTA-DASLab",
        "vram_gb": 6.6,
    },
}

# Test prompts
PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain quantum computing in simple terms. Be concise.",
    "coding": "Write a Python function to check if a number is prime. Include docstring.",
}


def wait_for_server(url: str, timeout: int = 300) -> bool:
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


def chat_completion(url: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send chat completion request."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
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

    return {
        "elapsed": elapsed,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "tokens_per_sec": data["usage"]["completion_tokens"] / elapsed,
        "content": data["choices"][0]["message"]["content"],
    }


def run_benchmark(url: str, model: str, runs: int = 3, batch_size: int = 4) -> dict:
    """Run full benchmark suite on a model."""
    results = {
        "single": {},
        "batch": None,
        "warmup_time": 0,
    }

    # Warmup run
    print("    Warmup...", end=" ", flush=True)
    warmup_start = time.perf_counter()
    try:
        chat_completion(url, model, "Hello", max_tokens=10)
        results["warmup_time"] = time.perf_counter() - warmup_start
        print(f"done ({results['warmup_time']:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    # Single request benchmarks
    for prompt_key, prompt in PROMPTS.items():
        speeds = []
        tokens_list = []

        for i in range(runs):
            try:
                r = chat_completion(url, model, prompt, max_tokens=200)
                speeds.append(r["tokens_per_sec"])
                tokens_list.append(r["completion_tokens"])
            except Exception as e:
                print(f"    {prompt_key} run {i+1}: FAILED - {e}")
                continue

        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            avg_tokens = sum(tokens_list) / len(tokens_list)
            print(f"    {prompt_key:8s}: {avg_speed:.1f} tok/s (avg {avg_tokens:.0f} tokens)")
            results["single"][prompt_key] = {
                "avg_tokens_per_sec": avg_speed,
                "min": min(speeds),
                "max": max(speeds),
                "runs": len(speeds),
            }

    # Batch benchmark
    print(f"    batch-{batch_size}:  ", end="", flush=True)
    try:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(chat_completion, url, model, PROMPTS["medium"], 128)
                for _ in range(batch_size)
            ]
            batch_results = [f.result() for f in as_completed(futures)]

        elapsed = time.perf_counter() - start
        total_tokens = sum(r["completion_tokens"] for r in batch_results)
        throughput = total_tokens / elapsed

        print(f"{throughput:.1f} tok/s aggregate ({total_tokens} tokens in {elapsed:.1f}s)")
        results["batch"] = {
            "batch_size": batch_size,
            "aggregate_throughput": throughput,
            "total_tokens": total_tokens,
            "elapsed": elapsed,
        }
    except Exception as e:
        print(f"FAILED - {e}")

    return results


def launch_server(model: str, tp: int = 1, port: int = 8000,
                  max_model_len: int = 4096, cuda_graphs: bool = True) -> subprocess.Popen:
    """Launch vLLM server with specified configuration."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1" if tp == 2 else "0"

    if tp == 2:
        env["CUDA_FORCE_P2P_ACCESS"] = "1"
        env["VLLM_SKIP_P2P_CHECK"] = "1"
        env["NCCL_P2P_LEVEL"] = "NVL"
        env["NCCL_BUFF_SIZE"] = "16777216"

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.90",
        "--port", str(port),
    ]

    if cuda_graphs and tp == 2:
        cmd.extend([
            "--disable-custom-all-reduce",
            "--compilation-config",
            '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
        ])
    elif not cuda_graphs:
        cmd.append("--enforce-eager")

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark all Gemma 3 model sizes")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()) + ["all", "1b", "4b", "12b"],
                        default=["all"], help="Models to benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Use existing server")
    parser.add_argument("--no-launch", action="store_true", help="Don't launch servers, use existing")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test")
    parser.add_argument("--output", default="benchmark-sizes.json", help="Output file")
    parser.add_argument("--tp", type=int, choices=[1, 2], default=None,
                        help="Force tensor parallel size (default: auto)")
    args = parser.parse_args()

    # Resolve model selection
    selected = set()
    for m in args.models:
        if m == "all":
            selected.update(MODELS.keys())
        elif m in ["1b", "4b", "12b"]:
            selected.update(k for k in MODELS.keys() if k.startswith(m))
        else:
            selected.add(m)

    print("=" * 70)
    print("GEMMA 3 MODEL SIZE BENCHMARK")
    print("=" * 70)
    print(f"Models to test: {len(selected)}")
    print(f"Runs per prompt: {args.runs}")
    print()

    all_results = {}

    for model_key in sorted(selected):
        model_info = MODELS[model_key]
        model_name = model_info["name"]

        # Determine optimal TP
        if args.tp:
            tp = args.tp
        else:
            # Auto: use TP=2 for 12B W8A8 (needs >24GB), single GPU otherwise
            tp = 2 if model_info["vram_gb"] > 20 else 1

        print("=" * 70)
        print(f"MODEL: {model_key}")
        print(f"  Name: {model_name}")
        print(f"  Size: {model_info['size']}, Quant: {model_info['quant']}")
        print(f"  Provider: {model_info['provider']}")
        print(f"  Est. VRAM: {model_info['vram_gb']:.1f} GB")
        print(f"  Tensor Parallel: {tp}")
        print("-" * 70)

        server_proc = None

        try:
            if not args.no_launch:
                print("  Launching server...")
                server_proc = launch_server(model_name, tp=tp, max_model_len=4096)

                print("  Waiting for server...", end=" ", flush=True)
                if not wait_for_server(args.url, timeout=300):
                    print("TIMEOUT")
                    continue
                print("READY")

            # Run benchmark
            results = run_benchmark(args.url, model_name, runs=args.runs)

            if results:
                all_results[model_key] = {
                    "model": model_info,
                    "config": {"tensor_parallel": tp, "max_model_len": 4096},
                    "results": results,
                }

        except KeyboardInterrupt:
            print("\nInterrupted!")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            if server_proc:
                print("  Stopping server...")
                stop_server(server_proc)
                time.sleep(3)  # Allow GPU memory to clear

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'Size':>5} {'Quant':>8} {'Single':>10} {'Batch':>10}")
    print("-" * 70)

    for key, data in sorted(all_results.items()):
        model = data["model"]
        results = data["results"]

        single = results["single"].get("medium", {}).get("avg_tokens_per_sec", 0)
        batch = results.get("batch", {}).get("aggregate_throughput", 0) if results.get("batch") else 0

        print(f"{key:<20} {model['size']:>5} {model['quant']:>8} {single:>9.1f} {batch:>9.1f}")

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
