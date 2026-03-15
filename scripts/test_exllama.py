#!/usr/bin/env python3
"""Quick ExLlamaV2 test for Gemma 3 27B EXL2."""

import sys
import time
import torch

# Check GPU status first
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"         {mem:.1f} GB total")

print("\nLoading ExLlamaV2...")
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator

model_dir = "/home/yeb/Developer/gemma/models/gemma-3-27b-it-exl2-4bpw"

print(f"\nLoading model from: {model_dir}")
config = ExLlamaV2Config(model_dir)
config.max_seq_len = 8192

# Use tensor parallelism across both GPUs
print("Initializing model with TENSOR PARALLELISM...")
model = ExLlamaV2(config)

# Load with tensor parallelism
print("Loading model with tensor parallelism (TP=2)...")
load_start = time.perf_counter()
model.load_tp(progress=True)  # TP mode
load_time = time.perf_counter() - load_start
print(f"Model loaded in {load_time:.1f}s")

# Create cache after TP load
print("Creating cache...")
cache = ExLlamaV2Cache(model, max_seq_len=8192)

# Show GPU memory usage
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"  GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

print("Creating generator...")
generator = ExLlamaV2DynamicGenerator(
    model=model,
    cache=cache,
    tokenizer=tokenizer,
)

# Warmup
print("\nWarmup...")
_ = generator.generate(prompt="Hello", max_new_tokens=10)

# Benchmark
print("\n" + "="*60)
print("BENCHMARK: Single request, 256 tokens")
print("="*60)

prompt = "Explain quantum computing in simple terms that anyone can understand."

runs = 3
times = []
for i in range(runs):
    cache.current_seq_len = 0  # Reset cache
    start = time.perf_counter()
    output = generator.generate(prompt=prompt, max_new_tokens=256)
    elapsed = time.perf_counter() - start
    tok_per_sec = 256 / elapsed
    times.append(tok_per_sec)
    print(f"  Run {i+1}: {tok_per_sec:.1f} tok/s ({elapsed:.2f}s)")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.1f} tok/s")
print(f"\nSample output:\n{output[:500]}...")
