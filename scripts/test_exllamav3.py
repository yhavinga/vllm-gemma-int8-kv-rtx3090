#!/usr/bin/env python3
"""ExLlamaV3 benchmark for Gemma 3 27B EXL3."""

import sys
import time
import torch

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

    print("\nLoading ExLlamaV3...")
    from exllamav3 import Model, Config, Cache, Tokenizer, DefaultSampler
    from exllamav3.generator import Generator

    model_dir = "/home/yeb/Developer/gemma/models/gemma-3-27b-it-exl3-4bpw"

    print(f"\nLoading model from: {model_dir}")
    config = Config.from_directory(model_dir)

    print("Creating model...")
    model = Model.from_config(config)

    # Create cache BEFORE loading model (so loader allocates cache tensors)
    print("Creating cache...")
    cache = Cache(model, max_num_tokens=8192)

    # Load with tensor parallelism on 2 GPUs
    print("Loading model (tensor parallel on 2 GPUs)...")
    load_start = time.perf_counter()
    model.load(progressbar=True, tensor_p=True)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Show GPU memory usage
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: {allocated:.1f} GB allocated")

    print("\nCreating tokenizer...")
    tokenizer = Tokenizer(config)

    print("Creating generator...")
    generator = Generator(model, cache, tokenizer)
    sampler = DefaultSampler()

    # Warmup with proper format
    print("\nWarmup...")
    warmup_prompt = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
    _ = generator.generate(warmup_prompt, sampler=sampler, max_new_tokens=10, add_bos=True)

    # Benchmark
    print("\n" + "="*60)
    print("BENCHMARK: Single request, 256 tokens")
    print("="*60)

    # Use proper Gemma 3 chat format
    raw_prompt = "Explain quantum computing in simple terms that anyone can understand."
    prompt = f"<start_of_turn>user\n{raw_prompt}<end_of_turn>\n<start_of_turn>model\n"

    runs = 3
    times = []
    for i in range(runs):
        start = time.perf_counter()
        output = generator.generate(prompt, sampler=sampler, max_new_tokens=256, add_bos=True)
        elapsed = time.perf_counter() - start
        tok_per_sec = 256 / elapsed
        times.append(tok_per_sec)
        print(f"  Run {i+1}: {tok_per_sec:.1f} tok/s ({elapsed:.2f}s)")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.1f} tok/s")

    # Show GPU memory after inference
    print("\nGPU memory after inference:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

    print(f"\nSample output:\n{output[:500]}...")

if __name__ == '__main__':
    main()
