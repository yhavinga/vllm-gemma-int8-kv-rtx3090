#!/usr/bin/env python3
"""ExLlamaV3 long context benchmark for Gemma 3 27B EXL3."""

import sys
import time
import torch

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

    # Load parliamentary text
    text_path = "/home/yeb/Developer/gemma/data/dutch_parliament_text.txt"
    print(f"\nLoading text from: {text_path}")
    with open(text_path, "r") as f:
        dutch_text = f.read()
    print(f"Text length: {len(dutch_text):,} characters")

    print("\nLoading ExLlamaV3...")
    from exllamav3 import Model, Config, Cache, Tokenizer, DefaultSampler
    from exllamav3.generator import Generator

    model_dir = "/home/yeb/Developer/gemma/models/gemma-3-27b-it-exl3-4bpw"

    print(f"\nLoading model from: {model_dir}")
    config = Config.from_directory(model_dir)

    print("Creating model...")
    model = Model.from_config(config)

    # Create cache with larger context for long context testing
    # 128K causes OOM - try 64K
    context_size = 65536
    print(f"Creating cache for {context_size:,} tokens...")
    cache = Cache(model, max_num_tokens=context_size)

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

    # Count tokens in text (ExLlamaV3 tokenizer returns tensor directly)
    text_tokens = tokenizer.encode(dutch_text).shape[-1]
    print(f"Parliamentary text: {text_tokens:,} tokens")

    print("Creating generator...")
    generator = Generator(model, cache, tokenizer)
    sampler = DefaultSampler()

    # Warmup
    print("\nWarmup...")
    warmup_prompt = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
    _ = generator.generate(warmup_prompt, sampler=sampler, max_new_tokens=10, add_bos=True)

    # Test different context lengths
    test_configs = [
        (8000, "8K context"),
        (32000, "32K context"),
        (60000, "~60K context (near max for 64K cache)"),
    ]

    for char_limit, label in test_configs:
        print("\n" + "="*60)
        print(f"BENCHMARK: {label}")
        print("="*60)

        # Truncate text to fit context
        truncated_text = dutch_text[:char_limit]

        # Create prompt
        prompt = f"""<start_of_turn>user
Here is a Dutch parliamentary debate transcript. Please provide a comprehensive summary in English, highlighting the main topics discussed, key arguments made, and any notable exchanges between speakers.

TEXT:
{truncated_text}

Please summarize the above Dutch parliamentary debate in English.<end_of_turn>
<start_of_turn>model
"""

        # Count prompt tokens
        prompt_tokens = tokenizer.encode(prompt).shape[-1]
        print(f"  Prompt: {prompt_tokens:,} tokens ({len(truncated_text):,} chars of text)")

        if prompt_tokens > context_size - 512:
            print(f"  SKIP: Prompt too long for {context_size} context")
            continue

        # Generate
        start = time.perf_counter()
        try:
            output = generator.generate(prompt, sampler=sampler, max_new_tokens=256, add_bos=True)
            elapsed = time.perf_counter() - start

            # Extract just the model's response
            response = output.split("<start_of_turn>model\n")[-1]

            gen_tokens = 256  # Requested
            total_time = elapsed
            prefill_estimate = elapsed - (256 / 44)  # Rough estimate based on ~44 tok/s generation

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Prefill (estimated): {prefill_estimate:.2f}s ({prompt_tokens / max(0.01, prefill_estimate):.0f} tok/s)")
            print(f"  Output preview: {response[:300]}...")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Show final GPU memory
    print("\n" + "="*60)
    print("GPU memory after all tests:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

if __name__ == '__main__':
    main()
