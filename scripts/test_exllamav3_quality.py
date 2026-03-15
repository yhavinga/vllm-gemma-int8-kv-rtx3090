#!/usr/bin/env python3
"""ExLlamaV3 quality comparison - Dutch parliamentary summarization."""

import json
import time
import torch
from datetime import datetime

def main():
    print("Loading text...")
    text_path = "/home/yeb/Developer/gemma/data/dutch_parliament_text.txt"
    with open(text_path, "r") as f:
        dutch_text = f.read()
    print(f"Text length: {len(dutch_text):,} characters")

    print("\nLoading ExLlamaV3...")
    from exllamav3 import Model, Config, Cache, Tokenizer, DefaultSampler
    from exllamav3.generator import Generator

    model_dir = "/home/yeb/Developer/gemma/models/gemma-3-27b-it-exl3-4bpw"
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)

    # 64K context (max that fits)
    cache = Cache(model, max_num_tokens=65536)
    model.load(progressbar=True, tensor_p=True)

    tokenizer = Tokenizer(config)
    generator = Generator(model, cache, tokenizer)

    # Use temperature 0.3 to match vLLM test
    sampler = DefaultSampler()
    sampler.temperature = 0.3

    # Same prompt structure as vLLM quality test
    prompt = f"""<start_of_turn>user
Summarize this Dutch parliamentary debate. Include:
1. Main topics discussed
2. Key arguments from each speaker
3. Any questions raised to ministers
4. The overall tone (critical, supportive, neutral)

TEXT:
{dutch_text}

Provide a comprehensive summary in Dutch.<end_of_turn>
<start_of_turn>model
"""

    prompt_tokens = tokenizer.encode(prompt).shape[-1]
    print(f"\nPrompt tokens: {prompt_tokens:,}")

    print("\nGenerating summary...")
    start = time.perf_counter()
    output = generator.generate(prompt, sampler=sampler, max_new_tokens=800, add_bos=True)
    elapsed = time.perf_counter() - start

    # Extract just the model response
    response = output.split("<start_of_turn>model\n")[-1]
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0]

    # Count actual output tokens
    output_tokens = tokenizer.encode(response).shape[-1]
    tokens_per_sec = output_tokens / elapsed

    print(f"\nGeneration complete:")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {tokens_per_sec:.1f} tok/s")

    # Save result
    result = {
        "timestamp": datetime.now().isoformat(),
        "test": "Dutch parliamentary summarization",
        "model": "turboderp/gemma-3-27b-it-exl3-4bpw",
        "backend": "ExLlamaV3",
        "context_size": 65536,
        "input_chars": len(dutch_text),
        "input_tokens": prompt_tokens,
        "temperature": 0.3,
        "content": response,
        "completion_tokens": output_tokens,
        "elapsed": elapsed,
        "tokens_per_sec": tokens_per_sec
    }

    output_path = "/home/yeb/Developer/gemma/results/quality-tests/quality-exllamav3-64k.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")

    print("\n" + "="*60)
    print("SUMMARY OUTPUT:")
    print("="*60)
    print(response)

if __name__ == '__main__':
    main()
