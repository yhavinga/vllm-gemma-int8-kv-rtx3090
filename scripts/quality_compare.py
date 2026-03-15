#!/usr/bin/env python3
"""
Quality comparison of quantized Gemma 3 27B models on Dutch text.
Tests summarization of Dutch parliamentary proceedings.
"""

import argparse
import json
import time
from datetime import datetime

import requests

# Load Dutch parliamentary text from file or use default
# Source: https://zoek.officielebekendmakingen.nl/h-ek-19941995-11-382-402.html
# Eerste Kamer debate, December 20, 1994 - Mr. Schinck (PvdA) on tax legislation
import os

_DEFAULT_TEXT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "dutch_parliament_text.txt")

def load_dutch_text(filepath=None, max_chars=None):
    """Load Dutch parliamentary text from file, truncated to fit context.

    Gemma 3 Dutch text ratio: ~3.94 chars/token

    Recommended max_chars by context size:
    - 8K context:   ~28,000 chars (~7K tokens, leaving room for prompt+output)
    - 32K context:  ~110,000 chars (~28K tokens)
    - 128K context: ~390,000 chars (~100K tokens)
    """
    if max_chars is None:
        max_chars = 28000  # Safe default for 8K context

    path = filepath or _DEFAULT_TEXT_FILE
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            if len(text) > max_chars:
                # Truncate at sentence boundary
                text = text[:max_chars]
                last_period = text.rfind('.')
                if last_period > max_chars * 0.8:
                    text = text[:last_period + 1]
            return text
    return None

# Character limits by context size (Gemma 3, ~3.94 chars/token)
CONTEXT_CHAR_LIMITS = {
    8192: 28000,      # ~7K tokens input
    32768: 110000,    # ~28K tokens input
    131072: 390000,   # ~100K tokens input
}

DUTCH_PARLIAMENT_TEXT = load_dutch_text() or """
[Fallback text if file not found]
"""

SUMMARIZATION_PROMPT = """Je bent een ervaren parlementair journalist die gespecialiseerd is in het samenvatten van Kamerdebatten.

Maak een gestructureerde samenvatting van onderstaande parlementaire bijdrage. De samenvatting moet:

1. **Hoofdpunten** (maximaal 4 bullets): De kernpunten van het betoog
2. **Vragen aan de minister** (genummerde lijst): Concrete vragen die gesteld worden
3. **Standpunt fractie** (1-2 zinnen): Het algemene standpunt van de spreker
4. **Toon** (1 woord): De toon van het betoog (bijv. kritisch, constructief, bezorgd)

Gebruik formele, zakelijke taal passend bij parlementaire verslaggeving.

---

PARLEMENTAIRE BIJDRAGE:

{text}

---

SAMENVATTING:"""


def query_model(url: str, model: str, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> dict:
    """Query a vLLM model and return response with timing."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
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
        "content": data["choices"][0]["message"]["content"],
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "elapsed": elapsed,
        "tokens_per_sec": data["usage"]["completion_tokens"] / elapsed,
    }


def run_comparison(url: str, models: list[str], temperature: float = 0.3):
    """Run the same prompt through multiple models and compare."""
    prompt = SUMMARIZATION_PROMPT.format(text=DUTCH_PARLIAMENT_TEXT)

    results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model}")
        print(f"{'='*60}")

        try:
            result = query_model(url, model, prompt, temperature)
            results[model] = result

            print(f"\nTokens: {result['completion_tokens']} in {result['elapsed']:.2f}s ({result['tokens_per_sec']:.1f} tok/s)")
            print(f"\n--- OUTPUT ---\n")
            print(result["content"])
            print(f"\n--- END ---\n")

        except Exception as e:
            print(f"ERROR: {e}")
            results[model] = {"error": str(e)}

    return results


def save_results(results: dict, filename: str, text: str = None, context_size: int = 8192):
    """Save results to JSON file."""
    text_preview = (text[:200] + "...") if text else "N/A"
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "Dutch parliamentary summarization",
        "context_size": context_size,
        "input_chars": len(text) if text else 0,
        "input_tokens_est": int(len(text) / 3.94) if text else 0,
        "temperature": 0.3,
        "input_text_preview": text_preview,
        "results": results,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Compare quantized model quality on Dutch text")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", required=True, help="Model name to test")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature (default: 0.3)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--context", type=int, default=8192, choices=[8192, 32768, 131072],
                        help="Context size: 8192, 32768, or 131072 (default: 8192)")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Max input chars (overrides --context calculation)")
    args = parser.parse_args()

    # Determine max chars based on context size
    max_chars = args.max_chars or CONTEXT_CHAR_LIMITS.get(args.context, 28000)

    # Reload text with appropriate limit
    text = load_dutch_text(max_chars=max_chars)
    estimated_tokens = len(text) / 3.94

    print("=" * 60)
    print("Dutch Parliamentary Text Summarization Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Context size: {args.context:,} tokens")
    print(f"Temperature: {args.temperature}")
    print(f"Input text: {len(text):,} characters (~{int(estimated_tokens):,} tokens)")

    prompt = SUMMARIZATION_PROMPT.format(text=text)

    try:
        result = query_model(args.url, args.model, prompt, args.temperature)

        print(f"\n{'='*60}")
        print(f"RESULT")
        print(f"{'='*60}")
        print(f"Tokens: {result['completion_tokens']} in {result['elapsed']:.2f}s ({result['tokens_per_sec']:.1f} tok/s)")
        print(f"\n{result['content']}")

        if args.output:
            save_results({args.model: result}, args.output, text=text, context_size=args.context)

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        return None


if __name__ == "__main__":
    main()
