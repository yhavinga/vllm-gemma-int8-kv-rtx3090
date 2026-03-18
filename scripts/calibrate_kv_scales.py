#!/usr/bin/env python3
"""One-time KV scale calibration helper for patched INT8 vLLM servers.

Sends a few real prompts (e.g. parliamentary text chunks) to trigger non-fallback
k/v scale updates before normal traffic.
"""

import argparse
import json
import textwrap
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_prompts_from_file(path: Path, num_chunks: int, chunk_chars: int) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    prompts: list[str] = []
    i = 0
    while len(prompts) < num_chunks and i < len(text):
        chunk = text[i : i + chunk_chars].strip()
        if chunk:
            prompts.append(
                "Calibrate KV ranges. Read and summarize in 3 bullets:\n\n" + chunk
            )
        i += chunk_chars
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--text-file",
        default="data/dutch_parliament_text.txt",
        help="Calibration corpus file",
    )
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--chunk-chars", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    prompts = make_prompts_from_file(Path(args.text_file), args.chunks, args.chunk_chars)
    prompts += [
        "Write a tiny Python function that reverses a string.",
        "Give 5 concise facts about the European Union budget process.",
    ]

    print(f"Sending {len(prompts)} calibration prompts to {args.url}")
    for idx, prompt in enumerate(prompts, 1):
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
        out = post_json(args.url, payload)
        choice = out.get("choices", [{}])[0]
        text = (choice.get("text") or "").strip().replace("\n", " ")
        print(f"[{idx}/{len(prompts)}] ok | sample: {text[:90]}")

    print("Calibration requests complete.")
    print(
        textwrap.dedent(
            """
            Note:
            - Start vLLM with --kv-cache-dtype int8 --calculate-kv-scales.
            - Enable V-only FP8 emulation with: VLLM_INT8_V_FP8_EMUL=1
            - With the patched logic, scales remain update-enabled until a non-zero
              K/V pass is observed, then they freeze.
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
