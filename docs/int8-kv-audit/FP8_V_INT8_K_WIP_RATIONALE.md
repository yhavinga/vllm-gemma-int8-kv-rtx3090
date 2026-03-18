# WIP Rationale: V-only FP8-emul + K INT8 (vLLM local patch)

## Goal
Reduce KV-cache quantization error on V (which showed broader/heavier tails on 27B) while preserving the existing INT8 K path and memory footprint.

## Design Summary
- K cache path stays symmetric INT8 (`round(x / k_scale)`, clamp to `[-128,127]`).
- V cache path uses software-emulated FP8 E4M3 byte encoding, stored in the same `int8` cache tensor.
- Dequant path decodes those bytes in Triton and multiplies by `v_scale`.
- Runtime toggle: `VLLM_INT8_V_FP8_EMUL=1`.

## Why This Split
- K quantization noise directly impacts attention logits and ranking stability; keeping K in standard symmetric INT8 avoids introducing a second numeric format in logits path.
- V quantization noise mostly impacts mixed value aggregation; FP8-like code points (E4M3) can better allocate dynamic range for heavy-tailed V activations than linear INT8 at equal storage.
- Storage cost remains 8-bit per element for both K and V.

## Local vLLM Changes (installed package)
These edits were made in local site-packages (not tracked by this repo's git):

- `venv/lib/python3.10/site-packages/vllm/model_executor/layers/attention/attention.py`
- `venv/lib/python3.10/site-packages/vllm/model_executor/layers/attention/mla_attention.py`
- `venv/lib/python3.10/site-packages/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `venv/lib/python3.10/site-packages/vllm/v1/attention/ops/triton_unified_attention.py`

## Behavior Changes
1. `set_default_quant_scales`:
- For `kv_cache_dtype=int8`, use `k_range=127`.
- Use `v_range=448` when `VLLM_INT8_V_FP8_EMUL=1`, else `127`.

2. `calc_kv_scales` (Attention + MLA):
- Keep existing fallback default (`absmax=20`) for near-zero warmup passes.
- Do **not** freeze `calculate_kv_scales` on fallback observations.
- Freeze only after at least one non-fallback pass, so per-layer scales can calibrate on real prompts.

3. KV write kernel:
- K: standard INT8 quantization.
- V: when env-flag enabled, normalize by `v_scale` and encode to FP8-E4M3 byte code (`int8` carrier).

4. KV read kernel:
- K: standard INT8 dequant (`int8 * k_scale`).
- V: FP8-E4M3 decode from int8 carrier, then multiply by `v_scale`.

## Expected Tradeoff
- Accuracy/quality: potentially better for V-heavy error modes vs pure INT8 V.
- Throughput: small overhead from encode/decode logic in Triton.
- Memory: unchanged vs INT8 KV.

## Intended Usage
```bash
export VLLM_INT8_V_FP8_EMUL=1
vllm serve RedHatAI/gemma-3-27b-it-quantized.w4a16 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype int8 \
  --calculate-kv-scales
```

Optional one-time calibration:
```bash
python scripts/calibrate_kv_scales.py \
  --model RedHatAI/gemma-3-27b-it-quantized.w4a16 \
  --text-file data/dutch_parliament_text.txt
```

## WIP Status
This is an experimental local patch. The next section in `INT8_KV_AUDIT.md` should be updated with:
- 27B long-context quality sanity check output
- benchmark numbers for INT8 baseline vs V-FP8-emul
