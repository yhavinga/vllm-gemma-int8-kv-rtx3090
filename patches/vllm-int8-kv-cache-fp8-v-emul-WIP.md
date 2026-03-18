# WIP Local Patch Record (site-packages edits)

Because `venv/` is gitignored, this file records the exact local edit locations for reproducibility.

## Files touched
- `venv/lib/python3.10/site-packages/vllm/model_executor/layers/attention/attention.py`
- `venv/lib/python3.10/site-packages/vllm/model_executor/layers/attention/mla_attention.py`
- `venv/lib/python3.10/site-packages/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `venv/lib/python3.10/site-packages/vllm/v1/attention/ops/triton_unified_attention.py`

## Key markers to verify
Run:
```bash
rg -n "VLLM_INT8_V_FP8_EMUL|fp8_e4m3_encode_to_int8|fp8_e4m3_decode_from_int8|used_default|v_range_val = 448" \
  venv/lib/python3.10/site-packages/vllm/model_executor/layers/attention/attention.py \
  venv/lib/python3.10/site-packages/vllm/model_executor/layers/attention/mla_attention.py \
  venv/lib/python3.10/site-packages/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py \
  venv/lib/python3.10/site-packages/vllm/v1/attention/ops/triton_unified_attention.py
```

## Patch intent
- Keep K in INT8.
- Optionally encode V as FP8-E4M3 emulation into int8 bytes.
- Preserve first non-fallback real-prompt calibration behavior for per-layer scales.
