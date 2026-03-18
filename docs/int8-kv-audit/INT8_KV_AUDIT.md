# INT8 KV Cache Audit (vLLM on RTX 3090)

## 1) Direct answer to your main question

No, the implementation does **not** assume one shared K/V range across all layers.

What it does:
- Uses one `k_scale` and one `v_scale` **per attention layer** (per-tensor scale, not per-head).
- Computes those scales from that layer's observed K/V activations when `--calculate-kv-scales` is enabled.
- Applies those scales in Triton KV write/read kernels for int8 quantize/dequantize.

Important nuance:
- Scales are computed once (first calculation pass) and then reused.
- If first-pass K/V is near-zero, code falls back to default `absmax=20` (`scale=20/127=0.15748`).

## 2) What is actually quantized (and what is not)

Your question says "quantize BF16 weights for K/V".

In this implementation, the int8 KV path quantizes **runtime K/V cache activations**, not model projection weights.

Flow:
1. Model computes `K` and `V` in BF16.
2. KV write path stores them as int8 using `round(x / scale)` and clamp.
3. KV read path dequantizes int8 back to BF16-ish values via `int8 * scale` before attention math.
4. Attention matmuls still run in normal precision backend paths.

## 3) Verified implementation details

Validated in installed vLLM code and runtime tests:
- `CacheDType` supports `int8`.
- `is_quantized_kv_cache("int8") == True`.
- Triton backend supports `int8` KV cache.
- Write kernel performs float32 divide + round + clamp + int8 cast.
- Read kernel performs int8->float32 * scale -> Q dtype.
- `calc_kv_scales` uses range 127 for int8 and sets scale once.

## 4) Range behavior evidence

### Code-level behavior

- `k_range`/`v_range` set to 127 when `kv_cache_dtype == 'int8'`.
- `calc_kv_scales` computes:
  - `k_scale = absmax(K) / 127`
  - `v_scale = absmax(V) / 127`
- Then `self.calculate_kv_scales = False` for that layer.

### Measured behavior (live GPU run)

Probe run:
- Model: `RedHatAI/gemma-3-1b-it-quantized.w8a8`
- GPU: RTX 3090
- Config: `kv_cache_dtype=int8`, `calculate_kv_scales=True`

Observed first-pass per-layer scales:
- Layers observed: 26
- `k_scale` min/median/max: `0.0384 / 0.0723 / 0.2470`
- `v_scale` min/median/max: `0.0182 / 0.0608 / 1.0630`
- Unique rounded scales: `k=25`, `v=26`

So in this run, ranges were clearly **not the same** across layers.

Data and plot:
- `data/per_layer_scales_gemma3_1b.csv`
- `plots/per_layer_scales_gemma3_1b.svg`

### 27B TP=2 measured behavior

Probe run:
- Model: `RedHatAI/gemma-3-27b-it-quantized.w4a16`
- GPUs: 2x RTX 3090 (`tensor_parallel_size=2`)
- Config: `kv_cache_dtype=int8`, `calculate_kv_scales=True`

Observed calibrated scales (after patch: do not freeze on fallback):
- Layers observed: 62
- `k_scale` min/median/max: `0.09744 / 0.15551 / 0.56299`
- `v_scale` min/median/max: `0.02042 / 0.75197 / 6.96063`
- Unique rounded scales: `k=51`, `v=59`

Interpretation:
- The implementation is still per-layer in code.
- Warmup still starts at fallback (`20/127`) when K/V are zero, but scales are now updated on first non-zero prompt pass and become layer-specific.

Data and plot:
- `data/per_layer_scales_gemma3_27b_tp2_raw.jsonl`
- `data/per_layer_scales_gemma3_27b_tp2.csv`
- `plots/per_layer_scales_gemma3_27b_tp2.svg`

## 5) Throughput context for PR/thread framing

Using your existing grid files, I consolidated 4B results:
- `data/throughput_4b_configs.csv`
- `plots/throughput_4b_configs.svg`

Highlights from those files:
- TP=2: INT8 is near BF16 (slight ±1-2%).
- DP=2 + INT8 unlocks 32K/64K/128K points with strong gains vs TP=2 at long context.

## 6) Claims audit

### Strongly supported
- INT8 KV cache path is integrated in vLLM attention write/read path.
- KV cache memory is ~2x smaller vs BF16.
- On long context, DP+INT8 throughput gains in your grid are substantial.

### Needs caveat wording
- "Per-tensor symmetric quantization" is correct for integrated vLLM path.
- But your standalone POC script is **per-head** quantization, which is not the same scheme as production path.

### Potentially misleading without qualifier
- "Scale computed once per layer, per forward pass" should be reworded to:
  - "Scale computed during first calculation pass and then reused." 
- If first-pass activations are near-zero, fallback default can dominate and may effectively become fixed-scale behavior.

## 7) Practical risk notes for upstream PR

- First-pass-only calibration is cheap, but can be brittle if first pass is not representative.
- Default fallback (`20/127`) gives coarse resolution near zero (see `data/default_scale_quant_error_examples.csv`; very small magnitudes can have high relative error).
- Production-safe follow-up options:
  - calibration over N warmup batches with real text prompts before serving,
  - defer scale freeze until non-zero K/V observed at least once per layer,
  - periodic scale refresh (bounded cadence),
  - per-head scales (higher metadata cost),
  - percentile-based absmax instead of strict max.

## 8) Repro commands used

```bash
# Sanity/integration test
source venv/bin/activate
python scripts/test_vllm_int8.py

# POC quantization/quality numbers
python scripts/int8_kv_cache_poc.py

# Export per-layer measured scales (this audit)
python /tmp/vllm_scale_probe_csv.py

# Regenerate audit artifacts/plots
python docs/int8-kv-audit/generate_artifacts.py
```
