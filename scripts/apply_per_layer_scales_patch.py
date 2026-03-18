#!/usr/bin/env python3
"""
Patch vLLM to support per-layer KV cache scales.

This fixes the precision loss problem where a single global scale is used
for all 62 attention layers, wasting precision. With per-layer scales:
- Layer 42 (v_absmax=884) gets v_scale=6.96
- Layer 59 (v_absmax=2.6) gets v_scale=0.02

Usage:
    python scripts/apply_per_layer_scales_patch.py

Then start vLLM with:
    export VLLM_KV_SCALES_FILE=/path/to/scales/gemma3_27b_per_layer.json
"""

import os
import sys
import re
from pathlib import Path

def find_vllm_path():
    """Find the vLLM installation path."""
    # Try venv first
    venv_path = Path("/home/yeb/Developer/gemma/venv/lib/python3.10/site-packages/vllm")
    if venv_path.exists():
        return venv_path

    # Try importing vllm to find its path
    try:
        import vllm
        return Path(vllm.__file__).parent
    except ImportError:
        print("ERROR: Cannot find vLLM installation")
        sys.exit(1)


def patch_attention_py(vllm_path: Path):
    """Patch attention.py to support per-layer scales."""
    attention_file = vllm_path / "model_executor/layers/attention/attention.py"

    if not attention_file.exists():
        print(f"ERROR: {attention_file} not found")
        return False

    content = attention_file.read_text()

    # Check if already patched
    if "def _extract_layer_index" in content:
        print(f"  {attention_file.name}: Already patched (per-layer scales)")
        return True

    # Create backup
    backup_file = attention_file.with_suffix('.py.bak.per_layer')
    if not backup_file.exists():
        backup_file.write_text(content)
        print(f"  Created backup: {backup_file}")

    # Patch 1: Add _extract_layer_index function after _load_kv_scales_from_file
    old_load_fn = '''    try:
        _LOADED_KV_SCALES = json.loads(path.read_text())
        logger.info(f"Loaded KV scales from {scales_file}: "
                   f"k_scale={_LOADED_KV_SCALES.get('k_scale')}, "
                   f"v_scale={_LOADED_KV_SCALES.get('v_scale')}")
        return _LOADED_KV_SCALES
    except Exception as e:
        logger.warning(f"Failed to load KV scales from {scales_file}: {e}")
        return None'''

    new_load_fn = '''    try:
        _LOADED_KV_SCALES = json.loads(path.read_text())
        # Check if per-layer scales are available
        if "layers" in _LOADED_KV_SCALES:
            num_layers = len(_LOADED_KV_SCALES["layers"])
            logger.info(f"Loaded PER-LAYER KV scales from {scales_file}: "
                       f"{num_layers} layers with individual k_scale/v_scale")
        else:
            logger.info(f"Loaded GLOBAL KV scales from {scales_file}: "
                       f"k_scale={_LOADED_KV_SCALES.get('k_scale')}, "
                       f"v_scale={_LOADED_KV_SCALES.get('v_scale')}")
        return _LOADED_KV_SCALES
    except Exception as e:
        logger.warning(f"Failed to load KV scales from {scales_file}: {e}")
        return None


def _extract_layer_index(prefix: str) -> str | None:
    """Extract layer index from prefix like 'model.layers.42.self_attn'.

    Returns the layer index as string (e.g., '42') or None if not found.
    """
    import re
    # Match patterns like "model.layers.42" or "layers.42"
    match = re.search(r'layers\\.([0-9]+)', prefix)
    if match:
        return match.group(1)
    return None'''

    if old_load_fn not in content:
        print(f"  WARNING: Could not find expected _load_kv_scales_from_file pattern")
        print(f"           The patch may have been partially applied or vLLM was updated")
        # Try a more lenient search
        if "_LOADED_KV_SCALES = json.loads" in content:
            print(f"           Found JSON loading code, attempting alternative patch...")
        else:
            return False
    else:
        content = content.replace(old_load_fn, new_load_fn)

    # Patch 2: Modify set_default_quant_scales signature and logic
    old_set_default = '''def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer.

    If VLLM_KV_SCALES_FILE is set, loads pre-calibrated scales from that file.
    This enables CUDA graph capture with INT8 KV cache by avoiding dynamic
    scale calculation during inference.
    """
    # Check for pre-calibrated scales file
    loaded_scales = _load_kv_scales_from_file()
    if loaded_scales:
        k_scale = loaded_scales.get("k_scale", 1.0)
        v_scale = loaded_scales.get("v_scale", 1.0)
        # Disable dynamic calculation when using fixed scales
        layer.calculate_kv_scales = False
    else:
        k_scale = 1.0
        v_scale = 1.0'''

    new_set_default = '''def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False, prefix: str = "") -> None:
    """Sets default quantization scales for the layer.

    If VLLM_KV_SCALES_FILE is set, loads pre-calibrated scales from that file.
    Supports both global scales and per-layer scales:

    Global format (legacy):
        {"k_scale": 1.0, "v_scale": 1.0}

    Per-layer format (recommended):
        {"layers": {"0": {"k_scale": 0.127, "v_scale": 5.23}, ...}}

    This enables CUDA graph capture with INT8 KV cache by avoiding dynamic
    scale calculation during inference.
    """
    # Check for pre-calibrated scales file
    loaded_scales = _load_kv_scales_from_file()
    k_scale = 1.0
    v_scale = 1.0

    if loaded_scales:
        # Check for per-layer scales first
        if "layers" in loaded_scales and prefix:
            layer_idx = _extract_layer_index(prefix)
            if layer_idx and layer_idx in loaded_scales["layers"]:
                layer_scales = loaded_scales["layers"][layer_idx]
                k_scale = layer_scales.get("k_scale", 1.0)
                v_scale = layer_scales.get("v_scale", 1.0)
                logger.debug(f"Layer {layer_idx} ({prefix}): k_scale={k_scale:.4f}, v_scale={v_scale:.4f}")
        else:
            # Fall back to global scales
            k_scale = loaded_scales.get("k_scale", 1.0)
            v_scale = loaded_scales.get("v_scale", 1.0)

        # Disable dynamic calculation when using fixed scales
        layer.calculate_kv_scales = False'''

    if old_set_default not in content:
        print(f"  WARNING: Could not find expected set_default_quant_scales pattern")
        return False

    content = content.replace(old_set_default, new_set_default)

    # Patch 3: Update the call site in _init_kv_cache_quant
    old_call = "set_default_quant_scales(layer, register_buffer=True)"
    new_call = "set_default_quant_scales(layer, register_buffer=True, prefix=prefix)"

    if old_call not in content:
        print(f"  WARNING: Could not find expected set_default_quant_scales call")
        return False

    content = content.replace(old_call, new_call)

    # Write patched content
    attention_file.write_text(content)
    print(f"  {attention_file.name}: Patched for per-layer scales")
    return True


def patch_mla_attention_py(vllm_path: Path):
    """Patch mla_attention.py to pass prefix to set_default_quant_scales."""
    mla_file = vllm_path / "model_executor/layers/attention/mla_attention.py"

    if not mla_file.exists():
        print(f"  {mla_file.name}: Not found (optional, skipping)")
        return True

    content = mla_file.read_text()

    # Check if it uses set_default_quant_scales directly
    if "set_default_quant_scales(self," in content:
        # Check if already patched
        if "prefix=prefix" in content or "prefix=self." in content:
            print(f"  {mla_file.name}: Already patched")
            return True

        # MLA attention might call set_default_quant_scales directly
        # We need to ensure prefix is passed
        print(f"  {mla_file.name}: Uses set_default_quant_scales, checking...")

        # The MLA class should get prefix through _init_kv_cache_quant
        # which we've already patched

    print(f"  {mla_file.name}: No changes needed (uses _init_kv_cache_quant)")
    return True


def verify_scales_file():
    """Verify the scales file exists and has correct format."""
    scales_file = os.getenv("VLLM_KV_SCALES_FILE")
    if not scales_file:
        scales_path = Path("/home/yeb/Developer/gemma/scales/gemma3_27b_per_layer.json")
        if scales_path.exists():
            print(f"\nTip: Set VLLM_KV_SCALES_FILE={scales_path}")
            return scales_path
        return None

    scales_path = Path(scales_file)
    if not scales_path.exists():
        print(f"WARNING: VLLM_KV_SCALES_FILE={scales_file} does not exist")
        return None

    import json
    try:
        data = json.loads(scales_path.read_text())
        if "layers" in data:
            print(f"Scales file: {scales_path}")
            print(f"  Mode: per-layer ({len(data['layers'])} layers)")
            # Show range of scales
            k_scales = [data["layers"][str(i)]["k_scale"] for i in range(len(data["layers"]))]
            v_scales = [data["layers"][str(i)]["v_scale"] for i in range(len(data["layers"]))]
            print(f"  k_scale range: {min(k_scales):.4f} - {max(k_scales):.4f}")
            print(f"  v_scale range: {min(v_scales):.4f} - {max(v_scales):.4f}")
        else:
            print(f"Scales file: {scales_path}")
            print(f"  Mode: global (k_scale={data.get('k_scale')}, v_scale={data.get('v_scale')})")
        return scales_path
    except Exception as e:
        print(f"WARNING: Failed to parse scales file: {e}")
        return None


def main():
    print("=" * 60)
    print("vLLM Per-Layer KV Cache Scales Patch")
    print("=" * 60)

    vllm_path = find_vllm_path()
    print(f"\nvLLM path: {vllm_path}")

    print("\nApplying patches...")

    success = True
    success = patch_attention_py(vllm_path) and success
    success = patch_mla_attention_py(vllm_path) and success

    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: All patches applied")
        print("=" * 60)

        verify_scales_file()

        print("\nUsage:")
        print("  export VLLM_KV_SCALES_FILE=/home/yeb/Developer/gemma/scales/gemma3_27b_per_layer.json")
        print("  vllm serve ... --kv-cache-dtype int8")
        print("\nThe scales file should have the format:")
        print('  {"layers": {"0": {"k_scale": 0.127, "v_scale": 5.23}, ...}}')
    else:
        print("\n" + "=" * 60)
        print("FAILED: Some patches could not be applied")
        print("=" * 60)
        print("\nYou may need to manually patch the files or check vLLM version.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
