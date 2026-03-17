#!/usr/bin/env python3
"""
Test vLLM INT8 KV cache modifications.

This script tests if the INT8 KV cache dtype is recognized and
if the basic infrastructure works.
"""

import sys
sys.path.insert(0, '/home/yeb/Developer/gemma/venv/lib/python3.10/site-packages')

def test_dtype_config():
    """Test that INT8 is recognized in config."""
    print("Testing dtype configuration...")

    from vllm.config.cache import CacheDType
    from typing import get_args

    valid_dtypes = get_args(CacheDType)
    print(f"Valid cache dtypes: {valid_dtypes}")

    assert "int8" in valid_dtypes, "int8 not in CacheDType!"
    print("✓ int8 in CacheDType")


def test_dtype_mapping():
    """Test that INT8 maps to torch.int8."""
    print("\nTesting dtype mapping...")

    import torch
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

    print(f"STR_DTYPE_TO_TORCH_DTYPE['int8'] = {STR_DTYPE_TO_TORCH_DTYPE.get('int8')}")

    assert STR_DTYPE_TO_TORCH_DTYPE.get('int8') == torch.int8, "int8 doesn't map to torch.int8!"
    print("✓ int8 maps to torch.int8")


def test_quantized_check():
    """Test is_quantized_kv_cache function."""
    print("\nTesting is_quantized_kv_cache...")

    from vllm.v1.attention.backend import is_quantized_kv_cache

    assert is_quantized_kv_cache("int8") == True, "int8 not recognized as quantized!"
    assert is_quantized_kv_cache("auto") == False
    assert is_quantized_kv_cache("fp8") == True
    print("✓ is_quantized_kv_cache recognizes int8")


def test_triton_backend_support():
    """Test that Triton backend supports INT8."""
    print("\nTesting Triton backend support...")

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    print(f"Supported dtypes: {TritonAttentionBackend.supported_kv_cache_dtypes}")

    assert "int8" in TritonAttentionBackend.supported_kv_cache_dtypes, \
        "int8 not in TritonAttentionBackend.supported_kv_cache_dtypes!"
    print("✓ Triton backend supports int8")


def test_kernel_compilation():
    """Test that the modified kernels compile."""
    print("\nTesting kernel compilation...")

    import torch

    # Import the modified kernel
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash
    )

    # Create dummy inputs
    num_tokens = 4
    num_heads = 2
    head_size = 64
    block_size = 16
    num_blocks = 2

    device = "cuda"

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device)

    # INT8 cache
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.int8, device=device)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.int8, device=device)

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Scale = absmax / 127 (typical for BF16 range)
    k_scale = torch.tensor(0.01, dtype=torch.float32, device=device)  # Assuming inputs are in ~[-1.27, 1.27]
    v_scale = torch.tensor(0.01, dtype=torch.float32, device=device)

    try:
        triton_reshape_and_cache_flash(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype="int8",
            k_scale=k_scale,
            v_scale=v_scale,
        )
        print("✓ Reshape and cache kernel compiles and runs")
        print(f"  Key cache dtype: {key_cache.dtype}")
        print(f"  Key cache range: [{key_cache.min().item()}, {key_cache.max().item()}]")
    except Exception as e:
        print(f"✗ Kernel failed: {e}")
        raise


def main():
    print("=" * 60)
    print("vLLM INT8 KV Cache Integration Tests")
    print("=" * 60)

    test_dtype_config()
    test_dtype_mapping()
    test_quantized_check()
    test_triton_backend_support()
    test_kernel_compilation()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
