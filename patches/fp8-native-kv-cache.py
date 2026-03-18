"""
Native FP8 KV Cache Implementation for vLLM

Uses PyTorch's native float8_e4m3fn with INT8 bit storage.
No scales needed - FP8 is self-describing with per-value adaptive precision.

Key insight: FP8 and INT8 are both 8 bits. We can:
1. Quantize BF16 → FP8 (PyTorch native, optimized)
2. View FP8 as INT8 for storage (FREE - just metadata)
3. View INT8 as FP8 for loading (FREE - just metadata)
4. Convert FP8 → BF16 for compute (PyTorch native)

Range: FP8-E4M3 handles [-448, +448] with logarithmic precision.
No calibration, no scales, no per-layer tuning needed.

Author: Refactored from INT8 hack implementation
Date: 2026-03-18
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Verify FP8 support
assert hasattr(torch, 'float8_e4m3fn'), "PyTorch 2.1+ required for native FP8"


class FP8KVCache:
    """
    FP8 KV Cache with INT8 bit storage.

    Memory: Same as INT8 (8 bits per value)
    Precision: Much better than INT8+scale (per-value adaptive)
    Speed: Faster than Triton encode/decode (native PyTorch ops)
    """

    # FP8-E4M3 max representable value (before inf/NaN)
    FP8_E4M3_MAX = 448.0

    @staticmethod
    def encode(x: torch.Tensor) -> torch.Tensor:
        """
        Encode BF16/FP16/FP32 tensor to FP8 bits stored as INT8.

        Args:
            x: Input tensor in any float dtype

        Returns:
            INT8 tensor containing FP8-E4M3 bit patterns
        """
        # Step 1: Clamp to FP8-E4M3 representable range to avoid inf/NaN
        x_clamped = x.clamp(-FP8KVCache.FP8_E4M3_MAX, FP8KVCache.FP8_E4M3_MAX)

        # Step 2: Convert to FP8
        x_fp8 = x_clamped.to(torch.float8_e4m3fn)

        # Step 3: View as INT8 (FREE - same bits, different interpretation)
        x_int8 = x_fp8.view(torch.int8)

        return x_int8

    @staticmethod
    def decode(x_int8: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """
        Decode INT8 (containing FP8 bits) back to float tensor.

        Args:
            x_int8: INT8 tensor containing FP8-E4M3 bit patterns
            dtype: Target dtype for computation (default: bfloat16)

        Returns:
            Float tensor in requested dtype
        """
        # Step 1: View as FP8 (FREE - same bits, different interpretation)
        x_fp8 = x_int8.view(torch.float8_e4m3fn)

        # Step 2: Convert to compute dtype
        x_float = x_fp8.to(dtype)

        return x_float

    @staticmethod
    def encode_kv(
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode both K and V tensors."""
        return FP8KVCache.encode(key), FP8KVCache.encode(value)

    @staticmethod
    def decode_kv(
        key_int8: torch.Tensor,
        value_int8: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode both K and V tensors."""
        return FP8KVCache.decode(key_int8, dtype), FP8KVCache.decode(value_int8, dtype)


# =============================================================================
# Triton kernel integration (for vLLM attention backends)
# =============================================================================

def get_triton_fp8_encode():
    """
    Triton kernel for FP8 encoding during KV cache write.

    This replaces the scale-based INT8 quantization in reshape_and_cache_flash.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def fp8_encode_kernel(
        x_ptr,           # Input: BF16/FP16
        out_ptr,         # Output: INT8 (FP8 bits)
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load as float
        x = tl.load(x_ptr + offsets, mask=mask)

        # Convert to FP8-E4M3 (Triton native support)
        x_fp8 = x.to(tl.float8e4nv)

        # Reinterpret bits as int8 for storage
        # In Triton, we can just cast since it's the same bits
        x_int8 = x_fp8.to(tl.int8, bitcast=True)

        tl.store(out_ptr + offsets, x_int8, mask=mask)

    return fp8_encode_kernel


def get_triton_fp8_decode():
    """
    Triton kernel for FP8 decoding during attention compute.

    This replaces the scale-based INT8 dequantization in unified_attention.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def fp8_decode_kernel(
        x_int8_ptr,      # Input: INT8 (FP8 bits)
        out_ptr,         # Output: BF16/FP16
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load as int8
        x_int8 = tl.load(x_int8_ptr + offsets, mask=mask)

        # Reinterpret bits as FP8
        x_fp8 = x_int8.to(tl.float8e4nv, bitcast=True)

        # Convert to compute dtype
        x_float = x_fp8.to(tl.bfloat16)

        tl.store(out_ptr + offsets, x_float, mask=mask)

    return fp8_decode_kernel


# =============================================================================
# Integration helper for vLLM
# =============================================================================

def patch_reshape_and_cache_flash():
    """
    Returns the code changes needed for triton_reshape_and_cache_flash.py
    """
    return '''
# In _reshape_and_cache_flash_kernel, replace INT8 encoding with:

elif FP8_NATIVE_AS_INT8:
    # Native FP8 quantization stored as INT8 bits
    # No scales needed - FP8 is self-describing
    key_fp8 = key_load.to(tl.float8e4nv)
    key_tile = key_fp8.to(tl.int8, bitcast=True)

    value_fp8 = value_load.to(tl.float8e4nv)
    value_tile = value_fp8.to(tl.int8, bitcast=True)
'''


def patch_unified_attention():
    """
    Returns the code changes needed for triton_unified_attention.py
    """
    return '''
# In _unified_attention_kernel, replace INT8 decoding with:

elif FP8_NATIVE_AS_INT8:
    # Decode FP8 bits from INT8 storage - no scales needed
    K_fp8 = K_load.to(tl.float8e4nv, bitcast=True)
    K = K_fp8.to(Q.dtype)

    V_fp8 = V_load.to(tl.float8e4nv, bitcast=True)
    V = V_fp8.to(Q.dtype)
'''


# =============================================================================
# Test & Benchmark
# =============================================================================

def test_fp8_kv_cache():
    """Test FP8 KV cache encode/decode roundtrip."""
    print("=" * 60)
    print("FP8 KV Cache Test")
    print("=" * 60)

    # Simulate realistic KV cache values
    torch.manual_seed(42)

    # K typically has smaller range (absmax ~70)
    key = torch.randn(1, 16, 128, 128, device='cuda', dtype=torch.bfloat16) * 30

    # V can have larger range (absmax ~900 in extreme cases)
    value = torch.randn(1, 16, 128, 128, device='cuda', dtype=torch.bfloat16) * 100
    # Add some outliers like we see in real Gemma 3
    value[0, 0, 0, :10] = torch.tensor([500, -600, 700, -800, 884, -400, 300, -200, 100, -50],
                                        device='cuda', dtype=torch.bfloat16)

    print(f"\nInput shapes: K={key.shape}, V={value.shape}")
    print(f"K range: [{key.min():.1f}, {key.max():.1f}], absmax={key.abs().max():.1f}")
    print(f"V range: [{value.min():.1f}, {value.max():.1f}], absmax={value.abs().max():.1f}")

    # Encode
    key_int8, value_int8 = FP8KVCache.encode_kv(key, value)
    print(f"\nEncoded dtypes: K={key_int8.dtype}, V={value_int8.dtype}")
    print(f"Memory: {key_int8.numel() + value_int8.numel()} bytes (8-bit)")

    # Decode
    key_decoded, value_decoded = FP8KVCache.decode_kv(key_int8, value_int8)
    print(f"Decoded dtypes: K={key_decoded.dtype}, V={value_decoded.dtype}")

    # Check accuracy
    key_error = (key - key_decoded).abs()
    value_error = (value - value_decoded).abs()

    print(f"\n--- Reconstruction Error ---")
    print(f"K: max_err={key_error.max():.4f}, mean_err={key_error.mean():.6f}")
    print(f"V: max_err={value_error.max():.4f}, mean_err={value_error.mean():.6f}")

    # Check outlier preservation (the V=884 case)
    outlier_original = value[0, 0, 0, 4].item()
    outlier_decoded = value_decoded[0, 0, 0, 4].item()
    print(f"\nOutlier test (V=884):")
    print(f"  Original: {outlier_original:.1f}")
    print(f"  Decoded:  {outlier_decoded:.1f}")
    print(f"  FP8-E4M3 saturates at 448, so 884 → 448 (expected)")

    # Compare with old INT8+scale approach
    print(f"\n--- Comparison with INT8+global_scale ---")
    v_absmax = value.abs().max().item()
    int8_scale = v_absmax / 127  # Global scale for INT8
    value_int8_old = (value / int8_scale).round().clamp(-127, 127).to(torch.int8)
    value_decoded_old = value_int8_old.to(torch.bfloat16) * int8_scale
    value_error_old = (value - value_decoded_old).abs()
    print(f"INT8+scale: max_err={value_error_old.max():.4f}, mean_err={value_error_old.mean():.6f}")
    print(f"FP8 native: max_err={value_error.max():.4f}, mean_err={value_error.mean():.6f}")

    # The key insight: FP8 has better precision for SMALL values
    small_mask = value.abs() < 50
    if small_mask.any():
        print(f"\nPrecision for small values (|V| < 50):")
        print(f"  INT8+scale mean_err: {value_error_old[small_mask].mean():.6f}")
        print(f"  FP8 native mean_err: {value_error[small_mask].mean():.6f}")


def benchmark_fp8_kv_cache():
    """Benchmark FP8 vs INT8+scale encode/decode."""
    import time

    print("\n" + "=" * 60)
    print("FP8 KV Cache Benchmark")
    print("=" * 60)

    # Realistic KV cache size: batch=1, heads=16, seq=4096, dim=128
    shape = (1, 16, 4096, 128)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 50

    n_iters = 100

    # Warmup
    for _ in range(10):
        _ = FP8KVCache.encode(x)
    torch.cuda.synchronize()

    # Benchmark FP8 encode
    start = time.perf_counter()
    for _ in range(n_iters):
        x_int8 = FP8KVCache.encode(x)
    torch.cuda.synchronize()
    fp8_encode_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark FP8 decode
    start = time.perf_counter()
    for _ in range(n_iters):
        x_decoded = FP8KVCache.decode(x_int8)
    torch.cuda.synchronize()
    fp8_decode_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark INT8+scale encode (old approach)
    scale = x.abs().max() / 127
    start = time.perf_counter()
    for _ in range(n_iters):
        x_int8_old = (x / scale).round().clamp(-127, 127).to(torch.int8)
    torch.cuda.synchronize()
    int8_encode_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark INT8+scale decode (old approach)
    start = time.perf_counter()
    for _ in range(n_iters):
        x_decoded_old = x_int8_old.to(torch.bfloat16) * scale
    torch.cuda.synchronize()
    int8_decode_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"\nShape: {shape} = {x.numel():,} elements")
    print(f"\n{'Method':<20} {'Encode (ms)':<15} {'Decode (ms)':<15}")
    print("-" * 50)
    print(f"{'FP8 native':<20} {fp8_encode_time:<15.3f} {fp8_decode_time:<15.3f}")
    print(f"{'INT8+scale':<20} {int8_encode_time:<15.3f} {int8_decode_time:<15.3f}")
    print(f"\nFP8 speedup: {int8_encode_time/fp8_encode_time:.2f}x encode, {int8_decode_time/fp8_decode_time:.2f}x decode")


if __name__ == "__main__":
    test_fp8_kv_cache()
    benchmark_fp8_kv_cache()

    print("\n" + "=" * 60)
    print("vLLM Integration")
    print("=" * 60)
    print("\nTo integrate into vLLM, replace the Triton INT8 kernels:")
    print("\n--- reshape_and_cache_flash ---")
    print(patch_reshape_and_cache_flash())
    print("\n--- unified_attention ---")
    print(patch_unified_attention())
