#!/usr/bin/env python3
"""
INT8 KV Cache Proof-of-Concept for RTX 3090 (Ampere)

Tests quality degradation of INT8 quantized KV cache vs BF16 baseline.
Uses per-head symmetric quantization following vLLM's INT8 patterns.

Target metrics:
- Cosine similarity > 0.999
- Max error < 0.01
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
import time


# ============ Triton Kernels for INT8 KV Cache ============

@triton.jit
def _per_head_quant_int8_kernel(
    # Input: [num_tokens, num_heads, head_size] contiguous
    x_ptr,
    # Output: [num_tokens, num_heads, head_size] as int8
    x_q_ptr,
    # Scales: [num_heads] - one scale per head
    scales_ptr,
    # Dimensions
    num_tokens,
    num_heads,
    head_size,
    # Strides
    stride_token,
    stride_head,
    # Meta
    HEAD_SIZE_BLOCK: tl.constexpr,
):
    """
    Quantize KV to INT8 with per-head symmetric scaling.
    Each head gets its own scale based on max(abs(values)) across all tokens.

    This follows vLLM's pattern: scale = absmax / 127
    """
    head_idx = tl.program_id(0)

    # First pass: find absmax across all tokens for this head
    absmax_val = 0.0

    for token_idx in range(num_tokens):
        offs = tl.arange(0, HEAD_SIZE_BLOCK)
        mask = offs < head_size

        ptr = x_ptr + token_idx * stride_token + head_idx * stride_head + offs
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

        # tl.max returns scalar when reducing a block
        block_max = tl.max(tl.abs(x))
        absmax_val = tl.maximum(absmax_val, block_max)

    # Compute scale (avoid div by zero)
    absmax_val = tl.maximum(absmax_val, 1e-10)
    scale = absmax_val / 127.0

    # Store scale for this head (scale is now a scalar)
    tl.store(scales_ptr + head_idx, scale)

    # Second pass: quantize all tokens for this head
    for token_idx in range(num_tokens):
        offs = tl.arange(0, HEAD_SIZE_BLOCK)
        mask = offs < head_size

        ptr_in = x_ptr + token_idx * stride_token + head_idx * stride_head + offs
        ptr_out = x_q_ptr + token_idx * stride_token + head_idx * stride_head + offs

        x = tl.load(ptr_in, mask=mask, other=0.0).to(tl.float32)

        # Quantize: x_q = round(x / scale), clamp to [-128, 127]
        x_q = x / scale
        x_q = tl.extra.cuda.libdevice.round(x_q)
        x_q = tl.maximum(tl.minimum(x_q, 127.0), -128.0).to(tl.int8)

        tl.store(ptr_out, x_q, mask=mask)


def per_head_quant_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8 with per-head symmetric scaling.

    Args:
        x: [num_tokens, num_heads, head_size] in BF16/FP16/FP32

    Returns:
        x_q: [num_tokens, num_heads, head_size] as INT8
        scales: [num_heads] as FP32 (multiply to dequantize)
    """
    assert x.is_contiguous(), "Input must be contiguous"
    num_tokens, num_heads, head_size = x.shape

    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(num_heads, dtype=torch.float32, device=x.device)

    HEAD_SIZE_BLOCK = triton.next_power_of_2(head_size)

    _per_head_quant_int8_kernel[(num_heads,)](
        x, x_q, scales,
        num_tokens, num_heads, head_size,
        x.stride(0), x.stride(1),
        HEAD_SIZE_BLOCK=HEAD_SIZE_BLOCK,
    )

    return x_q, scales


@triton.jit
def _per_head_dequant_int8_kernel(
    # Input: [num_tokens, num_heads, head_size] as int8
    x_q_ptr,
    # Scales: [num_heads]
    scales_ptr,
    # Output: [num_tokens, num_heads, head_size] as target dtype
    x_ptr,
    # Dimensions
    num_tokens,
    num_heads,
    head_size,
    # Strides
    stride_token,
    stride_head,
    # Meta
    HEAD_SIZE_BLOCK: tl.constexpr,
):
    """Dequantize INT8 back to float with per-head scaling."""
    head_idx = tl.program_id(0)
    token_idx = tl.program_id(1)

    scale = tl.load(scales_ptr + head_idx)

    offs = tl.arange(0, HEAD_SIZE_BLOCK)
    mask = offs < head_size

    ptr_in = x_q_ptr + token_idx * stride_token + head_idx * stride_head + offs
    ptr_out = x_ptr + token_idx * stride_token + head_idx * stride_head + offs

    x_q = tl.load(ptr_in, mask=mask, other=0).to(tl.float32)
    x = x_q * scale

    tl.store(ptr_out, x.to(ptr_out.dtype.element_ty), mask=mask)


def per_head_dequant_int8(
    x_q: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Dequantize INT8 tensor with per-head scaling.

    Args:
        x_q: [num_tokens, num_heads, head_size] as INT8
        scales: [num_heads] as FP32
        dtype: Output dtype (default BF16)

    Returns:
        x: [num_tokens, num_heads, head_size] in target dtype
    """
    num_tokens, num_heads, head_size = x_q.shape

    x = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device=x_q.device)

    HEAD_SIZE_BLOCK = triton.next_power_of_2(head_size)

    _per_head_dequant_int8_kernel[(num_heads, num_tokens)](
        x_q, scales, x,
        num_tokens, num_heads, head_size,
        x_q.stride(0), x_q.stride(1),
        HEAD_SIZE_BLOCK=HEAD_SIZE_BLOCK,
    )

    return x


# ============ PyTorch Reference Implementation ============

def per_head_quant_int8_pytorch(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for validation."""
    # x: [num_tokens, num_heads, head_size]
    # Compute absmax per head (across tokens and head_size)
    absmax = x.abs().amax(dim=(0, 2), keepdim=False)  # [num_heads]
    absmax = absmax.clamp(min=1e-10)
    scales = absmax / 127.0  # [num_heads]

    # Quantize
    x_scaled = x / scales.view(1, -1, 1)
    x_q = x_scaled.round().clamp(-128, 127).to(torch.int8)

    return x_q, scales


def per_head_dequant_int8_pytorch(x_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """PyTorch reference dequantization."""
    return x_q.float() * scales.view(1, -1, 1)


# ============ Quality Tests ============

def test_quantization_roundtrip():
    """Test that Triton kernels match PyTorch reference."""
    print("=" * 60)
    print("TEST: Quantization Roundtrip (Triton vs PyTorch)")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"

    # Gemma 3 27B dimensions
    num_tokens = 1024
    num_kv_heads = 16
    head_size = 256

    x = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)

    # PyTorch reference
    x_q_pt, scales_pt = per_head_quant_int8_pytorch(x)
    x_dq_pt = per_head_dequant_int8_pytorch(x_q_pt, scales_pt)

    # Triton implementation
    x_q_tr, scales_tr = per_head_quant_int8(x)
    x_dq_tr = per_head_dequant_int8(x_q_tr, scales_tr)

    # Compare scales
    scale_diff = (scales_pt - scales_tr).abs().max().item()
    print(f"Max scale difference: {scale_diff:.6e}")

    # Compare quantized values
    q_match = (x_q_pt == x_q_tr).float().mean().item()
    print(f"Quantized value match: {q_match * 100:.2f}%")

    # Compare dequantized values
    dq_diff = (x_dq_pt.float() - x_dq_tr.float()).abs().max().item()
    print(f"Max dequantized difference: {dq_diff:.6e}")

    # Roundtrip error
    roundtrip_error = (x.float() - x_dq_tr.float()).abs()
    print(f"Roundtrip max error: {roundtrip_error.max().item():.6f}")
    print(f"Roundtrip mean error: {roundtrip_error.mean().item():.6f}")

    return q_match > 0.99


def test_attention_quality():
    """Compare INT8 vs BF16 attention output quality."""
    print("\n" + "=" * 60)
    print("TEST: Attention Quality (INT8 KV Cache vs BF16)")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"

    # Gemma 3 27B dimensions
    num_kv_heads = 16
    num_query_heads = 32  # GQA: 2 query heads per KV head
    head_size = 256

    # Test multiple sequence lengths
    for num_tokens in [256, 1024, 4096]:
        print(f"\n--- Sequence length: {num_tokens} ---")

        # Generate random K, V, Q with realistic distributions
        # Use smaller range to simulate typical activation magnitudes
        k = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device) * 0.5
        v = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device) * 0.5
        q = torch.randn(1, num_query_heads, head_size, dtype=torch.bfloat16, device=device) * 0.5

        # Expand K, V for GQA (2 query heads share each KV head)
        k_expanded = k.unsqueeze(2).expand(-1, -1, 2, -1).reshape(num_tokens, num_query_heads, head_size)
        v_expanded = v.unsqueeze(2).expand(-1, -1, 2, -1).reshape(num_tokens, num_query_heads, head_size)

        # Baseline: BF16 attention
        attn_bf16 = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1),  # [num_heads, 1, head_size]
            k_expanded.transpose(0, 1),  # [num_heads, num_tokens, head_size]
            v_expanded.transpose(0, 1),  # [num_heads, num_tokens, head_size]
        ).transpose(0, 1)  # [1, num_heads, head_size]

        # INT8 quantized path
        k_q, k_scales = per_head_quant_int8(k)
        v_q, v_scales = per_head_quant_int8(v)

        k_dequant = per_head_dequant_int8(k_q, k_scales)
        v_dequant = per_head_dequant_int8(v_q, v_scales)

        k_dq_expanded = k_dequant.unsqueeze(2).expand(-1, -1, 2, -1).reshape(num_tokens, num_query_heads, head_size)
        v_dq_expanded = v_dequant.unsqueeze(2).expand(-1, -1, 2, -1).reshape(num_tokens, num_query_heads, head_size)

        attn_int8 = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1),
            k_dq_expanded.transpose(0, 1),
            v_dq_expanded.transpose(0, 1),
        ).transpose(0, 1)

        # Compute error metrics
        mse = torch.mean((attn_bf16.float() - attn_int8.float()) ** 2).item()
        max_error = torch.max(torch.abs(attn_bf16.float() - attn_int8.float())).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            attn_bf16.flatten().float(), attn_int8.flatten().float(), dim=0
        ).item()

        # Relative error
        rel_error = (torch.abs(attn_bf16.float() - attn_int8.float()) /
                     (torch.abs(attn_bf16.float()) + 1e-10)).mean().item()

        print(f"  MSE: {mse:.6e}")
        print(f"  Max Error: {max_error:.6f}")
        print(f"  Relative Error: {rel_error:.4%}")
        print(f"  Cosine Similarity: {cosine_sim:.6f}")

        # Check targets
        if cosine_sim < 0.999:
            print(f"  WARNING: Cosine similarity below 0.999 target!")
        if max_error > 0.01:
            print(f"  WARNING: Max error above 0.01 target!")

    return True


def test_memory_savings():
    """Verify memory savings from INT8 vs BF16."""
    print("\n" + "=" * 60)
    print("TEST: Memory Savings")
    print("=" * 60)

    device = "cuda"

    # Simulate 32K context KV cache for Gemma 3 27B
    # 46 layers, 16 KV heads, 256 head size
    num_tokens = 32768
    num_layers = 46
    num_kv_heads = 16
    head_size = 256

    # BF16: 2 bytes per element
    bf16_size = num_tokens * num_layers * num_kv_heads * head_size * 2 * 2  # *2 for K+V
    bf16_size_gb = bf16_size / (1024**3)

    # INT8: 1 byte per element + scales (negligible)
    int8_size = num_tokens * num_layers * num_kv_heads * head_size * 1 * 2  # *2 for K+V
    scales_size = num_layers * num_kv_heads * 4 * 2  # FP32 scales for K+V
    int8_total_gb = (int8_size + scales_size) / (1024**3)

    print(f"Context: {num_tokens:,} tokens")
    print(f"Model: 46 layers, 16 KV heads, 256 head size")
    print(f"")
    print(f"BF16 KV cache: {bf16_size_gb:.2f} GB")
    print(f"INT8 KV cache: {int8_total_gb:.2f} GB")
    print(f"Savings: {bf16_size_gb - int8_total_gb:.2f} GB ({(1 - int8_total_gb/bf16_size_gb)*100:.1f}%)")
    print(f"Scales overhead: {scales_size / 1024:.2f} KB (negligible)")

    # Verify actual allocation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Allocate BF16 cache (single layer for test)
    k_bf16 = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)
    bf16_allocated = torch.cuda.memory_allocated() / (1024**2)

    del k_bf16, v_bf16
    torch.cuda.empty_cache()

    # Allocate INT8 cache
    k_int8 = torch.randint(-128, 127, (num_tokens, num_kv_heads, head_size), dtype=torch.int8, device=device)
    v_int8 = torch.randint(-128, 127, (num_tokens, num_kv_heads, head_size), dtype=torch.int8, device=device)
    k_scales = torch.randn(num_kv_heads, dtype=torch.float32, device=device)
    v_scales = torch.randn(num_kv_heads, dtype=torch.float32, device=device)
    int8_allocated = torch.cuda.memory_allocated() / (1024**2)

    print(f"\nActual allocation (single layer K+V):")
    print(f"  BF16: {bf16_allocated:.1f} MB")
    print(f"  INT8: {int8_allocated:.1f} MB")
    print(f"  Ratio: {bf16_allocated / int8_allocated:.2f}x")


def test_kernel_performance():
    """Benchmark quantization/dequantization kernel performance."""
    print("\n" + "=" * 60)
    print("TEST: Kernel Performance")
    print("=" * 60)

    device = "cuda"
    num_kv_heads = 16
    head_size = 256

    for num_tokens in [1024, 4096, 16384, 32768]:
        x = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            x_q, scales = per_head_quant_int8(x)
            x_dq = per_head_dequant_int8(x_q, scales)
        torch.cuda.synchronize()

        # Benchmark quantization
        start = time.perf_counter()
        for _ in range(100):
            x_q, scales = per_head_quant_int8(x)
        torch.cuda.synchronize()
        quant_time = (time.perf_counter() - start) / 100 * 1000

        # Benchmark dequantization
        start = time.perf_counter()
        for _ in range(100):
            x_dq = per_head_dequant_int8(x_q, scales)
        torch.cuda.synchronize()
        dequant_time = (time.perf_counter() - start) / 100 * 1000

        print(f"Tokens: {num_tokens:>6} | Quant: {quant_time:.3f}ms | Dequant: {dequant_time:.3f}ms")


def main():
    print("INT8 KV Cache Proof-of-Concept")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Triton: {triton.__version__}")
    print()

    # Run all tests
    test_quantization_roundtrip()
    test_attention_quality()
    test_memory_savings()
    test_kernel_performance()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If cosine similarity > 0.999 at all sequence lengths:")
    print("  -> INT8 KV cache is viable for Gemma 3 27B")
    print("  -> Expected 2x memory savings")
    print("  -> Can double context length at same VRAM budget")


if __name__ == "__main__":
    main()
