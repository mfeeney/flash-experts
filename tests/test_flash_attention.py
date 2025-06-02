"""
Tests for Flash Attention implementation.
"""

import math
import time
from typing import Tuple

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.flash_attention.flash_attn import flash_attention, FlashAttention
from src.flash_attention.gpt2_flash_attn import GPT2FlashAttention

def create_test_inputs(
    batch_size: int = 2,
    num_heads: int = 12,
    seq_len: int = 512,
    head_dim: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test inputs for attention computation."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    return q, k, v

def test_flash_attention_output_shape():
    """Test that Flash Attention outputs have the correct shape."""
    batch_size, num_heads, seq_len, head_dim = 2, 12, 512, 64
    q, k, v = create_test_inputs(batch_size, num_heads, seq_len, head_dim)
    
    output, attn_weights = flash_attention(q, k, v)
    
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

def test_flash_attention_causal():
    """Test that causal attention works correctly."""
    batch_size, num_heads, seq_len, head_dim = 2, 12, 512, 64
    q, k, v = create_test_inputs(batch_size, num_heads, seq_len, head_dim)
    
    # Compute attention with causal=True
    output, attn_weights = flash_attention(q, k, v, causal=True)
    
    # Check that attention weights are upper triangular
    for b in range(batch_size):
        for h in range(num_heads):
            attn = attn_weights[b, h]
            # Convert to binary mask (0 or 1)
            mask = (attn > 0).float()
            # Check that mask is lower triangular
            assert torch.allclose(
                mask,
                torch.tril(torch.ones_like(mask)),
                rtol=1e-5,
                atol=1e-5
            )

def test_flash_attention_against_standard():
    """Test that Flash Attention gives the same results as standard attention."""
    batch_size, num_heads, seq_len, head_dim = 2, 12, 128, 64  # Use smaller seq_len for standard attention
    q, k, v = create_test_inputs(batch_size, num_heads, seq_len, head_dim)
    
    # Compute standard attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q * scale, k.transpose(-2, -1))
    attn_weights = torch.softmax(scores, dim=-1)
    standard_output = torch.matmul(attn_weights, v)
    
    # Compute Flash Attention
    flash_output, flash_weights = flash_attention(q, k, v)
    
    # Compare outputs
    assert torch.allclose(flash_output, standard_output, rtol=1e-5, atol=1e-5)
    assert torch.allclose(flash_weights, attn_weights, rtol=1e-5, atol=1e-5)

def test_gpt2_flash_attention():
    """Test GPT-2 Flash Attention integration."""
    # Create a small GPT-2 config
    config = GPT2Config(
        n_layer=1,
        n_head=12,
        n_embd=768,
        vocab_size=50257,
    )
    
    # Create model with Flash Attention
    model = GPT2LMHeadModel(config)
    model.attn = GPT2FlashAttention(config)  # Replace standard attention with Flash Attention
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    input_ids = input_ids.to(model.device)
    
    # Test forward pass
    outputs = model(input_ids)
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

def benchmark_attention():
    """Benchmark Flash Attention against standard attention."""
    batch_size, num_heads, seq_len, head_dim = 2, 12, 1024, 64
    q, k, v = create_test_inputs(batch_size, num_heads, seq_len, head_dim)
    
    # Warm up
    for _ in range(3):
        _ = flash_attention(q, k, v)
        torch.cuda.synchronize()
    
    # Benchmark Flash Attention
    start_time = time.time()
    for _ in range(10):
        _ = flash_attention(q, k, v)
        torch.cuda.synchronize()
    flash_time = (time.time() - start_time) / 10
    
    # Benchmark standard attention
    start_time = time.time()
    for _ in range(10):
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        attn_weights = torch.softmax(scores, dim=-1)
        _ = torch.matmul(attn_weights, v)
        torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / 10
    
    print(f"\nBenchmark results (seq_len={seq_len}):")
    print(f"Flash Attention: {flash_time*1000:.2f}ms")
    print(f"Standard Attention: {standard_time*1000:.2f}ms")
    print(f"Speedup: {standard_time/flash_time:.2f}x")

if __name__ == "__main__":
    # Run benchmarks if script is run directly
    if torch.cuda.is_available():
        benchmark_attention()
    else:
        print("CUDA not available, skipping benchmarks") 