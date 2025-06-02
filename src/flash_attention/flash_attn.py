"""
Core Flash Attention implementation.
This module implements the Flash Attention algorithm as described in:
"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
by Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def flash_attention(
    q: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    k: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    v: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len]
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    causal: bool = False,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention using the Flash Attention algorithm.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Optional attention mask
        dropout_p: Dropout probability
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
        causal: Whether to use causal attention
        block_size: Size of blocks for tiling
        
    Returns:
        Tuple of (output, attention_weights)
    """
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, k_seq_len, _ = k.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Scale queries
    q = q * scale
    
    # Initialize output and attention weights
    output = torch.zeros_like(q)
    attention_weights = torch.zeros(
        (batch_size, num_heads, q_seq_len, k_seq_len),
        device=q.device,
        dtype=q.dtype
    )
    
    # Compute attention in blocks
    for i in range(0, q_seq_len, block_size):
        i_end = min(i + block_size, q_seq_len)
        
        # Get current block of queries
        qi = q[:, :, i:i_end, :]
        
        # Compute attention scores for current block
        scores = torch.matmul(qi, k.transpose(-2, -1))  # [batch_size, num_heads, block_size, k_seq_len]
        
        if causal:
            # Create causal mask for current block
            # For each position i, we can only attend to positions <= i
            # We need to handle the case where k_seq_len > q_seq_len
            causal_mask = torch.ones(
                (i_end - i, k_seq_len),
                device=q.device,
                dtype=torch.bool
            )
            # For each query position, mask out future key positions
            for j in range(i_end - i):
                # Get the absolute position in the sequence
                abs_pos = i + j
                # Mask out positions after the current position
                causal_mask[j, abs_pos + 1:] = False
            
            # Apply mask to scores for current block
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            if mask.dim() == 2:
                # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask[:, :, i:i_end, :], float('-inf'))
        
        # Compute attention weights for current block
        attn_weights = F.softmax(scores, dim=-1)
        
        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        
        # Compute output for current block
        output[:, :, i:i_end, :] = torch.matmul(attn_weights, v)
        
        # Store attention weights for current block
        attention_weights[:, :, i:i_end, :] = attn_weights
    
    return output, attention_weights

class FlashAttention(nn.Module):
    """
    Flash Attention module that can be used as a drop-in replacement for standard attention.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        causal: bool = False,
        block_size: int = 128,
    ):
        """
        Initialize Flash Attention module.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            dropout_p: Dropout probability
            causal: Whether to use causal attention
            block_size: Size of blocks for tiling
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_p = dropout_p
        self.causal = causal
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Flash Attention.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        return flash_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
            causal=self.causal,
            block_size=self.block_size,
        ) 