"""
Flash Attention package for efficient attention computation.
"""

from .flash_attn import FlashAttention
from .gpt2_flash_attn import (
    GPT2FlashAttention,
    GPT2FlashAttentionConfig,
    GPT2FlashAttentionLMHeadModel,
)

__all__ = [
    'FlashAttention',
    'GPT2FlashAttention',
    'GPT2FlashAttentionConfig',
    'GPT2FlashAttentionLMHeadModel',
] 