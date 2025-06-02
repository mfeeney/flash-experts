"""
Mixture of Experts (MoE) package for efficient model scaling.
"""

from .moe_layer import Router, Expert, MoELayer
from .moe_config import GPT2MoEConfig
from .gpt2_moe import GPT2MoEBlock, GPT2MoEModel

__all__ = [
    'Router',
    'Expert',
    'MoELayer',
    'GPT2MoEConfig',
    'GPT2MoEBlock',
    'GPT2MoEModel',
] 