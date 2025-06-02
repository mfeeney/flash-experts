"""
GPT-2 integration with Flash Attention.
This module provides a modified version of GPT-2's attention mechanism
that uses Flash Attention for improved efficiency.
"""

from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel
)
from transformers.modeling_utils import PreTrainedModel

from .flash_attn import FlashAttention

class GPT2FlashAttentionConfig(GPT2Config):
    """Configuration class for GPT-2 with Flash Attention."""
    
    model_type = "gpt2_flash_attn"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_flash_attention = True

class GPT2FlashAttention(GPT2Attention):
    """
    Modified GPT-2 attention that uses Flash Attention.
    This class inherits from GPT2Attention but replaces the standard attention
    computation with Flash Attention.
    """
    
    config_class = GPT2FlashAttentionConfig
    
    def __init__(self, config, is_cross_attention=False):
        """
        Initialize GPT-2 Flash Attention.
        
        Args:
            config: GPT-2 configuration
            is_cross_attention: Whether this is a cross-attention layer
        """
        super().__init__(config, is_cross_attention)
        
        # Initialize attention parameters from config
        self.num_attention_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_attention_heads
        
        # Replace standard attention with Flash Attention
        self.flash_attn = FlashAttention(
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            dropout_p=config.attn_pdrop,
            causal=not is_cross_attention,  # Use causal attention for self-attention
            block_size=128,  # Can be tuned based on hardware
        )
        
        # Verify that the model has the required layers
        if not hasattr(self, 'c_attn'):
            raise ValueError("GPT2FlashAttention requires c_attn layer from GPT2Attention")
        if not hasattr(self, 'c_proj'):
            raise ValueError("GPT2FlashAttention requires c_proj layer from GPT2Attention")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the attention layers."""
        # Initialize c_attn weights
        if hasattr(self.c_attn, 'weight'):
            self.c_attn.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(self.c_attn, 'bias') and self.c_attn.bias is not None:
                self.c_attn.bias.data.zero_()
        
        # Initialize c_proj weights
        if hasattr(self.c_proj, 'weight'):
            self.c_proj.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(self.c_proj, 'bias') and self.c_proj.bias is not None:
                self.c_proj.bias.data.zero_()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained model and convert its attention layers to Flash Attention."""
        config = kwargs.pop("config", None)
        if config is None:
            config = GPT2FlashAttentionConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Load the base model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        
        # Convert attention layers to Flash Attention
        if hasattr(model, "transformer"):
            for layer in model.transformer.h:
                if isinstance(layer.attn, GPT2Attention):
                    # Create new Flash Attention layer with same weights
                    flash_attn = cls(config)
                    flash_attn.load_state_dict(layer.attn.state_dict())
                    layer.attn = flash_attn
        
        return model
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_attention_heads, head_dim).
        
        Args:
            tensor: Input tensor of shape [batch_size, seq_len, num_attention_heads * head_dim]
            
        Returns:
            Tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return tensor.transpose(1, 2)  # [batch_size, num_attention_heads, seq_len, head_dim]

    def _attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention using Flash Attention.
        This overrides the standard attention computation in GPT2Attention.
        
        Args:
            query: Query tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
            value: Value tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            
        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention using Flash Attention
        attn_output, attn_weights = self.flash_attn(
            q=query,
            k=key,
            v=value,
            mask=attention_mask,
        )
        
        # Reshape output back to expected shape
        batch_size = query.shape[0]
        attn_output = attn_output.reshape(batch_size, -1, self.embed_dim)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_output = attn_output * head_mask
        
        return attn_output, attn_weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        past_key_value: Optional[Union[Tuple[torch.Tensor, torch.Tensor], "DynamicCache"]] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of GPT-2 Flash Attention.
        This overrides the forward method in GPT2Attention to use Flash Attention.
        
        Args:
            hidden_states: Input hidden states
            layer_past: Optional past key/value states (deprecated, use past_key_value)
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            encoder_attention_mask: Optional encoder attention mask
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            past_key_value: Optional past key/value states (can be tuple or DynamicCache)
            cache_position: Optional tensor indicating positions in the cache
            **kwargs: Additional arguments for future compatibility
            
        Returns:
            Tuple containing:
            - attention output
            - present key/value states (if use_cache)
            - attention weights (if output_attentions)
        """
        # Handle both layer_past and past_key_value for compatibility
        past = past_key_value if past_key_value is not None else layer_past
        
        # Project query, key, value using c_attn
        qkv = self.c_attn(hidden_states)
        all_head_size = self.num_attention_heads * self.head_dim
        q, k, v = qkv.split(all_head_size, dim=2)
        
        # Split into heads
        query = self._split_heads(q)
        key = self._split_heads(k)
        value = self._split_heads(v)
        
        if past is not None:
            # Handle both DynamicCache and tuple-based caching
            if hasattr(past, "get_seq_length"):  # DynamicCache
                # For DynamicCache, we need to update the cache with the new key/value
                # and get the updated cache for the current layer
                if use_cache:
                    past.update(key, value, layer_idx=0)
                # Get the current layer's cache
                past_key = past.key_cache[0] if past.key_cache else None
                past_value = past.value_cache[0] if past.value_cache else None
                
                if past_key is not None and past_value is not None:
                    if cache_position is not None:
                        # For DynamicCache, we don't need to handle cache_position
                        # as it's managed internally by the cache
                        key = torch.cat((past_key, key), dim=-2)
                        value = torch.cat((past_value, value), dim=-2)
                    else:
                        key = torch.cat((past_key, key), dim=-2)
                        value = torch.cat((past_value, value), dim=-2)
            else:  # Tuple-based caching
                past_key, past_value = past
                if cache_position is not None:
                    # Ensure cache_position is on the same device
                    cache_position = cache_position.to(past_key.device)
                    # Use cache_position to select the relevant past states
                    past_key = past_key[:, :, cache_position, :]
                    past_value = past_value[:, :, cache_position, :]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
        
        if use_cache:
            if hasattr(past, "get_seq_length"):  # DynamicCache
                present = past  # Return the updated cache object
            else:
                present = (key, value)
        else:
            present = None
        
        # Compute attention
        attn_output, attn_weights = self._attn(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        
        # Project output
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs

class GPT2FlashAttentionLMHeadModel(GPT2LMHeadModel):
    """
    GPT-2 model with Flash Attention for language modeling.
    This class inherits from GPT2LMHeadModel but uses Flash Attention in its transformer layers.
    """
    
    config_class = GPT2FlashAttentionConfig
    
    def __init__(self, config):
        """Initialize GPT-2 with Flash Attention."""
        super().__init__(config)
        
        # Replace standard attention with Flash Attention in all transformer layers
        for layer in self.transformer.h:
            layer.attn = GPT2FlashAttention(config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained model and convert it to use Flash Attention."""
        config = kwargs.pop("config", None)
        if config is None:
            config = GPT2FlashAttentionConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Load the base model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        
        # Convert to Flash Attention model
        flash_model = cls(config)
        flash_model.load_state_dict(model.state_dict())
        
        return flash_model
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_attention_heads, head_dim).
        
        Args:
            tensor: Input tensor of shape [batch_size, seq_len, num_attention_heads * head_dim]
            
        Returns:
            Tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return tensor.transpose(1, 2)  # [batch_size, num_attention_heads, seq_len, head_dim]

    def _attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention using Flash Attention.
        This overrides the standard attention computation in GPT2Attention.
        
        Args:
            query: Query tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
            value: Value tensor of shape [batch_size, num_attention_heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            
        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention using Flash Attention
        attn_output, attn_weights = self.flash_attn(
            q=query,
            k=key,
            v=value,
            mask=attention_mask,
        )
        
        # Reshape output back to expected shape
        batch_size = query.shape[0]
        attn_output = attn_output.reshape(batch_size, -1, self.embed_dim)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_output = attn_output * head_mask
        
        return attn_output, attn_weights