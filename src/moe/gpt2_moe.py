"""
GPT-2 model with Mixture of Experts (MoE) implementation.
"""

from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .moe_layer import MoELayer
from .moe_config import GPT2MoEConfig

class GPT2MoEBlock(nn.Module):
    """
    GPT-2 block with optional MoE layer.
    """
    def __init__(
        self,
        config: GPT2MoEConfig,
        layer_idx: int,
    ):
        """
        Initialize GPT-2 block with optional MoE layer.
        
        Args:
            config: Model configuration
            layer_idx: Index of this layer in the transformer
        """
        super().__init__()
        
        # Get the base GPT-2 block
        base_config = GPT2Config.from_pretrained("gpt2")
        base_config.update({
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_positions": config.n_positions,
            "n_ctx": config.n_ctx,
            "activation_function": "gelu",  # Use standard GELU
        })
        base_model = GPT2LMHeadModel(base_config)
        self.block = base_model.transformer.h[layer_idx]
        
        # Replace feed-forward with MoE if needed
        if (layer_idx + 1) % config.moe_layer_freq == 0:
            self.moe = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.n_inner or 4 * config.hidden_size,
                num_experts=config.num_experts,
                k=config.k,
                router_bias=config.router_bias,
                router_jitter_noise=config.router_jitter_noise,
                activation_fn="gelu",  # Use standard GELU
                dropout=config.resid_pdrop,
                use_aux_loss=config.use_aux_loss,
            )
            # Remove the original feed-forward network
            del self.block.mlp
        else:
            self.moe = None
        
        # Initialize moe_outputs as None
        self.moe_outputs = None
    
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
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the block.
        
        Args:
            hidden_states: Input hidden states
            layer_past: Optional past key/value states (deprecated, use past_key_value)
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            encoder_hidden_states: Optional encoder hidden states
            encoder_attention_mask: Optional encoder attention mask
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            past_key_value: Optional past key/value states
            cache_position: Optional cache position tensor
            
        Returns:
            Tuple containing:
            - Output hidden states
            - Present key/value states (if use_cache)
            - Attention weights (if output_attentions)
            - MoE auxiliary loss (if using MoE)
        """
        # Reset moe_outputs at the start of each forward pass
        self.moe_outputs = None
        
        # Handle past_key_value (new) vs layer_past (old) naming
        if past_key_value is not None:
            layer_past = past_key_value
        
        # Get attention outputs
        attn_outputs = self.block.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Get attention output and present state
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        
        # Apply MoE if present
        if self.moe is not None:
            # Add attention output to hidden states
            hidden_states = hidden_states + attn_output
            
            # Apply MoE
            moe_output, moe_outputs = self.moe(hidden_states, training=self.training)
            
            # Store MoE outputs for later access
            self.moe_outputs = moe_outputs
            
            # Add MoE outputs to the return tuple
            if moe_outputs is not None:
                outputs = outputs + (moe_outputs,)
        else:
            # Use original feed-forward network
            hidden_states = hidden_states + attn_output
            hidden_states = self.block.mlp(hidden_states)
        
        return (hidden_states,) + outputs

class GPT2MoEModel(GPT2LMHeadModel):
    """
    GPT-2 model with Mixture of Experts.
    """
    
    config_class = GPT2MoEConfig
    
    def __init__(self, config: GPT2MoEConfig):
        """
        Initialize GPT-2 with MoE.
        
        Args:
            config: Model configuration
        """
        # Create base config with standard GELU
        base_config = GPT2Config.from_pretrained("gpt2")
        base_config.update({
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_positions": config.n_positions,
            "n_ctx": config.n_ctx,
            "activation_function": "gelu",  # Use standard GELU
        })
        
        super().__init__(base_config)
        
        # Replace transformer blocks with MoE blocks
        self.transformer.h = nn.ModuleList([
            GPT2MoEBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Optional past key/value states
            attention_mask: Optional attention mask
            token_type_ids: Optional token type IDs
            position_ids: Optional position IDs
            head_mask: Optional head mask
            inputs_embeds: Optional input embeddings
            encoder_hidden_states: Optional encoder hidden states
            encoder_attention_mask: Optional encoder attention mask
            labels: Optional labels for language modeling
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs including:
            - Language modeling loss (if labels provided)
            - Logits
            - Past key/value states
            - Hidden states
            - Attentions
            - Cross attentions
            - MoE auxiliary losses
        """
        # Get base model outputs
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Collect MoE auxiliary losses
        if self.training:
            moe_losses = []
            for layer in self.transformer.h:
                if hasattr(layer, 'moe') and layer.moe is not None:
                    # Get the last output which contains MoE outputs
                    moe_outputs = layer.moe_outputs
                    if moe_outputs is not None and "aux_loss" in moe_outputs:
                        moe_losses.append(moe_outputs["aux_loss"])
            
            # Add MoE losses to the total loss
            if moe_losses:
                moe_loss = torch.stack(moe_losses).mean()
                if isinstance(outputs, CausalLMOutputWithCrossAttentions):
                    outputs.loss = outputs.loss + moe_loss
                else:
                    outputs = (outputs[0] + moe_loss,) + outputs[1:]
        
        return outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained model and convert it to use MoE."""
        config = kwargs.pop("config", None)
        if config is None:
            config = GPT2MoEConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Load the base model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        
        # Convert to MoE model
        moe_model = cls(config)
        moe_model.load_state_dict(model.state_dict(), strict=False)
        
        return moe_model 