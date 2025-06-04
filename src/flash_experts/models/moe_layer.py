"""
Mixture of Experts (MoE) layer implementation.
This module provides a MoE layer that can be used in transformer models.
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel

class Router(nn.Module):
    """
    Router network that decides which experts to use for each token.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int = 2,  # Number of experts to route to per token
        router_bias: bool = True,
        router_jitter_noise: float = 0.0,
    ):
        """
        Initialize the router.
        
        Args:
            hidden_size: Size of the input hidden states
            num_experts: Number of expert networks
            k: Number of experts to route to per token
            router_bias: Whether to use bias in the router
            router_jitter_noise: Amount of noise to add to router logits
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.router_jitter_noise = router_jitter_noise
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=router_bias)
        
        # Initialize router weights
        nn.init.normal_(self.router.weight, std=0.02)
        if router_bias:
            nn.init.zeros_(self.router.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the router.
        
        Args:
            hidden_states: Input hidden states of shape [batch_size, seq_len, hidden_size]
            training: Whether the model is in training mode
            
        Returns:
            Tuple of:
            - Router logits of shape [batch_size, seq_len, num_experts]
            - Expert indices of shape [batch_size, seq_len, k]
            - Expert weights of shape [batch_size, seq_len, k]
        """
        # Get router logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # Add noise during training for better exploration
        if training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise
        
        # Get top-k experts and their weights
        expert_weights, expert_indices = torch.topk(
            router_logits,
            k=min(self.k, self.num_experts),  # Ensure k doesn't exceed num_experts
            dim=-1,
            sorted=False
        )
        
        # Convert weights to probabilities
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        return router_logits, expert_indices, expert_weights

class Expert(nn.Module):
    """
    Expert network that processes tokens routed to it.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_fn: str = "gelu",
        dropout: float = 0.1,
    ):
        """
        Initialize the expert network.
        
        Args:
            hidden_size: Size of the input hidden states
            intermediate_size: Size of the intermediate layer
            activation_fn: Activation function to use
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Expert network layers
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = getattr(F, activation_fn)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the expert network.
        
        Args:
            hidden_states: Input hidden states of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output hidden states of shape [batch_size, seq_len, hidden_size]
        """
        x = self.fc1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MoELayer(nn.Module):
    """
    Mixture of Experts layer that combines multiple expert networks with a router.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        k: int = 2,
        router_bias: bool = True,
        router_jitter_noise: float = 0.0,
        activation_fn: str = "gelu",
        dropout: float = 0.1,
        use_aux_loss: bool = True,
    ):
        """
        Initialize the MoE layer.
        
        Args:
            hidden_size: Size of the input hidden states
            intermediate_size: Size of the intermediate layer in experts
            num_experts: Number of expert networks
            k: Number of experts to route to per token
            router_bias: Whether to use bias in the router
            router_jitter_noise: Amount of noise to add to router logits
            activation_fn: Activation function to use in experts
            dropout: Dropout probability
            use_aux_loss: Whether to compute auxiliary load balancing loss
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = min(k, num_experts)  # Ensure k doesn't exceed num_experts
        self.use_aux_loss = use_aux_loss
        
        # Initialize router
        self.router = Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            k=self.k,
            router_bias=router_bias,
            router_jitter_noise=router_jitter_noise,
        )
        
        # Initialize expert networks
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_fn=activation_fn,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        
        # Layer norm for input
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of the MoE layer.
        
        Args:
            hidden_states: Input hidden states of shape [batch_size, seq_len, hidden_size]
            training: Whether the model is in training mode
            
        Returns:
            Tuple of:
            - Output hidden states of shape [batch_size, seq_len, hidden_size]
            - Optional dictionary containing auxiliary losses and routing statistics
        """
        # Layer norm
        x = self.layer_norm(hidden_states)
        
        # Get router outputs
        router_logits, expert_indices, expert_weights = self.router(x, training=training)
        
        # Initialize output tensor
        batch_size, seq_len, _ = x.shape
        final_output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Get tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]
            if not expert_mask.any():
                continue
            
            # Get expert weights for this expert
            expert_weight = expert_weights * (expert_indices == expert_idx).float()  # [batch_size, seq_len, k]
            expert_weight = expert_weight.sum(dim=-1)  # [batch_size, seq_len]
            
            # Process tokens through expert
            expert_input = x[expert_mask]  # [num_tokens, hidden_size]
            expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, hidden_size]
            
            # Add weighted expert output to final output
            final_output[expert_mask] += expert_output * expert_weight[expert_mask].unsqueeze(-1)
        
        # Add residual connection
        output = final_output + hidden_states
        
        # Compute auxiliary loss if needed
        aux_loss = None
        if self.use_aux_loss and training:
            # Compute load balancing loss
            router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
            mean_expert_probs = router_probs.mean(dim=[0, 1])  # [num_experts]
            mean_expert_probs = mean_expert_probs.mean()  # scalar
            
            # Compute variance of expert usage
            # Create a tensor of shape [batch_size, seq_len, num_experts] where each position
            # indicates whether that expert was used for that token
            expert_usage = torch.zeros(batch_size, seq_len, self.num_experts, device=expert_indices.device)
            for k_idx in range(self.k):
                expert_usage.scatter_(
                    dim=-1,
                    index=expert_indices[..., k_idx:k_idx+1],
                    src=torch.ones_like(expert_indices[..., k_idx:k_idx+1], dtype=torch.float),
                )
            expert_usage = expert_usage.mean(dim=[0, 1])  # [num_experts]
            expert_usage = expert_usage.mean()  # scalar
            
            # Load balancing loss
            aux_loss = (mean_expert_probs * expert_usage).sum()
            
            # Add router z-loss for better numerical stability
            router_z_loss = torch.logsumexp(router_logits, dim=-1).mean()
            aux_loss = aux_loss + 0.01 * router_z_loss
        
        # Prepare output dictionary
        output_dict = None
        if training:
            output_dict = {
                "aux_loss": aux_loss,
                "router_logits": router_logits,
                "expert_indices": expert_indices,
                "expert_weights": expert_weights,
            }
        
        return output, output_dict 