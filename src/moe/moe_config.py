"""
Configuration class for GPT-2 with Mixture of Experts.
"""

from transformers import GPT2Config

class GPT2MoEConfig(GPT2Config):
    """Configuration class for GPT-2 with Mixture of Experts."""
    
    model_type = "gpt2_moe"
    
    def __init__(
        self,
        num_experts: int = 8,
        k: int = 2,
        router_bias: bool = True,
        router_jitter_noise: float = 0.0,
        use_aux_loss: bool = True,
        moe_layer_freq: int = 2,  # Replace every nth layer with MoE
        **kwargs
    ):
        """
        Initialize GPT-2 MoE configuration.
        
        Args:
            num_experts: Number of expert networks in each MoE layer
            k: Number of experts to route to per token
            router_bias: Whether to use bias in the router
            router_jitter_noise: Amount of noise to add to router logits
            use_aux_loss: Whether to compute auxiliary load balancing loss
            moe_layer_freq: Replace every nth layer with MoE (1 = all layers, 2 = every other layer, etc.)
            **kwargs: Additional arguments for GPT2Config
        """
        super().__init__(**kwargs)
        
        # MoE specific configuration
        self.num_experts = num_experts
        self.k = k
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        self.use_aux_loss = use_aux_loss
        self.moe_layer_freq = moe_layer_freq
        
        # Verify configuration
        if self.moe_layer_freq < 1:
            raise ValueError("moe_layer_freq must be at least 1")
        if self.k > self.num_experts:
            raise ValueError("k cannot be greater than num_experts")
        if self.k < 1:
            raise ValueError("k must be at least 1") 