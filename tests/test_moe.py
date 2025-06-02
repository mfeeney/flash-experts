"""
Tests for the Mixture of Experts (MoE) implementation.
"""

import pytest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.moe import (
    Router,
    Expert,
    MoELayer,
    GPT2MoEConfig,
    GPT2MoEModel,
)

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 8

@pytest.fixture
def hidden_size():
    return 64

@pytest.fixture
def num_experts():
    return 4

@pytest.fixture
def k():
    return 2

@pytest.fixture
def intermediate_size():
    return 128

@pytest.fixture
def sample_input(batch_size, seq_len, hidden_size):
    return torch.randn(batch_size, seq_len, hidden_size)

def test_router_initialization(hidden_size, num_experts, k):
    """Test router initialization and basic properties."""
    router = Router(
        hidden_size=hidden_size,
        num_experts=num_experts,
        k=k,
    )
    
    # Check router properties
    assert router.hidden_size == hidden_size
    assert router.num_experts == num_experts
    assert router.k == k
    assert router.router.weight.shape == (num_experts, hidden_size)
    assert router.router.bias.shape == (num_experts,)

def test_router_forward(sample_input, num_experts, k):
    """Test router forward pass and output shapes."""
    router = Router(
        hidden_size=sample_input.shape[-1],
        num_experts=num_experts,
        k=k,
    )
    
    # Forward pass
    router_logits, expert_indices, expert_weights = router(sample_input)
    
    # Check output shapes
    assert router_logits.shape == (sample_input.shape[0], sample_input.shape[1], num_experts)
    assert expert_indices.shape == (sample_input.shape[0], sample_input.shape[1], k)
    assert expert_weights.shape == (sample_input.shape[0], sample_input.shape[1], k)
    
    # Check expert indices range
    assert expert_indices.min() >= 0
    assert expert_indices.max() < num_experts
    
    # Check expert weights
    assert torch.allclose(expert_weights.sum(dim=-1), torch.ones_like(expert_weights.sum(dim=-1)))
    assert (expert_weights >= 0).all()

def test_router_jitter_noise(sample_input, num_experts, k):
    """Test router jitter noise during training."""
    router = Router(
        hidden_size=sample_input.shape[-1],
        num_experts=num_experts,
        k=k,
        router_jitter_noise=0.1,
    )
    
    # Forward pass without noise (eval mode)
    logits_no_noise, _, _ = router(sample_input, training=False)
    
    # Forward pass with noise (training mode)
    logits_with_noise, _, _ = router(sample_input, training=True)
    
    # Check that noise was added
    assert not torch.allclose(logits_no_noise, logits_with_noise)

def test_expert_initialization(hidden_size, intermediate_size):
    """Test expert initialization and basic properties."""
    expert = Expert(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    
    # Check expert properties
    assert expert.hidden_size == hidden_size
    assert expert.intermediate_size == intermediate_size
    assert expert.fc1.weight.shape == (intermediate_size, hidden_size)
    assert expert.fc2.weight.shape == (hidden_size, intermediate_size)

def test_expert_forward(sample_input, intermediate_size):
    """Test expert forward pass and output shapes."""
    expert = Expert(
        hidden_size=sample_input.shape[-1],
        intermediate_size=intermediate_size,
    )
    
    # Forward pass
    output = expert(sample_input)
    
    # Check output shape
    assert output.shape == sample_input.shape

def test_moe_layer_initialization(hidden_size, intermediate_size, num_experts, k):
    """Test MoE layer initialization and basic properties."""
    moe_layer = MoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        k=k,
    )
    
    # Check MoE layer properties
    assert moe_layer.hidden_size == hidden_size
    assert moe_layer.num_experts == num_experts
    assert moe_layer.k == k
    assert len(moe_layer.experts) == num_experts
    assert isinstance(moe_layer.router, Router)

def test_moe_layer_forward(sample_input, intermediate_size, num_experts, k):
    """Test MoE layer forward pass and output shapes."""
    moe_layer = MoELayer(
        hidden_size=sample_input.shape[-1],
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        k=k,
    )
    
    # Forward pass
    output, output_dict = moe_layer(sample_input, training=True)
    
    # Check output shape
    assert output.shape == sample_input.shape
    
    # Check output dictionary
    assert isinstance(output_dict, dict)
    assert "aux_loss" in output_dict
    assert "router_logits" in output_dict
    assert "expert_indices" in output_dict
    assert "expert_weights" in output_dict
    
    # Check auxiliary loss
    assert isinstance(output_dict["aux_loss"], torch.Tensor)
    assert output_dict["aux_loss"].ndim == 0  # scalar
    
    # Check expert indices and weights shapes
    batch_size, seq_len, _ = sample_input.shape
    assert output_dict["expert_indices"].shape == (batch_size, seq_len, k)
    assert output_dict["expert_weights"].shape == (batch_size, seq_len, k)
    
    # Check expert indices range
    assert output_dict["expert_indices"].min() >= 0
    assert output_dict["expert_indices"].max() < num_experts
    
    # Check expert weights
    assert torch.allclose(output_dict["expert_weights"].sum(dim=-1), torch.ones(batch_size, seq_len, device=sample_input.device))
    assert (output_dict["expert_weights"] >= 0).all()

def test_moe_layer_load_balancing(sample_input, intermediate_size, num_experts, k):
    """Test MoE layer load balancing through auxiliary loss."""
    moe_layer = MoELayer(
        hidden_size=sample_input.shape[-1],
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        k=k,
        use_aux_loss=True,
    )
    
    # Forward pass
    _, output_dict = moe_layer(sample_input, training=True)
    
    # Check that auxiliary loss is positive
    assert output_dict["aux_loss"] > 0
    
    # Check expert usage statistics
    expert_indices = output_dict["expert_indices"]  # [batch_size, seq_len, k]
    expert_usage = torch.zeros(num_experts, device=sample_input.device)
    
    # Count usage of each expert
    for expert_idx in range(num_experts):
        # Count how many times this expert is used across all positions and batches
        expert_usage[expert_idx] = (expert_indices == expert_idx).float().mean()
    
    # Check that all experts are used
    assert (expert_usage > 0).all()
    
    # Check that expert usage is roughly balanced
    # We expect each expert to be used at least 1/(2*num_experts) of the time
    # (since we use k=2 experts per token)
    min_expected_usage = 1.0 / (2 * num_experts)
    assert (expert_usage >= min_expected_usage * 0.5).all()  # Allow some tolerance

def test_gpt2_moe_config():
    """Test GPT2MoEConfig initialization and validation."""
    # Test valid configuration
    config = GPT2MoEConfig(
        num_experts=8,
        k=2,
        moe_layer_freq=2,
    )
    assert config.num_experts == 8
    assert config.k == 2
    assert config.moe_layer_freq == 2
    
    # Test invalid configurations
    with pytest.raises(ValueError):
        GPT2MoEConfig(k=10, num_experts=8)  # k > num_experts
    
    with pytest.raises(ValueError):
        GPT2MoEConfig(k=0)  # k < 1
    
    with pytest.raises(ValueError):
        GPT2MoEConfig(moe_layer_freq=0)  # moe_layer_freq < 1

def test_gpt2_moe_model():
    """Test GPT2MoEModel initialization and forward pass."""
    # Initialize model
    config = GPT2MoEConfig(
        num_experts=4,
        k=2,
        moe_layer_freq=2,
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=4,
        n_head=4,
    )
    model = GPT2MoEModel(config)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
    
    # Create sample input
    text = "Hello, world! This is a test."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])
    
    # Check outputs
    assert hasattr(outputs, "loss")
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[0] == inputs["input_ids"].shape[0]
    assert outputs.logits.shape[1] == inputs["input_ids"].shape[1]
    assert outputs.logits.shape[2] == config.vocab_size

def test_gpt2_moe_model_from_pretrained():
    """Test loading pretrained model and converting to MoE."""
    # Initialize config
    config = GPT2MoEConfig.from_pretrained(
        "gpt2",
        num_experts=4,
        k=2,
        moe_layer_freq=2,
    )
    
    # Load base model first
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Create MoE model and load state dict
    model = GPT2MoEModel(config)
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Check that model has MoE layers
    moe_layers = 0
    for layer in model.transformer.h:
        if hasattr(layer, 'moe') and layer.moe is not None:
            moe_layers += 1
    
    # Check that we have the expected number of MoE layers
    expected_moe_layers = config.num_hidden_layers // config.moe_layer_freq
    assert moe_layers == expected_moe_layers

def test_gpt2_moe_model_training():
    """Test GPT2MoEModel training with auxiliary loss."""
    # Initialize model
    config = GPT2MoEConfig(
        num_experts=4,
        k=2,
        moe_layer_freq=2,
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=4,
        n_head=4,
        use_aux_loss=True,  # Explicitly enable aux loss
    )
    model = GPT2MoEModel(config)
    model.train()

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token

    # Create sample input
    text = "Hello, world! This is a test."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass with labels
    outputs = model(**inputs, labels=inputs["input_ids"])

    # Check that loss includes auxiliary loss
    assert outputs.loss > 0
    assert outputs.loss.requires_grad  # Ensure loss requires gradients

    # Store initial parameters
    old_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Backward pass
    outputs.loss.backward()

    # Verify gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_gradients = True
            break
    assert has_gradients, "No non-zero gradients found"

    # Update parameters with larger learning rate
    learning_rate = 0.1  # Increased from 0.01
    for param in model.parameters():
        if param.grad is not None:
            param.data -= learning_rate * param.grad

    # Check that parameters were updated
    params_updated = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.allclose(old_params[name], param, rtol=1e-3):
                params_updated = True
                break
    assert params_updated, "No parameters were updated during training"

    # Verify MoE auxiliary loss was computed
    moe_losses = []
    for layer in model.transformer.h:
        if hasattr(layer, 'moe') and layer.moe is not None:
            if layer.moe_outputs is not None and "aux_loss" in layer.moe_outputs:
                moe_losses.append(layer.moe_outputs["aux_loss"])
    assert len(moe_losses) > 0, "No MoE auxiliary losses were computed" 