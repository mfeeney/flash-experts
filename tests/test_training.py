import os
# Disable Flash Attention by default for standard tests
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

import logging
import tempfile
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from src.training.config import TrainingConfig
from src.training.train import Trainer
from src.flash_attention.gpt2_flash_attn import GPT2FlashAttentionConfig, GPT2FlashAttentionLMHeadModel

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_training_loop():
    """Test the training loop with a small sample of data."""
    # Create a temporary directory for our test outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a minimal config for testing
        config = TrainingConfig(
            train_file="data/sample/train.txt",
            val_file="data/sample/val.txt",
            output_dir=tmp_dir,
            max_epochs=1,  # Just one epoch for testing
            batch_size=2,
            eval_steps=2,  # Evaluate every 2 steps
            save_total_limit=1,  # Keep only the best checkpoint
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_steps=2,
            max_steps=4,  # Limit total steps for testing
            save_best_only=False,  # Save all checkpoints during testing
            use_flash_attention=False,  # Use standard attention for this test
        )
        
        # Initialize trainer
        trainer = Trainer(config)
        
        # Run training for a few steps
        trainer.train()
        
        # Verify that checkpoints were created
        checkpoint_dir = Path(tmp_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0, "No checkpoints were created"
        
        # Verify that the model was saved
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
        assert (latest_checkpoint / "pytorch_model.bin").exists(), "Model weights were not saved"
        
        # Verify that the training state was saved
        assert (latest_checkpoint / "training_state.pt").exists(), "Training state was not saved"
        
        # Load the saved model and verify it's a GPT2LMHeadModel
        loaded_model = GPT2LMHeadModel.from_pretrained(latest_checkpoint)
        loaded_model.to(config.device)  # Move model to the correct device
        assert isinstance(loaded_model, GPT2LMHeadModel), "Loaded model is not a GPT2LMHeadModel"
        
        # Verify that the model is in eval mode after loading
        assert not loaded_model.training, "Loaded model is in training mode"
        
        # Test that the model can generate text
        test_input = torch.tensor([[1, 2, 3]]).to(config.device)  # Simple test input
        with torch.no_grad():
            outputs = loaded_model(test_input)
            assert outputs.logits.shape[0] == 1, "Model output shape is incorrect"
        
        print("All training tests passed!")

def test_training_with_flash_attention():
    """Test the training loop with Flash Attention enabled."""
    # Create a temporary directory for our test outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a minimal config for testing with Flash Attention
        config = TrainingConfig(
            train_file="data/sample/train.txt",
            val_file="data/sample/val.txt",
            output_dir=tmp_dir,
            max_epochs=1,
            batch_size=2,
            eval_steps=2,
            save_total_limit=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_steps=2,
            max_steps=4,
            save_best_only=False,
            use_flash_attention=True,  # Enable Flash Attention
        )
        
        # Initialize trainer with Flash Attention model
        trainer = Trainer(config, model_class=GPT2FlashAttentionLMHeadModel)
        
        # Verify that Flash Attention is enabled
        for i, layer in enumerate(trainer.model.transformer.h):
            assert hasattr(layer.attn, 'flash_attn'), f"Layer {i} does not have Flash Attention"
        
        # Run training for a few steps
        trainer.train()
        
        # Verify that checkpoints were created
        checkpoint_dir = Path(tmp_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0, "No checkpoints were created"
        
        # Load the saved model and verify Flash Attention is preserved
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
        loaded_model = GPT2FlashAttentionLMHeadModel.from_pretrained(
            latest_checkpoint,
            config=GPT2FlashAttentionConfig.from_pretrained(latest_checkpoint)
        )
        loaded_model.to(config.device)
        
        # Verify that Flash Attention is still enabled after loading
        for i, layer in enumerate(loaded_model.transformer.h):
            assert hasattr(layer.attn, 'flash_attn'), f"Layer {i} lost Flash Attention after loading"
        
        print("All Flash Attention training tests passed!")

if __name__ == "__main__":
    test_training_loop()
    test_training_with_flash_attention()