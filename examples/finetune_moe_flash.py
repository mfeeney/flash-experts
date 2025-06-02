"""
Script for fine-tuning the GPT-2 MoE model with Flash Attention on a specific dataset.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)

try:
    from src.moe import GPT2MoEConfig, GPT2MoEModel
    from src.flash_attention import GPT2FlashAttentionConfig, GPT2FlashAttention
except ImportError:
    print("Error: Could not import required modules.")
    print("Please install the package in development mode:")
    print("pip install -e .")
    sys.exit(1)

def create_model_and_tokenizer(
    model_name: str = "gpt2",
    num_experts: int = 4,
    k: int = 2,
    moe_layer_freq: int = 2,
    use_flash_attention: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Create and initialize the model and tokenizer."""
    # Create base config
    base_config = GPT2MoEConfig.from_pretrained(
        model_name,
        num_experts=num_experts,
        k=k,
        moe_layer_freq=moe_layer_freq,
        n_positions=1024,
        n_ctx=1024,
    )
    
    # Add Flash Attention config
    if use_flash_attention:
        base_config.use_flash_attention = True
    
    # Create model
    model = GPT2MoEModel(base_config)
    
    # Replace attention layers with Flash Attention if enabled
    if use_flash_attention:
        for layer in model.transformer.h:
            layer.attn = GPT2FlashAttention(base_config)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_dataset(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer: GPT2Tokenizer = None,
    max_length: int = 128,
    batch_size: int = 8,
):
    """Prepare the dataset for training."""
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
    )
    
    return tokenized_dataset, data_collator

def main():
    """Main function for fine-tuning the model."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create model and tokenizer
    print("Creating model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(
        model_name="gpt2",
        num_experts=4,
        k=2,
        moe_layer_freq=2,
        use_flash_attention=True,
        device=device,
    )
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Count MoE layers
    moe_layers = sum(1 for layer in model.transformer.h if hasattr(layer, 'moe') and layer.moe is not None)
    print(f"Number of MoE layers: {moe_layers}")
    
    # Verify Flash Attention
    flash_attn_layers = sum(1 for layer in model.transformer.h if hasattr(layer.attn, 'flash_attn'))
    print(f"Number of Flash Attention layers: {flash_attn_layers}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset, data_collator = prepare_dataset(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",  # Start with a small dataset for testing
        tokenizer=tokenizer,
        max_length=128,
        batch_size=8,
    )
    
    # Set up training arguments with minimal parameters
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Reduced batch size for stability
        per_device_eval_batch_size=4,
        warmup_steps=100,  # Reduced warmup steps
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,  # More frequent logging
        save_steps=200,
        learning_rate=1e-5,  # Slightly lower learning rate
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=8,  # Increased to compensate for smaller batch size
        gradient_checkpointing=False,  # Disabled due to compatibility issues
        remove_unused_columns=False,
        max_grad_norm=1.0,  # Added gradient clipping
        optim="adamw_torch",  # Explicitly specify optimizer
    )
    
    # Create trainer with minimal configuration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    
    # Add GPU monitoring callback
    class GPUMonitorCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                if state.global_step % 100 == 0:  # Log every 100 steps
                    print(f"\nGPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    trainer.add_callback(GPUMonitorCallback())
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    print("\nSaving model...")
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    print("\nTraining complete! Model saved to ./fine_tuned_model")
    print("You can now use the fine-tuned model with the generate_text function in combined_moe_flash.py")

if __name__ == "__main__":
    main() 