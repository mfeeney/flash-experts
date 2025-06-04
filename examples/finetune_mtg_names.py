"""
Script for fine-tuning GPT-2 MoE model on MTG card names using curriculum learning.
This script implements:
1. Curriculum learning with 3 stages
2. Early stopping with patience
3. Learning rate scheduling with warmup
4. Proper validation and checkpointing
5. Card type-aware training
"""

import os
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MTGCardExample:
    """Represents a single MTG card example for training."""
    name: str
    card_type: str
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

class MTGCardDataset(Dataset):
    """Dataset for MTG card names with curriculum stages."""
    
    def __init__(self, data_path: str, stage: str, split: str = 'train', max_length: int = 12):
        """Initialize dataset from processed data."""
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        with open(os.path.join(data_path, stage, f'{split}.json'), 'r') as f:
            data = json.load(f)
        
        self.examples = []
        self.card_types = set()
        
        # Process each card
        for card in data['cards']:
            # Get input_ids and attention_mask
            input_ids = card['tokens']
            attention_mask = [1] * len(input_ids)
            
            # Add padding if needed
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            # Create labels (same as input_ids for language modeling)
            labels = input_ids.copy()
            
            # Create example
            example = MTGCardExample(
                name=card['name'],
                card_type=card['type'],
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            self.examples.append(example)
            self.card_types.add(card['type'])
        
        # Log dataset statistics
        logger.info(f"Loaded {len(self.examples)} examples for {stage}/{split}")
        logger.info(f"Card types in dataset: {sorted(self.card_types)}")
        
        # Calculate type distribution
        type_counts = {}
        for example in self.examples:
            type_counts[example.card_type] = type_counts.get(example.card_type, 0) + 1
        
        logger.info("Card type distribution:")
        for card_type, count in sorted(type_counts.items()):
            percentage = (count / len(self.examples)) * 100
            logger.info(f"  {card_type}: {count} ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        # Only return tensor fields, keep card_type as a separate field
        return {
            'input_ids': torch.tensor(example.input_ids),
            'attention_mask': torch.tensor(example.attention_mask),
            'labels': torch.tensor(example.labels),
            'card_type': example.card_type  # Keep as string
        }

class CustomTrainer(Trainer):
    """Custom trainer with card type-aware logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_type_metrics = {}
        # Set loss type in model config
        if hasattr(self.model, 'config'):
            self.model.config.loss_type = "ForCausalLMLoss"
            # Also set task_specific_params to ensure loss type is recognized
            if not hasattr(self.model.config, 'task_specific_params'):
                self.model.config.task_specific_params = {}
            self.model.config.task_specific_params['causal_lm'] = {'loss_type': 'ForCausalLMLoss'}
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss and track metrics by card type."""
        # Remove card_type from inputs for model forward pass
        card_types = inputs.pop('card_type', None)
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Track loss by card type if available
        if card_types is not None:
            # Get per-example losses if available, otherwise use the average loss
            if hasattr(outputs, 'loss_per_example'):
                per_example_losses = outputs.loss_per_example
            else:
                # If we don't have per-example losses, use the average loss for all examples
                per_example_losses = [loss.item()] * len(card_types)
            
            # Update metrics for each card type
            for card_type, example_loss in zip(card_types, per_example_losses):
                if card_type not in self.card_type_metrics:
                    self.card_type_metrics[card_type] = []
                self.card_type_metrics[card_type].append(example_loss)
        
        return (loss, outputs) if return_outputs else loss
    
    def log_metrics(self, split, metrics, epoch=None):
        """Log metrics including card type-specific ones."""
        # Call parent method without epoch for older transformers version
        if epoch is not None:
            super().log_metrics(split, metrics)
        else:
            super().log_metrics(split, metrics)
        
        # Log card type metrics
        if self.card_type_metrics:
            logger.info(f"\nCard type metrics for {split}:")
            for card_type, losses in sorted(self.card_type_metrics.items()):
                avg_loss = np.mean(losses)
                std_loss = np.std(losses)
                logger.info(f"  {card_type}: {avg_loss:.4f} ± {std_loss:.4f} (n={len(losses)})")
            self.card_type_metrics.clear()
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader with custom collate function."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Get evaluation dataloader with custom collate function."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def _collate_fn(self, examples):
        """Custom collate function to handle card types properly."""
        # Separate tensor fields and card types
        tensor_fields = ['input_ids', 'attention_mask', 'labels']
        card_types = [example.pop('card_type') for example in examples]
        
        # Pad tensor fields
        batch = {}
        for field in tensor_fields:
            batch[field] = torch.stack([example[field] for example in examples])
        
        # Add card types back as a list
        batch['card_type'] = card_types
        
        return batch

class CustomTrainingLoop(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_loss = float('inf')
        self.patience = 3
        self.patience_counter = 0
        self.best_model_path = None
        self.eval_steps = 100  # Evaluate every 100 steps
    
    def save_model_safely(self, output_dir):
        """Save model using a simpler approach that doesn't rely on DTensor."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state dict
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save config
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
        
        # Save tokenizer if available
        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def evaluate(self):
        """Evaluate the model on the validation set."""
        eval_dataloader = self.get_eval_dataloader()
        self.model.eval()
        
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for inputs in eval_dataloader:
                # Move tensor inputs to device, keep non-tensor fields as is
                tensor_inputs = {k: v.to(self.model.device) for k, v in inputs.items() if torch.is_tensor(v)}
                non_tensor_inputs = {k: v for k, v in inputs.items() if not torch.is_tensor(v)}
                inputs = {**tensor_inputs, **non_tensor_inputs}
                
                # Forward pass
                loss = self.compute_loss(self.model, inputs)
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity
        }

    def train(self):
        """Custom training loop with evaluation and early stopping."""
        train_dataloader = self.get_train_dataloader()
        total_steps = len(train_dataloader) * self.args.num_train_epochs
        steps_trained = 0
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {total_steps}")
        
        # Initialize optimizer and scheduler
        self.optimizer = self.create_optimizer()
        num_training_steps = total_steps
        num_warmup_steps = self.args.warmup_steps
        
        # Create scheduler with proper warmup
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info("Optimizer and scheduler initialized")
        logger.info(f"  Learning rate: {self.args.learning_rate}")
        logger.info(f"  Warmup steps: {num_warmup_steps}")
        logger.info(f"  Total steps: {num_training_steps}")
        
        # Initial evaluation
        logger.info("\nRunning initial evaluation...")
        self.model.eval()
        eval_results = self.evaluate()
        current_loss = eval_results.get('eval_loss', float('inf'))
        self.best_loss = current_loss
        logger.info(f"Initial loss: {current_loss:.4f}")
        # Log metrics without epoch parameter
        self.log_metrics("eval", eval_results)
        
        for epoch in range(int(self.args.num_train_epochs)):
            epoch_iterator = train_dataloader
            for step, inputs in enumerate(epoch_iterator):
                # Move tensor inputs to device, keep non-tensor fields as is
                tensor_inputs = {k: v.to(self.model.device) for k, v in inputs.items() if torch.is_tensor(v)}
                non_tensor_inputs = {k: v for k, v in inputs.items() if not torch.is_tensor(v)}
                inputs = {**tensor_inputs, **non_tensor_inputs}
                
                # Training step
                self.model.train()
                loss = self.compute_loss(self.model, inputs)
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                steps_trained += 1
                
                # Evaluation and early stopping
                if steps_trained % self.eval_steps == 0:
                    logger.info(f"\nEvaluating at step {steps_trained}...")
                    self.model.eval()
                    eval_results = self.evaluate()
                    current_loss = eval_results.get('eval_loss', float('inf'))
                    
                    # Log current learning rate
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    logger.info(f"Current learning rate: {current_lr:.2e}")
                    
                    # Check if this is the best model so far
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        self.patience_counter = 0
                        # Save best model using safe method
                        best_model_path = os.path.join(self.args.output_dir, "best_model")
                        self.save_model_safely(best_model_path)
                        self.best_model_path = best_model_path
                        logger.info(f"New best model saved with loss: {current_loss:.4f}")
                    else:
                        self.patience_counter += 1
                        logger.info(f"No improvement for {self.patience_counter} evaluations")
                        
                        if self.patience_counter >= self.patience:
                            logger.info("Early stopping triggered!")
                            return
                    
                    # Log metrics without epoch parameter
                    self.log_metrics("eval", eval_results)
                
                # Logging
                if steps_trained % self.args.logging_steps == 0:
                    logger.info(f"Step {steps_trained}: loss = {loss.item():.4f}")
                
                # Save checkpoint using safe method
                if steps_trained % self.args.save_steps == 0:
                    checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-{steps_trained}")
                    self.save_model_safely(checkpoint_path)
            
            # Log end of epoch
            logger.info(f"Completed epoch {epoch + 1}/{self.args.num_train_epochs}")

def train_stage(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    stage: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    stage_num: int,
    device: torch.device,
    total_stages: int = 3
) -> None:
    """Train model on a single curriculum stage."""
    logger.info(f"\n{'='*20} Training Stage {stage_num}/{total_stages} {'='*20}")
    logger.info(f"Stage: {stage}")
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Calculate learning rate based on stage
    # Start with higher LR for early stages, decrease for later stages
    base_lr = 5e-5 * (1.0 - (stage_num - 1) / total_stages)
    
    # Training arguments - using only the most basic parameters
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"stage_{stage_num}"),
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, f"stage_{stage_num}", "logs"),
        logging_steps=50,
        save_steps=100,
        learning_rate=base_lr,
        fp16=True,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        # Disable wandb
        report_to="none",
    )
    
    trainer = CustomTrainingLoop(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer  # Pass tokenizer to trainer
    )
    
    # Train
    trainer.train()
    
    # Load best model if available
    if trainer.best_model_path and os.path.exists(trainer.best_model_path):
        logger.info("Loading best model from training...")
        state_dict = torch.load(os.path.join(trainer.best_model_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        model.to(device)
    
    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"\nEvaluation results for stage {stage_num}:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save final model using safe method
    final_model_path = os.path.join(output_dir, f"stage_{stage_num}", "final_model")
    trainer.save_model_safely(final_model_path)
    
    # Log final metrics
    final_metrics = trainer.evaluate()
    logger.info(f"\nFinal metrics for stage {stage_num}:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

def main():
    """Main training function."""
    # Set up paths
    data_dir = "data/mtg/processed"
    output_dir = "models/mtg_names"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check CUDA availability and memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Initialize model and tokenizer
    logger.info("Initializing model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load tokenizer vocabulary from processed data
    tokenizer_path = os.path.join(data_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        logger.info("Loading custom tokenizer vocabulary...")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    
    # Create a temporary trainer for safe model saving
    temp_trainer = CustomTrainingLoop(
        model=model,
        args=TrainingArguments(output_dir=output_dir),
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer
    )
    
    # Train each stage
    stages = ['stage_0', 'stage_1', 'stage_2']
    for i, stage in enumerate(stages, 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load datasets
        train_dataset = MTGCardDataset(data_dir, stage, 'train')
        val_dataset = MTGCardDataset(data_dir, stage, 'validation')
        
        # Train stage
        train_stage(model, tokenizer, stage, train_dataset, val_dataset, output_dir, i, device)
        
        # Save checkpoint after each stage using safe method
        checkpoint_dir = os.path.join(output_dir, f"checkpoint_stage_{i}")
        temp_trainer.save_model_safely(checkpoint_dir)
        # Save tokenizer separately
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint for stage {i} to {checkpoint_dir}")
        
        # Log GPU memory after each stage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory after stage {i}:")
            logger.info(f"  Allocated: {allocated:.1f}GB")
            logger.info(f"  Reserved: {reserved:.1f}GB")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 