import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Type

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
)

from .config import TrainingConfig
from .data_utils import create_dataloaders
from flash_attention import GPT2FlashAttention, GPT2FlashAttentionConfig

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model_class: Optional[Type[PreTrainedModel]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            model_class: Optional custom model class to use (default: GPT2LMHeadModel)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        if model_class is None:
            model_class = GPT2LMHeadModel
        
        if config.use_flash_attention:
            # Use Flash Attention model
            self.model = model_class.from_pretrained(
                "gpt2",
                config=GPT2FlashAttentionConfig.from_pretrained("gpt2")
            )
        else:
            # Use standard model
            self.model = model_class.from_pretrained("gpt2")
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )
        
        # Initialize training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_state = {
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'optimizer_state': None,
            'scheduler_state': None,
        }

        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataloaders
        self.train_dataloader, self.val_dataloader = create_dataloaders(
            tokenizer=self.tokenizer,
            train_file=config.train_file,
            val_file=config.val_file,
            config=config
        )

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            self.model.train()
            
            # Training loop
            train_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Check if we've reached max_steps
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max_steps ({self.config.max_steps}). Stopping training.")
                    return
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Update progress bar
                train_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': train_loss / (step + 1),
                    'lr': self.scheduler.get_last_lr()[0],
                    'step': self.global_step
                })
                
                # Evaluate and save checkpoint
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    self.save_checkpoint(eval_loss)
            
            # End of epoch evaluation
            eval_loss = self.evaluate()
            self.save_checkpoint(eval_loss)
            
            # Check max_steps again after epoch
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                logger.info(f"Reached max_steps ({self.config.max_steps}). Stopping training.")
                return

    def evaluate(self) -> float:
        """Evaluate the model on the validation set."""
        self.model.eval()
        eval_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                eval_loss += outputs.loss.item()

        eval_loss /= len(self.val_dataloader)
        logger.info(f"Evaluation loss: {eval_loss:.4f}")
        return eval_loss
    
    def save_checkpoint(self, eval_loss: float):
        """Save model checkpoint if it's the best so far or if save_best_only is False."""
        should_save = not self.config.save_best_only or eval_loss < self.best_loss
        logger.info(f"Attempting to save checkpoint. eval_loss={eval_loss:.4f}, best_loss={self.best_loss:.4f}, should_save={should_save}")
        
        if should_save:
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                logger.info(f"New best eval loss: {self.best_loss:.4f}")
            
            checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
            logger.info(f"Creating checkpoint directory: {checkpoint_dir}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            logger.info("Saving model and tokenizer...")
            try:
                logger.info(f"Model state dict keys: {list(self.model.state_dict().keys())}")
                logger.info(f"Model device: {next(self.model.parameters()).device}")
                logger.info(f"Model type: {type(self.model)}")
                logger.info(f"Model config: {self.model.config}")
                
                # Try to save model state dict directly first
                state_dict = self.model.state_dict()
                logger.info(f"State dict size: {sum(p.numel() for p in state_dict.values())} parameters")
                
                # Save using save_pretrained with more detailed error handling
                logger.info(f"Saving model to {checkpoint_dir}")
                try:
                    self.model.save_pretrained(checkpoint_dir, safe_serialization=False)
                    logger.info("Model saved successfully using save_pretrained")
                except Exception as save_error:
                    logger.error(f"Error in save_pretrained: {str(save_error)}", exc_info=True)
                    # Fallback to direct torch.save
                    logger.info("Attempting fallback save using torch.save")
                    torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
                    logger.info("Model saved successfully using torch.save")
                
                # Verify the model was saved
                model_path = checkpoint_dir / "pytorch_model.bin"
                if model_path.exists():
                    logger.info(f"Model file exists at {model_path} with size {model_path.stat().st_size} bytes")
                else:
                    logger.error(f"Model file was not created at {model_path}")
                    # Try to list directory contents to see what was created
                    logger.error(f"Directory contents of {checkpoint_dir}: {list(checkpoint_dir.iterdir())}")
                
                logger.info(f"Saving tokenizer to {checkpoint_dir}")
                self.tokenizer.save_pretrained(checkpoint_dir)
                logger.info("Tokenizer saved successfully")
            except Exception as e:
                logger.error(f"Error saving model/tokenizer: {str(e)}", exc_info=True)
                raise
            
            # Save training state
            logger.info("Saving training state...")
            torch.save({
                'epoch': self.epoch,
                'global_step': self.global_step,
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'best_loss': self.best_loss,
            }, checkpoint_dir / 'training_state.pt')
            
            logger.info(f"Successfully saved checkpoint to {checkpoint_dir}")
            
            # Remove old checkpoints if we're over the limit
            if self.config.save_total_limit is not None:
                checkpoints = sorted(
                    [d for d in self.output_dir.iterdir() if d.is_dir()],
                    key=lambda x: int(x.name.split('-')[-1])
                )
                # Always keep the latest checkpoint
                if len(checkpoints) > self.config.save_total_limit:
                    # Remove all but the latest checkpoint and the best checkpoint
                    checkpoints_to_remove = checkpoints[:-self.config.save_total_limit]
                    for checkpoint in checkpoints_to_remove:
                        if checkpoint != checkpoint_dir:  # Don't remove the checkpoint we just created
                            logger.info(f"Removing old checkpoint {checkpoint}")
                            for file in checkpoint.iterdir():
                                file.unlink()
                            checkpoint.rmdir()
        else:
            logger.info("Skipping checkpoint save (not best model and save_best_only=True)")

def main():
    """Entry point for training."""
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
