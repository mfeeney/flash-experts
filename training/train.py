import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import ( GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup)
from .config import TrainingConfig
from .data_utils import create_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)

        # Initialize model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._setup_optimizer()

        # Initialize scheduler
        self.scheduler = None # We'll set this up when we have the dataloader

        # Create dataloaders
        self.train_dataloader, self.val_dataloader = create_dataloaders(
            tokenizer=self.tokenizer,
            train_file=config.train_file,
            val_file=config.val_file,
            config=config
        )

        # Now we can set up the scheduler with the correct number of steps
        num_training_steps = len(self.train_dataloader) * config.max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

    def _setup_optimizer(self):
        """Initialize the optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        # TODO: Implement data loading
        # TODO: Implement training loop
        # TODO: Implement evaluation
        # TODO: Implement checkpointing
        pass

def main():
    """Entry point for training."""
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

