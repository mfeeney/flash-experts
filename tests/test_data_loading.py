import logging
import sys
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.config import TrainingConfig
from src.training.data_utils import create_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    # Create a minimal config for testing
    config = TrainingConfig(
        batch_size=2,
        max_seq_length=32,
        preprocessing_num_workers=0  # Use 0 for easier debugging
    )
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        train_file=project_root / "data" / "train.txt",
        val_file=project_root / "data" / "val.txt",
        config=config
    )
    
    # Test training dataloader
    logger.info("\nTesting training dataloader:")
    for batch_idx, batch in enumerate(train_dataloader):
        logger.info(f"\nBatch {batch_idx + 1}:")
        logger.info(f"Input shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        
        # Decode and print the first example in the batch
        example = batch['input_ids'][0]
        decoded = tokenizer.decode(example, skip_special_tokens=True)
        logger.info(f"Example text: {decoded}")
        
        if batch_idx >= 1:  # Just show 2 batches
            break
    
    # Test validation dataloader
    logger.info("\nTesting validation dataloader:")
    for batch_idx, batch in enumerate(val_dataloader):
        logger.info(f"\nBatch {batch_idx + 1}:")
        logger.info(f"Input shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        
        # Decode and print the first example in the batch
        example = batch['input_ids'][0]
        decoded = tokenizer.decode(example, skip_special_tokens=True)
        logger.info(f"Example text: {decoded}")
        
        if batch_idx >= 1:  # Just show 2 batches
            break

if __name__ == "__main__":
    test_data_loading() 