import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Custom collate function to handle padding of sequences.

    Args:
        batch: List of dictionaries containing 'input_ids' and 'labels'

    Returns:
        Dictionary with padded tensors
    """
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    # Pad sequences to the maximum length in this batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids, 
        'labels': labels
    }

class TextDataset(Dataset):
    """Dataset for training GPT-2 on text data."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: Union[str, Path],
        max_length: int,
        block_size: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: The tokenizer to use for encoding text
            file_path: Path to the text file containing training data
            max_length: Maximum sequence length
            block_size: Size of text blocks to use for training
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        
        # Load the dataset
        logger.info(f"Loading dataset from {file_path}")
        self.examples = self._load_and_tokenize(file_path)
        
    def _load_and_tokenize(self, file_path: Union[str, Path]) -> List[Dict[str, torch.Tensor]]:
        """
        Load and tokenize the text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of tokenized examples
        """
        # Load the dataset using HuggingFace datasets
        dataset = load_dataset(
            'text',
            data_files=str(file_path),
            split='train'
        )
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize the texts
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True
            )
            
            # Create input_ids and labels (for causal language modeling)
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Convert to list of examples
        return [example for example in tokenized_dataset]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.examples[i]

def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    train_file: Union[str, Path],
    val_file: Union[str, Path],
    config: 'TrainingConfig'  # This will be imported from config.py
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        tokenizer: The tokenizer to use
        train_file: Path to training data file
        val_file: Path to validation data file
        config: Training configuration
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        max_length=config.max_seq_length,
        block_size=config.max_seq_length
    )
    
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=val_file,
        max_length=config.max_seq_length,
        block_size=config.max_seq_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.preprocessing_num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.preprocessing_num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader 