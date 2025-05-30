from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class TrainingConfig:
    # Model Configuration
    model_name: str = "gpt2"  # Base model to use
    model_size: str = "small"  # Size variant (small, medium, large)
    
    # Training Hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * gradient_accumulation_steps
    max_epochs: int = 3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Data Configuration
    train_file: str = "data/train.txt"  # Path to training data
    val_file: str = "data/val.txt"      # Path to validation data
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4
    
    # Optimization
    optimizer: str = "adamw"  # Options: adamw, adam, sgd
    scheduler: str = "linear"  # Options: linear, cosine, constant
    fp16: bool = True  # Mixed precision training
    
    # Logging and Checkpointing
    output_dir: str = "checkpoints"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: Optional[int] = 3  # Number of checkpoints to keep
    
    # Device Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Flash Attention Configuration (for later use)
    use_flash_attention: bool = False  # Will be enabled when we implement Flash Attention
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        if self.fp16 and self.device == "cpu":
            print("Warning: Mixed precision training is not supported on CPU. Disabling fp16.")
            self.fp16 = False 