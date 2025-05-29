# Training Module

This module handles the training pipeline for GPT-2 with Flash Attention and Mixture of Experts. It provides a flexible and configurable training framework that can be extended to incorporate different attention mechanisms and model architectures.

## Directory Structure

- `__init__.py`: Makes this directory a Python package
- `train.py`: Main training script that orchestrates the training process
- `config.py`: Configuration settings and hyperparameters for training
- `README.md`: This documentation file

## Usage

The training pipeline is designed to be run from the command line. Basic usage:

```bash
python -m training.train --config path/to/config.yaml
```

## Dependencies

The training module requires:
- PyTorch 2.5.1+ (with CUDA support)
- Flash Attention 2.7.4+
- Transformers library (for GPT-2)
- Datasets library (for data loading)
- Wandb (for experiment tracking, optional)

## Configuration

Training parameters are managed through `config.py` and can be overridden via command line arguments. Key configuration options include:

- Model parameters (size, architecture)
- Training hyperparameters (learning rate, batch size, etc.)
- Data loading settings
- Logging and checkpointing options

## Training Process

The training pipeline follows these steps:
1. Load and preprocess the dataset
2. Initialize the model and optimizer
3. Train the model with logging and checkpointing
4. Evaluate the model on validation data
5. Save the final model and training metrics

## Logging and Monitoring

Training progress is logged to:
- Console output (basic metrics)
- Tensorboard (detailed metrics and graphs)
- Wandb (experiment tracking, if enabled)

## Checkpointing

The training script automatically saves:
- Model checkpoints
- Optimizer state
- Training metrics
- Configuration used for the run

## Extending the Training Pipeline

To add new features:
1. Add new configuration options in `config.py`
2. Implement the feature in `train.py`
3. Update this documentation
4. Add appropriate tests

## Common Issues and Solutions

- **Out of Memory**: Adjust batch size or gradient accumulation steps
- **Slow Training**: Check data loading pipeline and GPU utilization
- **Poor Performance**: Verify learning rate and model configuration

## Contributing

When adding new features:
1. Follow the existing code style
2. Add appropriate documentation
3. Include tests for new functionality
4. Update this README if necessary