# GPT-2 with Mixture of Experts and Flash Attention for MTG Card Name Generation

This project demonstrates how to implement and experiment with advanced transformer architectures by extending GPT-2 with Mixture of Experts (MoE) and Flash Attention to generate Magic: The Gathering card names. It's designed as a learning resource for understanding these concepts in practice.

## Project Overview

The project consists of several key components:

1. **Model Architecture**:
   - Base GPT-2 model extended with Mixture of Experts
   - Flash Attention implementation for improved efficiency
   - Custom token filtering and text generation for MTG card names
   - GPU-optimized training and inference

2. **Training Pipeline**:
   - Fine-tuning script for the MoE model on MTG card names
   - Training configuration and optimization with wandb tracking
   - Model evaluation and comparison
   - Experiment tracking and visualization

3. **Example Scripts**:
   - `examples/prepare_mtg_data.py`: Data preparation and preprocessing
   - `examples/finetune_mtg_names.py`: Fine-tuning script for MTG card names
   - `examples/generate_mtg_names.py`: Main script for card name generation
   - `examples/generate_names.py`: Simple generation script for testing

## Learning Objectives

This project helps you understand:

1. **Model Architecture**:
   - How to modify transformer architectures
   - Implementation of Mixture of Experts
   - Integration of Flash Attention for GPU optimization
   - Model initialization and weight loading

2. **Training Process**:
   - Setting up fine-tuning pipelines for specialized text generation
   - Managing training configurations
   - Handling model checkpoints
   - Monitoring training progress with wandb

3. **Text Generation**:
   - Specialized generation for MTG card names
   - Token filtering and processing
   - Text cleaning and formatting
   - Model comparison techniques

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd flash-experts
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode (recommended for local development):
   ```bash
   pip install -e .
   ```
   If you encounter permission issues, you can add the source directory to your PYTHONPATH:
   ```bash
   source setup_env.sh
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Note: This project requires CUDA-compatible GPUs for optimal performance. The requirements include CUDA 12.1 support.

## Project Structure

```
.
├── configs/                        # Training and model configurations
├── data/                           # MTG card name datasets
├── examples/                       # Example scripts
│   ├── prepare_mtg_data.py         # Data preparation script
│   ├── finetune_mtg_names.py       # Fine-tuning script
│   ├── generate_mtg_names.py       # Main generation script
│   └── generate_names.py           # Simple generation script
├── models/                         # Saved model checkpoints
├── src/
│   └── flash_experts/              # Main source code package
│       ├── __init__.py
│       ├── models/                 # Model implementations
│       │   ├── __init__.py
│       │   ├── moe.py              # Mixture of Experts implementation
│       │   ├── moe_layer.py        # MoE layer components
│       │   ├── moe_config.py       # MoE configuration
│       │   ├── flash_attention.py  # Flash Attention implementation
│       │   └── flash_attn.py       # Flash Attention utilities
│       └── training/               # Training utilities
│           ├── __init__.py
│           ├── trainer.py          # Training loop implementation
│           └── training_config.py  # Training configuration
├── tests/                          # Unit tests
│   ├── test_data/                  # Test data fixtures
│   ├── test_models/                # Model-specific tests
│   ├── test_moe.py                 # MoE implementation tests
│   ├── test_training.py            # Training pipeline tests
│   ├── test_flash_attention.py     # Flash Attention tests
│   └── test_data_loading.py        # Data loading tests
├── wandb/                          # Weights & Biases experiment tracking
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── requirements.txt                # Project dependencies
├── setup.py                        # Package configuration
├── pyproject.toml                  # Project metadata
└── setup_env.sh                    # Script to set PYTHONPATH for development
```

## Usage Examples

### 1. Running Tests

To run the test suite:
```bash
pytest tests/
```

This will run all tests including:
- Model architecture tests
- Training pipeline tests
- Flash Attention implementation tests
- Data loading tests

### 2. Preparing Data

To prepare the MTG card name dataset:
```bash
python3 examples/prepare_mtg_data.py
```

This script:
- Processes raw MTG card data
- Prepares training and validation sets
- Handles tokenization and formatting

### 3. Fine-tuning the Model

To fine-tune the model on MTG card names:
```bash
python3 examples/finetune_mtg_names.py
```

This will:
- Load the base GPT-2 model
- Convert it to a MoE architecture
- Fine-tune on MTG card names
- Track training progress with wandb
- Save the model to `./models/`

### 4. Generating Card Names

To generate MTG card names using the fine-tuned model:
```bash
python3 examples/generate_mtg_names.py
```

For quick testing, you can also use:
```bash
python3 examples/generate_names.py
```

These scripts demonstrate:
- Loading the fine-tuned model
- Generating card names with different parameters
- Model comparison
- Text cleaning and formatting

## Model Files

The fine-tuned model files are stored in the `models/` directory. The repository includes:
- Training scripts
- Model architecture code
- Configuration files
- Documentation
- Example generated card names

## Learning Notes

1. **Key Features**:
   - GPU-optimized training with Flash Attention
   - Mixture of Experts for improved model capacity
   - Specialized for MTG card name generation
   - Experiment tracking with wandb
   - Comprehensive test suite

2. **Areas for Experimentation**:
   - Adjusting model architecture
   - Modifying training parameters
   - Trying different generation strategies
   - Implementing custom token filtering
   - Experimenting with different MoE configurations