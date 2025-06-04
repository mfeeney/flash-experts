# GPT-2 with Mixture of Experts and Flash Attention

This project demonstrates how to implement and experiment with advanced transformer architectures by extending GPT-2 with Mixture of Experts (MoE) and Flash Attention. It's designed as a learning resource for understanding these concepts in practice.

## Project Overview

The project consists of several key components:

1. **Model Architecture**:
   - Base GPT-2 model extended with Mixture of Experts
   - Flash Attention implementation for improved efficiency
   - Custom token filtering and text generation

2. **Training Pipeline**:
   - Fine-tuning script for the MoE model
   - Training configuration and optimization
   - Model evaluation and comparison

3. **Example Scripts**:
   - `combined_moe_flash.py`: Demonstrates model loading and text generation
   - `finetune_moe_flash.py`: Shows how to fine-tune the model

## Learning Objectives

This project helps you understand:

1. **Model Architecture**:
   - How to modify transformer architectures
   - Implementation of Mixture of Experts
   - Integration of Flash Attention
   - Model initialization and weight loading

2. **Training Process**:
   - Setting up fine-tuning pipelines
   - Managing training configurations
   - Handling model checkpoints
   - Monitoring training progress

3. **Text Generation**:
   - Different generation strategies
   - Token filtering and processing
   - Text cleaning and formatting
   - Model comparison techniques

## Setup and Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Install additional dependencies:
   ```bash
   pip install datasets transformers[torch] tensorboard
   ```

## Usage Examples

### 1. Fine-tuning the Model

To fine-tune the model on the WikiText-2 dataset:
```bash
python examples/finetune_moe_flash.py
```

This will:
- Load the base GPT-2 model
- Convert it to a MoE architecture
- Fine-tune on WikiText-2
- Save the model to `./fine_tuned_model`

Note: The fine-tuned model is not included in the repository (see "Model Files" section below).

### 2. Generating Text

To compare pre-trained and fine-tuned models:
```bash
python examples/combined_moe_flash.py
```

This script demonstrates:
- Loading both models
- Text generation with different parameters
- Model comparison
- Text cleaning and formatting

## Project Structure

```
.
├── examples/
│   ├── combined_moe_flash.py    # Text generation and comparison
│   └── finetune_moe_flash.py    # Fine-tuning script
├── src/
│   ├── moe.py                   # Mixture of Experts implementation
│   └── flash_attention.py       # Flash Attention implementation
├── tests/                       # Unit tests
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── requirements.txt             # Project dependencies
└── setup.py                     # Package configuration
```

## Model Files

The fine-tuned model files are not included in the repository because:
1. They are large (several GB)
2. They can be regenerated using the provided scripts
3. It's best practice to keep generated artifacts out of version control

Instead, the repository includes:
- Training scripts
- Model architecture code
- Configuration files
- Documentation

To get the fine-tuned model, run the fine-tuning script as described above.

## Learning Notes

1. **Current Limitations**:
   - The model generates somewhat incoherent text
   - Fine-tuning shows minimal improvement
   - These issues are valuable learning opportunities!

2. **Key Concepts Demonstrated**:
   - Model architecture modification
   - Training pipeline setup
   - Text generation techniques
   - Common challenges in language models

3. **Areas for Experimentation**:
   - Adjusting model architecture
   - Modifying training parameters
   - Trying different datasets
   - Implementing custom generation strategies

