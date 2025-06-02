"""
Script for testing both pre-trained and fine-tuned GPT-2 MoE models with Flash Attention.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import GPT2Tokenizer
from typing import Optional, List, Dict, Union

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.moe import GPT2MoEConfig, GPT2MoEModel
    from src.flash_attention import GPT2FlashAttentionConfig, GPT2FlashAttention
except ImportError:
    print("Error: Could not import required modules.")
    print("Please install the package in development mode:")
    print("pip install -e .")
    sys.exit(1)

def create_model_and_tokenizer(
    model_path: str = "gpt2",
    num_experts: int = 4,
    k: int = 2,
    moe_layer_freq: int = 2,
    use_flash_attention: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple:
    """Create and initialize the model and tokenizer."""
    print(f"\nLoading model from: {model_path}")
    
    # Create base config
    base_config = GPT2MoEConfig.from_pretrained(
        model_path if model_path == "gpt2" else os.path.join(model_path, "config.json"),
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
    model = GPT2MoEModel.from_pretrained(
        model_path if model_path == "gpt2" else model_path,
        config=base_config,
    )
    
    # Replace attention layers with Flash Attention if enabled
    if use_flash_attention:
        for layer in model.transformer.h:
            layer.attn = GPT2FlashAttention(base_config)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def clean_generated_text(text: str) -> str:
    """Clean and format the generated text."""
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Fix spacing around punctuation
    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
    text = text.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
    
    # Fix multiple spaces
    text = ' '.join(text.split())
    
    # Capitalize sentences
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    text = '. '.join(sentences)
    
    # Ensure proper spacing after punctuation
    text = text.replace('.', '. ').replace('!', '! ').replace('?', '? ')
    text = ' '.join(text.split())  # Clean up any double spaces
    
    return text

def filter_tokens(logits: torch.Tensor, tokenizer: GPT2Tokenizer) -> torch.Tensor:
    """Filter out unwanted tokens from the logits."""
    # Create a copy of logits to modify
    filtered_logits = logits.clone()
    
    # Get vocabulary size
    vocab_size = tokenizer.vocab_size
    
    # Filter out special tokens
    special_tokens = set(tokenizer.all_special_ids)
    
    # Filter out very short tokens (1-2 characters)
    short_tokens = set()
    for i in range(vocab_size):
        token = tokenizer.decode([i])
        if len(token.strip()) <= 2 and not token.strip().isalpha():
            short_tokens.add(i)
    
    # Filter out tokens that are just numbers or symbols
    symbol_tokens = set()
    for i in range(vocab_size):
        token = tokenizer.decode([i])
        if token.strip().isdigit() or (len(token.strip()) == 1 and not token.strip().isalpha()):
            symbol_tokens.add(i)
    
    # Combine all unwanted tokens
    unwanted_tokens = special_tokens | short_tokens | symbol_tokens
    
    # Set logits to very low value for unwanted tokens
    if unwanted_tokens:
        filtered_logits[..., list(unwanted_tokens)] = -float('inf')
    
    return filtered_logits

def generate_text(
    model: GPT2MoEModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Generate text using the model."""
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1,
        )
    
    # Decode and clean the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_text = clean_generated_text(generated_text)
    
    return cleaned_text

def compare_models(
    pretrained_model: GPT2MoEModel,
    finetuned_model: GPT2MoEModel,
    tokenizer: GPT2Tokenizer,
    prompts: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[str, str]]:
    """Compare text generation between pre-trained and fine-tuned models."""
    results = {}
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTesting prompt {i}: {prompt}")
        
        # Generate with pre-trained model
        print("\nPre-trained model output:")
        pretrained_output = generate_text(
            pretrained_model, tokenizer, prompt, device=device
        )
        print(pretrained_output)
        
        # Generate with fine-tuned model
        print("\nFine-tuned model output:")
        finetuned_output = generate_text(
            finetuned_model, tokenizer, prompt, device=device
        )
        print(finetuned_output)
        
        results[f"prompt_{i}"] = {
            "prompt": prompt,
            "pretrained_output": pretrained_output,
            "finetuned_output": finetuned_output
        }
    
    return results

def main():
    """Main function to test and compare models."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create pre-trained model
    print("\nLoading pre-trained model...")
    pretrained_model, tokenizer = create_model_and_tokenizer(
        model_path="gpt2",
        num_experts=4,
        k=2,
        moe_layer_freq=2,
        use_flash_attention=True,
        device=device,
    )
    
    # Create fine-tuned model
    print("\nLoading fine-tuned model...")
    finetuned_model, _ = create_model_and_tokenizer(
        model_path="./fine_tuned_model",
        num_experts=4,
        k=2,
        moe_layer_freq=2,
        use_flash_attention=True,
        device=device,
    )
    
    # Test prompts
    test_prompts = [
        "The history of artificial intelligence",
        "In the beginning of the universe",
        "The most important scientific discovery",
        "The future of technology",
        "The art of programming",
    ]
    
    # Compare models
    print("\nComparing model outputs...")
    results = compare_models(
        pretrained_model,
        finetuned_model,
        tokenizer,
        test_prompts,
        device=device,
    )
    
    # Print summary
    print("\nModel Comparison Summary:")
    print("------------------------")
    for prompt_id, result in results.items():
        print(f"\n{result['prompt']}")
        print("Pre-trained length:", len(result['pretrained_output']))
        print("Fine-tuned length:", len(result['finetuned_output']))
        print("-" * 50)

if __name__ == "__main__":
    main() 