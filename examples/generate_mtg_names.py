"""
Script for testing the fine-tuned GPT-2 MoE model's card name generation capabilities.
"""

import os
import sys
from pathlib import Path
import json
import torch
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import GPT2Tokenizer, GPT2Config
try:
    from flash_experts.models.moe import GPT2MoEModel, GPT2MoEConfig
    from flash_experts.models.flash_attention import GPT2FlashAttention
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please install the package in development mode:")
    print("pip install -e .")
    sys.exit(1)

@dataclass
class GenerationConfig:
    """Configuration for card name generation."""
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    max_length: int = 32
    num_return_sequences: int = 5
    repetition_penalty: float = 1.2
    name: str = "Default"
    min_length: int = 8  # Reduced minimum length
    max_length: int = 32  # Reduced maximum length
    no_repeat_ngram_size: int = 3  # Increased to prevent more repetition
    bad_words_ids: Optional[List[List[int]]] = None  # Added to prevent certain words

class CardNameGenerator:
    """Class for generating and analyzing MTG card names."""
    
    def __init__(
        self,
        model_path: str = "./fine_tuned_model/mtg_names/stage_2/",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the generator with the fine-tuned model."""
        self.device = device
        print(f"Loading model from {model_path}...")
        
        # Load config first
        config = GPT2MoEConfig.from_pretrained(
            model_path,
            num_experts=2,
            k=1,
            moe_layer_freq=2,
            use_flash_attention=True
        )
        
        # Create model with config
        self.model = GPT2MoEModel(config)
        
        # Load state dict directly with torch.load
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location=device,
            weights_only=True
        )
        
        # Filter out standard attention parameters since we're using Flash Attention
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(x in k for x in ['attn.c_attn', 'attn.c_proj'])
        }
        
        # Load the filtered state dict
        self.model.load_state_dict(filtered_state_dict, strict=False)
        
        # Replace attention layers with Flash Attention
        for layer in self.model.transformer.h:
            layer.attn = GPT2FlashAttention(config)
        
        self.model.to(device)
        self.model.eval()
        
        # Load tokenizer from base GPT-2 model
        print("Loading tokenizer from base GPT-2 model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load training data for analysis
        with open("data/mtg/training_data.json", 'r') as f:
            self.training_data = json.load(f)
        
        # Define prompts for different card types with more specific instructions
        self.prompts = {
            'artifact': "Create a short, unique artifact name (1-2 words, no card type words): ",
            'land': "Create a short, unique land name (1-2 words, no card type words): ",
            'instant': "Create a short, unique instant name (1-2 words, no card type words): ",
            'sorcery': "Create a short, unique sorcery name (1-2 words, no card type words): ",
            'enchantment': "Create a short, unique enchantment name (1-2 words, no card type words): ",
            'creature': "Create a short, unique creature name (1-2 words, no card type words): ",
            'planeswalker': "Create a short, unique planeswalker name (1-2 words, no card type words): ",
            'any': "Create a short, unique card name (1-2 words, no card type words): "
        }
        
        # Common suffixes to clean up
        self.common_suffixes = ['er', 'ist', 'ing', 'or', 'ed', 'ian', 'ish', 'ic', 'al', 'ar', 'ur', 'ra', 'ry']
        
        # Words to avoid in generated names
        self.bad_words = {
            'named', 'name', 'card', 'type', 'word', 'create', 'short', 'unique',
            'sorcery', 'instant', 'enchantment', 'artifact', 'land', 'creature', 'planeswalker',
            'of', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
            'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        # Convert bad words to token IDs
        self.bad_words_ids = [
            self.tokenizer.encode(word, add_special_tokens=False)
            for word in self.bad_words
        ]
        
        print("Model loaded successfully!")
    
    def clean_generated_name(self, name: str, card_type: str) -> str:
        """Clean and format a generated card name."""
        # Remove the prompt
        name = name.replace(self.prompts[card_type], "").strip()
        
        # Remove any card type words and bad words
        words = name.split()
        cleaned_words = []
        for word in words:
            word = word.lower()
            # Skip bad words
            if word in self.bad_words:
                continue
            # Skip words that are too short or too long
            if len(word) < 3 or len(word) > 12:
                continue
            # Skip words that are just numbers or special characters
            if not any(c.isalpha() for c in word):
                continue
            cleaned_words.append(word)
        
        if not cleaned_words:
            return ""
        
        # Take only the first 2 words
        cleaned_words = cleaned_words[:2]
        
        # Remove common suffixes from each word
        final_words = []
        for word in cleaned_words:
            # Keep the word if it's a proper noun or short
            if word[0].isupper() or len(word) <= 4:
                final_words.append(word)
                continue
            
            # Remove common suffixes
            for suffix in self.common_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 3:
                    word = word[:-len(suffix)]
                    break
            
            final_words.append(word)
        
        # Rejoin and clean up
        name = ' '.join(final_words)
        name = re.sub(r'\s+', ' ', name)  # Remove extra spaces
        name = name.strip()
        
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def generate_names(
        self,
        card_type: str = 'any',
        num_batches: int = 1,
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """Generate card names for a specific type."""
        if config is None:
            config = GenerationConfig()
        
        # Set bad words for this generation
        config.bad_words_ids = self.bad_words_ids
        
        prompt = self.prompts.get(card_type.lower(), self.prompts['any'])
        all_names = []
        
        with torch.no_grad():
            for _ in range(num_batches):
                # Encode prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Generate names with more controlled parameters
                outputs = self.model.generate(
                    **inputs,
                    max_length=config.max_length,
                    min_length=config.min_length,
                    num_return_sequences=config.num_return_sequences,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    bad_words_ids=config.bad_words_ids,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
                
                # Decode and clean names
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                names = [self.clean_generated_name(text, card_type) for text in generated_texts]
                # Filter out empty names
                names = [name for name in names if name]
                all_names.extend(names)
        
        return all_names
    
    def analyze_names(self, names: List[str]) -> Dict:
        """Analyze generated card names."""
        # Basic statistics
        total_names = len(names)
        avg_length = sum(len(name) for name in names) / total_names
        word_counts = [len(name.split()) for name in names]
        avg_words = sum(word_counts) / total_names
        
        # Word frequency analysis
        all_words = [word.lower() for name in names for word in name.split()]
        word_freq = Counter(all_words)
        
        # Length distribution
        length_dist = Counter(len(name) for name in names)
        
        # Compare with training data
        training_names = [card['name'] for card in self.training_data['full_cards']['data']]
        training_avg_length = sum(len(name) for name in training_names) / len(training_names)
        
        # Analyze word patterns
        pattern_stats = {
            'proper_nouns': sum(1 for name in names if any(word[0].isupper() for word in name.split())),
            'single_word_names': sum(1 for name in names if len(name.split()) == 1),
            'multi_word_names': sum(1 for name in names if len(name.split()) > 1)
        }
        
        return {
            'total_names': total_names,
            'average_length': avg_length,
            'average_words': avg_words,
            'length_distribution': dict(length_dist),
            'most_common_words': word_freq.most_common(10),
            'training_data_comparison': {
                'generated_avg_length': avg_length,
                'training_avg_length': training_avg_length,
                'length_difference': avg_length - training_avg_length
            },
            'pattern_statistics': pattern_stats
        }

def main():
    """Main function to test the model's card name generation."""
    # Initialize generator
    generator = CardNameGenerator()
    
    # Test different card types
    card_types = ['artifact', 'land', 'instant', 'sorcery', 'enchantment', 'creature', 'planeswalker']
    
    # Generation configs to test with improved parameters
    configs = [
        GenerationConfig(
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.5,  # Increased to reduce repetition
            name="Balanced",
            min_length=8,
            max_length=32,
            no_repeat_ngram_size=3
        ),
        GenerationConfig(
            temperature=0.9,  # Slightly reduced for more coherence
            top_p=0.92,
            repetition_penalty=1.6,  # Increased to reduce repetition
            name="Creative",
            min_length=8,
            max_length=32,
            no_repeat_ngram_size=3
        ),
        GenerationConfig(
            temperature=0.6,  # Slightly increased for more variety
            top_p=0.8,
            repetition_penalty=1.4,  # Increased to reduce repetition
            name="Conservative",
            min_length=8,
            max_length=32,
            no_repeat_ngram_size=3
        )
    ]
    
    print("\nTesting card name generation...")
    print("=" * 80)
    
    for card_type in card_types:
        print(f"\nGenerating {card_type} names:")
        print("-" * 40)
        
        for config in configs:
            print(f"\nUsing {config.name} generation settings:")
            names = generator.generate_names(card_type, num_batches=2, config=config)
            
            # Print generated names
            print("\nGenerated names:")
            for i, name in enumerate(names, 1):
                print(f"{i}. {name}")
            
            # Analyze and print statistics
            analysis = generator.analyze_names(names)
            print("\nAnalysis:")
            print(f"Average length: {analysis['average_length']:.1f} characters")
            print(f"Average words: {analysis['average_words']:.1f}")
            print("\nPattern Statistics:")
            print(f"Proper nouns: {analysis['pattern_statistics']['proper_nouns']}")
            print(f"Single-word names: {analysis['pattern_statistics']['single_word_names']}")
            print(f"Multi-word names: {analysis['pattern_statistics']['multi_word_names']}")
            print("\nMost common words:")
            for word, count in analysis['most_common_words']:
                print(f"  {word}: {count}")
            print("\n" + "=" * 40)

if __name__ == "__main__":
    main() 