"""
Script for cleaning and preparing MTG card name training data.
This script:
1. Cleans card names (removes invalid names, enforces length limits)
2. Validates tokenization
3. Splits data into train/validation sets
4. Creates curriculum stages with proper validation
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import Counter
import random
from transformers import GPT2Tokenizer
import sys

@dataclass
class CardName:
    """Represents a cleaned and validated card name."""
    name: str
    card_type: str
    tokens: List[int]
    token_count: int
    word_count: int
    is_valid: bool
    validation_notes: List[str]

class MTGDataCleaner:
    """Class for cleaning and preparing MTG card name data."""
    
    def __init__(self, min_name_length: int = 3, max_name_length: int = 40):
        """Initialize the cleaner with validation parameters."""
        self.min_name_length = min_name_length
        self.max_name_length = max_name_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Card type patterns
        self.card_type_patterns = {
            'artifact': r'\bartifact\b',
            'land': r'\bland\b',
            'instant': r'\binstant\b',
            'sorcery': r'\bsorcery\b',
            'enchantment': r'\benchantment\b',
            'creature': r'\bcreature\b',
            'planeswalker': r'\bplaneswalker\b',
            'battle': r'\bbattle\b',
            'tribal': r'\btribal\b'
        }
        
        # Words that shouldn't appear in card names (only actual card types and mechanics)
        self.invalid_words = {
            'card', 'type', 'word', 'create', 'short', 'unique',
            'sorcery', 'instant', 'enchantment', 'artifact', 'land', 'creature', 'planeswalker',
            'battle', 'tribal', 'token', 'counter', 'mana', 'cost', 'power', 'toughness'
        }
        
        # Common suffixes to check for repetition (only check if they appear 3+ times)
        self.common_suffixes = ['er', 'ist', 'ing', 'or', 'ed', 'ian', 'ish', 'ic', 'al', 'ar', 'ur', 'ra', 'ry']
        
        # Track statistics
        self.stats = {
            'total_cards': 0,
            'valid_cards': 0,
            'invalid_cards': 0,
            'reasons': Counter(),
            'card_types': Counter(),
            'name_lengths': Counter(),
            'word_counts': Counter(),
            'token_counts': Counter()
        }
    
    def extract_card_type(self, type_line: str) -> str:
        """Extract the primary card type from the type line."""
        if not type_line or type_line == 'unknown':
            return 'unknown'
            
        # Debug output for first few cards
        if self.stats['total_cards'] < 5:
            print(f"\nDebug - Type line: '{type_line}'")
        
        # Convert to lowercase and handle em dash
        type_line = type_line.lower()
        type_line = type_line.replace('—', '-')  # Normalize em dash to hyphen
        
        # Split on hyphen to separate main type from subtypes
        parts = type_line.split('-')
        main_type = parts[0].strip()
        
        # Split main type into words
        words = main_type.split()
        
        # Handle special cases
        if '//' in type_line:  # Double-faced cards
            return 'double-faced'
        
        # Look for card types in the main type part
        # First, try to find the last card type word (after any supertypes)
        for word in reversed(words):
            if word in self.card_type_patterns:
                return word
        
        # If no card type found, try to match against patterns
        for card_type, pattern in self.card_type_patterns.items():
            if re.search(pattern, main_type):
                return card_type
        
        # If still no match, try to extract from the first word after any supertypes
        # Common supertypes: basic, legendary, snow, world, ongoing
        supertypes = {'basic', 'legendary', 'snow', 'world', 'ongoing'}
        for word in words:
            if word not in supertypes and word in self.card_type_patterns:
                return word
        
        # Debug output for first few unknown types
        if self.stats['total_cards'] < 5:
            print(f"Debug - Could not extract type from: '{type_line}'")
            print(f"Debug - Main type part: '{main_type}'")
            print(f"Debug - Words: {words}")
        
        return 'unknown'
    
    def validate_name(self, name: str, card_type: str) -> Tuple[bool, List[str]]:
        """Validate a card name and return validation notes."""
        notes = []
        
        # Check length
        if len(name) < self.min_name_length:
            notes.append(f"Name too short: {len(name)} chars")
            return False, notes
        if len(name) > self.max_name_length:
            notes.append(f"Name too long: {len(name)} chars")
            return False, notes
        
        # Check for invalid words (only card types and mechanics)
        words = name.lower().split()
        for word in words:
            if word in self.invalid_words:
                notes.append(f"Contains invalid word: {word}")
                return False, notes
        
        # Check for repetitive suffixes (only if they appear 3+ times)
        for word in words:
            for suffix in self.common_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 3:
                    # Check if this suffix appears too often in the name
                    suffix_count = sum(1 for w in words if w.endswith(suffix))
                    if suffix_count >= 3:  # Changed from > 1 to >= 3
                        notes.append(f"Repetitive suffix: {suffix}")
                        return False, notes
        
        # Check for proper capitalization (allow some exceptions)
        words = name.split()
        if not all(word[0].isupper() for word in words if word.lower() not in {'of', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as'}):
            notes.append("Improper capitalization")
            return False, notes
        
        # Check for special characters (allow more punctuation)
        if re.search(r'[^a-zA-Z\s\'\-,]', name):  # Added comma and hyphen
            notes.append("Contains invalid characters")
            return False, notes
        
        return True, notes
    
    def process_card(self, card: Dict) -> CardName:
        """Process a single card and return a CardName object."""
        name = card['name'].strip()
        
        # Get type line, handling different possible field names
        type_line = card.get('type_line', card.get('type', 'unknown'))
        if isinstance(type_line, list):
            type_line = ' '.join(type_line)
        
        # Debug output for first few cards
        if self.stats['total_cards'] < 5:
            print(f"\nDebug - Processing card: '{name}'")
            print(f"Debug - Raw type line: '{type_line}'")
            print(f"Debug - Available fields: {list(card.keys())}")
            if 'type' in card:
                print(f"Debug - Type field: '{card['type']}'")
            if 'type_line' in card:
                print(f"Debug - Type line field: '{card['type_line']}'")
        
        # Extract card type
        card_type = self.extract_card_type(type_line)
        
        # Debug output for first few cards
        if self.stats['total_cards'] < 5:
            print(f"Debug - Extracted type: '{card_type}'")
        
        # Validate name
        is_valid, notes = self.validate_name(name, card_type)
        
        # Tokenize
        tokens = self.tokenizer.encode(name, add_special_tokens=False)
        token_count = len(tokens)
        word_count = len(name.split())
        
        # Update statistics
        self.stats['total_cards'] += 1
        if is_valid:
            self.stats['valid_cards'] += 1
            self.stats['card_types'][card_type] += 1
            self.stats['name_lengths'][len(name)] += 1
            self.stats['word_counts'][word_count] += 1
            self.stats['token_counts'][token_count] += 1
        else:
            self.stats['invalid_cards'] += 1
            for note in notes:
                self.stats['reasons'][note] += 1
        
        return CardName(
            name=name,
            card_type=card_type,
            tokens=tokens,
            token_count=token_count,
            word_count=word_count,
            is_valid=is_valid,
            validation_notes=notes
        )
    
    def create_curriculum_stages(self, cards: List[CardName], validation_split: float = 0.1) -> Dict:
        """Create curriculum stages with validation sets."""
        # Filter valid cards
        valid_cards = [card for card in cards if card.is_valid]
        
        if not valid_cards:
            raise ValueError("No valid cards found after filtering! Check the validation criteria.")
        
        # Print token count distribution
        token_counts = [card.token_count for card in valid_cards]
        print("\nToken count distribution:")
        print(f"Min tokens: {min(token_counts)}")
        print(f"Max tokens: {max(token_counts)}")
        print(f"Mean tokens: {sum(token_counts) / len(token_counts):.1f}")
        
        # Sort by token count for curriculum
        valid_cards.sort(key=lambda x: x.token_count)
        
        # Create stages based on token count
        stages = {
            'stage_0': [],  # 1-10 tokens
            'stage_1': [],  # 11-20 tokens
            'stage_2': []   # 21+ tokens
        }
        
        # Distribute cards into stages
        for card in valid_cards:
            if card.token_count <= 10:
                stages['stage_0'].append(card)
            elif card.token_count <= 20:
                stages['stage_1'].append(card)
            else:
                stages['stage_2'].append(card)
        
        # Print stage distribution
        print("\nInitial stage distribution:")
        for stage_name, stage_cards in stages.items():
            print(f"{stage_name}: {len(stage_cards)} cards")
        
        # Verify no empty stages
        empty_stages = [stage for stage, cards in stages.items() if not cards]
        if empty_stages:
            print("\nWarning: Empty stages detected!")
            print("Empty stages:", empty_stages)
            print("\nAdjusting stage boundaries to ensure non-empty stages...")
            
            # Redistribute cards to ensure no empty stages
            stages = {
                'stage_0': [],  # First third
                'stage_1': [],  # Middle third
                'stage_2': []   # Last third
            }
            
            # Calculate split points
            total_cards = len(valid_cards)
            split1 = total_cards // 3
            split2 = 2 * total_cards // 3
            
            # Redistribute
            stages['stage_0'] = valid_cards[:split1]
            stages['stage_1'] = valid_cards[split1:split2]
            stages['stage_2'] = valid_cards[split2:]
            
            print("\nNew stage distribution:")
            for stage_name, stage_cards in stages.items():
                print(f"{stage_name}: {len(stage_cards)} cards")
        
        # Split each stage into train/validation
        result = {}
        for stage_name, stage_cards in stages.items():
            if not stage_cards:
                continue
                
            # Shuffle cards
            random.shuffle(stage_cards)
            
            # Calculate split point
            split_idx = max(1, int(len(stage_cards) * (1 - validation_split)))
            
            # Split into train/validation
            train_cards = stage_cards[:split_idx]
            val_cards = stage_cards[split_idx:]
            
            # Calculate statistics
            avg_tokens = sum(c.token_count for c in stage_cards) / len(stage_cards)
            avg_words = sum(c.word_count for c in stage_cards) / len(stage_cards)
            
            result[stage_name] = {
                'train': train_cards,
                'validation': val_cards,
                'stats': {
                    'total': len(stage_cards),
                    'train': len(train_cards),
                    'validation': len(val_cards),
                    'avg_tokens': avg_tokens,
                    'avg_words': avg_words,
                    'min_tokens': min(c.token_count for c in stage_cards),
                    'max_tokens': max(c.token_count for c in stage_cards)
                }
            }
        
        return result
    
    def save_processed_data(self, curriculum_data: Dict, output_dir: str):
        """Save processed data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save curriculum data
        for stage_name, stage_data in curriculum_data.items():
            stage_path = output_path / stage_name
            stage_path.mkdir(exist_ok=True)
            
            # Save train data
            train_data = {
                'cards': [
                    {
                        'name': card.name,
                        'type': card.card_type,
                        'tokens': card.tokens,
                        'token_count': card.token_count,
                        'word_count': card.word_count
                    }
                    for card in stage_data['train']
                ],
                'stats': stage_data['stats']
            }
            
            with open(stage_path / 'train.json', 'w') as f:
                json.dump(train_data, f, indent=2)
            
            # Save validation data
            val_data = {
                'cards': [
                    {
                        'name': card.name,
                        'type': card.card_type,
                        'tokens': card.tokens,
                        'token_count': card.token_count,
                        'word_count': card.word_count
                    }
                    for card in stage_data['validation']
                ],
                'stats': stage_data['stats']
            }
            
            with open(stage_path / 'validation.json', 'w') as f:
                json.dump(val_data, f, indent=2)
        
        # Save overall statistics
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save tokenizer vocabulary
        self.tokenizer.save_pretrained(output_path / 'tokenizer')

def main():
    """Main function to process MTG card data."""
    # Initialize cleaner
    cleaner = MTGDataCleaner()
    
    # Load raw data
    print("Loading raw MTG card data...")
    with open("data/mtg/training_data.json", 'r') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data['full_cards']['data'])} cards from raw data")
    
    # Print sample of raw data for debugging
    print("\nSample of raw data (first 5 cards):")
    for i, card in enumerate(raw_data['full_cards']['data'][:5]):
        print(f"\nCard {i + 1}:")
        print(f"  Name: {card.get('name', 'N/A')}")
        print(f"  Type: {card.get('type_line', card.get('type', 'N/A'))}")
        print(f"  Raw data: {card}")  # Print full card data for debugging
    
    # Process cards
    print("\nProcessing cards...")
    processed_cards = []
    for card in raw_data['full_cards']['data']:
        processed_card = cleaner.process_card(card)
        processed_cards.append(processed_card)
    
    # Print validation statistics before curriculum
    print("\nValidation Statistics:")
    valid_count = sum(1 for card in processed_cards if card.is_valid)
    print(f"Valid cards: {valid_count} ({valid_count/len(processed_cards)*100:.1f}%)")
    print(f"Invalid cards: {len(processed_cards) - valid_count}")
    
    # Print card type distribution before curriculum
    print("\nCard type distribution before curriculum:")
    type_counts = Counter(card.card_type for card in processed_cards if card.is_valid)
    total_valid = sum(type_counts.values())
    for card_type, count in type_counts.most_common():
        print(f"  {card_type}: {count} ({count/total_valid*100:.1f}%)")
    
    # Print sample of processed cards for debugging
    print("\nSample of processed cards (first 3 valid cards):")
    valid_cards = [card for card in processed_cards if card.is_valid][:3]
    for i, card in enumerate(valid_cards):
        print(f"\nProcessed Card {i + 1}:")
        print(f"  Name: {card.name}")
        print(f"  Type: {card.card_type}")
        print(f"  Tokens: {card.token_count}")
        print(f"  Words: {card.word_count}")
    
    if valid_count == 0:
        print("\nNo valid cards found! Most common validation failures:")
        for reason, count in cleaner.stats['reasons'].most_common(5):
            print(f"  {reason}: {count}")
        sys.exit(1)
    
    # Create curriculum stages
    print("\nCreating curriculum stages...")
    try:
        curriculum_data = cleaner.create_curriculum_stages(processed_cards)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Save processed data
    print("\nSaving processed data...")
    cleaner.save_processed_data(curriculum_data, "data/mtg/processed")
    
    # Print final statistics
    print("\nProcessing Statistics:")
    print(f"Total cards processed: {cleaner.stats['total_cards']}")
    print(f"Valid cards: {cleaner.stats['valid_cards']}")
    print(f"Invalid cards: {cleaner.stats['invalid_cards']}")
    
    print("\nInvalid card reasons:")
    for reason, count in cleaner.stats['reasons'].most_common():
        print(f"  {reason}: {count}")
    
    print("\nCard type distribution:")
    for card_type, count in cleaner.stats['card_types'].most_common():
        print(f"  {card_type}: {count} ({count/valid_count*100:.1f}%)")
    
    print("\nCurriculum stage statistics:")
    for stage_name, stage_data in curriculum_data.items():
        stats = stage_data['stats']
        print(f"\n{stage_name}:")
        print(f"  Total cards: {stats['total']}")
        print(f"  Train cards: {stats['train']}")
        print(f"  Validation cards: {stats['validation']}")
        print(f"  Average tokens: {stats['avg_tokens']:.1f}")
        print(f"  Average words: {stats['avg_words']:.1f}")
        print(f"  Token range: {stats['min_tokens']} - {stats['max_tokens']}")
        
        # Print card type distribution for this stage
        stage_cards = stage_data['train'] + stage_data['validation']
        stage_types = Counter(card.card_type for card in stage_cards)
        print("  Card type distribution:")
        for card_type, count in stage_types.most_common():
            print(f"    {card_type}: {count} ({count/stats['total']*100:.1f}%)")

if __name__ == "__main__":
    main() 