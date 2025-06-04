"""
Enhanced Magic: The Gathering card data processing and cleaning utilities.

This module provides functions to:
1. Clean and validate card data
2. Structure card data for training
3. Prepare data for both name-only and full card generation
4. Add metadata and relationships between cards
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CardData:
    """Structured card data for training."""
    name: str
    type_line: str
    mana_cost: str
    colors: List[str]
    rarity: str
    set_name: str
    power: Optional[str] = None
    toughness: Optional[str] = None
    loyalty: Optional[str] = None
    keywords: List[str] = None
    flavor_text: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'type_line': self.type_line,
            'mana_cost': self.mana_cost,
            'colors': self.colors,
            'rarity': self.rarity,
            'set_name': self.set_name,
            'power': self.power,
            'toughness': self.toughness,
            'loyalty': self.loyalty,
            'keywords': self.keywords or [],
            'flavor_text': self.flavor_text
        }

class MTGDataProcessor:
    """Processes and cleans Magic: The Gathering card data."""
    
    # Common card types and subtypes
    CARD_TYPES = {
        'creature', 'instant', 'sorcery', 'enchantment', 'artifact',
        'planeswalker', 'land', 'tribal', 'legendary'
    }
    
    # Common creature types
    CREATURE_TYPES = {
        'angel', 'dragon', 'wizard', 'knight', 'elf', 'goblin', 'zombie',
        'demon', 'beast', 'elemental', 'human', 'warrior', 'shaman'
    }
    
    # Common keywords
    KEYWORDS = {
        'flying', 'first strike', 'double strike', 'haste', 'vigilance',
        'deathtouch', 'lifelink', 'menace', 'reach', 'trample', 'hexproof',
        'indestructible', 'flash', 'defender'
    }
    
    def __init__(self, data_dir: str = "data/mtg"):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing card data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cards_cache = self.data_dir / "cards.json"
        self.processed_data = self.data_dir / "processed_cards.json"
        self.training_data = self.data_dir / "training_data.json"
        
        # Load common words and patterns
        self._load_common_patterns()
    
    def _load_common_patterns(self):
        """Load common words and patterns for validation."""
        # Common prefixes in card names
        self.name_prefixes = {
            'ancient', 'eternal', 'mighty', 'great', 'powerful', 'mystic',
            'dark', 'light', 'shadow', 'crystal', 'magic', 'arcane', 'divine'
        }
        
        # Common suffixes in card names
        self.name_suffixes = {
            'mage', 'knight', 'dragon', 'beast', 'spirit', 'elemental',
            'warrior', 'wizard', 'shaman', 'priest', 'guardian', 'lord'
        }
        
        # Common name patterns
        self.name_patterns = [
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Properly capitalized words
            r'^[A-Z][a-z]+\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # "X of Y" pattern
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Three word names
        ]
    
    def clean_card_name(self, name: str) -> Optional[str]:
        """
        Clean and validate a card name.
        
        Args:
            name: Raw card name
            
        Returns:
            Cleaned name or None if invalid
        """
        # Remove special characters and normalize whitespace
        name = re.sub(r'[®©™]', '', name)
        name = ' '.join(name.split())
        
        # Skip if name is too short or too long
        if len(name) < 3 or len(name) > 50:
            return None
        
        # Skip if name contains numbers or special characters
        if re.search(r'[0-9]', name) or re.search(r'[^a-zA-Z\s\'-]', name):
            return None
        
        # Skip joke cards and special cards
        joke_indicators = {
            'joke', 'test', 'token', 'playtest', 'promo', 'special',
            'customer service', 'market research', 'longest card name'
        }
        if any(indicator in name.lower() for indicator in joke_indicators):
            return None
        
        # Validate name format
        if not any(re.match(pattern, name) for pattern in self.name_patterns):
            # Allow some exceptions for special cases
            if not (name.startswith('The ') or  # Allow "The X" names
                   any(name.lower().startswith(prefix) for prefix in self.name_prefixes) or
                   any(name.lower().endswith(suffix) for suffix in self.name_suffixes)):
                return None
        
        return name
    
    def extract_card_type(self, type_line: str) -> Tuple[str, List[str]]:
        """
        Extract main type and subtypes from type line.
        
        Args:
            type_line: Raw type line (e.g., "Legendary Creature — Angel")
            
        Returns:
            Tuple of (main_type, subtypes)
        """
        # Split on em dash or hyphen
        parts = re.split(r'[—-]', type_line)
        main_type = parts[0].strip().lower()
        
        # Extract subtypes if present
        subtypes = []
        if len(parts) > 1:
            subtypes = [s.strip().lower() for s in parts[1].split()]
        
        return main_type, subtypes
    
    def process_card(self, card: Dict) -> Optional[CardData]:
        """
        Process a single card's data.
        
        Args:
            card: Raw card data dictionary
            
        Returns:
            Processed CardData or None if invalid
        """
        # Clean and validate name
        name = self.clean_card_name(card.get('name', ''))
        if not name:
            return None
        
        # Extract and validate type
        type_line = card.get('type_line', '')
        main_type, subtypes = self.extract_card_type(type_line)
        
        # Skip if main type is not recognized
        if main_type not in self.CARD_TYPES:
            return None
        
        # Process colors
        colors = [c.lower() for c in card.get('colors', [])]
        
        # Process power/toughness for creatures
        power = toughness = loyalty = None
        if main_type == 'creature':
            if 'power' in card and 'toughness' in card:
                power = card['power']
                toughness = card['toughness']
        elif main_type == 'planeswalker':
            loyalty = card.get('loyalty')
        
        # Extract keywords
        keywords = []
        if 'keywords' in card:
            keywords = [k.lower() for k in card['keywords'] if k.lower() in self.KEYWORDS]
        
        # Create structured card data
        return CardData(
            name=name,
            type_line=type_line,
            mana_cost=card.get('mana_cost', ''),
            colors=colors,
            rarity=card.get('rarity', '').lower(),
            set_name=card.get('set_name', ''),
            power=power,
            toughness=toughness,
            loyalty=loyalty,
            keywords=keywords,
            flavor_text=card.get('flavor_text')
        )
    
    def process_all_cards(self, force_refresh: bool = False) -> List[CardData]:
        """
        Process all cards in the dataset.
        
        Args:
            force_refresh: If True, reprocess all cards
            
        Returns:
            List of processed CardData objects
        """
        # Load raw card data
        if not force_refresh and self.processed_data.exists():
            try:
                with open(self.processed_data, 'r') as f:
                    return [CardData(**card) for card in json.load(f)]
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error loading processed data: {e}")
        
        # Load raw cards
        if not self.cards_cache.exists():
            raise FileNotFoundError("No card data found. Please run data collection first.")
        
        with open(self.cards_cache, 'r') as f:
            raw_cards = json.load(f)
        
        # Process cards
        processed_cards = []
        for card in tqdm(raw_cards, desc="Processing cards"):
            processed = self.process_card(card)
            if processed:
                processed_cards.append(processed)
        
        # Save processed data
        with open(self.processed_data, 'w') as f:
            json.dump([card.to_dict() for card in processed_cards], f, indent=2)
        
        logger.info(f"Processed {len(processed_cards)} valid cards")
        return processed_cards
    
    def prepare_training_data(self, cards: List[CardData]) -> Dict:
        """
        Prepare structured training data for the model.
        
        Args:
            cards: List of processed CardData objects
            
        Returns:
            Dictionary containing training data in various formats
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([card.to_dict() for card in cards])
        
        # Helper function to convert numpy types to native Python types
        def convert_to_native_types(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_to_native_types(x) for x in obj]
            return obj
        
        # Prepare different training formats
        training_data = {
            # Basic name-only format
            'names': {
                'data': df['name'].tolist(),
                'stats': {
                    'total': int(len(df)),
                    'avg_length': float(df['name'].str.len().mean()),
                    'min_length': int(df['name'].str.len().min()),
                    'max_length': int(df['name'].str.len().max())
                }
            },
            
            # Name with type format
            'names_with_types': {
                'data': df.apply(lambda x: f"{x['name']} | {x['type_line']}", axis=1).tolist(),
                'stats': {
                    'total': int(len(df)),
                    'type_distribution': convert_to_native_types(df['type_line'].value_counts().to_dict())
                }
            },
            
            # Full card format
            'full_cards': {
                'data': df.apply(lambda x: {
                    'name': x['name'],
                    'type': x['type_line'],
                    'cost': x['mana_cost'],
                    'colors': x['colors'],
                    'rarity': x['rarity'],
                    'power': x['power'],
                    'toughness': x['toughness'],
                    'keywords': x['keywords']
                }, axis=1).tolist(),
                'stats': {
                    'total': int(len(df)),
                    'color_distribution': convert_to_native_types(df['colors'].apply(lambda x: len(x)).value_counts().to_dict()),
                    'rarity_distribution': convert_to_native_types(df['rarity'].value_counts().to_dict())
                }
            }
        }
        
        # Convert all numpy types to native Python types
        training_data = convert_to_native_types(training_data)
        
        # Save training data
        with open(self.training_data, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Print statistics
        logger.info("\nTraining Data Statistics:")
        logger.info(f"Total cards: {len(df)}")
        logger.info(f"Average name length: {df['name'].str.len().mean():.1f} characters")
        logger.info("\nType Distribution:")
        for type_, count in df['type_line'].value_counts().head().items():
            logger.info(f"  {type_}: {count}")
        logger.info("\nColor Distribution:")
        for colors, count in df['colors'].apply(lambda x: len(x)).value_counts().items():
            logger.info(f"  {colors} colors: {count}")
        
        return training_data

def main():
    """Example usage of the MTGDataProcessor."""
    try:
        processor = MTGDataProcessor()
        
        # Process all cards
        logger.info("Processing card data...")
        cards = processor.process_all_cards(force_refresh=True)
        
        if not cards:
            raise ValueError("No valid cards found after processing")
        
        # Prepare training data
        logger.info("\nPreparing training data...")
        training_data = processor.prepare_training_data(cards)
        
        # Print some example cards
        print("\nExample processed cards:")
        print("-" * 50)
        for card in cards[:5]:
            print(f"Name: {card.name}")
            print(f"Type: {card.type_line}")
            print(f"Colors: {', '.join(card.colors)}")
            if card.power and card.toughness:
                print(f"P/T: {card.power}/{card.toughness}")
            print("-" * 50)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 