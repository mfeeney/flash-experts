"""
Magic: The Gathering card data collection and processing utilities.

This module provides functions to:
1. Fetch card data from the Scryfall API
2. Process and clean card names
3. Prepare data for model training
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MTGDataCollector:
    """Handles collection and processing of Magic: The Gathering card data."""
    
    BASE_URL = "https://api.scryfall.com"
    
    def __init__(self, cache_dir: str = "data/mtg"):
        """
        Initialize the data collector.
        
        Args:
            cache_dir: Directory to store cached card data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cards_cache = self.cache_dir / "cards.json"
        
    def fetch_all_cards(self, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch all Magic: The Gathering cards from Scryfall API.
        
        Args:
            force_refresh: If True, ignore cache and fetch fresh data
            
        Returns:
            List of card data dictionaries
            
        Raises:
            RuntimeError: If the API request fails and no cached data is available
        """
        # Check if cache exists and is valid
        if not force_refresh and self.cards_cache.exists():
            try:
                with open(self.cards_cache, 'r') as f:
                    cached_data = json.load(f)
                    if isinstance(cached_data, list) and len(cached_data) > 0:
                        logger.info("Loading card data from cache...")
                        return cached_data
                    else:
                        logger.warning("Cache file is empty or invalid, fetching fresh data...")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Cache file is corrupted: {e}, fetching fresh data...")
        
        logger.info("Fetching card data from Scryfall API...")
        cards = []
        has_more = True
        # Use a more specific query that the API accepts
        next_page = f"{self.BASE_URL}/cards/search?q=game:paper+is:booster"
        
        try:
            with tqdm(desc="Fetching cards") as pbar:
                while has_more:
                    try:
                        response = requests.get(next_page)
                        response.raise_for_status()
                        data = response.json()
                        
                        if 'data' not in data:
                            logger.error(f"Unexpected API response format: {data}")
                            break
                        
                        # Add cards from current page
                        cards.extend(data['data'])
                        pbar.update(len(data['data']))
                        
                        # Check if there are more pages
                        has_more = data.get('has_more', False)
                        next_page = data.get('next_page')
                        
                        if next_page:
                            # Respect API rate limits
                            time.sleep(0.1)
                        else:
                            break
                            
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Error fetching data: {e}")
                        if not cards and not self.cards_cache.exists():
                            raise RuntimeError(
                                "Failed to fetch card data and no cache available. "
                                "Please check your internet connection and try again."
                            )
                        break
            
            if not cards:
                if self.cards_cache.exists():
                    logger.warning("Using cached data due to API fetch failure")
                    with open(self.cards_cache, 'r') as f:
                        return json.load(f)
                else:
                    raise RuntimeError("No cards fetched and no cache available")
            
            # Cache the results
            with open(self.cards_cache, 'w') as f:
                json.dump(cards, f)
            
            logger.info(f"Fetched {len(cards)} cards")
            return cards
            
        except Exception as e:
            logger.error(f"Unexpected error during data collection: {e}")
            if self.cards_cache.exists():
                logger.warning("Falling back to cached data")
                with open(self.cards_cache, 'r') as f:
                    return json.load(f)
            raise
    
    def process_card_names(self, cards: List[Dict]) -> pd.DataFrame:
        """
        Process card data to extract and clean card names.
        
        Args:
            cards: List of card data dictionaries
            
        Returns:
            DataFrame containing processed card names and metadata
            
        Raises:
            ValueError: If no valid cards are found
        """
        if not cards:
            raise ValueError("No cards provided for processing")
            
        processed_data = []
        
        for card in cards:
            # Skip cards without names or with special characters
            if not card.get('name') or '//' in card['name']:
                continue
                
            processed_data.append({
                'name': card['name'],
                'type_line': card.get('type_line', ''),
                'rarity': card.get('rarity', ''),
                'set_name': card.get('set_name', ''),
                'mana_cost': card.get('mana_cost', ''),
                'colors': ','.join(card.get('colors', [])),
            })
        
        if not processed_data:
            raise ValueError("No valid card names found in the provided data")
            
        df = pd.DataFrame(processed_data)
        
        # Clean and normalize names
        df['name'] = df['name'].str.strip()
        
        # Remove duplicates (some cards appear in multiple sets)
        df = df.drop_duplicates(subset=['name'])
        
        logger.info(f"Processed {len(df)} unique card names")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, output_file: str = "data/mtg/card_names.txt") -> None:
        """
        Prepare card names for model training.
        
        Args:
            df: DataFrame containing processed card data
            output_file: Path to save the training data
            
        Raises:
            ValueError: If the DataFrame is empty
        """
        if df.empty:
            raise ValueError("No card names to save")
            
        # Sort by name length to help model learn patterns
        df = df.sort_values(by='name', key=lambda x: x.str.len())
        
        # Save names to file, one per line
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for name in df['name']:
                f.write(f"{name}\n")
        
        logger.info(f"Saved {len(df)} card names to {output_file}")

def main():
    """Example usage of the MTGDataCollector."""
    try:
        collector = MTGDataCollector()
        
        # Force refresh the data to ensure we have valid cards
        logger.info("Forcing refresh of card data...")
        cards = collector.fetch_all_cards(force_refresh=True)
        
        if not cards:
            raise ValueError("Failed to fetch any cards from the API")
            
        df = collector.process_card_names(cards)
        
        # Prepare training data
        collector.prepare_training_data(df)
        
        # Print some statistics
        print("\nCard name statistics:")
        print(f"Total unique cards: {len(df)}")
        print(f"Average name length: {df['name'].str.len().mean():.1f} characters")
        print(f"Shortest name: {df['name'].iloc[0]}")
        print(f"Longest name: {df['name'].iloc[-1]}")
        
        # Print a few example card names
        print("\nExample card names:")
        print("-" * 50)
        for name in df['name'].sample(n=5):
            print(f"- {name}")
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 