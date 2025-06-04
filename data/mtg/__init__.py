"""
Magic: The Gathering data collection and processing utilities.
"""

from .collector import MTGDataCollector
from .processor import MTGDataProcessor, CardData

__all__ = [
    'MTGDataCollector',
    'MTGDataProcessor',
    'CardData',
]
