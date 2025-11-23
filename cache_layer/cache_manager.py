"""Cache manager with warm fetch and offline mode"""

from typing import Optional
import pandas as pd
from config import CACHE_CONFIG
from .sqlite_cache import SQLiteCache
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CacheManager:
    """High-level cache manager with warm fetch support"""
    
    def __init__(self):
        """Initialize cache manager"""
        self.cache = SQLiteCache() if CACHE_CONFIG["USE_SQLITE"] else None
        self.enabled = CACHE_CONFIG["ENABLED"]
    
    def get_cached_data(self, ticker: str, days: int = None) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and valid
        
        Args:
            ticker: Ticker symbol
            days: Number of days to retrieve
        
        Returns:
            DataFrame or None
        """
        if not self.enabled or not self.cache:
            return None
        
        if self.cache.is_valid(ticker):
            return self.cache.get_data(ticker, days)
        
        return None
    
    def store_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Store data in cache
        
        Args:
            ticker: Ticker symbol
            df: DataFrame to store
        
        Returns:
            True if successful
        """
        if not self.enabled or not self.cache:
            return False
        
        return self.cache.store_data(ticker, df)
    
    def warm_fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Warm fetch: get data from cache if available
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            DataFrame or None
        """
        if not CACHE_CONFIG["WARM_FETCH"]:
            return None
        
        return self.get_cached_data(ticker)
    
    def is_offline_mode(self) -> bool:
        """Check if we should use offline mode (cache only)"""
        # This can be enhanced with network connectivity check
        return False

