"""Main data fetcher with caching and fallback support"""

import pandas as pd
from typing import Optional
from config import CACHE_CONFIG
from .yahoo_fetcher import YahooFetcher
from .fallback_fetcher import FallbackFetcher
from cache_layer.cache_manager import CacheManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataFetcher:
    """Main data fetcher with caching and fallback"""
    
    def __init__(self):
        """Initialize data fetcher"""
        self.yahoo_fetcher = YahooFetcher()
        self.fallback_fetcher = FallbackFetcher()
        self.cache_manager = CacheManager()
    
    def fetch(self, ticker: str, use_cache: bool = True, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch data with caching and fallback
        
        Args:
            ticker: Ticker symbol
            use_cache: Whether to use cache
            force_refresh: Force refresh even if cache is valid
        
        Returns:
            DataFrame with OHLCV data or None
        """
        # Try cache first (warm fetch)
        if use_cache and not force_refresh:
            cached_data = self.cache_manager.warm_fetch(ticker)
            if cached_data is not None:
                logger.debug(f"Using cached data for {ticker}")
                return cached_data
        
        # Try Yahoo Finance
        df = self.yahoo_fetcher.fetch_data(ticker)
        
        if df is None:
            # Try fallback APIs
            logger.info(f"Yahoo Finance failed for {ticker}, trying fallback...")
            df = self.fallback_fetcher.fetch(ticker)
        
        if df is not None and use_cache:
            # Store in cache
            self.cache_manager.store_data(ticker, df)
        
        return df
    
    def fetch_multiple(self, tickers: list, use_cache: bool = True) -> dict:
        """
        Fetch data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            use_cache: Whether to use cache
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            try:
                df = self.fetch(ticker, use_cache=use_cache)
                if df is not None:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue
        
        return results
    
    def is_liquid(self, ticker: str, min_volume: int = 1000000) -> bool:
        """
        Check if ticker is liquid enough
        
        Args:
            ticker: Ticker symbol
            min_volume: Minimum average volume
        
        Returns:
            True if liquid
        """
        df = self.fetch(ticker)
        if df is None or df.empty:
            return False
        
        avg_volume = df['Volume'].tail(20).mean()
        return avg_volume >= min_volume

