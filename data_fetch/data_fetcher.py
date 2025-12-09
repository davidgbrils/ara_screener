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
            force_refresh: Force refresh even if cache is valid (for fresh data)
        
        Returns:
            DataFrame with OHLCV data or None
        """
        from datetime import datetime, timezone
        
        # Check if we need fresh data (today's data)
        need_fresh = force_refresh
        
        # If not forcing refresh, check if cache has today's data
        if use_cache and not force_refresh:
            cached_data = self.cache_manager.warm_fetch(ticker)
            if cached_data is not None:
                # Check if cache has today's data
                if len(cached_data) > 0:
                    last_date = cached_data.index[-1]
                    today = datetime.now(timezone.utc).date()
                    
                    # Convert last_date to date if it's datetime
                    if hasattr(last_date, 'date'):
                        last_date_only = last_date.date()
                    else:
                        last_date_only = last_date
                    
                    # If cache has today's data, use it
                    if last_date_only >= today:
                        logger.debug(f"Using cached data for {ticker} (has today's data)")
                        return cached_data
                    else:
                        logger.debug(f"Cache outdated for {ticker}, fetching fresh data")
                        need_fresh = True
        
        # Fetch fresh data
        df = self.yahoo_fetcher.fetch_data(ticker)
        
        if df is None:
            # Try fallback APIs
            logger.info(f"Yahoo Finance failed for {ticker}, trying fallback...")
            df = self.fallback_fetcher.fetch(ticker)
        
        if df is not None:
            # Validate data freshness
            if len(df) > 0:
                last_date = df.index[-1]
                today = datetime.now(timezone.utc).date()
                
                if hasattr(last_date, 'date'):
                    last_date_only = last_date.date()
                else:
                    last_date_only = last_date
                
                if last_date_only < today:
                    logger.warning(f"Data for {ticker} may not include today: last date {last_date_only}, today {today}")
            
            if use_cache:
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

