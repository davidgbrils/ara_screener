"""Yahoo Finance data fetcher with retry and error handling"""

import yfinance as yf
import pandas as pd
import time
from typing import Optional
from config import YAHOO_FINANCE_SETTINGS, IDX_TICKER_SUFFIX
from utils.logger import setup_logger
from utils.helpers import retry_on_failure

logger = setup_logger(__name__)

class YahooFetcher:
    """Yahoo Finance data fetcher with robust error handling"""
    
    def __init__(self):
        """Initialize Yahoo fetcher"""
        self.settings = YAHOO_FINANCE_SETTINGS
        self.retry_count = self.settings["retry_count"]
        self.retry_delay = self.settings["retry_delay"]
    
    def _normalize_ticker(self, ticker: str) -> str:
        """
        Normalize ticker to Yahoo Finance format
        
        Args:
            ticker: Ticker symbol (with or without .JK)
        
        Returns:
            Normalized ticker
        """
        ticker = ticker.upper().strip()
        if not ticker.endswith(IDX_TICKER_SUFFIX):
            ticker = f"{ticker}{IDX_TICKER_SUFFIX}"
        return ticker
    
    @retry_on_failure(max_retries=3, delay=2, exceptions=(Exception,))
    def fetch_data(self, ticker: str, period: str = None, interval: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance
        
        Args:
            ticker: Ticker symbol
            period: Data period (defaults to config)
            interval: Data interval (defaults to config)
        
        Returns:
            DataFrame with OHLCV data or None
        """
        ticker = self._normalize_ticker(ticker)
        period = period or self.settings["period"]
        interval = interval or self.settings["interval"]
        
        try:
            logger.debug(f"Fetching {ticker} from Yahoo Finance...")
            
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Fix multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for {ticker}")
                return None
            
            # Clean data
            df = df[required_cols].copy()
            df.dropna(inplace=True)
            
            if len(df) < 50:  # Minimum data requirement
                logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
                return None
            
            logger.info(f"Successfully fetched {ticker}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def fetch_multiple(self, tickers: list, delay: float = 0.1) -> dict:
        """
        Fetch data for multiple tickers with rate limiting
        
        Args:
            tickers: List of ticker symbols
            delay: Delay between requests (seconds)
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            df = self.fetch_data(ticker)
            if df is not None:
                results[ticker] = df
            time.sleep(delay)  # Rate limiting
        
        return results
    
    def is_ticker_valid(self, ticker: str) -> bool:
        """
        Check if ticker is valid and has data
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            True if valid
        """
        df = self.fetch_data(ticker, period="5d")
        return df is not None and not df.empty

