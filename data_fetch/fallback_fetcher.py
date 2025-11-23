"""Fallback data fetcher for alternative APIs"""

import requests
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
from config import FALLBACK_APIS
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FallbackFetcher:
    """Fallback fetcher using alternative APIs"""
    
    def __init__(self):
        """Initialize fallback fetcher"""
        self.idx_enabled = FALLBACK_APIS["IDX_API"]["ENABLED"]
        self.eodhd_enabled = FALLBACK_APIS["EODHD"]["ENABLED"]
        self.tv_enabled = FALLBACK_APIS["TRADINGVIEW"]["ENABLED"]
    
    def fetch_from_idx(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from IDX API (if available)
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            DataFrame or None
        """
        if not self.idx_enabled:
            return None
        
        try:
            # IDX API implementation would go here
            # This is a placeholder
            logger.warning("IDX API not fully implemented")
            return None
        except Exception as e:
            logger.error(f"Error fetching from IDX API: {e}")
            return None
    
    def fetch_from_eodhd(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from EODHD API
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            DataFrame or None
        """
        if not self.eodhd_enabled or not FALLBACK_APIS["EODHD"]["API_KEY"]:
            return None
        
        try:
            api_key = FALLBACK_APIS["EODHD"]["API_KEY"]
            base_url = FALLBACK_APIS["EODHD"]["BASE_URL"]
            
            # EODHD API implementation would go here
            # This is a placeholder
            logger.warning("EODHD API not fully implemented")
            return None
        except Exception as e:
            logger.error(f"Error fetching from EODHD: {e}")
            return None
    
    def fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Try to fetch from fallback APIs in order
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            DataFrame or None
        """
        # Try IDX first
        df = self.fetch_from_idx(ticker)
        if df is not None:
            return df
        
        # Try EODHD
        df = self.fetch_from_eodhd(ticker)
        if df is not None:
            return df
        
        return None

