"""Data normalization and cleaning"""

import pandas as pd
import numpy as np
from typing import Optional
from config import INDICATOR_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataNormalizer:
    """Normalize and clean OHLCV data"""
    
    def __init__(self):
        """Initialize normalizer"""
        self.min_history_days = INDICATOR_CONFIG["MIN_HISTORY_DAYS"]
    
    def normalize(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Normalize DataFrame structure and clean data
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Normalized DataFrame or None
        """
        if df is None or df.empty:
            return None
        
        try:
            df = df.copy()
            
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return None
            
            # Select only required columns
            df = df[required_cols].copy()
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Remove rows with invalid data
            df = df[
                (df['High'] >= df['Low']) &
                (df['High'] >= df['Open']) &
                (df['High'] >= df['Close']) &
                (df['Low'] <= df['Open']) &
                (df['Low'] <= df['Close']) &
                (df['Volume'] >= 0)
            ]
            
            # Fill missing values (forward fill for OHLC, 0 for Volume)
            df['Volume'] = df['Volume'].fillna(0)
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')
            
            # Remove rows with all NaN
            df.dropna(how='all', inplace=True)
            
            # Check minimum history requirement
            if len(df) < self.min_history_days:
                logger.warning(f"Insufficient history: {len(df)} < {self.min_history_days}")
                return None
            
            logger.debug(f"Normalized data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return None
    
    def fix_multi_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix multi-index columns from yfinance
        
        Args:
            df: DataFrame with potential multi-index columns
        
        Returns:
            DataFrame with fixed columns
        """
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """
        Remove outliers from price data
        
        Args:
            df: DataFrame
            method: Outlier detection method ('iqr' or 'zscore')
        
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        if method == "iqr":
            # IQR method
            for col in ['Open', 'High', 'Low', 'Close']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == "zscore":
            # Z-score method
            for col in ['Open', 'High', 'Low', 'Close']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
        
        return df

