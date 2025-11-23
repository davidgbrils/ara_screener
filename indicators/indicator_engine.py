"""Technical indicator calculation engine"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from config import INDICATOR_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

class IndicatorEngine:
    """Calculate all technical indicators"""
    
    def __init__(self):
        """Initialize indicator engine"""
        self.ma_periods = INDICATOR_CONFIG["MA_PERIODS"]
        self.rsi_period = INDICATOR_CONFIG["RSI_PERIOD"]
        self.bb_period = INDICATOR_CONFIG["BB_PERIOD"]
        self.bb_std = INDICATOR_CONFIG["BB_STD"]
        self.atr_period = INDICATOR_CONFIG["ATR_PERIOD"]
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicator columns
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Moving Averages
        for period in self.ma_periods:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # Bollinger Bands
        bb = self._calculate_bollinger_bands(df['Close'], self.bb_period, self.bb_std)
        df['BB_Upper'] = bb['upper']
        df['BB_Mid'] = bb['mid']
        df['BB_Lower'] = bb['lower']
        
        # OBV
        if INDICATOR_CONFIG["OBV_ENABLED"]:
            df['OBV'] = self._calculate_obv(df)
        
        # VWAP
        if INDICATOR_CONFIG["VWAP_ENABLED"]:
            df['VWAP'] = self._calculate_vwap(df)
        
        # ATR
        df['ATR'] = self._calculate_atr(df, self.atr_period)
        
        # Relative Volume
        df['RVOL'] = self._calculate_rvol(df)
        
        # 1-week percentage change
        df['PCT_CHANGE_1W'] = df['Close'].pct_change(periods=5) * 100
        
        # MA20 slope
        df['MA20_SLOPE'] = df['MA20'].diff()
        df['MA20_SLOPE_ACCEL'] = df['MA20_SLOPE'].diff()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std_dev * std),
            'mid': sma,
            'lower': sma - (std_dev * std),
        }
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['Volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_rvol(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Calculate Relative Volume"""
        avg_volume = df['Volume'].rolling(window=lookback).mean()
        rvol = df['Volume'] / avg_volume
        return rvol
    
    def get_latest_values(self, df: pd.DataFrame) -> Dict:
        """
        Get latest indicator values
        
        Args:
            df: DataFrame with indicators
        
        Returns:
            Dictionary of latest values
        """
        if df is None or df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            'close': latest['Close'],
            'volume': latest['Volume'],
            'ma20': latest.get('MA20', None),
            'ma50': latest.get('MA50', None),
            'ma200': latest.get('MA200', None),
            'rsi': latest.get('RSI', None),
            'bb_upper': latest.get('BB_Upper', None),
            'bb_mid': latest.get('BB_Mid', None),
            'bb_lower': latest.get('BB_Lower', None),
            'obv': latest.get('OBV', None),
            'vwap': latest.get('VWAP', None),
            'atr': latest.get('ATR', None),
            'rvol': latest.get('RVOL', None),
            'pct_change_1w': latest.get('PCT_CHANGE_1W', None),
            'ma20_slope': latest.get('MA20_SLOPE', None),
        }

