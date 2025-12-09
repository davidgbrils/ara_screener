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
        df['ATR_PCT'] = (df['ATR'] / df['Close']).replace([np.inf, -np.inf], np.nan)
        
        # Relative Volume
        df['RVOL'] = self._calculate_rvol(df)
        
        # 1-week percentage change
        df['PCT_CHANGE_1W'] = df['Close'].pct_change(periods=5) * 100
        df['PCT_CHANGE_1M'] = df['Close'].pct_change(periods=21) * 100
        df['PCT_CHANGE_3M'] = df['Close'].pct_change(periods=63) * 100
        
        # MA20 slope
        df['MA20_SLOPE'] = df['MA20'].diff()
        df['MA20_SLOPE_ACCEL'] = df['MA20_SLOPE'].diff()
        df['MA20_PCT_CHANGE_5D'] = df['MA20'].pct_change(periods=5) * 100
        
        # Chaikin Money Flow
        df['CMF'] = self._calculate_chaikin_money_flow(df)
        
        # VWAP Profile (distance from VWAP)
        if 'VWAP' in df.columns:
            df['VWAP_DISTANCE'] = ((df['Close'] - df['VWAP']) / df['VWAP']) * 100

        # 52-week high/low and distance to high
        df['HIGH_52W'] = df['High'].rolling(window=252, min_periods=1).max()
        df['LOW_52W'] = df['Low'].rolling(window=252, min_periods=1).min()
        df['DIST_52W_HIGH'] = ((df['Close'] - df['HIGH_52W']) / df['HIGH_52W']).replace([np.inf, -np.inf], np.nan) * 100

        # 20-day high/low
        df['HIGH_20D'] = df['High'].rolling(window=20, min_periods=1).max()
        df['LOW_20D'] = df['Low'].rolling(window=20, min_periods=1).min()

        # Bollinger Band width
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'BB_Mid' in df.columns:
            width = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'].replace(0, np.nan)
            df['BB_WIDTH'] = width.replace([np.inf, -np.inf], np.nan)

        # StdDev ratios
        df['STDDEV_10D'] = df['Close'].rolling(window=10).std()
        df['STDDEV_20D'] = df['Close'].rolling(window=20).std()

        # MA200 slope
        if 'MA200' in df.columns:
            df['MA200_SLOPE'] = df['MA200'].diff()

        # MACD
        macd = self._calculate_macd(df['Close'])
        df['MACD'] = macd['macd']
        df['MACD_SIGNAL'] = macd['signal']
        df['MACD_HIST'] = macd['hist']

        # OBV slope 5d
        if 'OBV' in df.columns:
            df['OBV_SLOPE_5D'] = df['OBV'].diff(periods=5)

        # Intraday today change
        df['PCT_CHANGE_TODAY'] = ((df['Close'] - df['Open']) / df['Open']).replace([np.inf, -np.inf], np.nan) * 100
        
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
    
    def _calculate_chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF)
        
        CMF = Sum(AD) / Sum(Volume) over period
        AD = ((Close - Low) - (High - Close)) / (High - Low) * Volume
        """
        # Calculate Money Flow Multiplier
        high_low = df['High'] - df['Low']
        close_low = df['Close'] - df['Low']
        high_close = df['High'] - df['Close']
        
        # Avoid division by zero
        mfm = (close_low - high_close) / high_low.replace(0, np.nan)
        mfm = mfm.fillna(0)
        
        # Calculate Accumulation/Distribution
        ad = mfm * df['Volume']
        
        # Calculate CMF
        cmf = ad.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
        
        return cmf.fillna(0)

    def _ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        hist = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'hist': hist}
    
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
            'atr_pct': latest.get('ATR_PCT', None),
            'rvol': latest.get('RVOL', None),
            'pct_change_1w': latest.get('PCT_CHANGE_1W', None),
            'pct_change_1m': latest.get('PCT_CHANGE_1M', None),
            'pct_change_3m': latest.get('PCT_CHANGE_3M', None),
            'pct_change_today': latest.get('PCT_CHANGE_TODAY', None),
            'ma20_slope': latest.get('MA20_SLOPE', None),
            'ma20_pct_change_5d': latest.get('MA20_PCT_CHANGE_5D', None),
            'cmf': latest.get('CMF', None),
            'vwap_distance': latest.get('VWAP_DISTANCE', None),
            'high_52w': latest.get('HIGH_52W', None),
            'low_52w': latest.get('LOW_52W', None),
            'dist_52w_high': latest.get('DIST_52W_HIGH', None),
            'high_20d': latest.get('HIGH_20D', None),
            'low_20d': latest.get('LOW_20D', None),
            'bb_width': latest.get('BB_WIDTH', None),
            'ma200_slope': latest.get('MA200_SLOPE', None),
            'macd': latest.get('MACD', None),
            'macd_signal': latest.get('MACD_SIGNAL', None),
            'macd_hist': latest.get('MACD_HIST', None),
            'obv_slope_5d': latest.get('OBV_SLOPE_5D', None),
            'stddev_10d': latest.get('STDDEV_10D', None),
            'stddev_20d': latest.get('STDDEV_20D', None),
        }

