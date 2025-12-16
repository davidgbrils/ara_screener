"""
Indicator Engine - Technical Analysis Indicators

Calculates all required technical indicators for multi-timeframe analysis:
- Trend: EMA 9, 20, 50
- Momentum: RSI, MACD
- Volatility: ATR
- Volume: VWAP, Volume MA, Volume Spike
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class IndicatorSet:
    """Complete set of indicators for a timeframe"""
    # Trend
    ema_9: float
    ema_20: float
    ema_50: float
    ema_trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    
    # Momentum
    rsi: float
    rsi_signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_cross: str  # "BULLISH_CROSS", "BEARISH_CROSS", "NONE"
    
    # Volatility
    atr: float
    atr_percent: float  # ATR as % of price
    
    # Volume
    vwap: float
    volume_ma: float
    volume_spike: float  # Current volume / Volume MA
    volume_signal: str  # "HIGH", "NORMAL", "LOW"
    
    # Price position
    price_vs_ema: str  # "ABOVE_ALL", "BELOW_ALL", "MIXED"
    price_vs_vwap: str  # "ABOVE", "BELOW"


class IndicatorEngine:
    """
    Calculate technical indicators for stock analysis
    
    Supports:
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - ATR (Average True Range)
    - VWAP (Volume Weighted Average Price)
    - Volume analysis
    """
    
    def __init__(self):
        """Initialize indicator engine"""
        self.ema_periods = [9, 20, 50]
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.atr_period = 14
        self.volume_ma_period = 20
    
    # =========================================================================
    # TREND INDICATORS
    # =========================================================================
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            series: Price series
            period: EMA period
        
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_all_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all EMAs and add to DataFrame
        
        Args:
            df: DataFrame with OHLCV
        
        Returns:
            DataFrame with EMA columns added
        """
        df = df.copy()
        for period in self.ema_periods:
            df[f'EMA_{period}'] = self.calculate_ema(df['Close'], period)
        return df
    
    def get_ema_trend(self, ema_9: float, ema_20: float, ema_50: float) -> str:
        """
        Determine trend based on EMA stack
        
        Bullish: EMA9 > EMA20 > EMA50
        Bearish: EMA9 < EMA20 < EMA50
        """
        if ema_9 > ema_20 > ema_50:
            return "BULLISH"
        elif ema_9 < ema_20 < ema_50:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            series: Price series
            period: RSI period (default 14)
        
        Returns:
            RSI series
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_rsi_signal(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi >= 70:
            return "OVERBOUGHT"
        elif rsi <= 30:
            return "OVERSOLD"
        else:
            return "NEUTRAL"
    
    def calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD, Signal Line, and Histogram
        
        Args:
            series: Price series
        
        Returns:
            Tuple of (MACD Line, Signal Line, Histogram)
        """
        ema_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def get_macd_cross(self, macd: float, signal: float, 
                       prev_macd: float, prev_signal: float) -> str:
        """Detect MACD crossover"""
        if prev_macd <= prev_signal and macd > signal:
            return "BULLISH_CROSS"
        elif prev_macd >= prev_signal and macd < signal:
            return "BEARISH_CROSS"
        else:
            return "NONE"
    
    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with High, Low, Close
            period: ATR period
        
        Returns:
            ATR series
        """
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (intraday)
        
        Args:
            df: DataFrame with OHLCV
        
        Returns:
            VWAP series
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def calculate_volume_ma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Moving Average"""
        return df['Volume'].rolling(window=period).mean()
    
    def calculate_volume_spike(self, volume: float, volume_ma: float) -> float:
        """Calculate volume spike ratio"""
        if volume_ma > 0:
            return volume / volume_ma
        return 1.0
    
    def get_volume_signal(self, spike_ratio: float) -> str:
        """Interpret volume spike"""
        if spike_ratio >= 2.0:
            return "HIGH"
        elif spike_ratio <= 0.5:
            return "LOW"
        else:
            return "NORMAL"
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate_all(self, df: pd.DataFrame) -> Optional[IndicatorSet]:
        """
        Calculate all indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            IndicatorSet with all calculated indicators
        """
        if df is None or len(df) < 50:
            logger.warning("Insufficient data for indicator calculation")
            return None
        
        try:
            # Add all indicators to DataFrame
            df = df.copy()
            
            # EMAs
            df['EMA_9'] = self.calculate_ema(df['Close'], 9)
            df['EMA_20'] = self.calculate_ema(df['Close'], 20)
            df['EMA_50'] = self.calculate_ema(df['Close'], 50)
            
            # RSI
            df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
            
            # MACD
            macd, signal, hist = self.calculate_macd(df['Close'])
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
            
            # ATR
            df['ATR'] = self.calculate_atr(df, self.atr_period)
            
            # Volume
            df['VWAP'] = self.calculate_vwap(df)
            df['Volume_MA'] = self.calculate_volume_ma(df, self.volume_ma_period)
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Calculate derived values
            ema_9 = float(latest['EMA_9'])
            ema_20 = float(latest['EMA_20'])
            ema_50 = float(latest['EMA_50'])
            close = float(latest['Close'])
            
            ema_trend = self.get_ema_trend(ema_9, ema_20, ema_50)
            
            rsi = float(latest['RSI'])
            rsi_signal = self.get_rsi_signal(rsi)
            
            macd_val = float(latest['MACD'])
            macd_sig = float(latest['MACD_Signal'])
            macd_hist = float(latest['MACD_Hist'])
            macd_cross = self.get_macd_cross(
                macd_val, macd_sig,
                float(prev['MACD']), float(prev['MACD_Signal'])
            )
            
            atr = float(latest['ATR'])
            atr_percent = (atr / close) * 100 if close > 0 else 0
            
            vwap = float(latest['VWAP'])
            volume_ma = float(latest['Volume_MA'])
            volume_spike = self.calculate_volume_spike(
                float(latest['Volume']), volume_ma
            )
            volume_signal = self.get_volume_signal(volume_spike)
            
            # Price position
            if close > ema_9 and close > ema_20 and close > ema_50:
                price_vs_ema = "ABOVE_ALL"
            elif close < ema_9 and close < ema_20 and close < ema_50:
                price_vs_ema = "BELOW_ALL"
            else:
                price_vs_ema = "MIXED"
            
            price_vs_vwap = "ABOVE" if close > vwap else "BELOW"
            
            return IndicatorSet(
                ema_9=ema_9,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_trend=ema_trend,
                rsi=rsi,
                rsi_signal=rsi_signal,
                macd=macd_val,
                macd_signal=macd_sig,
                macd_histogram=macd_hist,
                macd_cross=macd_cross,
                atr=atr,
                atr_percent=atr_percent,
                vwap=vwap,
                volume_ma=volume_ma,
                volume_spike=volume_spike,
                volume_signal=volume_signal,
                price_vs_ema=price_vs_ema,
                price_vs_vwap=price_vs_vwap
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def add_indicators_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicator columns to DataFrame
        
        Args:
            df: Original DataFrame with OHLCV
        
        Returns:
            DataFrame with all indicator columns
        """
        df = df.copy()
        
        # EMAs
        df['EMA_9'] = self.calculate_ema(df['Close'], 9)
        df['EMA_20'] = self.calculate_ema(df['Close'], 20)
        df['EMA_50'] = self.calculate_ema(df['Close'], 50)
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        
        # MACD
        macd, signal, hist = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # ATR
        df['ATR'] = self.calculate_atr(df, self.atr_period)
        
        # Volume
        df['VWAP'] = self.calculate_vwap(df)
        df['Volume_MA'] = self.calculate_volume_ma(df, self.volume_ma_period)
        df['Volume_Spike'] = df['Volume'] / df['Volume_MA']
        
        return df
