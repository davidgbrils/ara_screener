"""Market Regime Filter - Bull/Bear Detection"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RegimeFilter:
    """Detect market regime (Bull/Bear/Neutral)"""
    
    def __init__(self):
        """Initialize regime filter"""
        pass
    
    def detect_regime(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Detect market regime
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period for regime detection
        
        Returns:
            Dictionary with regime info
        """
        if len(df) < lookback:
            return {
                'regime': 'NEUTRAL',
                'confidence': 0.0,
                'trend': 0.0,
            }
        
        recent = df.tail(lookback)
        prices = recent['Close']
        
        # Calculate trend using linear regression
        x = np.arange(len(prices))
        y = prices.values
        
        # Remove NaN
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            return {
                'regime': 'NEUTRAL',
                'confidence': 0.0,
                'trend': 0.0,
            }
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Linear regression
        coeffs = np.polyfit(x_clean, y_clean, 1)
        slope = coeffs[0]
        trend_pct = (slope / prices.iloc[0]) * 100 if prices.iloc[0] > 0 else 0
        
        # Calculate volatility
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate MA structure
        ma20 = recent['Close'].rolling(20).mean().iloc[-1]
        ma50 = recent['Close'].rolling(50).mean().iloc[-1]
        ma200 = recent['Close'].rolling(200).mean().iloc[-1] if len(recent) >= 200 else None
        
        # Determine regime
        bullish_signals = 0
        bearish_signals = 0
        
        # Trend
        if trend_pct > 5:
            bullish_signals += 1
        elif trend_pct < -5:
            bearish_signals += 1
        
        # MA structure
        if ma200 and ma20 > ma50 > ma200:
            bullish_signals += 1
        elif ma200 and ma20 < ma50 < ma200:
            bearish_signals += 1
        
        # Price above/below MA
        current_price = prices.iloc[-1]
        if current_price > ma20 > ma50:
            bullish_signals += 1
        elif current_price < ma20 < ma50:
            bearish_signals += 1
        
        # Volatility (low volatility in bull, high in bear)
        if volatility < 0.15:
            bullish_signals += 0.5
        elif volatility > 0.30:
            bearish_signals += 0.5
        
        # Determine regime
        if bullish_signals > bearish_signals + 1:
            regime = 'BULL'
            confidence = min(bullish_signals / 3.0, 1.0)
        elif bearish_signals > bullish_signals + 1:
            regime = 'BEAR'
            confidence = min(bearish_signals / 3.0, 1.0)
        else:
            regime = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'trend': trend_pct,
            'volatility': volatility,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
        }
    
    def is_bull_market(self, df: pd.DataFrame) -> bool:
        """Check if market is in bull regime"""
        regime_info = self.detect_regime(df)
        return regime_info['regime'] == 'BULL' and regime_info['confidence'] > 0.6

