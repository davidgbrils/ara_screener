"""Advanced pattern detection for multi-bagger identification"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from config import PATTERN_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PatternDetector:
    """Detect advanced patterns for multi-bagger stocks"""
    
    def __init__(self):
        """Initialize pattern detector"""
        self.config = PATTERN_CONFIG
    
    def detect_parabolic(self, df: pd.DataFrame, window: int = 20) -> bool:
        """
        Detect parabolic price movement
        
        Args:
            df: DataFrame with price data
            window: Lookback window
        
        Returns:
            True if parabolic pattern detected
        """
        if len(df) < window:
            return False
        
        prices = df['Close'].tail(window)
        returns = prices.pct_change().dropna()
        
        # Check for accelerating returns
        if len(returns) < 5:
            return False
        
        recent_returns = returns.tail(5).mean()
        earlier_returns = returns.head(len(returns) - 5).mean()
        
        # Parabolic: recent returns much higher than earlier
        return recent_returns > earlier_returns * 2 and recent_returns > 0.05
    
    def detect_volume_climax(self, df: pd.DataFrame, multiplier: float = 3.0) -> bool:
        """
        Detect volume climax (exhaustion volume)
        
        Args:
            df: DataFrame with volume data
            multiplier: Volume multiplier threshold
        
        Returns:
            True if volume climax detected
        """
        if len(df) < 20:
            return False
        
        latest_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].tail(20).mean()
        
        return latest_volume >= avg_volume * multiplier
    
    def detect_vcp(self, df: pd.DataFrame, contraction_days: int = 30) -> Optional[Dict]:
        """
        Detect Volatility Contraction Pattern (VCP)
        
        Args:
            df: DataFrame with price data
            contraction_days: Days to analyze
        
        Returns:
            VCP pattern info or None
        """
        if len(df) < contraction_days:
            return None
        
        recent = df.tail(contraction_days)
        
        # Calculate volatility (ATR or price range)
        volatility = (recent['High'] - recent['Low']) / recent['Close']
        
        # Check for decreasing volatility
        first_half_vol = volatility.head(contraction_days // 2).mean()
        second_half_vol = volatility.tail(contraction_days // 2).mean()
        
        if second_half_vol < first_half_vol * 0.7:  # 30% contraction
            return {
                'detected': True,
                'contraction_ratio': second_half_vol / first_half_vol,
                'days': contraction_days,
            }
        
        return None
    
    def detect_darvas_box(self, df: pd.DataFrame, window: int = 20) -> Optional[Dict]:
        """
        Detect Darvas Box pattern
        
        Args:
            df: DataFrame with price data
            window: Lookback window
        
        Returns:
            Darvas Box info or None
        """
        if len(df) < window:
            return None
        
        recent = df.tail(window)
        
        # Find pivot highs and lows
        highs = recent['High']
        lows = recent['Low']
        
        top = highs.max()
        bottom = lows.min()
        
        # Check if price is near top (breakout zone)
        current_price = df['Close'].iloc[-1]
        box_height = top - bottom
        
        if box_height > 0 and current_price >= top * 0.95:  # Within 5% of top
            return {
                'detected': True,
                'top': top,
                'bottom': bottom,
                'height': box_height,
                'current_price': current_price,
            }
        
        return None
    
    def detect_pocket_pivot(self, df: pd.DataFrame) -> bool:
        """
        Detect Pocket Pivot (volume surge on up day)
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            True if pocket pivot detected
        """
        if len(df) < 50:
            return False
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Price up and volume surge
        price_up = latest['Close'] > prev['Close']
        volume_surge = latest['Volume'] > df['Volume'].tail(50).mean() * 1.5
        
        return price_up and volume_surge
    
    def detect_gap_up(self, df: pd.DataFrame, min_gap_pct: float = 2.0) -> Optional[Dict]:
        """
        Detect gap-up pattern
        
        Args:
            df: DataFrame with OHLCV data
            min_gap_pct: Minimum gap percentage
        
        Returns:
            Gap info or None
        """
        if len(df) < 2:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        gap = latest['Low'] - prev['High']
        gap_pct = (gap / prev['High']) * 100
        
        if gap_pct >= min_gap_pct:
            return {
                'detected': True,
                'gap_pct': gap_pct,
                'gap_amount': gap,
                'prev_high': prev['High'],
                'current_low': latest['Low'],
            }
        
        return None
    
    def detect_reaccumulation_base(self, df: pd.DataFrame, days: int = 50) -> Optional[Dict]:
        """
        Detect reaccumulation base pattern
        
        Args:
            df: DataFrame with price data
            days: Lookback period
        
        Returns:
            Base pattern info or None
        """
        if len(df) < days:
            return None
        
        recent = df.tail(days)
        
        # Check for sideways movement with volume
        price_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].mean()
        avg_volume = recent['Volume'].mean()
        recent_volume = recent['Volume'].tail(10).mean()
        
        # Base: tight price range, increasing volume
        if price_range < 0.15 and recent_volume > avg_volume * 1.2:
            return {
                'detected': True,
                'price_range_pct': price_range * 100,
                'volume_increase': (recent_volume / avg_volume - 1) * 100,
                'days': days,
            }
        
        return None
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect all patterns
        
        Args:
            df: DataFrame with indicators
        
        Returns:
            Dictionary of all detected patterns
        """
        patterns = {}
        
        if self.config["PARABOLIC_DETECTION"]:
            patterns['parabolic'] = self.detect_parabolic(df)
        
        if self.config["VOLUME_CLIMAX"]:
            patterns['volume_climax'] = self.detect_volume_climax(df)
        
        if self.config["VCP_DETECTION"]:
            patterns['vcp'] = self.detect_vcp(df)
        
        if self.config["DARVAS_BOX"]:
            patterns['darvas_box'] = self.detect_darvas_box(df)
        
        if self.config["POCKET_PIVOT"]:
            patterns['pocket_pivot'] = self.detect_pocket_pivot(df)
        
        if self.config["GAP_UP_DETECTION"]:
            patterns['gap_up'] = self.detect_gap_up(df)
        
        if self.config["REACCUMULATION_BASE"]:
            patterns['reaccumulation_base'] = self.detect_reaccumulation_base(df)
        
        return patterns

