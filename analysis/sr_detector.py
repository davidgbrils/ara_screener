"""
Support & Resistance Detector

Automatically detects support and resistance zones from price action:
- Swing High/Low detection
- Consolidation zone detection
- EMA confluence zones
- Zone strength classification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SRZone:
    """Support or Resistance Zone"""
    price_low: float
    price_high: float
    strength: str  # "STRONG", "WEAK"
    touches: int
    zone_type: str  # "SUPPORT", "RESISTANCE"
    source: str  # "SWING", "CONSOLIDATION", "EMA", "VWAP"


@dataclass
class SRLevels:
    """Complete S/R levels for analysis"""
    supports: List[SRZone]
    resistances: List[SRZone]
    closest_support: Optional[SRZone]
    closest_resistance: Optional[SRZone]
    current_price: float


class SRDetector:
    """
    Automatic Support & Resistance Detection
    
    Methods:
    - Swing High/Low detection
    - Consolidation zone detection (high volume areas)
    - EMA confluence zones
    - Zone strength classification based on touches
    """
    
    def __init__(self, swing_window: int = 5, zone_tolerance: float = 0.02):
        """
        Initialize S/R Detector
        
        Args:
            swing_window: Window for swing point detection
            zone_tolerance: Tolerance for zone merging (2% default)
        """
        self.swing_window = swing_window
        self.zone_tolerance = zone_tolerance
    
    # =========================================================================
    # SWING POINT DETECTION
    # =========================================================================
    
    def detect_swing_highs(self, df: pd.DataFrame, window: int = None) -> List[Tuple[int, float]]:
        """
        Detect swing highs (local maxima)
        
        Args:
            df: DataFrame with High column
            window: Lookback window
        
        Returns:
            List of (index, price) tuples
        """
        window = window or self.swing_window
        swing_highs = []
        
        highs = df['High'].values
        
        for i in range(window, len(highs) - window):
            is_swing_high = True
            for j in range(1, window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append((i, highs[i]))
        
        return swing_highs
    
    def detect_swing_lows(self, df: pd.DataFrame, window: int = None) -> List[Tuple[int, float]]:
        """
        Detect swing lows (local minima)
        
        Args:
            df: DataFrame with Low column
            window: Lookback window
        
        Returns:
            List of (index, price) tuples
        """
        window = window or self.swing_window
        swing_lows = []
        
        lows = df['Low'].values
        
        for i in range(window, len(lows) - window):
            is_swing_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append((i, lows[i]))
        
        return swing_lows
    
    # =========================================================================
    # ZONE DETECTION
    # =========================================================================
    
    def detect_consolidation_zones(self, df: pd.DataFrame, 
                                    volume_threshold: float = 1.5,
                                    min_candles: int = 5) -> List[Tuple[float, float]]:
        """
        Detect consolidation zones with high volume
        
        Args:
            df: DataFrame with OHLCV
            volume_threshold: Minimum volume spike ratio
            min_candles: Minimum candles in zone
        
        Returns:
            List of (low, high) price ranges
        """
        zones = []
        
        # Calculate volume MA
        volume_ma = df['Volume'].rolling(20).mean()
        volume_spike = df['Volume'] / volume_ma
        
        # Find high volume clusters
        high_vol_mask = volume_spike >= volume_threshold
        
        # Group consecutive high volume candles
        df_temp = df.copy()
        df_temp['high_vol'] = high_vol_mask
        
        # Find contiguous regions
        current_zone_start = None
        current_zone = []
        
        for i, row in df_temp.iterrows():
            if row['high_vol']:
                if current_zone_start is None:
                    current_zone_start = i
                current_zone.append(row)
            else:
                if len(current_zone) >= min_candles:
                    zone_low = min(c['Low'] for c in current_zone)
                    zone_high = max(c['High'] for c in current_zone)
                    zones.append((zone_low, zone_high))
                current_zone_start = None
                current_zone = []
        
        # Check last zone
        if len(current_zone) >= min_candles:
            zone_low = min(c['Low'] for c in current_zone)
            zone_high = max(c['High'] for c in current_zone)
            zones.append((zone_low, zone_high))
        
        return zones
    
    def get_ema_zones(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """
        Get EMA confluence zones (where EMAs cluster)
        
        Args:
            df: DataFrame with EMA columns
        
        Returns:
            List of (low, high) price ranges
        """
        zones = []
        
        # Check if EMA columns exist
        ema_cols = ['EMA_9', 'EMA_20', 'EMA_50']
        if not all(col in df.columns for col in ema_cols):
            return zones
        
        latest = df.iloc[-1]
        ema_values = [latest['EMA_9'], latest['EMA_20'], latest['EMA_50']]
        
        # Check if EMAs are clustered (within 2% of each other)
        ema_range = max(ema_values) - min(ema_values)
        ema_avg = sum(ema_values) / len(ema_values)
        
        if ema_range / ema_avg < 0.03:  # 3% cluster
            zones.append((min(ema_values), max(ema_values)))
        
        return zones
    
    # =========================================================================
    # ZONE CLASSIFICATION
    # =========================================================================
    
    def count_touches(self, price: float, df: pd.DataFrame, 
                      tolerance: float = None) -> int:
        """
        Count how many times price touched a level
        
        Args:
            price: Price level
            df: DataFrame with OHLCV
            tolerance: Price tolerance (default: zone_tolerance)
        
        Returns:
            Number of touches
        """
        tolerance = tolerance or self.zone_tolerance
        tolerance_value = price * tolerance
        
        touches = 0
        
        for _, row in df.iterrows():
            # Check if price touched this level
            if (row['Low'] <= price + tolerance_value and 
                row['High'] >= price - tolerance_value):
                touches += 1
        
        return touches
    
    def classify_strength(self, touches: int) -> str:
        """
        Classify zone strength based on touches
        
        Args:
            touches: Number of price touches
        
        Returns:
            "STRONG" or "WEAK"
        """
        if touches >= 3:
            return "STRONG"
        else:
            return "WEAK"
    
    # =========================================================================
    # MERGE AND FILTER ZONES
    # =========================================================================
    
    def merge_overlapping_zones(self, zones: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Merge overlapping price zones
        
        Args:
            zones: List of (low, high) tuples
        
        Returns:
            Merged zones
        """
        if not zones:
            return []
        
        # Sort by low price
        sorted_zones = sorted(zones, key=lambda x: x[0])
        merged = [sorted_zones[0]]
        
        for current in sorted_zones[1:]:
            last = merged[-1]
            
            # Check if overlapping (with tolerance)
            if current[0] <= last[1] * (1 + self.zone_tolerance):
                # Merge
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    # =========================================================================
    # MAIN DETECTION
    # =========================================================================
    
    def detect_levels(self, df: pd.DataFrame) -> Optional[SRLevels]:
        """
        Detect all support and resistance levels
        
        Args:
            df: DataFrame with OHLCV (and optionally EMA columns)
        
        Returns:
            SRLevels with all detected zones
        """
        if df is None or len(df) < 50:
            logger.warning("Insufficient data for S/R detection")
            return None
        
        try:
            current_price = float(df['Close'].iloc[-1])
            supports = []
            resistances = []
            
            # 1. Detect swing points
            swing_highs = self.detect_swing_highs(df)
            swing_lows = self.detect_swing_lows(df)
            
            # Convert swing points to zones
            for idx, price in swing_highs:
                zone_low = price * (1 - self.zone_tolerance / 2)
                zone_high = price * (1 + self.zone_tolerance / 2)
                touches = self.count_touches(price, df)
                
                zone = SRZone(
                    price_low=zone_low,
                    price_high=zone_high,
                    strength=self.classify_strength(touches),
                    touches=touches,
                    zone_type="RESISTANCE" if price > current_price else "SUPPORT",
                    source="SWING"
                )
                
                if price > current_price:
                    resistances.append(zone)
                else:
                    supports.append(zone)
            
            for idx, price in swing_lows:
                zone_low = price * (1 - self.zone_tolerance / 2)
                zone_high = price * (1 + self.zone_tolerance / 2)
                touches = self.count_touches(price, df)
                
                zone = SRZone(
                    price_low=zone_low,
                    price_high=zone_high,
                    strength=self.classify_strength(touches),
                    touches=touches,
                    zone_type="SUPPORT" if price < current_price else "RESISTANCE",
                    source="SWING"
                )
                
                if price < current_price:
                    supports.append(zone)
                else:
                    resistances.append(zone)
            
            # 2. Detect consolidation zones
            consolidation_zones = self.detect_consolidation_zones(df)
            for zone_low, zone_high in consolidation_zones:
                mid_price = (zone_low + zone_high) / 2
                touches = 5  # Consolidation zones are inherently multi-touch
                
                zone = SRZone(
                    price_low=zone_low,
                    price_high=zone_high,
                    strength="STRONG",
                    touches=touches,
                    zone_type="SUPPORT" if mid_price < current_price else "RESISTANCE",
                    source="CONSOLIDATION"
                )
                
                if mid_price < current_price:
                    supports.append(zone)
                else:
                    resistances.append(zone)
            
            # 3. EMA zones (if available)
            ema_zones = self.get_ema_zones(df)
            for zone_low, zone_high in ema_zones:
                mid_price = (zone_low + zone_high) / 2
                
                zone = SRZone(
                    price_low=zone_low,
                    price_high=zone_high,
                    strength="WEAK",  # Dynamic, can break easily
                    touches=1,
                    zone_type="SUPPORT" if mid_price < current_price else "RESISTANCE",
                    source="EMA"
                )
                
                if mid_price < current_price:
                    supports.append(zone)
                else:
                    resistances.append(zone)
            
            # Sort by distance from current price
            supports.sort(key=lambda z: current_price - z.price_high, reverse=False)
            resistances.sort(key=lambda z: z.price_low - current_price, reverse=False)
            
            # Get closest S/R
            closest_support = supports[0] if supports else None
            closest_resistance = resistances[0] if resistances else None
            
            # Limit to top 5 each
            supports = supports[:5]
            resistances = resistances[:5]
            
            return SRLevels(
                supports=supports,
                resistances=resistances,
                closest_support=closest_support,
                closest_resistance=closest_resistance,
                current_price=current_price
            )
            
        except Exception as e:
            logger.error(f"Error detecting S/R levels: {e}")
            return None
