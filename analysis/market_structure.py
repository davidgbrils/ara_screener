"""
Market Structure Analysis

Detects market structure patterns:
- Higher High / Higher Low (Bullish structure)
- Lower High / Lower Low (Bearish structure)
- Break of Structure (BOS)
- Market phase detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StructurePoint:
    """A structure point (swing high or low)"""
    index: int
    price: float
    point_type: str  # "HH", "HL", "LH", "LL"


@dataclass
class MarketStructureResult:
    """Complete market structure analysis"""
    structure_type: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    current_phase: str  # "ACCUMULATION", "MARKUP", "DISTRIBUTION", "BREAKDOWN", "UNKNOWN"
    structure_points: List[StructurePoint]
    bos_detected: bool
    bos_direction: Optional[str]  # "BULLISH_BOS", "BEARISH_BOS"
    trend_strength: str  # "STRONG", "WEAK"


class MarketStructureAnalyzer:
    """
    Analyze market structure using swing points
    
    Bullish: HH + HL pattern
    Bearish: LH + LL pattern
    Sideways: Mixed or no clear pattern
    """
    
    def __init__(self, swing_window: int = 5):
        """
        Initialize analyzer
        
        Args:
            swing_window: Window for swing detection
        """
        self.swing_window = swing_window
    
    # =========================================================================
    # SWING DETECTION
    # =========================================================================
    
    def detect_swing_highs(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Detect swing highs"""
        swing_highs = []
        highs = df['High'].values
        
        for i in range(self.swing_window, len(highs) - self.swing_window):
            is_swing = True
            for j in range(1, self.swing_window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swing_highs.append((i, highs[i]))
        
        return swing_highs
    
    def detect_swing_lows(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Detect swing lows"""
        swing_lows = []
        lows = df['Low'].values
        
        for i in range(self.swing_window, len(lows) - self.swing_window):
            is_swing = True
            for j in range(1, self.swing_window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swing_lows.append((i, lows[i]))
        
        return swing_lows
    
    # =========================================================================
    # HH/HL/LH/LL CLASSIFICATION
    # =========================================================================
    
    def classify_swing_highs(self, swing_highs: List[Tuple[int, float]]) -> List[StructurePoint]:
        """
        Classify swing highs as HH (Higher High) or LH (Lower High)
        """
        points = []
        
        for i, (idx, price) in enumerate(swing_highs):
            if i == 0:
                point_type = "HH"  # First point is baseline
            else:
                prev_price = swing_highs[i - 1][1]
                point_type = "HH" if price > prev_price else "LH"
            
            points.append(StructurePoint(
                index=idx,
                price=price,
                point_type=point_type
            ))
        
        return points
    
    def classify_swing_lows(self, swing_lows: List[Tuple[int, float]]) -> List[StructurePoint]:
        """
        Classify swing lows as HL (Higher Low) or LL (Lower Low)
        """
        points = []
        
        for i, (idx, price) in enumerate(swing_lows):
            if i == 0:
                point_type = "HL"  # First point is baseline
            else:
                prev_price = swing_lows[i - 1][1]
                point_type = "HL" if price > prev_price else "LL"
            
            points.append(StructurePoint(
                index=idx,
                price=price,
                point_type=point_type
            ))
        
        return points
    
    # =========================================================================
    # STRUCTURE DETERMINATION
    # =========================================================================
    
    def determine_structure(self, high_points: List[StructurePoint], 
                           low_points: List[StructurePoint]) -> str:
        """
        Determine overall market structure
        
        Bullish: Majority HH and HL
        Bearish: Majority LH and LL
        Sideways: Mixed
        """
        if not high_points or not low_points:
            return "SIDEWAYS"
        
        # Count recent points (last 4 of each)
        recent_highs = high_points[-4:]
        recent_lows = low_points[-4:]
        
        hh_count = sum(1 for p in recent_highs if p.point_type == "HH")
        lh_count = sum(1 for p in recent_highs if p.point_type == "LH")
        hl_count = sum(1 for p in recent_lows if p.point_type == "HL")
        ll_count = sum(1 for p in recent_lows if p.point_type == "LL")
        
        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count
        
        if bullish_score >= 3 and bullish_score > bearish_score:
            return "BULLISH"
        elif bearish_score >= 3 and bearish_score > bullish_score:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    # =========================================================================
    # BREAK OF STRUCTURE (BOS)
    # =========================================================================
    
    def detect_bos(self, df: pd.DataFrame, 
                   high_points: List[StructurePoint],
                   low_points: List[StructurePoint]) -> Tuple[bool, Optional[str]]:
        """
        Detect Break of Structure
        
        Bullish BOS: Price breaks above last Lower High
        Bearish BOS: Price breaks below last Higher Low
        """
        if len(df) < 10 or not high_points or not low_points:
            return False, None
        
        current_close = df['Close'].iloc[-1]
        recent_high = df['High'].tail(5).max()
        recent_low = df['Low'].tail(5).min()
        
        # Find last LH and HL
        last_lh = None
        last_hl = None
        
        for p in reversed(high_points):
            if p.point_type == "LH":
                last_lh = p
                break
        
        for p in reversed(low_points):
            if p.point_type == "HL":
                last_hl = p
                break
        
        # Check for BOS
        if last_lh and recent_high > last_lh.price:
            return True, "BULLISH_BOS"
        
        if last_hl and recent_low < last_hl.price:
            return True, "BEARISH_BOS"
        
        return False, None
    
    # =========================================================================
    # PHASE DETECTION
    # =========================================================================
    
    def detect_phase(self, df: pd.DataFrame, structure: str) -> str:
        """
        Detect market phase based on structure and momentum
        
        - ACCUMULATION: Sideways after downtrend, volume increasing
        - MARKUP: Bullish structure with momentum
        - DISTRIBUTION: Sideways after uptrend, high volume
        - BREAKDOWN: Bearish structure with momentum
        """
        if len(df) < 50:
            return "UNKNOWN"
        
        # Calculate momentum
        close = df['Close'].values
        momentum_20 = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
        momentum_50 = (close[-1] - close[-50]) / close[-50] if len(close) >= 50 else 0
        
        # Volume analysis
        volume = df['Volume'].values
        volume_ma = np.mean(volume[-20:])
        recent_volume = np.mean(volume[-5:])
        volume_increasing = recent_volume > volume_ma * 1.2
        
        # Determine phase
        if structure == "BULLISH":
            if momentum_20 > 0.05:  # Strong upward momentum
                return "MARKUP"
            else:
                return "ACCUMULATION"
        
        elif structure == "BEARISH":
            if momentum_20 < -0.05:  # Strong downward momentum
                return "BREAKDOWN"
            else:
                return "DISTRIBUTION"
        
        else:  # SIDEWAYS
            if momentum_50 > 0.1:  # After uptrend
                return "DISTRIBUTION"
            elif momentum_50 < -0.1:  # After downtrend
                return "ACCUMULATION"
            else:
                return "UNKNOWN"
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def analyze(self, df: pd.DataFrame) -> Optional[MarketStructureResult]:
        """
        Perform complete market structure analysis
        
        Args:
            df: DataFrame with OHLCV
        
        Returns:
            MarketStructureResult
        """
        if df is None or len(df) < 50:
            logger.warning("Insufficient data for market structure analysis")
            return None
        
        try:
            # Detect swing points
            swing_highs = self.detect_swing_highs(df)
            swing_lows = self.detect_swing_lows(df)
            
            # Classify points
            high_points = self.classify_swing_highs(swing_highs)
            low_points = self.classify_swing_lows(swing_lows)
            
            # Merge and sort all points
            all_points = high_points + low_points
            all_points.sort(key=lambda p: p.index)
            
            # Determine structure
            structure = self.determine_structure(high_points, low_points)
            
            # Detect BOS
            bos_detected, bos_direction = self.detect_bos(df, high_points, low_points)
            
            # Detect phase
            phase = self.detect_phase(df, structure)
            
            # Determine trend strength
            if structure in ["BULLISH", "BEARISH"]:
                # Count consecutive same-direction points
                recent_points = all_points[-6:] if len(all_points) >= 6 else all_points
                if structure == "BULLISH":
                    bullish_points = sum(1 for p in recent_points if p.point_type in ["HH", "HL"])
                    strength = "STRONG" if bullish_points >= 4 else "WEAK"
                else:
                    bearish_points = sum(1 for p in recent_points if p.point_type in ["LH", "LL"])
                    strength = "STRONG" if bearish_points >= 4 else "WEAK"
            else:
                strength = "WEAK"
            
            return MarketStructureResult(
                structure_type=structure,
                current_phase=phase,
                structure_points=all_points[-10:],  # Last 10 points
                bos_detected=bos_detected,
                bos_direction=bos_direction,
                trend_strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return None
