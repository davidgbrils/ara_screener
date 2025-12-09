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
        Detect parabolic price movement with Sequential TD (Tom DeMark)
        
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
        
        recent_returns = float(returns.tail(5).mean())
        earlier_returns = float(returns.head(len(returns) - 5).mean())
        
        # Basic parabolic check
        basic_parabolic = (recent_returns > earlier_returns * 2) and (recent_returns > 0.05)
        
        # Sequential TD check (Tom DeMark Sequential)
        # Count consecutive up closes
        up_closes = 0
        max_up_closes = 0
        for i in range(1, len(prices)):
            current_price = float(prices.iloc[i])
            prev_price = float(prices.iloc[i-1])
            if current_price > prev_price:
                up_closes += 1
                max_up_closes = max(max_up_closes, up_closes)
            else:
                up_closes = 0
        
        # Sequential TD: 9+ consecutive up closes indicates parabolic
        sequential_td = max_up_closes >= 9
        
        # Check for exponential acceleration
        if len(returns) >= 9:
            # Calculate acceleration (second derivative)
            acceleration = float(returns.tail(9).diff().mean())
            exponential = acceleration > 0 and recent_returns > 0.08
        else:
            exponential = False
        
        return basic_parabolic or sequential_td or exponential
    
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
        Detect Volatility Contraction Pattern (VCP) - Mark Minervini Style
        
        VCP characteristics:
        1. Price consolidation with decreasing volatility
        2. Volume dry-up during consolidation
        3. Tight price action (low range)
        4. Usually 3-5 contractions
        5. Breakout with volume
        
        Args:
            df: DataFrame with price data
            contraction_days: Days to analyze
        
        Returns:
            VCP pattern info or None
        """
        if len(df) < contraction_days * 2:
            return None
        
        # Look at longer period for VCP
        lookback = min(contraction_days * 2, len(df))
        recent = df.tail(lookback)
        
        # Calculate price range percentage
        price_ranges = (recent['High'] - recent['Low']) / recent['Close']
        
        # Divide into segments to detect contractions
        segments = 3
        segment_size = len(recent) // segments
        
        segment_vols = []
        segment_volumes = []
        
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segments - 1 else len(recent)
            segment = recent.iloc[start_idx:end_idx]
            
            # Average volatility
            seg_vol = price_ranges.iloc[start_idx:end_idx].mean()
            segment_vols.append(seg_vol)
            
            # Average volume
            seg_vol_avg = segment['Volume'].mean()
            segment_volumes.append(seg_vol_avg)
        
        # Check for decreasing volatility (contractions)
        contractions = 0
        for i in range(1, len(segment_vols)):
            current_vol = float(segment_vols[i])
            prev_vol = float(segment_vols[i-1])
            if current_vol < prev_vol * 0.85:  # 15% reduction
                contractions += 1
        
        # Check volume dry-up
        current_volume = float(segment_volumes[-1])
        initial_volume = float(segment_volumes[0])
        volume_dry_up = current_volume < initial_volume * 0.7
        
        # Check tight price action (latest segment has low range)
        latest_range = float(price_ranges.tail(segment_size).mean())
        tight_action = latest_range < 0.03  # Less than 3% range
        
        # Check if price is near recent high (breakout potential)
        recent_high = float(recent['High'].max())
        current_price = float(df['Close'].iloc[-1])
        near_high = current_price >= recent_high * 0.95
        
        # VCP detected if: contractions + volume dry-up + tight action
        if contractions >= 2 and volume_dry_up and tight_action:
            return {
                'detected': True,
                'contractions': contractions,
                'contraction_ratio': float(segment_vols[-1]) / float(segment_vols[0]) if float(segment_vols[0]) > 0 else 1.0,
                'volume_dry_up': volume_dry_up,
                'tight_action': tight_action,
                'near_high': near_high,
                'days': lookback,
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
        
        top = float(highs.max())
        bottom = float(lows.min())
        
        # Check if price is near top (breakout zone)
        current_price = float(df['Close'].iloc[-1])
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
        price_up = float(latest['Close']) > float(prev['Close'])
        volume_surge = float(latest['Volume']) > float(df['Volume'].tail(50).mean() * 1.5)
        
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
        
        gap = float(latest['Low']) - float(prev['High'])
        gap_pct = (gap / float(prev['High'])) * 100
        
        if gap_pct >= min_gap_pct:
            return {
                'detected': True,
                'gap_pct': gap_pct,
                'gap_amount': gap,
                'prev_high': float(prev['High']),
                'current_low': float(latest['Low']),
            }
        
        return None

    def detect_volume_dry_up(self, df: pd.DataFrame, days: int = 20) -> Optional[Dict]:
        """
        Detect Volume Dry-Up pattern
        
        Volume dry-up occurs when:
        1. Volume decreases significantly during consolidation
        2. Price moves sideways or slightly down
        3. Often precedes breakout
        
        Args:
            df: DataFrame with OHLCV data
            days: Lookback period
        
        Returns:
            Volume dry-up info or None
        """
        if len(df) < days * 2:
            return None
        
        recent = df.tail(days)
        earlier = df.tail(days * 2).head(days)
        
        # Calculate average volumes
        recent_avg_vol = recent['Volume'].mean()
        earlier_avg_vol = earlier['Volume'].mean()
        
        # Volume dry-up: recent volume < 70% of earlier
        volume_ratio = recent_avg_vol / earlier_avg_vol if earlier_avg_vol > 0 else 1.0
        
        if volume_ratio < 0.7:
            # Check price action (should be sideways or slight down)
            price_change = (recent['Close'].iloc[-1] - earlier['Close'].iloc[0]) / earlier['Close'].iloc[0]
            sideways = abs(price_change) < 0.10  # Less than 10% change
            
            # Check volatility (should be decreasing)
            recent_volatility = (recent['High'] - recent['Low']).mean() / recent['Close'].mean()
            earlier_volatility = (earlier['High'] - earlier['Low']).mean() / earlier['Close'].mean()
            decreasing_vol = recent_volatility < earlier_volatility * 0.8
            
            if sideways and decreasing_vol:
                return {
                    'detected': True,
                    'volume_ratio': volume_ratio,
                    'price_change_pct': price_change * 100,
                    'volatility_decrease': (1 - recent_volatility / earlier_volatility) * 100 if earlier_volatility > 0 else 0,
                    'days': days,
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

    def detect_money_flow(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Money Flow using Chaikin + OBV + VWAP profile
        
        Args:
            df: DataFrame with indicators (must have CMF, OBV, VWAP)
        
        Returns:
            Money flow info or None
        """
        if len(df) < 20:
            return None
        
        if 'CMF' not in df.columns or 'OBV' not in df.columns or 'VWAP' not in df.columns:
            return None
        
        latest = df.iloc[-1]
        
        # Chaikin Money Flow
        cmf = latest.get('CMF', 0)
        cmf_positive = cmf > 0.1  # Strong positive
        
        # OBV trend
        obv_trend = df['OBV'].tail(20).iloc[-1] > df['OBV'].tail(20).iloc[0]
        obv_rising = obv_trend and df['OBV'].tail(5).is_monotonic_increasing
        
        # VWAP position
        vwap = latest.get('VWAP', 0)
        close = latest['Close']
        above_vwap = close > vwap * 1.02  # 2% above VWAP
        
        # VWAP distance
        vwap_distance = latest.get('VWAP_DISTANCE', 0) if 'VWAP_DISTANCE' in df.columns else 0
        
        # Money flow score
        score = 0
        if cmf_positive:
            score += 1
        if obv_rising:
            score += 1
        if above_vwap:
            score += 1
        
        if score >= 2:
            return {
                'detected': True,
                'cmf': cmf,
                'cmf_positive': cmf_positive,
                'obv_rising': obv_rising,
                'above_vwap': above_vwap,
                'vwap_distance': vwap_distance,
                'score': score,
            }
        
        return None

    def detect_52w_breakout(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 60:
            return None
        latest = df.iloc[-1]
        high_52w = df['High'].rolling(window=252, min_periods=1).max().iloc[-1] if 'HIGH_52W' not in df.columns else df['HIGH_52W'].iloc[-1]
        prev_high = df['High'].rolling(window=252, min_periods=1).max().iloc[-2]
        price = latest['Close']
        volume = latest['Volume']
        avg_vol = df['Volume'].tail(50).mean()
        breakout = price >= high_52w * 0.995 and high_52w >= prev_high
        vol_confirm = volume >= avg_vol * 1.5
        if breakout and vol_confirm:
            return {
                'detected': True,
                'high_52w': float(high_52w),
                'price': float(price),
                'volume_surge': float(volume / max(avg_vol, 1)),
            }
        return None

    def detect_candlestick_bullish(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 2:
            return None
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        engulfing = latest['Close'] > latest['Open'] and prev['Close'] < prev['Open'] and latest['Close'] >= prev['Open'] and latest['Open'] <= prev['Close']
        body = abs(latest['Close'] - latest['Open'])
        range_c = latest['High'] - latest['Low']
        lower_wick = latest['Open'] - latest['Low'] if latest['Close'] >= latest['Open'] else latest['Close'] - latest['Low']
        hammer = body <= range_c * 0.4 and lower_wick >= range_c * 0.5 and latest['Close'] > latest['Open']
        if engulfing or hammer:
            return {
                'detected': True,
                'engulfing': bool(engulfing),
                'hammer': bool(hammer),
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
        
        # New patterns
        patterns['volume_dry_up'] = self.detect_volume_dry_up(df)
        patterns['money_flow'] = self.detect_money_flow(df)
        patterns['breakout_52w'] = self.detect_52w_breakout(df)
        patterns['candlestick_bullish'] = self.detect_candlestick_bullish(df)
        
        return patterns

