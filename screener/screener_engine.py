"""Main screener engine for ARA signal generation"""

import pandas as pd
from typing import Dict, Optional, List
from config import SCREENER_CONFIG, SIGNAL_THRESHOLDS
from .pattern_detector import PatternDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ScreenerEngine:
    """Main screener for identifying ARA/multi-bagger candidates"""
    
    def __init__(self):
        """Initialize screener engine"""
        self.config = SCREENER_CONFIG
        self.pattern_detector = PatternDetector()
    
    def screen(self, df: pd.DataFrame, ticker: str) -> Dict:
        """
        Screen ticker and generate signal
        
        Args:
            df: DataFrame with indicators
            ticker: Ticker symbol
        
        Returns:
            Dictionary with signal and reasons
        """
        if df is None or df.empty or len(df) < 50:
            return {
                'ticker': ticker,
                'signal': 'NONE',
                'score': 0.0,
                'reasons': [],
                'patterns': {},
            }
        
        latest = df.iloc[-1]
        
        # Check basic filters
        if not self._passes_basic_filters(df, latest):
            return {
                'ticker': ticker,
                'signal': 'NONE',
                'score': 0.0,
                'reasons': ['Failed basic filters'],
                'patterns': {},
            }
        
        # Calculate score
        score, reasons, parameter_count = self._calculate_score(df, latest)
        
        # Detect patterns
        patterns = self.pattern_detector.detect_all_patterns(df)
        
        # Determine signal
        signal = self._determine_signal(score)
        
        # Calculate confidence/accuracy score
        confidence = self._calculate_confidence(score, parameter_count, patterns, df, latest)
        
        # Get entry/exit levels
        entry_levels = self._calculate_entry_levels(df, latest)
        
        # Data validation
        data_quality = self._validate_data_quality(df, latest)
        
        return {
            'ticker': ticker,
            'signal': signal,
            'score': score,
            'confidence': confidence,
            'parameter_count': parameter_count,
            'data_quality': data_quality,
            'reasons': reasons,
            'patterns': patterns,
            'entry_levels': entry_levels,
            'latest_price': float(latest['Close']),
            'latest_volume': int(latest['Volume']),
        }
    
    def _passes_basic_filters(self, df: pd.DataFrame, latest: pd.Series) -> bool:
        """Check basic filters"""
        # Price filter
        if self.config["MIN_PRICE"] and latest['Close'] < self.config["MIN_PRICE"]:
            return False
        if self.config["MAX_PRICE"] and latest['Close'] > self.config["MAX_PRICE"]:
            return False
        
        # Volume filter
        avg_volume = df['Volume'].tail(20).mean()
        if avg_volume < self.config["MIN_VOLUME"]:
            return False
        
        return True
    
    def _calculate_score(self, df: pd.DataFrame, latest: pd.Series) -> tuple:
        """
        Calculate ARA score with parameter count
        
        Returns:
            Tuple of (score, reasons, parameter_count)
        """
        score = 0.0
        reasons = []
        parameter_count = 0
        max_parameters = 7  # Total number of parameters checked
        
        # RVOL check
        rvol = latest.get('RVOL', 0)
        if rvol >= self.config["RVOL_THRESHOLD"]:
            score += 0.25
            reasons.append(f"RVOL {rvol:.2f}x")
            parameter_count += 1
        
        # Bollinger Breakout
        if pd.notna(latest.get('BB_Upper')) and latest['Close'] > latest['BB_Upper']:
            score += 0.20
            reasons.append("Bollinger Breakout")
            parameter_count += 1
        
        # MA Structure
        ma20 = latest.get('MA20', 0)
        ma50 = latest.get('MA50', 0)
        ma200 = latest.get('MA200', 0)
        
        if ma20 > ma50 > ma200 and all(pd.notna([ma20, ma50, ma200])):
            score += 0.15
            reasons.append("Bullish MA Structure")
            parameter_count += 1
        
        # RSI Momentum
        rsi = latest.get('RSI', 50)
        if self.config["RSI_MIN"] <= rsi <= self.config["RSI_MAX"]:
            score += 0.15
            reasons.append(f"RSI {rsi:.1f}")
            parameter_count += 1
        
        # OBV Rising
        if len(df) >= 20:
            obv_trend = df['OBV'].tail(20).iloc[-1] > df['OBV'].tail(20).iloc[0]
            if obv_trend:
                score += 0.10
                reasons.append("OBV Rising")
                parameter_count += 1
        
        # Close > VWAP
        vwap = latest.get('VWAP', 0)
        if pd.notna(vwap) and latest['Close'] > vwap:
            score += 0.10
            reasons.append("Above VWAP")
            parameter_count += 1
        
        # MA20 Slope
        ma20_slope = latest.get('MA20_SLOPE', 0)
        if ma20_slope > self.config["MA_SLOPE_MIN"]:
            score += 0.05
            reasons.append("MA20 Accelerating")
            parameter_count += 1
        
        return min(score, 1.0), reasons, parameter_count
    
    def _calculate_confidence(
        self, 
        score: float, 
        parameter_count: int, 
        patterns: Dict, 
        df: pd.DataFrame,
        latest: pd.Series
    ) -> float:
        """
        Calculate confidence/accuracy score based on multiple factors
        
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0
        
        # Base confidence from score (40%)
        confidence += score * 0.4
        
        # Parameter count (30%) - more parameters = higher confidence
        max_params = 7
        param_ratio = min(parameter_count / max_params, 1.0)
        confidence += param_ratio * 0.3
        
        # Pattern detection (20%) - more patterns = higher confidence
        detected_patterns = sum(1 for v in patterns.values() 
                              if v and (isinstance(v, bool) or (isinstance(v, dict) and v.get('detected'))))
        pattern_score = min(detected_patterns / 5.0, 1.0)  # Max 5 patterns
        confidence += pattern_score * 0.2
        
        # Data quality (10%) - based on volume and price consistency
        if len(df) >= 200:  # Sufficient history
            confidence += 0.05
        if latest['Volume'] > df['Volume'].tail(20).mean() * 2:  # High volume
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _validate_data_quality(self, df: pd.DataFrame, latest: pd.Series) -> Dict:
        """
        Validate data quality and return validation info
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0,
        }
        
        # Check data completeness
        if len(df) < 200:
            validation['issues'].append(f"Insufficient history: {len(df)} days")
            validation['quality_score'] -= 0.1
        
        # Check for missing values
        missing_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                       if pd.isna(latest.get(col))]
        if missing_cols:
            validation['issues'].append(f"Missing data: {', '.join(missing_cols)}")
            validation['quality_score'] -= 0.2
            validation['is_valid'] = False
        
        # Check price consistency
        if latest['High'] < latest['Low'] or latest['High'] < latest['Close'] or latest['Low'] > latest['Close']:
            validation['issues'].append("Price data inconsistency")
            validation['quality_score'] -= 0.3
            validation['is_valid'] = False
        
        # Check volume
        if latest['Volume'] <= 0:
            validation['issues'].append("Invalid volume")
            validation['quality_score'] -= 0.2
            validation['is_valid'] = False
        
        # Check for zero prices
        if latest['Close'] <= 0:
            validation['issues'].append("Invalid price")
            validation['quality_score'] -= 0.3
            validation['is_valid'] = False
        
        validation['quality_score'] = max(validation['quality_score'], 0.0)
        
        return validation
    
    def _determine_signal(self, score: float) -> str:
        """Determine signal based on score"""
        if score >= SIGNAL_THRESHOLDS["STRONG_AURA"]:
            return "STRONG_AURA"
        elif score >= SIGNAL_THRESHOLDS["WATCHLIST"]:
            return "WATCHLIST"
        elif score >= SIGNAL_THRESHOLDS["POTENTIAL"]:
            return "POTENTIAL"
        else:
            return "NONE"
    
    def _calculate_entry_levels(self, df: pd.DataFrame, latest: pd.Series) -> Dict:
        """
        Calculate entry, stop loss, and take profit levels
        
        Returns:
            Dictionary with entry levels
        """
        from config import ENTRY_CONFIG
        
        current_price = latest['Close']
        atr = latest.get('ATR', current_price * 0.02)  # Default 2% if no ATR
        
        # Entry zone
        entry_zone_pct = ENTRY_CONFIG["ENTRY_ZONE_PCT"]
        entry_low = current_price * (1 - entry_zone_pct)
        entry_high = current_price * (1 + entry_zone_pct)
        
        # Stop Loss (below support or ATR-based)
        sl_atr = current_price - (atr * ENTRY_CONFIG["ATR_MULTIPLIER_SL"])
        
        # Find recent support
        lookback = min(ENTRY_CONFIG["SUPPORT_RESISTANCE_WINDOW"], len(df))
        recent_low = df['Low'].tail(lookback).min()
        sl_support = recent_low * 0.98  # 2% below support
        
        stop_loss = min(sl_atr, sl_support)
        
        # Take Profit levels
        tp1 = current_price + (atr * ENTRY_CONFIG["ATR_MULTIPLIER_TP1"])
        tp2 = current_price + (atr * ENTRY_CONFIG["ATR_MULTIPLIER_TP2"])
        
        # Reward to Risk
        risk = current_price - stop_loss
        reward1 = tp1 - current_price
        reward2 = tp2 - current_price
        rr1 = reward1 / risk if risk > 0 else 0
        rr2 = reward2 / risk if risk > 0 else 0
        
        return {
            'entry_low': float(entry_low),
            'entry_high': float(entry_high),
            'stop_loss': float(stop_loss),
            'take_profit_1': float(tp1),
            'take_profit_2': float(tp2),
            'reward_risk_1': float(rr1),
            'reward_risk_2': float(rr2),
        }

