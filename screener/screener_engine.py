"""
Main screener engine for ARA signal generation - Clean and Optimized Version

This module provides a comprehensive stock screening engine for identifying
ARA (Accumulation, Reaccumulation, and Advance) patterns and multi-bagger candidates.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Union, Any
from dataclasses import dataclass
from fundamentals.financial_fetcher import FinancialFetcher
from config import SCREENER_CONFIG, SIGNAL_THRESHOLDS, PATTERN_CONFIG, ENTRY_CONFIG
from .pattern_detector import PatternDetector
from .regime_filter import RegimeFilter
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ScreeningResult:
    """Data class for screening results"""
    ticker: str
    signal: str
    score: float
    confidence: float
    parameter_count: int
    data_quality: Dict[str, Any]
    reasons: List[str]
    patterns: Dict[str, Any]
    entry_levels: Dict[str, float]
    latest_price: float
    latest_volume: int
    regime: Dict[str, Any]
    classifications: Dict[str, Dict]
    fundamentals: Dict[str, Any]


@dataclass
class ClassificationResult:
    """Base data class for classification results"""
    score: int
    reasons: List[str]
    class_name: str = "Unknown"


@dataclass
class MaxARAResult(ClassificationResult):
    """MAX ARA classification result"""
    pass


@dataclass
class BPJSResult:
    """BPJS classification result"""
    score: int
    reasons: List[str]
    advice: Dict[str, str]
    candidate: bool
    strong_candidate: bool
    suitability: str
    class_name: str = "BPJS"


@dataclass
class GorenganResult:
    """Gorengan classification result"""
    score: int
    reasons: List[str]
    risk: str
    class_name: str = "Gorengan"


class ScreenerEngine:
    """
    Main screener for identifying ARA/multi-bagger candidates
    
    This engine performs comprehensive technical analysis including:
    - Volume and price momentum analysis
    - Technical indicator scoring
    - Pattern detection
    - Market regime analysis
    - Risk classification
    - Entry/exit level calculation
    """
    
    def __init__(self):
        """Initialize screener engine with all components"""
        self.config = SCREENER_CONFIG
        self.pattern_detector = PatternDetector()
        self.regime_filter = RegimeFilter() if PATTERN_CONFIG.get("MARKET_REGIME", True) else None
        self.fin_fetcher = FinancialFetcher()
        
        # Cache for fundamentals to avoid repeated API calls
        self._fundamentals_cache = {}
    
    def screen(self, df: pd.DataFrame, ticker: str) -> Dict:
        """
        Screen ticker and generate comprehensive signal
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            ticker: Stock ticker symbol
            
        Returns:
            ScreeningResult with complete analysis
        """
        try:
            logger.debug(f"Starting screen for {ticker}")
            
            # Input validation
            if not self._validate_input(df, ticker):
                logger.debug(f"Input validation failed for {ticker}")
                return self._create_empty_result(ticker, "Invalid input data")
            
            latest = df.iloc[-1]
            logger.debug(f"Latest data shape: {latest.shape}")
            
            # Basic filtering
            if not self._passes_basic_filters(df, latest):
                logger.debug(f"Basic filters failed for {ticker}")
                return self._create_empty_result(ticker, "Failed basic filters")
            
            logger.debug(f"Basic filters passed for {ticker}")
            
            # Core analysis
            score, reasons, parameter_count = self._calculate_score(df, latest)
            # patterns = self.pattern_detector.detect_all_patterns(df)  # Temporarily disabled
            patterns = {}  # Empty patterns for now
            regime_info = self._get_regime_info(df, ticker)
            signal = self._determine_signal(score)
            confidence = self._calculate_confidence(score, parameter_count, patterns, df, latest, regime_info)
            entry_levels = self._calculate_entry_levels(df, latest)
            data_quality = self._validate_data_quality(df, latest)
            
            # Classifications
            fundamentals = self._get_fundamentals_cached(ticker)
            max_ara = self._classify_max_ara(df, latest, fundamentals)
            bpjs = self._classify_bpjs(df, latest, fundamentals)
            gorengan = self._classify_gorengan(df, latest, fundamentals)
            
            # Convert ScreeningResult to dictionary for compatibility
            result = ScreeningResult(
                ticker=ticker,
                signal=signal,
                score=score,
                confidence=confidence,
                parameter_count=parameter_count,
                data_quality=data_quality,
                reasons=reasons,
                patterns=patterns,
                entry_levels=entry_levels,
                latest_price=float(latest['Close']),
                latest_volume=int(latest['Volume']),
                regime=regime_info,
                classifications={
                    'max_ara': max_ara.__dict__,
                    'bpjs': bpjs.__dict__,
                    'gorengan': gorengan.__dict__,
                },
                fundamentals=fundamentals or {}
            )
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"Error screening {ticker}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_empty_result(ticker, f"Screening error: {str(e)}")
    
    def _validate_input(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validate input data"""
        if df is None or df.empty:
            return False
        if len(df) < 50:
            return False
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in df.columns for col in required_columns)
    
    def _create_empty_result(self, ticker: str, reason: str) -> Dict:
        """Create empty result for failed screenings"""
        result = ScreeningResult(
            ticker=ticker,
            signal='NONE',
            score=0.0,
            confidence=0.0,
            parameter_count=0,
            data_quality={'is_valid': False, 'issues': [reason], 'quality_score': 0.0},
            reasons=[reason],
            patterns={},
            entry_levels={},
            latest_price=0.0,
            latest_volume=0,
            regime={'regime': 'NEUTRAL', 'confidence': 0.0},
            classifications={'max_ara': {}, 'bpjs': {}, 'gorengan': {}},
            fundamentals={}
        )
        return result.__dict__
    
    def _passes_basic_filters(self, df: pd.DataFrame, latest: pd.Series) -> bool:
        """Apply basic price, volume, and volatility filters"""
        try:
            # Helper function to safely get numeric values
            def get_numeric(value, default=0):
                try:
                    if hasattr(value, 'iloc'):
                        # It's a Series, get the first value
                        value = value.iloc[0]
                    if pd.isna(value):
                        return default
                    return float(value)
                except (AttributeError, IndexError, TypeError):
                    return default
            
            # Price filters
            min_price = self.config.get("MIN_PRICE")
            max_price = self.config.get("MAX_PRICE")
            close_price = get_numeric(latest['Close'])
            if min_price is not None and close_price < min_price:
                return False
            if max_price is not None and close_price > max_price:
                return False
            
            # Volume filter
            avg_volume = float(df['Volume'].tail(20).mean())
            if avg_volume < self.config["MIN_VOLUME"]:
                return False
            
            # Volatility filter
            atr_pct = latest.get('ATR_PCT')
            atr_range = self.config.get("ATR_PCT_RANGE")
            if atr_range and pd.notna(atr_pct):
                low, high = atr_range
                atr_pct_float = get_numeric(atr_pct)
                if not (low <= atr_pct_float <= high):
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in basic filters: {e}")
            return False
    
    def _calculate_score(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[float, List[str], int]:
        """
        Calculate comprehensive ARA score
        
        Returns:
            Tuple of (score, reasons, parameter_count)
        """
        try:
            score = 0.0
            reasons = []
            parameter_count = 0
            
            # Helper function to safely get numeric values
            def get_numeric(value, default=0):
                try:
                    if hasattr(value, 'iloc'):
                        # It's a Series, get the first value
                        value = value.iloc[0]
                    if pd.isna(value):
                        return default
                    return float(value)
                except (AttributeError, IndexError, TypeError):
                    return default
            
            logger.debug(f"Starting score calculation for {latest.get('ticker', 'Unknown')}")
            
            # RVOL check (25% weight)
            rvol = get_numeric(latest.get('RVOL', 0))
            logger.debug(f"RVOL: {rvol}")
            if rvol >= self.config["RVOL_THRESHOLD"]:
                score += 0.25
                reasons.append(f"RVOL {rvol:.2f}x")
                parameter_count += 1
            
            # Bollinger Breakout (20% weight)
            bb_upper = latest.get('BB_Upper')
            close_price = get_numeric(latest['Close'])
            logger.debug(f"BB_Upper: {bb_upper}, Close: {close_price}")
            if pd.notna(bb_upper) and close_price > get_numeric(bb_upper):
                score += 0.20
                reasons.append("Bollinger Breakout")
                parameter_count += 1
            
            # MA Structure (15% weight)
            logger.debug("Checking MA structure")
            if self._check_bullish_ma_structure(latest):
                score += 0.15
                reasons.append("Bullish MA Structure")
                parameter_count += 1
            
            # RSI Momentum (15% weight)
            rsi = get_numeric(latest.get('RSI', 50))
            logger.debug(f"RSI: {rsi}")
            if self.config["RSI_MIN"] <= rsi <= self.config["RSI_MAX"]:
                score += 0.15
                reasons.append(f"RSI {rsi:.1f}")
                parameter_count += 1
            
            # OBV Rising (10% weight)
            logger.debug("Checking OBV rising")
            if self._check_obv_rising(df):
                score += 0.10
                reasons.append("OBV Rising")
                parameter_count += 1
            
            # VWAP Position (10% weight)
            vwap = latest.get('VWAP')
            logger.debug(f"VWAP: {vwap}, Close: {close_price}")
            if pd.notna(vwap) and close_price > get_numeric(vwap):
                score += 0.10
                reasons.append("Above VWAP")
                parameter_count += 1
            
            # Additional momentum factors
            logger.debug("Calculating momentum factors")
            additional_score = self._calculate_momentum_factors(latest, reasons, parameter_count)
            score += additional_score
            
            logger.debug(f"Final score: {score}")
            return min(score, 1.0), reasons, parameter_count
            
        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0, ["Score calculation error"], 0
    
    def _check_bullish_ma_structure(self, latest: pd.Series) -> bool:
        """Check for bullish moving average structure"""
        def get_numeric(value, default=0):
            try:
                if hasattr(value, 'iloc'):
                    # It's a Series, get the first value
                    value = value.iloc[0]
                if pd.isna(value):
                    return default
                return float(value)
            except (AttributeError, IndexError, TypeError):
                return default
        
        ma20 = get_numeric(latest.get('MA20'))
        ma50 = get_numeric(latest.get('MA50'))
        ma200 = get_numeric(latest.get('MA200'))
        
        return (ma20 > 0 and ma50 > 0 and ma200 > 0 and 
                ma20 > ma50 > ma200)
    
    def _check_obv_rising(self, df: pd.DataFrame) -> bool:
        """Check if OBV is rising over the last 20 days"""
        if len(df) < 20:
            return False
        try:
            return df['OBV'].tail(20).iloc[-1] > df['OBV'].tail(20).iloc[0]
        except (KeyError, IndexError):
            return False
    
    def _calculate_momentum_factors(self, latest: pd.Series, reasons: List[str], parameter_count: int) -> float:
        """Calculate additional momentum factors"""
        score = 0.0
        
        # Helper function to safely get numeric values
        def get_numeric(value, default=0):
            try:
                if hasattr(value, 'iloc'):
                    # It's a Series, get the first value
                    value = value.iloc[0]
                if pd.isna(value):
                    return default
                return float(value)
            except (AttributeError, IndexError, TypeError):
                return default
        
        # MA20 Slope (5% weight)
        ma20_slope = get_numeric(latest.get('MA20_SLOPE', 0))
        if ma20_slope > self.config["MA_SLOPE_MIN"]:
            score += 0.05
            reasons.append("MA20 Accelerating")
            parameter_count += 1
        
        # MACD confirmation (10% weight)
        if self._check_macd_bullish(latest):
            score += 0.10
            reasons.append("MACD Bullish")
            parameter_count += 1
        
        # Multi-horizon momentum (up to 15% weight)
        momentum_score = self._check_multi_horizon_momentum(latest, reasons, parameter_count)
        score += momentum_score
        
        # 52-week proximity (5% weight)
        dist_52w = latest.get('DIST_52W_HIGH')
        if pd.notna(dist_52w) and get_numeric(dist_52w) >= -3:
            score += 0.05
            reasons.append("Near 52W High")
            parameter_count += 1
        
        return score
    
    def _check_macd_bullish(self, latest: pd.Series) -> bool:
        """Check for bullish MACD configuration"""
        def get_numeric(value, default=0):
            try:
                if hasattr(value, 'iloc'):
                    # It's a Series, get the first value
                    value = value.iloc[0]
                if pd.isna(value):
                    return default
                return float(value)
            except (AttributeError, IndexError, TypeError):
                return default
        
        macd = get_numeric(latest.get('MACD'))
        macd_signal = get_numeric(latest.get('MACD_SIGNAL'))
        macd_hist = get_numeric(latest.get('MACD_HIST'))
        
        return (macd != 0 and macd_signal != 0 and macd_hist != 0 and
                macd > macd_signal and macd > 0 and macd_hist > 0)
    
    def _check_multi_horizon_momentum(self, latest: pd.Series, reasons: List[str], parameter_count: int) -> float:
        """Check momentum across different timeframes"""
        score = 0.0
        
        # Helper function to safely get numeric values
        def get_numeric(value, default=0):
            try:
                if hasattr(value, 'iloc'):
                    # It's a Series, get the first value
                    value = value.iloc[0]
                if pd.isna(value):
                    return default
                return float(value)
            except (AttributeError, IndexError, TypeError):
                return default
        
        pct_1w = get_numeric(latest.get('PCT_CHANGE_1W', 0))
        pct_1m = get_numeric(latest.get('PCT_CHANGE_1M', 0))
        pct_3m = get_numeric(latest.get('PCT_CHANGE_3M', 0))
        
        if pct_1w >= 5:
            score += 0.05
            reasons.append("1W Momentum")
            parameter_count += 1
        if pct_1m >= 10:
            score += 0.05
            reasons.append("1M Momentum")
            parameter_count += 1
        if pct_3m >= 20:
            score += 0.05
            reasons.append("3M Momentum")
            parameter_count += 1
        
        return score
    
    def _get_regime_info(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Get market regime information"""
        if not self.regime_filter:
            return {'regime': 'NEUTRAL', 'confidence': 0.0}
        
        try:
            return self.regime_filter.detect_regime(df)
        except Exception as e:
            logger.warning(f"Regime detection failed for {ticker}: {e}")
            return {'regime': 'NEUTRAL', 'confidence': 0.0}
    
    def _determine_signal(self, score: float) -> str:
        """Determine signal based on score thresholds"""
        if score >= SIGNAL_THRESHOLDS["STRONG_AURA"]:
            return "STRONG_AURA"
        elif score >= SIGNAL_THRESHOLDS["WATCHLIST"]:
            return "WATCHLIST"
        elif score >= SIGNAL_THRESHOLDS["POTENTIAL"]:
            return "POTENTIAL"
        else:
            return "NONE"
    
    def _calculate_confidence(
        self, 
        score: float, 
        parameter_count: int, 
        patterns: Dict, 
        df: pd.DataFrame,
        latest: pd.Series,
        regime_info: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on multiple factors"""
        try:
            confidence = 0.0
            
            # Base confidence from score (40%)
            confidence += score * 0.4
            
            # Parameter count (30%)
            max_params = 12
            param_ratio = min(parameter_count / max_params, 1.0)
            confidence += param_ratio * 0.3
            
            # Pattern detection (20%)
            detected_patterns = self._count_detected_patterns(patterns)
            pattern_score = min(detected_patterns / 5.0, 1.0)
            confidence += pattern_score * 0.2
            
            # Data quality (5%)
            confidence += self._calculate_data_quality_boost(df, latest)
            
            # Market regime (5%)
            if regime_info.get('regime') == 'BULL':
                confidence += 0.05 * regime_info.get('confidence', 0)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _count_detected_patterns(self, patterns: Dict) -> int:
        """Count number of detected patterns"""
        return sum(1 for v in patterns.values() 
                  if v and (isinstance(v, bool) or (isinstance(v, dict) and v.get('detected'))))
    
    def _calculate_data_quality_boost(self, df: pd.DataFrame, latest: pd.Series) -> float:
        """Calculate confidence boost from data quality"""
        boost = 0.0
        
        if len(df) >= 200:
            boost += 0.03
        if latest['Volume'] > df['Volume'].tail(20).mean() * 2:
            boost += 0.02
        
        return boost
    
    def _validate_data_quality(self, df: pd.DataFrame, latest: pd.Series) -> Dict[str, Any]:
        """Validate data quality and return validation info"""
        validation = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0,
        }
        
        try:
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
            if not self._check_price_consistency(latest):
                validation['issues'].append("Price data inconsistency")
                validation['quality_score'] -= 0.3
                validation['is_valid'] = False
            
            # Check volume validity
            if latest['Volume'] <= 0:
                validation['issues'].append("Invalid volume")
                validation['quality_score'] -= 0.2
                validation['is_valid'] = False
            
            # Check price validity
            if latest['Close'] <= 0:
                validation['issues'].append("Invalid price")
                validation['quality_score'] -= 0.3
                validation['is_valid'] = False
            
            validation['quality_score'] = max(validation['quality_score'], 0.0)
            
        except Exception as e:
            logger.error(f"Error in data quality validation: {e}")
            validation['is_valid'] = False
            validation['quality_score'] = 0.0
            validation['issues'].append("Validation error")
        
        return validation
    
    def _check_price_consistency(self, latest: pd.Series) -> bool:
        """Check for price data consistency"""
        try:
            return not (latest['High'] < latest['Low'] or 
                       latest['High'] < latest['Close'] or 
                       latest['Low'] > latest['Close'])
        except (KeyError, TypeError):
            return False
    
    def _calculate_entry_levels(self, df: pd.DataFrame, latest: pd.Series) -> Dict[str, float]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            current_price = latest['Close']
            atr = latest.get('ATR', current_price * 0.02)  # Default 2% if no ATR
            
            # Entry zone
            entry_zone_pct = ENTRY_CONFIG["ENTRY_ZONE_PCT"]
            entry_low = current_price * (1 - entry_zone_pct)
            entry_high = current_price * (1 + entry_zone_pct)
            
            # Stop Loss calculation
            stop_loss = self._calculate_stop_loss(df, latest, current_price, atr)
            
            # Take Profit levels
            tp1 = current_price + (atr * ENTRY_CONFIG["ATR_MULTIPLIER_TP1"])
            tp2 = current_price + (atr * ENTRY_CONFIG["ATR_MULTIPLIER_TP2"])
            
            # Reward to Risk ratios
            risk = current_price - stop_loss
            rr1 = (tp1 - current_price) / risk if risk > 0 else 0
            rr2 = (tp2 - current_price) / risk if risk > 0 else 0
            
            return {
                'entry_low': float(entry_low),
                'entry_high': float(entry_high),
                'stop_loss': float(stop_loss),
                'take_profit_1': float(tp1),
                'take_profit_2': float(tp2),
                'reward_risk_1': float(rr1),
                'reward_risk_2': float(rr2),
            }
            
        except Exception as e:
            logger.error(f"Error calculating entry levels: {e}")
            return {}
    
    def _calculate_stop_loss(self, df: pd.DataFrame, latest: pd.Series, current_price: float, atr: float) -> float:
        """Calculate optimal stop loss level"""
        try:
            # ATR-based stop loss
            sl_atr = current_price - (atr * ENTRY_CONFIG["ATR_MULTIPLIER_SL"])
            
            # Support-based stop loss
            lookback = min(ENTRY_CONFIG["SUPPORT_RESISTANCE_WINDOW"], len(df))
            recent_low = df['Low'].tail(lookback).min()
            sl_support = recent_low * 0.98  # 2% below support
            
            return min(sl_atr, sl_support)
            
        except Exception:
            return current_price * 0.95  # Default 5% stop loss
    
    def _classify_max_ara(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Optional[Dict] = None) -> MaxARAResult:
        """Classify for MAX ARA (Upper Auto Rejection) potential"""
        try:
            score = 0
            reasons = []
            
            # Get technical indicators
            indicators = self._extract_technical_indicators(df, latest)
            
            # 1. Basic Liquidity Check (Crucial for ARA)
            # Ensure stock has enough value to be tradeable but not too heavy
            tx_value = latest['Close'] * latest['Volume']
            if tx_value >= 1_000_000_000: # Min 1B transaction value
                score += 10
            elif tx_value < 500_000_000:
                reasons.append('Low Liquidity')
                # Penalize low liquidity significantly
                score -= 20
            
            # 2. Basic structure checks
            if self._check_max_ara_structure(indicators, latest, reasons):
                score += 15  # MA structure
                score += 15  # Breakout
            
            # 3. Volume and momentum checks (Modified for accuracy)
            score += self._check_max_ara_volume(indicators, latest, reasons)
            score += self._check_max_ara_momentum(indicators, latest, reasons)
            
            # 4. Volatility and trend checks
            score += self._check_max_ara_volatility(indicators, reasons)
            score += self._check_max_ara_ratios(indicators, reasons)
            
            # 5. ARA Proximity Check
            # Check if close is nearing ARA limit (simplified estimation)
            # Assuming ARA limits: <200: 35%, 200-5000: 25%, >5000: 20%
            price = latest['Close']
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else price
            pct_change = ((price - prev_close) / prev_close) * 100
            
            is_near_ara = False
            if price < 200 and pct_change >= 20: 
                is_near_ara = True
            elif 200 <= price < 5000 and pct_change >= 15:
                is_near_ara = True
            elif price >= 5000 and pct_change >= 10:
                is_near_ara = True
                
            if is_near_ara:
                score += 20
                reasons.append('Near ARA momentum')
            
            # 6. Fundamentals check
            if fundamentals and self._check_max_ara_fundamentals(fundamentals, reasons):
                score += 5
            
            # Determine classification
            cls = self._determine_max_ara_class(score)
            
            return MaxARAResult(score=score, reasons=reasons, class_name=cls)
            
        except Exception as e:
            logger.error(f"Error in MAX ARA classification: {e}")
            return MaxARAResult(score=0, reasons=[], class_name='Low')
    
    def _extract_technical_indicators(self, df: pd.DataFrame, latest: pd.Series) -> Dict[str, float]:
        """Extract common technical indicators for classification"""
        return {
            'vol20': df['Volume'].tail(20).mean(),
            'vol5': df['Volume'].tail(5).mean(),
            'rv': latest.get('RVOL', 0),
            'rsi': latest.get('RSI', 0),
            'vwap': latest.get('VWAP', 0),
            'obv_slope': latest.get('OBV_SLOPE_5D', 0),
            'ma20': latest.get('MA20', 0),
            'ma50': latest.get('MA50', 0),
            'ma200': latest.get('MA200', 0),
            'ma20_slope': latest.get('MA20_SLOPE', 0),
            'ma20_pct5': latest.get('MA20_PCT_CHANGE_5D', 0),
            'atr_pct': latest.get('ATR_PCT', 0) * 100 if latest.get('ATR_PCT') is not None else None,
            'high20': latest.get('HIGH_20D', 0),
            'close': latest.get('Close', 0),
            'std10': latest.get('STDDEV_10D', 0),
            'std20': latest.get('STDDEV_20D', 0),
        }
    
    def _check_max_ara_structure(self, indicators: Dict, latest: pd.Series, reasons: List[str]) -> bool:
        """Check basic MAX ARA structure requirements"""
        close = indicators['close']
        ma20 = indicators['ma20']
        ma50 = indicators['ma50']
        ma200 = indicators['ma200']
        high20 = indicators['high20']
        
        structure_ok = True
        
        if close > ma20 and close > ma50 and close > ma200:
            reasons.append('MA structure')
        else:
            structure_ok = False
        
        if high20 and close >= high20:
            reasons.append('Breakout High20D')
        else:
            structure_ok = False
        
        return structure_ok
    
    def _check_max_ara_volume(self, indicators: Dict, latest: pd.Series, reasons: List[str]) -> int:
        """Check volume criteria for MAX ARA"""
        score = 0
        
        # Consistent High Volume is better than just a spike
        if indicators['vol5'] and indicators['vol20'] and indicators['vol5'] > indicators['vol20']:
            score += 10
            reasons.append('Vol5 > Vol20')

        # RVOL logic refined
        rvol = indicators['rv']
        if rvol >= 2.5: # Lowered from 4 to catch early moves
            score += 20
            reasons.append(f'RVOL {rvol:.1f}x')
        elif rvol >= 1.5:
            score += 10
        
        return score
    
    def _check_max_ara_momentum(self, indicators: Dict, latest: pd.Series, reasons: List[str]) -> int:
        """Check momentum criteria for MAX ARA"""
        score = 0
        
        if indicators['vwap'] and indicators['close'] > indicators['vwap']:
            score += 8
            reasons.append('Close>VWAP')
        
        if 55 <= indicators['rsi'] <= 75:
            score += 6
            reasons.append('RSI zone')
        
        if indicators['obv_slope'] and indicators['obv_slope'] > 0:
            score += 6
            reasons.append('OBV rising')
        
        if indicators['ma20_pct5'] and indicators['ma20_pct5'] >= 8:
            score += 10
            reasons.append('MA20 pct_change 5d>=8%')
        
        return score
    
    def _check_max_ara_volatility(self, indicators: Dict, reasons: List[str]) -> int:
        """Check volatility criteria for MAX ARA"""
        score = 0
        
        if indicators['atr_pct'] is not None and indicators['atr_pct'] <= 8:
            score += 5
            reasons.append('ATR%<=8')
        
        return score
    
    def _check_max_ara_ratios(self, indicators: Dict, reasons: List[str]) -> int:
        """Check various ratios for MAX ARA"""
        score = 0
        
        # MA ratios
        if indicators['ma50'] and indicators['ma20'] / indicators['ma50'] >= 1.02:
            score += 2
        if indicators['ma200'] and indicators['ma20'] / indicators['ma200'] >= 1.05:
            score += 2
        
        # Price ratios
        if indicators['high20'] and indicators['close'] / indicators['high20'] >= 1.0:
            score += 2
        
        # Volume ratios
        if indicators['vol5'] and indicators['close'] / indicators['vol5'] >= 2:  # Note: This might be a bug in original
            score += 2
        
        # Volatility ratios
        if indicators['std10'] and indicators['std20'] and indicators['std10'] / indicators['std20'] <= 0.8:
            score += 2
        
        return score
    
    def _check_max_ara_fundamentals(self, fundamentals: Dict, reasons: List[str]) -> bool:
        """Check fundamental criteria for MAX ARA"""
        mc = fundamentals.get('market_cap')
        if mc and mc >= 500_000_000_000:
            reasons.append('MarketCap>=500B')
            return True
        return False
    
    def _determine_max_ara_class(self, score: int) -> str:
        """Determine MAX ARA classification based on score"""
        if score >= 85:
            return 'Very High'
        elif score >= 70:
            return 'High'
        elif score >= 55:
            return 'Medium'
        else:
            return 'Low'
    
    def _classify_bpjs(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Optional[Dict] = None) -> BPJSResult:
        """Classify for BPJS (Buy Pagi Jual Sore) - Day Trading potential"""
        try:
            score = 0
            reasons = []
            
            # --- 1. Liquidity Check (Transaction Value) ---
            tx_value = latest['Close'] * latest['Volume']
            if tx_value >= 5_000_000_000: # Ideally > 5B
                score += 15
            elif tx_value >= 1_000_000_000:
                score += 5
            else:
                score -= 30 # Penalize low liquidity heavily
                reasons.append('Too Illiquid for BPJS')

            # --- 2. Trend & Structure Alignment (High Accuracy) ---
            # Ensure we are trading with the trend
            ma20 = latest.get('MA20', 0)
            ma50 = latest.get('MA50', 0)
            close = latest['Close']
            
            trend_aligned = False
            if ma20 > ma50:
                score += 10
                trend_aligned = True
            
            if close > ma20:
                score += 10
            else:
                score -= 10 # Avoid trading below MA20 for momentum

            # --- 3. Intraday momentum checks ---
            score += self._check_bpjs_intraday_momentum(latest, reasons)
            
            # --- 4. Candlestick Quality (Strong Close) ---
            # Ideally close near high
            high = latest['High']
            low = latest['Low']
            if high > low:
                range_pos = (close - low) / (high - low)
                if range_pos >= 0.7: # Top 30% close
                    score += 15
                    reasons.append(f'Strong Close ({range_pos:.0%})')
                elif range_pos < 0.3: # Weak close
                    score -= 10
                    reasons.append('Weak Close')

            # --- 5. Gap Analysis ---
            if len(df) >= 2:
                prev_close = df['Close'].iloc[-2]
                open_price = latest['Open']
                if open_price > prev_close * 1.01:
                    score += 10
                    reasons.append('Gap Up > 1%')

            # --- 6. Volume surge patterns ---
            score += self._check_bpjs_volume_patterns(df, latest, reasons)
            
            # --- 7. Liquidity fundamentals ---
            score += self._check_bpjs_liquidity(fundamentals, reasons)
            
            # Determine candidate status
            candidate, strong_candidate, suitability = self._determine_bpjs_status(score, reasons)
            
            # Generate Dynamic Advice
            advice = self._get_bpjs_advice_dynamic(score, latest)
            
            return BPJSResult(
                score=score,
                reasons=reasons,
                advice=advice,
                candidate=candidate,
                strong_candidate=strong_candidate,
                suitability=suitability,
                class_name=suitability
            )
            
        except Exception as e:
            logger.error(f"Error in BPJS classification: {e}")
            return BPJSResult(
                score=0, reasons=[], advice={},
                candidate=False, strong_candidate=False, suitability='NONE',
                class_name='NONE'
            )
    
    def _check_bpjs_intraday_momentum(self, latest: pd.Series, reasons: List[str]) -> int:
        """Check intraday momentum for BPJS"""
        score = 0
        
        openp = latest.get('Open')
        close = latest.get('Close')
        pct_today = latest.get('PCT_CHANGE_TODAY', 0)
        
        # Bullish intraday
        if openp is not None and openp < close:
            score += 10
            reasons.append('Open<Close (bullish intraday)')
        
        # Strong daily performance
        if pct_today >= 1.0:
            score += 15
            reasons.append(f'PctToday>={pct_today:.1f}%')
        elif pct_today >= 0.5:
            score += 8
            reasons.append(f'PctToday>={pct_today:.1f}%')
        
        return score
    
    def _check_bpjs_volume_patterns(self, df: pd.DataFrame, latest: pd.Series, reasons: List[str]) -> int:
        """Check volume patterns for BPJS"""
        score = 0
        vol20 = df['Volume'].tail(20).mean()
        vol5 = df['Volume'].tail(5).mean()
        
        # Short-term volume surge
        if vol5 and latest['Volume'] / vol5 >= 4:
            score += 20
            reasons.append('Vol/Avg5>=4 (strong surge)')
        elif vol5 and latest['Volume'] / vol5 >= 2.5:
            score += 12
            reasons.append('Vol/Avg5>=2.5')
        
        # Medium-term volume increase
        if vol20 and latest['Volume'] / vol20 >= 3:
            score += 15
            reasons.append('Vol/Avg20>=3')
        elif vol20 and latest['Volume'] / vol20 >= 1.8:
            score += 8
            reasons.append('Vol/Avg20>=1.8')
        
        # Intraday volume trend
        if len(df) >= 3:
            vol_trend = df['Volume'].tail(3).iloc[-1] > df['Volume'].tail(3).iloc[0]
            if vol_trend:
                score += 8
                reasons.append('Volume increasing intraday')
        
        return score
    
    def _check_bpjs_price_position(self, df: pd.DataFrame, latest: pd.Series, reasons: List[str]) -> int:
        """Check price position for BPJS"""
        score = 0
        
        close = latest.get('Close', 0)
        high = latest.get('High', 0)
        low = latest.get('Low', 0)
        vwap = latest.get('VWAP')
        atr_pct = latest.get('ATR_PCT', 0) * 100 if latest.get('ATR_PCT') is not None else None
        
        # Intraday range
        if close and high and low:
            hl_range = (high - low) / close
            if hl_range <= 0.03:
                score += 15
                reasons.append('Range<=3% (tight intraday)')
            elif hl_range <= 0.05:
                score += 8
                reasons.append('Range<=5%')
        
        # VWAP position
        if vwap and close > vwap * 1.02:
            score += 20
            reasons.append('Close>VWAP+2% (strong position)')
        elif vwap and close > vwap:
            score += 10
            reasons.append('Close>VWAP')
        
        # Volatility
        if atr_pct is not None and atr_pct <= 4:
            score += 20
            reasons.append('ATR%<=4 (low volatility)')
        elif atr_pct is not None and atr_pct <= 6:
            score += 10
            reasons.append('ATR%<=6')
        
        # Near recent highs
        if len(df) >= 5:
            high5 = df['High'].tail(5).max()
            if close >= high5 * 0.98:
                score += 12
                reasons.append('Close near 5-day high')
        
        return score
    
    def _check_bpjs_technical_structure(self, df: pd.DataFrame, latest: pd.Series, reasons: List[str]) -> int:
        """Check technical structure for BPJS"""
        score = 0
        
        # Short-term MA structure
        ma5 = df['Close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else None
        ma10 = df['Close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else None
        ma20 = latest.get('MA20', 0)
        
        if ma5 and ma10 and ma5 > ma10:
            score += 15
            reasons.append('MA5>MA10 (short-term momentum)')
        
        if ma10 and ma20 and ma10 > ma20:
            score += 10
            reasons.append('MA10>MA20 (medium-term momentum)')
        
        return score
    
    def _check_bpjs_liquidity(self, fundamentals: Optional[Dict], reasons: List[str]) -> int:
        """Check liquidity factors for BPJS"""
        score = 0
        
        if not fundamentals:
            return score
        
        spread = fundamentals.get('bid_ask_spread_pct')
        if spread is not None and spread <= 1.0:
            score += 8
            reasons.append('BidAskSpread<=1.0% (good liquidity)')
        elif spread is not None and spread <= 1.5:
            score += 5
            reasons.append('BidAskSpread<=1.5%')
        
        # Market cap filter
        mc = fundamentals.get('market_cap')
        if mc and mc >= 100_000_000_000:
            score += 5
            reasons.append('MarketCap>=100B (good liquidity)')
        
        return score
    
    def _determine_bpjs_status(self, score: int, reasons: List[str]) -> Tuple[bool, bool, str]:
        """Determine BPJS candidate status"""
        candidate = False
        strong_candidate = False
        
        if score >= 80:
            candidate = True
            strong_candidate = True
            reasons.append('STRONG_BPJS_CANDIDATE')
            suitability = 'HIGH'
        elif score >= 65:
            candidate = True
            reasons.append('BPJS_CANDIDATE')
            suitability = 'MEDIUM'
        elif score >= 50:
            candidate = True
            reasons.append('POTENTIAL_BPJS_CANDIDATE')
            suitability = 'LOW'
        else:
            suitability = 'NONE'
        
        return candidate, strong_candidate, suitability
    
    def _get_bpjs_advice_dynamic(self, score: int, latest: pd.Series) -> Dict[str, str]:
        """Get Dynamic BPJS trading advice based on score and volatility"""
        
        close = latest['Close']
        atr = latest.get('ATR', close * 0.02)
        vwap = latest.get('VWAP', close)
        
        # Determine aggression based on score
        if score >= 80:
            entry_strategy = f"Aggressive: Market Open or > {int(close)}"
            sl_level = int(close - (atr * 0.8))
            tp_min = int(close + (atr * 1.5))
            tp_max = int(close + (atr * 3.0))
        else:
            entry_strategy = f"Conservative: Wait pullback to VWAP area (~{int(vwap)})"
            sl_level = int(close - (atr * 1.0))
            tp_min = int(close + (atr * 1.2))
            tp_max = int(close + (atr * 2.0))
            
        advice = {
            'action': 'BUY' if score >= 65 else 'WATCH',
            'enter': entry_strategy,
            'sl_price': f"{sl_level} (approx -{(close-sl_level)/close:.1%})",
            'tp_area': f"{tp_min} - {tp_max} (approx +{(tp_min-close)/close:.1%} to +{(tp_max-close)/close:.1%})",
            'timeframe': 'Day Trade (Close before 15:50)',
            'risk_note': 'High Risk - Strict Stop Loss required'
        }
        
        return advice

    def _get_bpjs_advice(self, score: int) -> Dict[str, str]:
        """Legacy advice - kept for compatibility"""
        return self._get_bpjs_advice_dynamic(score, pd.Series({'Close': 100, 'ATR': 2}))
    
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Optional[Dict] = None) -> GorenganResult:
        """Classify for gorengan (speculative) potential"""
        try:
            score = 0
            reasons = []
            
            # Volume explosion
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            
            # Price explosion
            pct1d = self._calculate_daily_change(df, latest)
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            
            # Breakout
            high20 = latest.get('HIGH_20D')
            close = latest.get('Close', 0)
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            
            # Volatility expansion
            score += self._check_volatility_expansion(df, reasons)
            
            # Speculative fundamentals
            score += self._check_speculative_fundamentals(fundamentals, reasons)
            
            # Determine risk level
            risk = self._determine_gorengan_risk(score)
            
            return GorenganResult(score=score, reasons=reasons, risk=risk, class_name=risk)
            
        except Exception as e:
            logger.error(f"Error in gorengan classification: {e}")
            return GorenganResult(score=0, reasons=[], class_name='Low', risk='Low')
    
    def _calculate_daily_change(self, df: pd.DataFrame, latest: pd.Series) -> float:
        """Calculate daily percentage change"""
        if len(df) < 2:
            return 0
        try:
            prev_close = df['Close'].iloc[-2]
            current_close = latest['Close']
            return ((current_close - prev_close) / prev_close) * 100
        except (IndexError, ZeroDivisionError):
            return 0
    
    def _check_volatility_expansion(self, df: pd.DataFrame, reasons: List[str]) -> int:
        """Check for volatility expansion"""
        score = 0
        
        try:
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
        except Exception:
            pass
        
        return score
    
    def _check_speculative_fundamentals(self, fundamentals: Optional[Dict], reasons: List[str]) -> int:
        """Check for speculative fundamental characteristics"""
        score = 0
        
        if not fundamentals:
            return score
        
        # Small cap (more speculative)
        mc = fundamentals.get('market_cap')
        if mc and mc < 200_000_000_000:
            score += 10
            reasons.append('SmallCap<200B')
        
        # Wide spread (less liquid, more speculative)
        spread = fundamentals.get('bid_ask_spread_pct')
        if spread and spread >= 3:
            score += 10
            reasons.append('WideSpread>=3%')
        
        return score
    
    def _determine_gorengan_risk(self, score: int) -> str:
        """Determine risk level for gorengan classification"""
        if score >= 70:
            return 'High'
        elif score >= 50:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_fundamentals_cached(self, ticker: str) -> Optional[Dict]:
        """Get fundamentals with caching to avoid repeated API calls"""
        try:
            # Check cache first
            if ticker in self._fundamentals_cache:
                return self._fundamentals_cache[ticker]
            
            # Fetch and cache
            fundamentals = self.fin_fetcher.get(ticker)
            if fundamentals:
                self._fundamentals_cache[ticker] = fundamentals
            
            return fundamentals
            
        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {ticker}: {e}")
            return None
