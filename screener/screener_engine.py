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
from config import SCREENER_CONFIG, SIGNAL_THRESHOLDS, PATTERN_CONFIG, ENTRY_CONFIG, GORENGAN_CONFIG, GORENGAN_SCORING, GORENGAN_RISK_LEVELS
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


@dataclass
class UMAResult:
    """UMA (Unusual Market Activity) classification result"""
    score: int
    risk_level: str  # 'NONE', 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    indicators: Dict[str, Any]
    reasons: List[str]
    class_name: str = "UMA"


@dataclass
class InsiderActivityResult:
    """Insider Activity detection result"""
    score: int
    detected: bool
    activity_type: str  # 'NONE', 'ACCUMULATION', 'DISTRIBUTION', 'SMART_MONEY'
    indicators: Dict[str, Any]
    reasons: List[str]
    class_name: str = "InsiderActivity"


@dataclass 
class EnhancedGorenganResult:
    """Enhanced Gorengan classification with full scoring system"""
    total_score: int
    risk_level: str          # 'LOW', 'ACTIVE_GORENGAN', 'HIGH_UMA_RISK', 'INSIDER_STRONG'
    is_gorengan: bool
    is_uma_risk: bool
    has_insider_activity: bool
    score_breakdown: Dict[str, int]
    indicators: Dict[str, Any]
    reasons: List[str]
    warnings: List[str]
    uma_result: Optional[UMAResult] = None
    insider_result: Optional[InsiderActivityResult] = None
    class_name: str = "EnhancedGorengan"


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
    - Multi-mode screening (BPJS, ARA, Multi-bagger, Scalping, Gorengan)
    """
    
    def __init__(self):
        """Initialize screener engine with all components"""
        self.config = SCREENER_CONFIG
        self.pattern_detector = PatternDetector()
        self.regime_filter = RegimeFilter() if PATTERN_CONFIG.get("MARKET_REGIME", True) else None
        self.fin_fetcher = FinancialFetcher()
        
        # Initialize mode screener
        from .mode_screener import ModeScreener
        self.mode_screener = ModeScreener()
        
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
            patterns = self.pattern_detector.detect_all_patterns(df)  # Re-enabled for Super Setup!
            
            regime_info = self._get_regime_info(df, ticker)
            
            # Super Setup Detection (The "Overpower" Feature)
            is_super_setup, super_reasons = self._detect_super_setup(df, latest, patterns, regime_info)
            if is_super_setup:
                score = max(score, 0.95) # Mandatorily high score
                reasons.extend(super_reasons)
                signal = "SUPER_ALPHA" # Override signal
            else:
                signal = self._determine_signal(score)
            
            confidence = self._calculate_confidence(score, parameter_count, patterns, df, latest, regime_info)
            
            # Boost confidence for Super Setups
            if signal == "SUPER_ALPHA":
                confidence = max(confidence, 0.85)

            entry_levels = self._calculate_entry_levels(df, latest)
            data_quality = self._validate_data_quality(df, latest)
            
            # Classifications
            fundamentals = self._get_fundamentals_cached(ticker)
            max_ara = self._classify_max_ara(df, latest, fundamentals)
            bpjs = self._classify_bpjs(df, latest, fundamentals)
            gorengan = self._classify_gorengan(df, latest, fundamentals)
            
            # Enhanced Gorengan/UMA/Insider Detection
            enhanced_gorengan = self._classify_enhanced_gorengan(df, latest, fundamentals)
            
            # Multi-Mode Screening (BPJS, ARA, Multi-bagger, Scalping)
            try:
                all_modes = self.mode_screener.screen_all_modes(
                    df, latest, ticker, fundamentals, 
                    enhanced_gorengan.__dict__ if hasattr(enhanced_gorengan, '__dict__') else enhanced_gorengan
                )
                all_modes_dict = {
                    'best_mode': all_modes.best_mode,
                    'overall_score': all_modes.overall_score,
                    'overall_risk': all_modes.overall_risk,
                    'bandar_timing': all_modes.bandar_timing,
                    'recommendation': all_modes.recommendation,
                    'bpjs': all_modes.bpjs.__dict__ if all_modes.bpjs else {},
                    'ara': all_modes.ara.__dict__ if all_modes.ara else {},
                    'multibagger': all_modes.multibagger.__dict__ if all_modes.multibagger else {},
                    'scalping': all_modes.scalping.__dict__ if all_modes.scalping else {},
                }
            except Exception as mode_err:
                logger.warning(f"Mode screening failed for {ticker}: {mode_err}")
                all_modes_dict = {'best_mode': 'NONE', 'error': str(mode_err)}
            
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
                    'enhanced_gorengan': enhanced_gorengan.__dict__,
                    'all_modes': all_modes_dict,
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
            
            # --- SUSPENSION & UMA RISK FILTER ---
            
            # 1. Suspended Check (Zero Volume today)
            if get_numeric(latest['Volume']) <= 0:
                logger.debug(f"{latest.name} rejected: SUSPENDED (Zero Volume)")
                return False

            # 2. UMA Risk: Extreme Pump Check
            # If price rose > 45% in last 3 days or > 75% in last 5 days -> High UMA Risk
            if len(df) >= 5:
                # 3-day change
                p3 = df['Close'].iloc[-3] if len(df) >= 3 else df['Close'].iloc[0]
                chg_3d = (close_price - p3) / p3
                
                # 5-day change
                p5 = df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[0]
                chg_5d = (close_price - p5) / p5
                
                if chg_3d > 0.45 or chg_5d > 0.75:
                   logger.debug(f"{latest.name} rejected: UMA RISK (Extreme Pump > 45-75%)") 
                   return False

            # ------------------------------------
            
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

    def _detect_super_setup(self, df: pd.DataFrame, latest: pd.Series, patterns: Dict, regime_info: Dict) -> Tuple[bool, List[str]]:
        """
        Detect 'Super Alpha' Setup - The Overpower Feature
        
        Criteria:
        1. Bullish Market Regime (or Neutral with strong specific strength)
        2. Strong Smart Money Flow (Volume/VWAP)
        3. High Quality Pattern (VCP or Parabolic or Breakout)
        4. No major resistance overhead
        """
        reasons = []
        setup_score = 0
        
        # 1. Regime Check (Filter out noise in Bear markets)
        regime = regime_info.get('regime', 'NEUTRAL')
        if regime == 'BEAR':
            return False, [] # Strict: No super setups in bear market
        
        # 2. Smart Money / Volume Check
        money_flow = patterns.get('money_flow')
        vol_dry_up = patterns.get('volume_dry_up')
        
        # Strong Flow OR (Dry Up + Breakout)
        has_smart_money = False
        if money_flow and money_flow.get('detected'):
            has_smart_money = True
            setup_score += 1
            reasons.append("Smart Money Flow Detected")
        
        # 3. Premium Pattern Check
        vcp = patterns.get('vcp')
        parabolic = patterns.get('parabolic')
        breakout_52w = patterns.get('breakout_52w')
        
        has_premium_pattern = False
        if vcp and vcp.get('detected'):
            has_premium_pattern = True
            reasons.append(f"VCP Pattern ({vcp.get('contractions')} contractions)")
        elif parabolic:
            has_premium_pattern = True
            reasons.append("Parabolic State")
        elif breakout_52w and breakout_52w.get('detected'):
            has_premium_pattern = True
            reasons.append("52-Week Breakout")
            
        if has_premium_pattern:
            setup_score += 1

        # 4. Momentum Confirmation
        rsi = latest.get('RSI', 50)
        ma20_slope = latest.get('MA20_SLOPE', 0)
        
        has_momentum = (50 <= rsi <= 80) and (ma20_slope > 0)
        if has_momentum:
            setup_score += 1
            
        # FINAL VERDICT
        # Need at least Smart Money + Pattern + Momentum
        if setup_score >= 3:
            return True, ["🚀 SUPER ALPHA SETUP"] + reasons
            
        return False, []
        
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
            
            # Base confidence from score (70%)
            confidence += score * 0.7
            
            # Parameter count (20%)
            max_params = 12
            param_ratio = min(parameter_count / max_params, 1.0)
            confidence += param_ratio * 0.2
            
            # Pattern detection (10%)
            detected_patterns = self._count_detected_patterns(patterns)
            pattern_score = min(detected_patterns / 5.0, 1.0)
            confidence += pattern_score * 0.1
            
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

    # =========================================================================
    # ENHANCED GORENGAN / UMA / INSIDER ACTIVITY DETECTION
    # =========================================================================
    
    def _classify_enhanced_gorengan(self, df: pd.DataFrame, latest: pd.Series, 
                                      fundamentals: Optional[Dict] = None) -> EnhancedGorenganResult:
        """
        Enhanced Gorengan classification with full scoring system
        
        Based on user specification:
        - Filter Awal: Ciri Saham Gorengan
        - UMA Detection
        - Insider Activity Detection
        - Distribution/Pump & Dump Detection
        - Comprehensive Scoring System
        """
        try:
            total_score = 0
            score_breakdown = {}
            indicators = {}
            reasons = []
            warnings = []
            
            # === 1. FILTER AWAL - CIRI SAHAM GORENGAN ===
            
            # 1A. Likuiditas & Harga
            close_price = float(latest['Close'])
            indicators['close_price'] = close_price
            
            # Low price stock (< 500)
            if close_price < GORENGAN_CONFIG["MAX_PRICE_GORENGAN"]:
                score_breakdown['low_price'] = GORENGAN_SCORING["LOW_PRICE"]
                total_score += GORENGAN_SCORING["LOW_PRICE"]
                reasons.append(f"Harga < Rp{GORENGAN_CONFIG['MAX_PRICE_GORENGAN']} (mudah dimainkan)")
            
            # Market Cap check
            market_cap = fundamentals.get('market_cap', 0) if fundamentals else 0
            indicators['market_cap'] = market_cap
            if market_cap > 0 and market_cap < GORENGAN_CONFIG["MAX_MARKET_CAP"]:
                score_breakdown['low_market_cap'] = GORENGAN_SCORING["LOW_MARKET_CAP"]
                total_score += GORENGAN_SCORING["LOW_MARKET_CAP"]
                reasons.append(f"Market Cap < 1T (small cap, mudah dikontrol)")
            
            # 1B. Volume Spike Tidak Normal
            volume_ratio = self._calculate_volume_ratio(df, latest)
            indicators['volume_ratio'] = volume_ratio
            
            if volume_ratio >= GORENGAN_CONFIG["VR_SUSPICIOUS"]:  # >= 10
                score_breakdown['volume_ratio'] = GORENGAN_SCORING["VR_10"]
                total_score += GORENGAN_SCORING["VR_10"]
                reasons.append(f"VR {volume_ratio:.1f}x (>={GORENGAN_CONFIG['VR_SUSPICIOUS']}x = SANGAT MENCURIGAKAN)")
                warnings.append("⚠️ Volume Ratio sangat tidak normal!")
            elif volume_ratio >= GORENGAN_CONFIG["VR_ABNORMAL"]:  # >= 5
                score_breakdown['volume_ratio'] = GORENGAN_SCORING["VR_5"]
                total_score += GORENGAN_SCORING["VR_5"]
                reasons.append(f"VR {volume_ratio:.1f}x (>={GORENGAN_CONFIG['VR_ABNORMAL']}x = tidak normal)")
            
            # 1C. Lonjakan Harga Tidak Sehat
            price_spike = self._calculate_price_spike(df, latest)
            indicators['price_spike_pct'] = price_spike
            
            if price_spike >= GORENGAN_CONFIG["PRICE_SPIKE_UMA"]:  # >= 20%
                score_breakdown['price_spike'] = GORENGAN_SCORING["PRICE_SPIKE_20"]
                total_score += GORENGAN_SCORING["PRICE_SPIKE_20"]
                reasons.append(f"Price Spike {price_spike:.1f}% (>={GORENGAN_CONFIG['PRICE_SPIKE_UMA']}% = KANDIDAT UMA)")
                warnings.append("⚠️ Lonjakan harga kandidat UMA!")
            elif price_spike >= GORENGAN_CONFIG["PRICE_SPIKE_GORENGAN"]:  # >= 10%
                score_breakdown['price_spike'] = GORENGAN_SCORING["PRICE_SPIKE_10"]
                total_score += GORENGAN_SCORING["PRICE_SPIKE_10"]
                reasons.append(f"Price Spike {price_spike:.1f}% (>={GORENGAN_CONFIG['PRICE_SPIKE_GORENGAN']}% = indikasi gorengan)")
            
            # === 2. FILTER UMA (UNUSUAL MARKET ACTIVITY) ===
            
            # 2A. Candle Tidak Wajar (Body Ratio)
            body_ratio = self._calculate_body_ratio(latest)
            indicators['body_ratio'] = body_ratio
            
            if body_ratio >= GORENGAN_CONFIG["BODY_RATIO_THRESHOLD"]:  # >= 0.7
                score_breakdown['body_ratio'] = GORENGAN_SCORING["BODY_RATIO_HIGH"]
                total_score += GORENGAN_SCORING["BODY_RATIO_HIGH"]
                reasons.append(f"Body Ratio {body_ratio:.2f} (>={GORENGAN_CONFIG['BODY_RATIO_THRESHOLD']} = candle tidak wajar)")
            
            # 2B. Multi-Day Pump
            multi_day_pump = self._calculate_multi_day_pump(df, GORENGAN_CONFIG["MULTI_DAY_PUMP_DAYS"])
            indicators['multi_day_pump_pct'] = multi_day_pump
            
            if multi_day_pump >= GORENGAN_CONFIG["MULTI_DAY_PUMP_PCT"]:  # >= 25%
                score_breakdown['multi_day_pump'] = GORENGAN_SCORING["MULTI_DAY_PUMP"]
                total_score += GORENGAN_SCORING["MULTI_DAY_PUMP"]
                reasons.append(f"{GORENGAN_CONFIG['MULTI_DAY_PUMP_DAYS']}-Day Pump {multi_day_pump:.1f}% (>={GORENGAN_CONFIG['MULTI_DAY_PUMP_PCT']}%)")
                warnings.append(f"⚠️ Naik {multi_day_pump:.1f}% dalam {GORENGAN_CONFIG['MULTI_DAY_PUMP_DAYS']} hari!")
            
            # 2C. Volume Tidak Turun Saat Harga Turun (Distribution)
            is_distribution = self._detect_volume_price_divergence(df, latest)
            indicators['volume_price_divergence'] = is_distribution
            
            if is_distribution:
                score_breakdown['distribution'] = 2
                total_score += 2
                reasons.append("Harga turun + Volume tinggi = Distribusi")
                warnings.append("⚠️ Kemungkinan bandar sedang distribusi!")
            
            # === 3. DETEKSI INSIDER ACTIVITY / BANDAR MASUK ===
            
            # 3A. Broker Summary Anomali (placeholder - need broker data)
            # Note: This requires broker summary data which is not available from Yahoo Finance
            indicators['broker_concentration'] = None  # Requires external data
            
            # 3B. Net Buy Diam-Diam (Accumulation) - Using price+volume proxy
            is_accumulation = self._detect_accumulation_pattern(df, latest)
            indicators['accumulation_detected'] = is_accumulation
            
            if is_accumulation:
                score_breakdown['sideways_netbuy'] = GORENGAN_SCORING["SIDEWAYS_NETBUY"]
                total_score += GORENGAN_SCORING["SIDEWAYS_NETBUY"]
                reasons.append("Pola Accumulation: Sideways + Volume meningkat")
            
            # 3C. Volume Tinggi Tapi Range Sempit (Smart Money)
            is_smart_money = self._detect_smart_money(df, latest)
            indicators['smart_money_detected'] = is_smart_money
            
            if is_smart_money:
                score_breakdown['high_vol_low_atr'] = GORENGAN_SCORING["HIGH_VOL_LOW_ATR"]
                total_score += GORENGAN_SCORING["HIGH_VOL_LOW_ATR"]
                reasons.append("High Volume + Low Range = Smart Money")
            
            # === 4. DETEKSI DISTRIBUSI / PUMP & DUMP ===
            
            # 4A. RSI Divergence
            rsi_divergence = self._detect_rsi_divergence(df)
            indicators['rsi_divergence'] = rsi_divergence.get('detected', False)
            
            if rsi_divergence.get('detected'):
                score_breakdown['rsi_divergence'] = GORENGAN_SCORING["RSI_DIVERGENCE"]
                total_score += GORENGAN_SCORING["RSI_DIVERGENCE"]
                reasons.append(f"RSI Bearish Divergence: {rsi_divergence.get('description', '')}")
                warnings.append("⚠️ RSI Divergence = Potensi distribusi!")
            
            # 4B. Fake Breakout
            fake_breakout = self._detect_fake_breakout(df, latest)
            indicators['fake_breakout'] = fake_breakout
            
            if fake_breakout:
                score_breakdown['fake_breakout'] = GORENGAN_SCORING["FAKE_BREAKOUT"]
                total_score += GORENGAN_SCORING["FAKE_BREAKOUT"]
                reasons.append("Fake Breakout: Break resistance lalu tutup di bawah")
                warnings.append("⚠️ Fake Breakout detected!")
            
            # === 5. DETERMINE RISK LEVEL ===
            risk_level = self._determine_enhanced_gorengan_risk(total_score)
            is_gorengan = total_score >= GORENGAN_RISK_LEVELS["ACTIVE_GORENGAN"]
            is_uma_risk = total_score >= GORENGAN_RISK_LEVELS["HIGH_UMA_RISK"]
            has_insider_activity = total_score >= GORENGAN_RISK_LEVELS["INSIDER_STRONG"]
            
            # Add final warnings based on risk level
            if has_insider_activity:
                warnings.insert(0, "🚨 INSIDER/BANDAR KUAT - Risiko sangat tinggi!")
            elif is_uma_risk:
                warnings.insert(0, "⚠️ RISIKO UMA TINGGI - Hati-hati!")
            elif is_gorengan:
                warnings.insert(0, "⚠️ SAHAM GORENGAN AKTIF")
            
            # Create UMA and Insider results
            uma_result = self._create_uma_result(indicators, reasons, total_score)
            insider_result = self._create_insider_result(indicators, reasons, total_score)
            
            return EnhancedGorenganResult(
                total_score=total_score,
                risk_level=risk_level,
                is_gorengan=is_gorengan,
                is_uma_risk=is_uma_risk,
                has_insider_activity=has_insider_activity,
                score_breakdown=score_breakdown,
                indicators=indicators,
                reasons=reasons,
                warnings=warnings,
                uma_result=uma_result,
                insider_result=insider_result,
                class_name=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced gorengan classification: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return EnhancedGorenganResult(
                total_score=0,
                risk_level='LOW',
                is_gorengan=False,
                is_uma_risk=False,
                has_insider_activity=False,
                score_breakdown={},
                indicators={},
                reasons=[],
                warnings=[],
                class_name='LOW'
            )
    
    def _calculate_volume_ratio(self, df: pd.DataFrame, latest: pd.Series) -> float:
        """
        Calculate Volume Ratio = Volume Hari Ini / Rata-rata Volume 20 Hari
        
        VR >= 5 → tidak normal
        VR >= 10 → sangat mencurigakan
        """
        try:
            vol_20d_avg = df['Volume'].tail(20).mean()
            if vol_20d_avg > 0:
                return float(latest['Volume']) / vol_20d_avg
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_price_spike(self, df: pd.DataFrame, latest: pd.Series) -> float:
        """
        Calculate daily price spike percentage
        Price Spike = (Close - Close[-1]) / Close[-1] × 100%
        """
        try:
            if len(df) < 2:
                return 0.0
            prev_close = df['Close'].iloc[-2]
            if prev_close > 0:
                return ((float(latest['Close']) - prev_close) / prev_close) * 100
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_body_ratio(self, latest: pd.Series) -> float:
        """
        Calculate Body Ratio = |Close - Open| / (High - Low)
        
        Body Ratio >= 0.7 indicates candle tidak wajar (manipulation sign)
        Upper shadow kecil = tanda dipompa
        """
        try:
            high_low_range = float(latest['High']) - float(latest['Low'])
            if high_low_range > 0:
                body = abs(float(latest['Close']) - float(latest['Open']))
                return body / high_low_range
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_multi_day_pump(self, df: pd.DataFrame, days: int = 3) -> float:
        """
        Calculate N-day gain percentage
        3-Day Gain = (Close_today - Close_3days_ago) / Close_3days_ago * 100
        
        >= 25% dalam <= 3 hari = Multi-Day Pump
        """
        try:
            if len(df) < days + 1:
                return 0.0
            close_n_days_ago = df['Close'].iloc[-(days + 1)]
            close_today = df['Close'].iloc[-1]
            if close_n_days_ago > 0:
                return ((close_today - close_n_days_ago) / close_n_days_ago) * 100
            return 0.0
        except Exception:
            return 0.0
    
    def _detect_volume_price_divergence(self, df: pd.DataFrame, latest: pd.Series) -> bool:
        """
        Detect distribution pattern:
        Harga turun AND Volume tetap tinggi → Distribusi
        """
        try:
            if len(df) < 5:
                return False
            
            # Check if price is declining over last 3 days
            price_3d_ago = df['Close'].iloc[-4]
            price_today = float(latest['Close'])
            price_declining = price_today < price_3d_ago
            
            # Check if volume is still high (above average)
            vol_avg = df['Volume'].tail(20).mean()
            vol_recent_avg = df['Volume'].tail(3).mean()
            volume_high = vol_recent_avg > vol_avg * 1.2  # 20% above average
            
            return price_declining and volume_high
        except Exception:
            return False
    
    def _detect_accumulation_pattern(self, df: pd.DataFrame, latest: pd.Series) -> bool:
        """
        Detect accumulation (insider masuk) pattern:
        - Harga sideways
        - Volume meningkat (proxy for Net Buy)
        
        Jika ΔHarga kecil AND Volume besar → Accumulation
        """
        try:
            if len(df) < 20:
                return False
            
            # Check if price is sideways (small price change over 10 days)
            price_10d_ago = df['Close'].iloc[-11]
            price_today = float(latest['Close'])
            price_change_pct = abs((price_today - price_10d_ago) / price_10d_ago * 100)
            is_sideways = price_change_pct < 5  # Less than 5% change = sideways
            
            # Check if volume is increasing
            vol_early = df['Volume'].iloc[-20:-10].mean()
            vol_recent = df['Volume'].iloc[-10:].mean()
            volume_increasing = vol_recent > vol_early * 1.3  # 30% increase
            
            return is_sideways and volume_increasing
        except Exception:
            return False
    
    def _detect_smart_money(self, df: pd.DataFrame, latest: pd.Series) -> bool:
        """
        Detect Smart Money pattern:
        - Volume melonjak (> 2x average)
        - Harga tidak naik signifikan (range sempit)
        
        High Volume + Low ATR → Smart Money
        """
        try:
            # Volume check
            vol_avg = df['Volume'].tail(20).mean()
            vol_today = float(latest['Volume'])
            high_volume = vol_today > vol_avg * GORENGAN_CONFIG["HIGH_VOL_LOW_ATR_VOL_MULT"]
            
            # Price range check (today's range as percentage)
            high = float(latest['High'])
            low = float(latest['Low'])
            close = float(latest['Close'])
            
            if close > 0:
                range_pct = ((high - low) / close) * 100
                low_range = range_pct < GORENGAN_CONFIG["HIGH_VOL_LOW_ATR_RANGE_MAX"]
                
                return high_volume and low_range
            
            return False
        except Exception:
            return False
    
    def _detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 14) -> Dict:
        """
        Detect RSI bearish divergence:
        - Harga Higher High
        - RSI Lower High
        → Distribusi
        """
        try:
            if len(df) < lookback + 5:
                return {'detected': False, 'description': ''}
            
            # Get recent price and RSI data
            prices = df['Close'].tail(lookback).values
            rsi_col = 'RSI' if 'RSI' in df.columns else None
            
            if rsi_col is None:
                return {'detected': False, 'description': 'No RSI data'}
            
            rsi_values = df[rsi_col].tail(lookback).values
            
            # Find peaks in price and RSI
            # Simple peak detection: compare first half max with second half max
            mid = lookback // 2
            
            price_first_half_max = max(prices[:mid])
            price_second_half_max = max(prices[mid:])
            
            rsi_first_half_max = max(rsi_values[:mid])
            rsi_second_half_max = max(rsi_values[mid:])
            
            # Bearish divergence: Price Higher High + RSI Lower High
            price_higher_high = price_second_half_max > price_first_half_max
            rsi_lower_high = rsi_second_half_max < rsi_first_half_max - 3  # Allow some tolerance
            
            if price_higher_high and rsi_lower_high:
                return {
                    'detected': True,
                    'description': f'Price HH but RSI LH (RSI: {rsi_second_half_max:.1f} < {rsi_first_half_max:.1f})'
                }
            
            return {'detected': False, 'description': ''}
        except Exception:
            return {'detected': False, 'description': ''}
    
    def _detect_fake_breakout(self, df: pd.DataFrame, latest: pd.Series) -> bool:
        """
        Detect Fake Breakout:
        - Break resistance (close above previous high)
        - Tutup di bawah resistance (fail to hold)
        - Volume besar
        
        Indicates distribution / pump & dump
        """
        try:
            if len(df) < 20:
                return False
            
            # Find resistance (20-day high excluding today)
            resistance = df['High'].iloc[-21:-1].max()
            
            # Today's data
            high_today = float(latest['High'])
            close_today = float(latest['Close'])
            
            # Check if broke above resistance
            broke_resistance = high_today > resistance
            
            # Check if closed below resistance (with tolerance)
            tolerance = GORENGAN_CONFIG.get("FAKE_BREAKOUT_TOLERANCE", 0.02)
            closed_below = close_today < resistance * (1 + tolerance)
            
            # Check high volume
            vol_avg = df['Volume'].tail(20).mean()
            high_volume = float(latest['Volume']) > vol_avg * 1.5
            
            return broke_resistance and closed_below and high_volume
        except Exception:
            return False
    
    def _determine_enhanced_gorengan_risk(self, score: int) -> str:
        """
        Determine risk level based on total score
        
        Skor >= 8 → Saham Gorengan Aktif
        Skor >= 12 → Risiko UMA tinggi
        Skor >= 15 → Insider/Bandar kuat
        """
        if score >= GORENGAN_RISK_LEVELS["INSIDER_STRONG"]:
            return 'INSIDER_STRONG'
        elif score >= GORENGAN_RISK_LEVELS["HIGH_UMA_RISK"]:
            return 'HIGH_UMA_RISK'
        elif score >= GORENGAN_RISK_LEVELS["ACTIVE_GORENGAN"]:
            return 'ACTIVE_GORENGAN'
        else:
            return 'LOW'
    
    def _create_uma_result(self, indicators: Dict, reasons: List[str], score: int) -> UMAResult:
        """Create UMA result from analysis"""
        uma_indicators = {
            'volume_ratio': indicators.get('volume_ratio', 0),
            'price_spike': indicators.get('price_spike_pct', 0),
            'body_ratio': indicators.get('body_ratio', 0),
            'multi_day_pump': indicators.get('multi_day_pump_pct', 0),
        }
        
        uma_reasons = [r for r in reasons if 'UMA' in r.upper() or 'SPIKE' in r.upper() 
                       or 'PUMP' in r.upper() or 'BODY' in r.upper()]
        
        if score >= GORENGAN_RISK_LEVELS["HIGH_UMA_RISK"]:
            risk_level = 'EXTREME'
        elif score >= GORENGAN_RISK_LEVELS["ACTIVE_GORENGAN"]:
            risk_level = 'HIGH'
        elif score >= 5:
            risk_level = 'MEDIUM'
        elif score >= 2:
            risk_level = 'LOW'
        else:
            risk_level = 'NONE'
        
        return UMAResult(
            score=score,
            risk_level=risk_level,
            indicators=uma_indicators,
            reasons=uma_reasons,
            class_name="UMA"
        )
    
    def _create_insider_result(self, indicators: Dict, reasons: List[str], score: int) -> InsiderActivityResult:
        """Create Insider Activity result from analysis"""
        insider_indicators = {
            'accumulation_detected': indicators.get('accumulation_detected', False),
            'smart_money_detected': indicators.get('smart_money_detected', False),
            'broker_concentration': indicators.get('broker_concentration'),
        }
        
        insider_reasons = [r for r in reasons if 'ACCUM' in r.upper() or 'SMART' in r.upper() 
                          or 'INSIDER' in r.upper() or 'BANDAR' in r.upper()]
        
        # Determine activity type
        if indicators.get('accumulation_detected') and indicators.get('smart_money_detected'):
            activity_type = 'ACCUMULATION'
        elif indicators.get('smart_money_detected'):
            activity_type = 'SMART_MONEY'
        elif indicators.get('volume_price_divergence'):
            activity_type = 'DISTRIBUTION'
        else:
            activity_type = 'NONE'
        
        detected = score >= GORENGAN_RISK_LEVELS["INSIDER_STRONG"]
        
        return InsiderActivityResult(
            score=score,
            detected=detected,
            activity_type=activity_type,
            indicators=insider_indicators,
            reasons=insider_reasons,
            class_name="InsiderActivity"
        )
