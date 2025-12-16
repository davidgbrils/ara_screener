"""
Multi-Timeframe Technical Analyzer

Orchestrates analysis across multiple timeframes:
- Fetches data for each timeframe
- Calculates indicators
- Detects S/R levels
- Analyzes market structure
- Calculates confluence and confidence
- Generates entry/tp/sl levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .indicator_engine import IndicatorEngine, IndicatorSet
from .sr_detector import SRDetector, SRLevels
from .market_structure import MarketStructureAnalyzer, MarketStructureResult
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe"""
    timeframe: str  # "1m", "5m", "15m", "1h", "4h", "1D"
    trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    trend_emoji: str  # "🟢", "🔴", "🟡"
    indicators: IndicatorSet
    sr_levels: Optional[SRLevels]
    market_structure: Optional[MarketStructureResult]
    summary: str  # Brief description


@dataclass
class TradingPlan:
    """Entry, TP, SL recommendations"""
    entry_low: float
    entry_high: float
    tp1: float
    tp1_pct: float
    tp2: float
    tp2_pct: float
    sl: float
    sl_pct: float
    risk_reward: float


@dataclass
class MultiTFAnalysis:
    """Complete multi-timeframe analysis result"""
    ticker: str
    timestamp: datetime
    timeframes: Dict[str, TimeframeAnalysis]
    
    # Synthesis
    primary_trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    confluence: str  # "HIGH", "MEDIUM", "LOW"
    confidence: float  # 0-100
    bias: str  # "SCALPING", "INTRADAY", "SWING", "AVOID"
    
    # Key levels
    key_support: float
    key_resistance: float
    current_price: float
    
    # Trading plan
    trading_plan: Optional[TradingPlan]
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


class TechnicalAnalyzer:
    """
    Multi-Timeframe Technical Analysis Engine
    
    Analyzes:
    - 1D (Daily) - Primary trend
    - 4H - Swing bias
    - 1H - Intraday trend
    - 15M - Entry timing
    - 5M - Scalping (optional)
    """
    
    # Timeframe configurations
    TIMEFRAMES = {
        "1D": {"interval": "1d", "period": "1y", "min_candles": 200},
        "4h": {"interval": "1h", "period": "60d", "min_candles": 200},  # 4h simulated
        "1h": {"interval": "1h", "period": "30d", "min_candles": 200},
        "15m": {"interval": "15m", "period": "7d", "min_candles": 200},
        "5m": {"interval": "5m", "period": "5d", "min_candles": 200},
    }
    
    def __init__(self):
        """Initialize analyzer components"""
        self.indicator_engine = IndicatorEngine()
        self.sr_detector = SRDetector()
        self.structure_analyzer = MarketStructureAnalyzer()
    
    # =========================================================================
    # SINGLE TIMEFRAME ANALYSIS
    # =========================================================================
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Optional[TimeframeAnalysis]:
        """
        Analyze a single timeframe
        
        Args:
            df: DataFrame with OHLCV
            timeframe: Timeframe label
        
        Returns:
            TimeframeAnalysis result
        """
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for {timeframe} analysis")
            return None
        
        try:
            # Calculate indicators
            df_with_indicators = self.indicator_engine.add_indicators_to_df(df)
            indicators = self.indicator_engine.calculate_all(df)
            
            if not indicators:
                return None
            
            # Detect S/R levels
            sr_levels = self.sr_detector.detect_levels(df_with_indicators)
            
            # Analyze market structure
            structure = self.structure_analyzer.analyze(df)
            
            # Determine trend based on multiple factors
            trend = self._determine_trend(indicators, structure)
            trend_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}.get(trend, "⚪")
            
            # Generate summary
            summary = self._generate_tf_summary(indicators, structure, trend)
            
            return TimeframeAnalysis(
                timeframe=timeframe,
                trend=trend,
                trend_emoji=trend_emoji,
                indicators=indicators,
                sr_levels=sr_levels,
                market_structure=structure,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe}: {e}")
            return None
    
    def _determine_trend(self, indicators: IndicatorSet, 
                         structure: Optional[MarketStructureResult]) -> str:
        """Determine trend from indicators and structure"""
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA trend
        if indicators.ema_trend == "BULLISH":
            bullish_signals += 2
        elif indicators.ema_trend == "BEARISH":
            bearish_signals += 2
        
        # Price vs EMA
        if indicators.price_vs_ema == "ABOVE_ALL":
            bullish_signals += 1
        elif indicators.price_vs_ema == "BELOW_ALL":
            bearish_signals += 1
        
        # MACD
        if indicators.macd_histogram > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # MACD Cross
        if indicators.macd_cross == "BULLISH_CROSS":
            bullish_signals += 2
        elif indicators.macd_cross == "BEARISH_CROSS":
            bearish_signals += 2
        
        # RSI
        if 50 < indicators.rsi < 70:
            bullish_signals += 1
        elif 30 < indicators.rsi < 50:
            bearish_signals += 1
        
        # Market structure
        if structure:
            if structure.structure_type == "BULLISH":
                bullish_signals += 2
            elif structure.structure_type == "BEARISH":
                bearish_signals += 2
        
        # Determine
        if bullish_signals >= 5 and bullish_signals > bearish_signals + 2:
            return "BULLISH"
        elif bearish_signals >= 5 and bearish_signals > bullish_signals + 2:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    def _generate_tf_summary(self, indicators: IndicatorSet,
                             structure: Optional[MarketStructureResult],
                             trend: str) -> str:
        """Generate brief summary for timeframe"""
        parts = []
        
        # EMA summary
        if indicators.ema_trend == "BULLISH":
            parts.append("EMA Stack Up")
        elif indicators.ema_trend == "BEARISH":
            parts.append("EMA Stack Down")
        
        # RSI
        parts.append(f"RSI {indicators.rsi:.0f}")
        
        # MACD
        if indicators.macd_cross == "BULLISH_CROSS":
            parts.append("MACD Bull Cross")
        elif indicators.macd_cross == "BEARISH_CROSS":
            parts.append("MACD Bear Cross")
        
        # Volume
        if indicators.volume_signal == "HIGH":
            parts.append("High Volume")
        
        # Structure
        if structure and structure.bos_detected:
            parts.append(f"{structure.bos_direction}")
        
        return ", ".join(parts) if parts else "Consolidating"
    
    # =========================================================================
    # MULTI-TIMEFRAME SYNTHESIS
    # =========================================================================
    
    def calculate_confluence(self, analyses: Dict[str, TimeframeAnalysis]) -> Tuple[str, float]:
        """
        Calculate confluence across timeframes
        
        Returns:
            Tuple of (confluence_level, confidence_score)
        """
        if not analyses:
            return "LOW", 0.0
        
        trends = [a.trend for a in analyses.values()]
        
        bullish = trends.count("BULLISH")
        bearish = trends.count("BEARISH")
        total = len(trends)
        
        # Calculate confluence
        max_agreement = max(bullish, bearish)
        
        if max_agreement >= 4:
            confluence = "HIGH"
            confidence = 80 + (max_agreement - 4) * 5
        elif max_agreement >= 3:
            confluence = "MEDIUM"
            confidence = 60 + (max_agreement - 3) * 10
        else:
            confluence = "LOW"
            confidence = 30 + max_agreement * 10
        
        # Penalty for conflicting higher TFs
        if "1D" in analyses and "4h" in analyses:
            if analyses["1D"].trend != analyses["4h"].trend:
                confidence -= 15
                confluence = "LOW" if confluence == "MEDIUM" else confluence
        
        return confluence, min(confidence, 95)
    
    def determine_primary_trend(self, analyses: Dict[str, TimeframeAnalysis]) -> str:
        """Determine primary trend prioritizing higher timeframes"""
        # Priority: 1D > 4h > 1h
        priority_tfs = ["1D", "4h", "1h"]
        
        for tf in priority_tfs:
            if tf in analyses:
                return analyses[tf].trend
        
        # Fallback to majority
        trends = [a.trend for a in analyses.values()]
        bullish = trends.count("BULLISH")
        bearish = trends.count("BEARISH")
        
        if bullish > bearish:
            return "BULLISH"
        elif bearish > bullish:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    def determine_bias(self, confluence: str, primary_trend: str,
                       analyses: Dict[str, TimeframeAnalysis]) -> str:
        """Determine trading bias"""
        if confluence == "LOW":
            return "AVOID"
        
        if primary_trend == "SIDEWAYS":
            return "AVOID"
        
        # Check lower TF alignment
        lower_tfs = ["15m", "5m"]
        lower_bullish = sum(1 for tf in lower_tfs if tf in analyses and analyses[tf].trend == "BULLISH")
        lower_bearish = sum(1 for tf in lower_tfs if tf in analyses and analyses[tf].trend == "BEARISH")
        
        if confluence == "HIGH":
            if primary_trend == "BULLISH" and lower_bullish >= 1:
                return "SWING"
            elif primary_trend == "BEARISH" and lower_bearish >= 1:
                return "SWING"
        
        if "1h" in analyses and analyses["1h"].trend == primary_trend:
            return "INTRADAY"
        
        if lower_bullish >= 2 or lower_bearish >= 2:
            return "SCALPING"
        
        return "AVOID"
    
    # =========================================================================
    # TRADING PLAN
    # =========================================================================
    
    def generate_trading_plan(self, analyses: Dict[str, TimeframeAnalysis],
                              current_price: float) -> Optional[TradingPlan]:
        """Generate entry, TP, SL based on analysis"""
        if not analyses:
            return None
        
        # Get key levels from highest TF with SR data
        key_support = None
        key_resistance = None
        atr = None
        
        for tf in ["1D", "4h", "1h", "15m"]:
            if tf in analyses and analyses[tf].sr_levels:
                sr = analyses[tf].sr_levels
                if sr.closest_support:
                    key_support = sr.closest_support.price_high
                if sr.closest_resistance:
                    key_resistance = sr.closest_resistance.price_low
                break
        
        # Get ATR from daily or 1h
        for tf in ["1D", "1h"]:
            if tf in analyses:
                atr = analyses[tf].indicators.atr
                break
        
        if not atr:
            atr = current_price * 0.02  # Default 2% ATR
        
        # Calculate levels
        if key_support:
            entry_low = key_support
            entry_high = key_support + atr * 0.3
        else:
            entry_low = current_price - atr * 0.5
            entry_high = current_price
        
        sl = entry_low - atr * 0.5
        sl_pct = ((entry_low - sl) / entry_low) * 100
        
        tp1 = entry_high + atr * 1.0
        tp1_pct = ((tp1 - entry_high) / entry_high) * 100
        
        tp2 = entry_high + atr * 2.0
        tp2_pct = ((tp2 - entry_high) / entry_high) * 100
        
        # Risk/Reward
        risk = entry_high - sl
        reward = tp1 - entry_high
        rr = reward / risk if risk > 0 else 0
        
        return TradingPlan(
            entry_low=entry_low,
            entry_high=entry_high,
            tp1=tp1,
            tp1_pct=tp1_pct,
            tp2=tp2,
            tp2_pct=tp2_pct,
            sl=sl,
            sl_pct=sl_pct,
            risk_reward=rr
        )
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def analyze(self, ticker: str, data: Dict[str, pd.DataFrame]) -> Optional[MultiTFAnalysis]:
        """
        Perform complete multi-timeframe analysis
        
        Args:
            ticker: Stock ticker
            data: Dict of {timeframe: DataFrame}
        
        Returns:
            MultiTFAnalysis result
        """
        if not data:
            logger.error(f"No data provided for {ticker}")
            return None
        
        try:
            # Analyze each timeframe
            analyses = {}
            for tf, df in data.items():
                analysis = self.analyze_timeframe(df, tf)
                if analysis:
                    analyses[tf] = analysis
            
            if not analyses:
                logger.error(f"No successful analyses for {ticker}")
                return None
            
            # Get current price from most recent data
            current_price = 0
            for tf in ["1D", "4h", "1h", "15m", "5m"]:
                if tf in data and not data[tf].empty:
                    current_price = float(data[tf]['Close'].iloc[-1])
                    break
            
            # Calculate confluence
            confluence, confidence = self.calculate_confluence(analyses)
            
            # Determine primary trend
            primary_trend = self.determine_primary_trend(analyses)
            
            # Determine bias
            bias = self.determine_bias(confluence, primary_trend, analyses)
            
            # Get key levels
            key_support = current_price * 0.95  # Default
            key_resistance = current_price * 1.05  # Default
            
            for tf in ["1D", "4h", "1h"]:
                if tf in analyses and analyses[tf].sr_levels:
                    sr = analyses[tf].sr_levels
                    if sr.closest_support:
                        key_support = sr.closest_support.price_high
                    if sr.closest_resistance:
                        key_resistance = sr.closest_resistance.price_low
                    break
            
            # Generate trading plan
            trading_plan = None
            if bias != "AVOID":
                trading_plan = self.generate_trading_plan(analyses, current_price)
            
            # Check for warnings
            warnings = []
            
            # Higher TF conflict
            if "1D" in analyses and "1h" in analyses:
                if analyses["1D"].trend != analyses["1h"].trend:
                    warnings.append("TF conflict: Daily vs 1H trend mismatch")
            
            # RSI extremes
            for tf, a in analyses.items():
                if a.indicators.rsi_signal == "OVERBOUGHT":
                    warnings.append(f"RSI overbought on {tf}")
                elif a.indicators.rsi_signal == "OVERSOLD":
                    warnings.append(f"RSI oversold on {tf}")
            
            # Low volume
            if any(a.indicators.volume_signal == "LOW" for a in analyses.values()):
                warnings.append("Low volume detected")
            
            return MultiTFAnalysis(
                ticker=ticker,
                timestamp=datetime.now(),
                timeframes=analyses,
                primary_trend=primary_trend,
                confluence=confluence,
                confidence=confidence,
                bias=bias,
                key_support=key_support,
                key_resistance=key_resistance,
                current_price=current_price,
                trading_plan=trading_plan,
                warnings=warnings[:3]  # Limit warnings
            )
            
        except Exception as e:
            logger.error(f"Error in multi-TF analysis for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
