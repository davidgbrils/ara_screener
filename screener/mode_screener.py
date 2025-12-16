"""
Multi-Mode Stock Screener for Indonesian Market (IDX)

This module provides comprehensive screening across 5 trading modes:
- Mode A: BPJS (Saham Sehat - Safe & Stable)
- Mode B: ARA Potential (Fast Move)
- Mode C: Multi-Bagger Early Stage
- Mode D: Scalping Intraday
- Mode E: Gorengan/UMA Filter

Author: ARA Bot V3 - Professional Quant Trader
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from config import (
    MODE_BPJS_CONFIG, MODE_ARA_CONFIG, MODE_MULTIBAGGER_CONFIG,
    MODE_SCALPING_CONFIG, GORENGAN_CONFIG, ENHANCED_SCORING,
    SCORE_INTERPRETATION, CAPITAL_ADVISOR_CONFIG
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# DATA CLASSES FOR MODE RESULTS
# =============================================================================

@dataclass
class ModeResult:
    """Base result for all screening modes"""
    ticker: str
    mode: str
    passed: bool
    score: int
    trend: str  # 'BULLISH', 'SIDEWAYS', 'BEARISH'
    reasons: List[str]
    entry: Dict[str, float]  # {'low': x, 'high': y}
    tp: Dict[str, float]     # {'tp1': x, 'tp2': y}
    sl: float
    risk_level: str          # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    warnings: List[str]
    notes: str
    indicators: Dict[str, Any]


@dataclass
class BPJSResult(ModeResult):
    """BPJS Mode - Saham Sehat untuk Scalping/Swing Aman"""
    suitability: str = "BPJS - Aman untuk scalping"
    holding_period: str = "1-5 hari"


@dataclass
class ARAResult(ModeResult):
    """ARA Mode - Potensi ARA (Fast Move)"""
    ara_probability: float = 0.0
    is_already_ara: bool = False
    has_uma: bool = False


@dataclass
class MultibaggerResult(ModeResult):
    """Multi-Bagger Mode - Saham Undervalue Early"""
    phase: str = "BASE"  # 'BASE', 'EARLY_MARKUP', 'MARKUP', 'DISTRIBUTION'
    potential_rr: float = 0.0  # Risk-Reward ratio
    accumulation_days: int = 0


@dataclass
class ScalpingResult(ModeResult):
    """Scalping Mode - Intraday Quick Trade"""
    entry_type: str = "MARKET"  # 'MARKET', 'LIMIT', 'PULLBACK'
    scalping_advice: str = ""
    max_hold_time: str = "1 hari (close before 15:50)"


@dataclass
class AllModesResult:
    """Combined result from all screening modes"""
    ticker: str
    best_mode: str
    overall_score: int
    overall_risk: str
    bpjs: Optional[BPJSResult] = None
    ara: Optional[ARAResult] = None
    multibagger: Optional[MultibaggerResult] = None
    scalping: Optional[ScalpingResult] = None
    gorengan_warning: Optional[Dict] = None
    recommendation: str = ""
    bandar_timing: str = ""  # "SEBELUM", "BERSAMA", "TELAT"


# =============================================================================
# MODE SCREENER CLASS
# =============================================================================

class ModeScreener:
    """
    Multi-mode stock screener for different trading strategies.
    
    Implements comprehensive analysis based on:
    - Price action
    - Volume behavior
    - Technical indicators
    - Risk assessment
    """
    
    def __init__(self):
        """Initialize mode screener"""
        self.bpjs_config = MODE_BPJS_CONFIG
        self.ara_config = MODE_ARA_CONFIG
        self.multibagger_config = MODE_MULTIBAGGER_CONFIG
        self.scalping_config = MODE_SCALPING_CONFIG
        self.gorengan_config = GORENGAN_CONFIG
        self.scoring = ENHANCED_SCORING
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_numeric(self, value, default=0):
        """Safely extract numeric value"""
        try:
            if hasattr(value, 'iloc'):
                value = value.iloc[0]
            if pd.isna(value):
                return default
            return float(value)
        except (AttributeError, IndexError, TypeError):
            return default
    
    def _calculate_trend(self, df: pd.DataFrame, latest: pd.Series) -> str:
        """Determine current trend"""
        try:
            ema5 = self._get_numeric(latest.get('EMA5', latest.get('MA20', 0)))
            ema20 = self._get_numeric(latest.get('EMA20', latest.get('MA50', 0)))
            ema50 = self._get_numeric(latest.get('EMA50', latest.get('MA200', 0)))
            close = self._get_numeric(latest['Close'])
            
            # Strong bullish: EMA5 > EMA20 > EMA50 and price > EMA5
            if ema5 > ema20 > ema50 and close > ema5:
                return "BULLISH"
            # Weak bullish
            elif close > ema20:
                return "BULLISH"
            # Bearish
            elif close < ema20 and close < ema50:
                return "BEARISH"
            else:
                return "SIDEWAYS"
        except Exception:
            return "UNKNOWN"
    
    def _calculate_gap_pct(self, df: pd.DataFrame, latest: pd.Series) -> float:
        """Calculate gap percentage from previous close"""
        try:
            if len(df) < 2:
                return 0.0
            prev_close = df['Close'].iloc[-2]
            today_open = self._get_numeric(latest['Open'])
            return ((today_open - prev_close) / prev_close) * 100
        except Exception:
            return 0.0
    
    def _calculate_close_position(self, latest: pd.Series) -> float:
        """Calculate close position in day's range (0-100%)"""
        try:
            high = self._get_numeric(latest['High'])
            low = self._get_numeric(latest['Low'])
            close = self._get_numeric(latest['Close'])
            
            if high > low:
                return ((close - low) / (high - low)) * 100
            return 50.0
        except Exception:
            return 50.0
    
    def _calculate_body_pct(self, latest: pd.Series) -> float:
        """Calculate candle body percentage"""
        try:
            high = self._get_numeric(latest['High'])
            low = self._get_numeric(latest['Low'])
            open_p = self._get_numeric(latest['Open'])
            close = self._get_numeric(latest['Close'])
            
            if high > low:
                return (abs(close - open_p) / (high - low)) * 100
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_volume_ratio(self, df: pd.DataFrame, latest: pd.Series) -> float:
        """Calculate volume ratio vs 20-day average"""
        try:
            vol_avg = df['Volume'].tail(20).mean()
            if vol_avg > 0:
                return self._get_numeric(latest['Volume']) / vol_avg
            return 0.0
        except Exception:
            return 0.0
    
    def _is_break_resistance(self, df: pd.DataFrame, latest: pd.Series, days: int = 20) -> bool:
        """Check if breaking resistance"""
        try:
            if len(df) < days + 1:
                return False
            resistance = df['High'].iloc[-(days+1):-1].max()
            close = self._get_numeric(latest['Close'])
            return close > resistance
        except Exception:
            return False
    
    def _is_already_ara(self, df: pd.DataFrame, latest: pd.Series) -> bool:
        """Check if stock hit ARA (Auto Rejection Atas) today"""
        try:
            if len(df) < 2:
                return False
            prev_close = df['Close'].iloc[-2]
            close = self._get_numeric(latest['Close'])
            price = prev_close
            
            # ARA limits based on price
            if price < 200:
                limit = 0.35
            elif price < 5000:
                limit = 0.25
            else:
                limit = 0.20
            
            change_pct = (close - prev_close) / prev_close
            return change_pct >= limit * 0.95  # 95% of limit = considered ARA
        except Exception:
            return False
    
    def _calculate_entry_tp_sl(self, df: pd.DataFrame, latest: pd.Series, 
                                mode: str = "swing") -> Tuple[Dict, Dict, float]:
        """Calculate entry, TP, and SL levels based on mode"""
        try:
            close = self._get_numeric(latest['Close'])
            atr = self._get_numeric(latest.get('ATR', close * 0.02))
            vwap = self._get_numeric(latest.get('VWAP', close))
            
            if mode == "scalping":
                # Scalping: tight levels
                entry = {'low': vwap * 0.995, 'high': close}
                tp = {'tp1': close + atr * 0.5, 'tp2': close + atr * 1.0}
                sl = close - atr * 0.5
            elif mode == "ara":
                # ARA: aggressive
                entry = {'low': close * 0.98, 'high': close * 1.02}
                tp = {'tp1': close * 1.10, 'tp2': close * 1.25}
                sl = close * 0.93
            elif mode == "multibagger":
                # Multi-bagger: wide levels
                entry = {'low': close * 0.95, 'high': close}
                tp = {'tp1': close * 1.30, 'tp2': close * 1.50}
                sl = close * 0.85
            else:  # swing/bpjs
                entry = {'low': close - atr * 0.5, 'high': close + atr * 0.3}
                tp = {'tp1': close + atr * 1.5, 'tp2': close + atr * 2.5}
                sl = close - atr * 1.0
            
            return entry, tp, sl
        except Exception:
            close = self._get_numeric(latest['Close'])
            return (
                {'low': close * 0.98, 'high': close * 1.02},
                {'tp1': close * 1.05, 'tp2': close * 1.10},
                close * 0.95
            )
    
    def _calculate_enhanced_score(self, df: pd.DataFrame, latest: pd.Series,
                                   mode_indicators: Dict) -> Tuple[int, List[str]]:
        """Calculate enhanced score based on ENHANCED_SCORING config"""
        score = 0
        reasons = []
        
        try:
            # Volume scoring
            vol_ratio = mode_indicators.get('volume_ratio', 0)
            if vol_ratio >= 10:
                score += self.scoring["VOLUME_RATIO_10"]
                reasons.append(f"Volume Ratio {vol_ratio:.1f}x (+{self.scoring['VOLUME_RATIO_10']})")
            elif vol_ratio >= 5:
                score += self.scoring["VOLUME_RATIO_5"]
                reasons.append(f"Volume Ratio {vol_ratio:.1f}x (+{self.scoring['VOLUME_RATIO_5']})")
            
            # Break resistance
            if mode_indicators.get('break_resistance'):
                score += self.scoring["BREAK_RESISTANCE_VALID"]
                reasons.append(f"Break Resistance (+{self.scoring['BREAK_RESISTANCE_VALID']})")
            
            # RSI healthy
            rsi = self._get_numeric(latest.get('RSI', 50))
            if 50 <= rsi <= 70:
                score += self.scoring["RSI_HEALTHY"]
                reasons.append(f"RSI Sehat {rsi:.0f} (+{self.scoring['RSI_HEALTHY']})")
            
            # EMA structure
            if mode_indicators.get('ema_bullish'):
                score += self.scoring["EMA_STRUCTURE_BULLISH"]
                reasons.append(f"EMA Structure Bullish (+{self.scoring['EMA_STRUCTURE_BULLISH']})")
            
            # VWAP
            vwap = self._get_numeric(latest.get('VWAP', 0))
            close = self._get_numeric(latest['Close'])
            if vwap > 0 and close > vwap:
                score += self.scoring["ABOVE_VWAP"]
                reasons.append(f"Above VWAP (+{self.scoring['ABOVE_VWAP']})")
            
            # Body candle
            body_pct = mode_indicators.get('body_pct', 0)
            if body_pct >= 70:
                score += self.scoring["BODY_CANDLE_STRONG"]
                reasons.append(f"Body {body_pct:.0f}% (+{self.scoring['BODY_CANDLE_STRONG']})")
            
            # Close position
            close_pos = mode_indicators.get('close_position', 0)
            if close_pos >= 90:
                score += self.scoring["CLOSE_NEAR_HIGH"]
                reasons.append(f"Close {close_pos:.0f}% High (+{self.scoring['CLOSE_NEAR_HIGH']})")
            
            # Gap
            gap = mode_indicators.get('gap_pct', 0)
            if gap >= 1:
                score += self.scoring["GAP_UP"]
                reasons.append(f"Gap Up {gap:.1f}% (+{self.scoring['GAP_UP']})")
            
            # === NEGATIVE FACTORS ===
            
            # Divergence
            if mode_indicators.get('rsi_divergence'):
                score += self.scoring["DIVERGENCE"]
                reasons.append(f"RSI Divergence ({self.scoring['DIVERGENCE']})")
            
            # Distribution
            if mode_indicators.get('distribution'):
                score += self.scoring["DISTRIBUTION"]
                reasons.append(f"Distribusi ({self.scoring['DISTRIBUTION']})")
            
            # Fake breakout
            if mode_indicators.get('fake_breakout'):
                score += self.scoring["FAKE_BREAKOUT"]
                reasons.append(f"Fake Breakout ({self.scoring['FAKE_BREAKOUT']})")
            
            # Already ARA
            if mode_indicators.get('already_ara'):
                score += self.scoring["ALREADY_ARA"]
                reasons.append(f"Sudah ARA ({self.scoring['ALREADY_ARA']})")
            
            # Overbought
            if rsi > 80:
                score += self.scoring["OVERBOUGHT"]
                reasons.append(f"Overbought RSI {rsi:.0f} ({self.scoring['OVERBOUGHT']})")
            
        except Exception as e:
            logger.error(f"Error calculating enhanced score: {e}")
        
        return score, reasons
    
    def _determine_risk_level(self, score: int, mode_indicators: Dict) -> str:
        """Determine risk level based on score and indicators"""
        if score < SCORE_INTERPRETATION["AVOID_THRESHOLD"]:
            return "EXTREME"
        elif mode_indicators.get('gorengan_risk', False):
            return "HIGH"
        elif score >= SCORE_INTERPRETATION["STRONG_MOMENTUM"]:
            return "LOW"
        elif score >= SCORE_INTERPRETATION["HIGH_OPPORTUNITY"]:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _determine_bandar_timing(self, df: pd.DataFrame, latest: pd.Series, 
                                  mode_indicators: Dict) -> str:
        """
        Determine timing relative to bandar/insider:
        - SEBELUM: Early accumulation phase
        - BERSAMA: Momentum phase
        - TELAT: Distribution/late phase
        """
        try:
            # Accumulation signs
            if mode_indicators.get('accumulation_phase'):
                return "SEBELUM BANDAR - Fase akumulasi awal"
            
            # Check if in early momentum
            vol_ratio = mode_indicators.get('volume_ratio', 0)
            close_pos = mode_indicators.get('close_position', 0)
            rsi = self._get_numeric(latest.get('RSI', 50))
            
            if vol_ratio >= 5 and close_pos >= 80 and rsi < 70:
                return "BERSAMA BANDAR - Momentum awal"
            
            # Distribution signs
            if mode_indicators.get('distribution') or mode_indicators.get('rsi_divergence'):
                return "TELAT - Potensi distribusi"
            
            # Overbought / extended
            if rsi > 75 or mode_indicators.get('already_ara'):
                return "TELAT - Extended / sudah naik tinggi"
            
            # Default
            if vol_ratio >= 3:
                return "BERSAMA BANDAR - Dalam pergerakan"
            
            return "NETRAL - Tidak ada sinyal jelas"
            
        except Exception:
            return "UNKNOWN"
    
    # =========================================================================
    # MODE A: BPJS - SAHAM SEHAT
    # =========================================================================
    
    def screen_bpjs(self, df: pd.DataFrame, latest: pd.Series, ticker: str,
                    fundamentals: Optional[Dict] = None) -> BPJSResult:
        """
        Mode A - BPJS (Buy Pagi Jual Sore) / Saham Sehat
        
        Tujuan: Saham kuat, likuid, cocok swing & scalping aman
        """
        reasons = []
        warnings = []
        indicators = {}
        passed = True
        
        try:
            close = self._get_numeric(latest['Close'])
            
            # === FILTER CHECKS ===
            
            # 1. Market Cap check (proxy: high volume implies large cap)
            market_cap = fundamentals.get('market_cap', 0) if fundamentals else 0
            avg_vol = df['Volume'].tail(20).mean()
            tx_value = close * avg_vol
            
            indicators['market_cap'] = market_cap
            indicators['tx_value_avg'] = tx_value
            
            if market_cap > 0:
                if market_cap < self.bpjs_config["MARKET_CAP_MIN"]:
                    passed = False
                    reasons.append(f"Market Cap < 5T (actual: {market_cap/1e12:.1f}T)")
            else:
                # Use tx value as proxy
                if tx_value < 5_000_000_000:  # < 5B daily value
                    passed = False
                    reasons.append("Transaksi harian kurang likuid")
            
            # 2. Average Volume check
            if avg_vol < self.bpjs_config["AVG_VOL_20D_MIN"]:
                passed = False
                reasons.append(f"Avg Volume 20D < 10M (actual: {avg_vol/1e6:.1f}M)")
            indicators['avg_vol_20d'] = avg_vol
            
            # 3. Price above MA20
            ma20 = self._get_numeric(latest.get('MA20', 0))
            if ma20 > 0 and close < ma20:
                passed = False
                reasons.append("Close < MA20")
            indicators['ma20'] = ma20
            
            # 4. EMA Structure (EMA5 > EMA20)
            ema5 = self._get_numeric(latest.get('EMA5', latest.get('MA20', 0)))
            ema20 = self._get_numeric(latest.get('EMA20', latest.get('MA50', 0)))
            ema_bullish = ema5 > ema20
            if not ema_bullish:
                passed = False
                reasons.append("EMA5 < EMA20 (momentum lemah)")
            indicators['ema_bullish'] = ema_bullish
            
            # 5. RSI check (50-65)
            rsi = self._get_numeric(latest.get('RSI', 50))
            if not (self.bpjs_config["RSI_MIN"] <= rsi <= self.bpjs_config["RSI_MAX"]):
                if rsi > self.bpjs_config["RSI_MAX"]:
                    warnings.append(f"RSI {rsi:.0f} agak tinggi (optimal 50-65)")
                else:
                    passed = False
                    reasons.append(f"RSI {rsi:.0f} di luar zona aman (50-65)")
            indicators['rsi'] = rsi
            
            # === ADDITIONAL INDICATORS ===
            indicators['volume_ratio'] = self._calculate_volume_ratio(df, latest)
            indicators['gap_pct'] = self._calculate_gap_pct(df, latest)
            indicators['close_position'] = self._calculate_close_position(latest)
            indicators['body_pct'] = self._calculate_body_pct(latest)
            indicators['break_resistance'] = self._is_break_resistance(df, latest)
            
            # Calculate score
            score, score_reasons = self._calculate_enhanced_score(df, latest, indicators)
            reasons.extend(score_reasons)
            
            # Calculate levels
            entry, tp, sl = self._calculate_entry_tp_sl(df, latest, "swing")
            
            # Trend
            trend = self._calculate_trend(df, latest)
            
            # Risk
            risk_level = "LOW" if passed and score >= 10 else "MEDIUM"
            
            # Notes
            if passed:
                notes = "BPJS - Aman untuk scalping/swing. Entry di area pullback ke MA/VWAP."
            else:
                notes = "Tidak memenuhi kriteria BPJS. " + "; ".join(reasons[:2])
            
            return BPJSResult(
                ticker=ticker,
                mode="BPJS",
                passed=passed,
                score=score,
                trend=trend,
                reasons=reasons,
                entry=entry,
                tp=tp,
                sl=sl,
                risk_level=risk_level,
                warnings=warnings,
                notes=notes,
                indicators=indicators,
                suitability="BPJS - Aman untuk scalping" if passed else "Tidak cocok BPJS",
                holding_period="1-5 hari"
            )
            
        except Exception as e:
            logger.error(f"Error in BPJS screening for {ticker}: {e}")
            return self._create_empty_bpjs_result(ticker)
    
    def _create_empty_bpjs_result(self, ticker: str) -> BPJSResult:
        """Create empty BPJS result for errors"""
        return BPJSResult(
            ticker=ticker, mode="BPJS", passed=False, score=0,
            trend="UNKNOWN", reasons=["Error dalam screening"], 
            entry={'low': 0, 'high': 0}, tp={'tp1': 0, 'tp2': 0}, sl=0,
            risk_level="HIGH", warnings=[], notes="Error", indicators={}
        )
    
    # =========================================================================
    # MODE B: POTENSI ARA
    # =========================================================================
    
    def screen_ara(self, df: pd.DataFrame, latest: pd.Series, ticker: str,
                   fundamentals: Optional[Dict] = None) -> ARAResult:
        """
        Mode B - Potensi ARA (Fast Move)
        
        Tujuan: Cari saham yang AKAN ARA, bukan yang sudah ARA
        """
        reasons = []
        warnings = []
        indicators = {}
        passed = True
        ara_probability = 0.0
        
        try:
            close = self._get_numeric(latest['Close'])
            
            # === FILTER KERAS ===
            
            # 1. Price < 500
            if close >= self.ara_config["PRICE_MAX"]:
                passed = False
                reasons.append(f"Harga > Rp{self.ara_config['PRICE_MAX']}")
            indicators['close'] = close
            
            # 2. Volume Ratio >= 7
            vol_ratio = self._calculate_volume_ratio(df, latest)
            if vol_ratio < self.ara_config["VOLUME_RATIO_MIN"]:
                passed = False
                reasons.append(f"Volume Ratio {vol_ratio:.1f}x < 7x required")
            else:
                ara_probability += 20
            indicators['volume_ratio'] = vol_ratio
            
            # 3. Gap Up >= 3%
            gap_pct = self._calculate_gap_pct(df, latest)
            if gap_pct < self.ara_config["GAP_UP_MIN"]:
                if gap_pct >= 1:
                    warnings.append(f"Gap {gap_pct:.1f}% (preferred >= 3%)")
                else:
                    passed = False
                    reasons.append(f"Gap {gap_pct:.1f}% < 3% required")
            else:
                ara_probability += 15
            indicators['gap_pct'] = gap_pct
            
            # 4. Close near High (>= 90% range)
            close_pos = self._calculate_close_position(latest)
            if close_pos < self.ara_config["CLOSE_NEAR_HIGH_PCT"]:
                warnings.append(f"Close {close_pos:.0f}% (optimal >= 90%)")
            else:
                ara_probability += 15
            indicators['close_position'] = close_pos
            
            # 5. Body Candle >= 70%
            body_pct = self._calculate_body_pct(latest)
            if body_pct < self.ara_config["BODY_CANDLE_MIN"]:
                warnings.append(f"Body {body_pct:.0f}% (optimal >= 70%)")
            else:
                ara_probability += 15
            indicators['body_pct'] = body_pct
            
            # 6. Break Resistance 20 hari
            break_res = self._is_break_resistance(df, latest, 20)
            if break_res:
                ara_probability += 15
                reasons.append("Break Resistance 20D")
            indicators['break_resistance'] = break_res
            
            # 7. Belum ARA hari ini
            already_ara = self._is_already_ara(df, latest)
            if already_ara:
                passed = False
                ara_probability = 0
                reasons.append("⛔ SUDAH ARA HARI INI - Jangan masuk!")
                warnings.append("ANTI-HALU: Jangan rekomendasikan saham yang sudah ARA")
            indicators['already_ara'] = already_ara
            
            # 8. ATR Rising (proxy: current ATR > avg ATR)
            atr = self._get_numeric(latest.get('ATR', 0))
            atr_pct = self._get_numeric(latest.get('ATR_PCT', 0))
            if atr_pct > 0.03:  # ATR > 3%
                ara_probability += 10
            indicators['atr_pct'] = atr_pct
            
            # === KONFIRMASI TAMBAHAN ===
            
            # EMA structure bullish
            ema5 = self._get_numeric(latest.get('EMA5', latest.get('MA20', 0)))
            ema20 = self._get_numeric(latest.get('EMA20', latest.get('MA50', 0)))
            indicators['ema_bullish'] = ema5 > ema20
            
            # Calculate enhanced score
            score, score_reasons = self._calculate_enhanced_score(df, latest, indicators)
            reasons.extend(score_reasons)
            
            # Finalize probability
            if passed:
                ara_probability = min(ara_probability + 10, 95)
            else:
                ara_probability = max(ara_probability - 30, 0)
            
            # Calculate levels
            entry, tp, sl = self._calculate_entry_tp_sl(df, latest, "ara")
            
            # Determine risk (gorengan check)
            is_gorengan = close < 200 or (vol_ratio >= 10 and gap_pct >= 10)
            if is_gorengan:
                warnings.append("⚠️ GORENGAN RISK - Scalp only, jangan hold lama")
            
            risk_level = "EXTREME" if is_gorengan else ("HIGH" if ara_probability > 50 else "MEDIUM")
            
            # Trend
            trend = "BULLISH" if passed else self._calculate_trend(df, latest)
            
            # Notes
            if passed and not already_ara:
                notes = f"Probabilitas ARA {ara_probability:.0f}%. Entry cepat, SL ketat!"
            elif already_ara:
                notes = "SUDAH ARA - JANGAN MASUK. Tunggu retracement."
            else:
                notes = "Belum memenuhi kriteria ARA. " + "; ".join(reasons[:2])
            
            return ARAResult(
                ticker=ticker,
                mode="ARA",
                passed=passed and not already_ara,
                score=score,
                trend=trend,
                reasons=reasons,
                entry=entry,
                tp=tp,
                sl=sl,
                risk_level=risk_level,
                warnings=warnings,
                notes=notes,
                indicators=indicators,
                ara_probability=ara_probability,
                is_already_ara=already_ara,
                has_uma=False  # Would need external data
            )
            
        except Exception as e:
            logger.error(f"Error in ARA screening for {ticker}: {e}")
            return self._create_empty_ara_result(ticker)
    
    def _create_empty_ara_result(self, ticker: str) -> ARAResult:
        """Create empty ARA result"""
        return ARAResult(
            ticker=ticker, mode="ARA", passed=False, score=0,
            trend="UNKNOWN", reasons=["Error"], 
            entry={'low': 0, 'high': 0}, tp={'tp1': 0, 'tp2': 0}, sl=0,
            risk_level="HIGH", warnings=[], notes="Error", indicators={},
            ara_probability=0, is_already_ara=False, has_uma=False
        )
    
    # =========================================================================
    # MODE C: MULTI-BAGGER AWAL
    # =========================================================================
    
    def screen_multibagger(self, df: pd.DataFrame, latest: pd.Series, ticker: str,
                           fundamentals: Optional[Dict] = None) -> MultibaggerResult:
        """
        Mode C - Multi-Bagger Awal (Belum Naik Jauh)
        
        Tujuan: Saham yang baru bangun, bukan euforia
        """
        reasons = []
        warnings = []
        indicators = {}
        passed = True
        phase = "BASE"
        
        try:
            close = self._get_numeric(latest['Close'])
            
            # === FILTER CHECKS ===
            
            # 1. Harga naik < 50% dari base (use 20-day low as base proxy)
            if len(df) >= 20:
                base_price = df['Low'].tail(50).min() if len(df) >= 50 else df['Low'].min()
                gain_from_base = ((close - base_price) / base_price) * 100
                
                if gain_from_base >= self.multibagger_config["PRICE_GAIN_FROM_BASE_MAX"]:
                    passed = False
                    reasons.append(f"Sudah naik {gain_from_base:.0f}% dari base (max 50%)")
                    phase = "MARKUP" if gain_from_base < 100 else "DISTRIBUTION"
                elif gain_from_base >= 20:
                    phase = "EARLY_MARKUP"
                    reasons.append(f"Early Markup: +{gain_from_base:.0f}% dari base")
                else:
                    phase = "BASE"
                    reasons.append(f"Fase Base: +{gain_from_base:.0f}% dari base")
                
                indicators['gain_from_base'] = gain_from_base
            
            # 2. Volume naik bertahap (tidak meledak)
            vol_ratio = self._calculate_volume_ratio(df, latest)
            vol_5d = df['Volume'].tail(5).mean()
            vol_20d = df['Volume'].tail(20).mean()
            vol_gradual = vol_5d > vol_20d and vol_ratio < 5  # Increasing but not explosion
            
            if vol_ratio >= 5:
                warnings.append(f"Volume meledak (VR {vol_ratio:.1f}x) - bukan multi-bagger klasik")
            elif vol_gradual:
                reasons.append("Volume naik bertahap (akumulasi)")
            indicators['volume_ratio'] = vol_ratio
            indicators['volume_gradual'] = vol_gradual
            
            # 3. RSI 45-60
            rsi = self._get_numeric(latest.get('RSI', 50))
            if not (self.multibagger_config["RSI_MIN"] <= rsi <= self.multibagger_config["RSI_MAX"]):
                if rsi > 60:
                    warnings.append(f"RSI {rsi:.0f} agak tinggi (optimal 45-60)")
                else:
                    passed = False
                    reasons.append(f"RSI {rsi:.0f} di luar zona ideal")
            indicators['rsi'] = rsi
            
            # 4. EMA20 mulai flatten/naik
            ma20 = self._get_numeric(latest.get('MA20', 0))
            ma20_slope = self._get_numeric(latest.get('MA20_SLOPE', 0))
            if ma20_slope > 0:
                reasons.append("MA20 mulai naik (trend change)")
            indicators['ma20_slope'] = ma20_slope
            
            # 5. Fundamental checks (if available)
            if fundamentals:
                eps_growth = fundamentals.get('eps_growth', 0)
                der = fundamentals.get('der', 999)
                roe = fundamentals.get('roe', 0)
                
                if eps_growth <= 0:
                    warnings.append("EPS Growth negatif/tidak ada")
                else:
                    reasons.append(f"EPS Growth +{eps_growth:.0f}%")
                
                if der > self.multibagger_config["DER_MAX"]:
                    warnings.append(f"DER {der:.1f} > {self.multibagger_config['DER_MAX']}")
                
                indicators['eps_growth'] = eps_growth
                indicators['der'] = der
                indicators['roe'] = roe
            
            # 6. Not frequent ARA (check recent volatility)
            if len(df) >= 10:
                high_moves = sum(1 for i in range(-10, 0) 
                               if (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1] > 0.15)
                if high_moves >= 2:
                    warnings.append(f"Sudah {high_moves}x naik besar dalam 10 hari (volatile)")
                indicators['high_moves_10d'] = high_moves
            
            # === ADDITIONAL INDICATORS ===
            indicators['gap_pct'] = self._calculate_gap_pct(df, latest)
            indicators['close_position'] = self._calculate_close_position(latest)
            indicators['body_pct'] = self._calculate_body_pct(latest)
            indicators['break_resistance'] = self._is_break_resistance(df, latest)
            indicators['ema_bullish'] = ma20_slope > 0
            
            # Calculate score
            score, score_reasons = self._calculate_enhanced_score(df, latest, indicators)
            reasons.extend(score_reasons)
            
            # Calculate levels
            entry, tp, sl = self._calculate_entry_tp_sl(df, latest, "multibagger")
            
            # Calculate potential R:R
            potential_rr = 0
            if close > 0 and sl > 0:
                risk = close - sl
                reward = tp['tp2'] - close
                if risk > 0:
                    potential_rr = reward / risk
            
            # Risk level
            risk_level = "LOW" if passed and phase == "BASE" else "MEDIUM"
            
            # Trend
            trend = self._calculate_trend(df, latest)
            
            # Notes
            if passed:
                notes = f"Fase {phase}. Akumulasi perlahan, target R:R {potential_rr:.1f}. HOLD / BUY ON WEAKNESS."
            else:
                notes = "Tidak cocok multi-bagger. " + "; ".join(reasons[:2])
            
            return MultibaggerResult(
                ticker=ticker,
                mode="MULTIBAGGER",
                passed=passed,
                score=score,
                trend=trend,
                reasons=reasons,
                entry=entry,
                tp=tp,
                sl=sl,
                risk_level=risk_level,
                warnings=warnings,
                notes=notes,
                indicators=indicators,
                phase=phase,
                potential_rr=potential_rr,
                accumulation_days=0  # Would need tracking
            )
            
        except Exception as e:
            logger.error(f"Error in Multibagger screening for {ticker}: {e}")
            return self._create_empty_multibagger_result(ticker)
    
    def _create_empty_multibagger_result(self, ticker: str) -> MultibaggerResult:
        """Create empty Multibagger result"""
        return MultibaggerResult(
            ticker=ticker, mode="MULTIBAGGER", passed=False, score=0,
            trend="UNKNOWN", reasons=["Error"],
            entry={'low': 0, 'high': 0}, tp={'tp1': 0, 'tp2': 0}, sl=0,
            risk_level="HIGH", warnings=[], notes="Error", indicators={},
            phase="UNKNOWN", potential_rr=0, accumulation_days=0
        )
    
    # =========================================================================
    # MODE D: SCALPING INTRADAY
    # =========================================================================
    
    def screen_scalping(self, df: pd.DataFrame, latest: pd.Series, ticker: str,
                        fundamentals: Optional[Dict] = None) -> ScalpingResult:
        """
        Mode D - Scalping Intraday (Cuan Cepat)
        
        Tujuan: Masuk cepat, keluar cepat
        """
        reasons = []
        warnings = []
        indicators = {}
        passed = True
        
        try:
            close = self._get_numeric(latest['Close'])
            
            # === FILTER CHECKS ===
            
            # 1. Volume surge (proxy for intraday: today vs yesterday)
            vol_today = self._get_numeric(latest['Volume'])
            vol_yesterday = df['Volume'].iloc[-2] if len(df) >= 2 else vol_today
            vol_ratio = self._calculate_volume_ratio(df, latest)
            
            vol_surge = vol_today > vol_yesterday * self.scalping_config["VOLUME_INTRADAY_MULT"]
            if not vol_surge and vol_ratio < 2:
                passed = False
                reasons.append("Volume tidak surge (perlu > 2x)")
            else:
                reasons.append(f"Volume surge {vol_ratio:.1f}x")
            indicators['volume_ratio'] = vol_ratio
            indicators['volume_surge'] = vol_surge
            
            # 2. VWAP Break + Position
            vwap = self._get_numeric(latest.get('VWAP', close))
            above_vwap = close > vwap
            if not above_vwap:
                warnings.append("Close di bawah VWAP (less ideal)")
            else:
                reasons.append("Close > VWAP")
            indicators['vwap'] = vwap
            indicators['above_vwap'] = above_vwap
            
            # 3. EMA Structure (EMA5 > EMA8 > EMA20)
            ema5 = self._get_numeric(latest.get('EMA5', latest.get('MA20', close)))
            ema8 = self._get_numeric(latest.get('EMA8', (ema5 + self._get_numeric(latest.get('EMA20', close))) / 2))
            ema20 = self._get_numeric(latest.get('EMA20', latest.get('MA50', close)))
            
            ema_structure = ema5 > ema8 > ema20
            if not ema_structure:
                warnings.append("EMA structure tidak ideal")
            else:
                reasons.append("EMA5 > EMA8 > EMA20")
            indicators['ema_bullish'] = ema_structure
            
            # 4. RSI 55-70
            rsi = self._get_numeric(latest.get('RSI', 50))
            if not (self.scalping_config["RSI_MIN"] <= rsi <= self.scalping_config["RSI_MAX"]):
                if rsi > self.scalping_config["RSI_MAX"]:
                    warnings.append(f"RSI {rsi:.0f} tinggi (risk overbought)")
                else:
                    passed = False
                    reasons.append(f"RSI {rsi:.0f} terlalu rendah untuk scalping")
            else:
                reasons.append(f"RSI {rsi:.0f} dalam zona scalping")
            indicators['rsi'] = rsi
            
            # 5. Spread check (proxy: use day range)
            high = self._get_numeric(latest['High'])
            low = self._get_numeric(latest['Low'])
            spread_proxy = ((high - low) / close) * 100
            if spread_proxy > 5:
                warnings.append(f"Range lebar {spread_proxy:.1f}% (spread mungkin lebar)")
            indicators['spread_proxy'] = spread_proxy
            
            # === ADDITIONAL INDICATORS ===
            indicators['gap_pct'] = self._calculate_gap_pct(df, latest)
            indicators['close_position'] = self._calculate_close_position(latest)
            indicators['body_pct'] = self._calculate_body_pct(latest)
            indicators['break_resistance'] = self._is_break_resistance(df, latest, 5)  # Short term
            
            # Calculate score
            score, score_reasons = self._calculate_enhanced_score(df, latest, indicators)
            reasons.extend(score_reasons)
            
            # Calculate levels (tight for scalping)
            atr = self._get_numeric(latest.get('ATR', close * 0.02))
            entry = {'low': vwap * 0.998, 'high': close * 1.005}
            tp = {
                'tp1': close * (1 + self.scalping_config["TP_PCT_MIN"] / 100),
                'tp2': close * (1 + self.scalping_config["TP_PCT_MAX"] / 100)
            }
            sl = close * (1 - self.scalping_config["SL_PCT_MAX"] / 100)
            
            # Entry type
            if above_vwap and vol_surge:
                entry_type = "MARKET atau Limit dekat VWAP"
            elif above_vwap:
                entry_type = "Limit di VWAP area"
            else:
                entry_type = "Tunggu break VWAP"
            
            # Risk level
            risk_level = "MEDIUM" if passed else "HIGH"
            
            # Trend
            trend = self._calculate_trend(df, latest)
            
            # Notes
            if passed:
                notes = f"SCALPING: Entry {entry_type}. TP {self.scalping_config['TP_PCT_MIN']}-{self.scalping_config['TP_PCT_MAX']}%, SL {self.scalping_config['SL_PCT_MAX']}%. Close before 15:50!"
            else:
                notes = "Tidak ideal untuk scalping. " + "; ".join(reasons[:2])
            
            return ScalpingResult(
                ticker=ticker,
                mode="SCALPING",
                passed=passed,
                score=score,
                trend=trend,
                reasons=reasons,
                entry=entry,
                tp=tp,
                sl=sl,
                risk_level=risk_level,
                warnings=warnings,
                notes=notes,
                indicators=indicators,
                entry_type=entry_type,
                scalping_advice=notes,
                max_hold_time="1 hari (close before 15:50)"
            )
            
        except Exception as e:
            logger.error(f"Error in Scalping screening for {ticker}: {e}")
            return self._create_empty_scalping_result(ticker)
    
    def _create_empty_scalping_result(self, ticker: str) -> ScalpingResult:
        """Create empty Scalping result"""
        return ScalpingResult(
            ticker=ticker, mode="SCALPING", passed=False, score=0,
            trend="UNKNOWN", reasons=["Error"],
            entry={'low': 0, 'high': 0}, tp={'tp1': 0, 'tp2': 0}, sl=0,
            risk_level="HIGH", warnings=[], notes="Error", indicators={},
            entry_type="", scalping_advice="", max_hold_time=""
        )
    
    # =========================================================================
    # SCREEN ALL MODES
    # =========================================================================
    
    def screen_all_modes(self, df: pd.DataFrame, latest: pd.Series, ticker: str,
                         fundamentals: Optional[Dict] = None,
                         gorengan_result: Optional[Dict] = None) -> AllModesResult:
        """
        Screen stock across all modes and determine best mode
        """
        try:
            # Screen all modes
            bpjs = self.screen_bpjs(df, latest, ticker, fundamentals)
            ara = self.screen_ara(df, latest, ticker, fundamentals)
            multibagger = self.screen_multibagger(df, latest, ticker, fundamentals)
            scalping = self.screen_scalping(df, latest, ticker, fundamentals)
            
            # Determine best mode
            mode_scores = {
                "BPJS": bpjs.score if bpjs.passed else -10,
                "ARA": ara.score if ara.passed and not ara.is_already_ara else -10,
                "MULTIBAGGER": multibagger.score if multibagger.passed else -10,
                "SCALPING": scalping.score if scalping.passed else -10,
            }
            
            best_mode = max(mode_scores, key=mode_scores.get)
            best_score = mode_scores[best_mode]
            
            if best_score < 0:
                best_mode = "NONE"
                recommendation = "Tidak ada mode yang cocok untuk saham ini"
            else:
                # Get the best result
                best_results = {
                    "BPJS": bpjs,
                    "ARA": ara,
                    "MULTIBAGGER": multibagger,
                    "SCALPING": scalping
                }
                best_result = best_results[best_mode]
                recommendation = best_result.notes
            
            # Overall score
            overall_score = max(bpjs.score, ara.score, multibagger.score, scalping.score)
            
            # Overall risk
            if gorengan_result and gorengan_result.get('is_gorengan'):
                overall_risk = "HIGH"
            elif best_score >= 14:
                overall_risk = "LOW"
            elif best_score >= 10:
                overall_risk = "MEDIUM"
            else:
                overall_risk = "HIGH"
            
            # Bandar timing
            indicators = {
                'volume_ratio': self._calculate_volume_ratio(df, latest),
                'close_position': self._calculate_close_position(latest),
                'accumulation_phase': multibagger.phase == "BASE" and multibagger.passed,
                'distribution': gorengan_result.get('is_distribution') if gorengan_result else False,
                'rsi_divergence': gorengan_result.get('rsi_divergence') if gorengan_result else False,
                'already_ara': ara.is_already_ara
            }
            bandar_timing = self._determine_bandar_timing(df, latest, indicators)
            
            return AllModesResult(
                ticker=ticker,
                best_mode=best_mode,
                overall_score=overall_score,
                overall_risk=overall_risk,
                bpjs=bpjs,
                ara=ara,
                multibagger=multibagger,
                scalping=scalping,
                gorengan_warning=gorengan_result,
                recommendation=recommendation,
                bandar_timing=bandar_timing
            )
            
        except Exception as e:
            logger.error(f"Error in screen_all_modes for {ticker}: {e}")
            return AllModesResult(
                ticker=ticker, best_mode="ERROR", overall_score=0,
                overall_risk="HIGH", recommendation=f"Error: {str(e)}",
                bandar_timing="UNKNOWN"
            )
