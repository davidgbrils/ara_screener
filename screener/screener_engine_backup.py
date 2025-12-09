"""Main screener engine for ARA signal generation"""

import pandas as pd
from typing import Dict, Optional, List
from fundamentals.financial_fetcher import FinancialFetcher
from config import SCREENER_CONFIG, SIGNAL_THRESHOLDS, PATTERN_CONFIG
from .pattern_detector import PatternDetector
from .regime_filter import RegimeFilter
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ScreenerEngine:
    """Main screener for identifying ARA/multi-bagger candidates"""
    
    def __init__(self):
        """Initialize screener engine"""
        self.config = SCREENER_CONFIG
        self.pattern_detector = PatternDetector()
        self.regime_filter = RegimeFilter() if PATTERN_CONFIG.get("MARKET_REGIME", True) else None
        self.fin_fetcher = FinancialFetcher()
    
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

        # Detect market regime if enabled
        if self.regime_filter:
            try:
                regime_info = self.regime_filter.detect_regime(df)
            except Exception as e:
                logger.warning(f"Regime detection failed for {ticker}: {e}")
                regime_info = {'regime': 'NEUTRAL', 'confidence': 0.0}
        else:
            regime_info = {'regime': 'NEUTRAL', 'confidence': 0.0}

        # Determine signal
        signal = self._determine_signal(score)

        # Calculate confidence/accuracy score (include regime info)
        confidence = self._calculate_confidence(score, parameter_count, patterns, df, latest, regime_info)
        
        # Get entry/exit levels
        entry_levels = self._calculate_entry_levels(df, latest)
        
        # Data validation
        data_quality = self._validate_data_quality(df, latest)
        
        # Additional classifications
        fundamentals = self._get_fundamentals(ticker)
        max_ara = self._classify_max_ara(df, latest, fundamentals)
        bpjs = self._classify_bpjs(df, latest, fundamentals)
        gorengan = self._classify_gorengan(df, latest, fundamentals)

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
            'regime': regime_info,
            'classifications': {
                'max_ara': max_ara,
                'bpjs': bpjs,
                'gorengan': gorengan,
            },
            'fundamentals': fundamentals or {},
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
        
        # Volatility filter
        atr_pct = latest.get('ATR_PCT', None)
        atr_range = self.config.get("ATR_PCT_RANGE")
        if atr_range and pd.notna(atr_pct):
            low, high = atr_range
            if not (low <= atr_pct <= high):
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
        max_parameters = 12
        
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

        # MACD confirmation
        macd = latest.get('MACD', None)
        macd_signal = latest.get('MACD_SIGNAL', None)
        macd_hist = latest.get('MACD_HIST', None)
        if pd.notna(macd) and pd.notna(macd_signal) and pd.notna(macd_hist):
            macd_ok = macd > macd_signal and macd > 0 and macd_hist > 0
            if macd_ok:
                score += 0.10
                reasons.append("MACD Bullish")
                parameter_count += 1

        # Momentum multi-horizon
        pct_1w = latest.get('PCT_CHANGE_1W', 0)
        pct_1m = latest.get('PCT_CHANGE_1M', 0)
        pct_3m = latest.get('PCT_CHANGE_3M', 0)
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

        # 52W proximity
        dist_52w = latest.get('DIST_52W_HIGH', None)
        if pd.notna(dist_52w) and dist_52w >= -3:
            score += 0.05
            reasons.append("Near 52W High")
            parameter_count += 1

        return min(score, 1.0), reasons, parameter_count
    
    def _calculate_confidence(
        self, 
        score: float, 
        parameter_count: int, 
        patterns: Dict, 
        df: pd.DataFrame,
        latest: pd.Series,
        regime_info: Dict = None
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
        max_params = 12
        param_ratio = min(parameter_count / max_params, 1.0)
        confidence += param_ratio * 0.3
        
        # Pattern detection (20%) - more patterns = higher confidence
        detected_patterns = sum(1 for v in patterns.values() 
                              if v and (isinstance(v, bool) or (isinstance(v, dict) and v.get('detected'))))
        pattern_score = min(detected_patterns / 5.0, 1.0)  # Max 5 patterns
        confidence += pattern_score * 0.2
        
        # Data quality (5%) - based on volume and price consistency
        if len(df) >= 200:  # Sufficient history
            confidence += 0.03
        if latest['Volume'] > df['Volume'].tail(20).mean() * 2:  # High volume
            confidence += 0.02
        
        # Market regime (5%) - boost in bull market
        if regime_info and regime_info.get('regime') == 'BULL':
            confidence += 0.05 * regime_info.get('confidence', 0)
        
        # Advanced patterns boost (10%)
        advanced_patterns = ['vcp', 'money_flow', 'volume_dry_up', 'breakout_52w', 'candlestick_bullish']
        detected_advanced = sum(1 for p in advanced_patterns 
                              if patterns.get(p) and 
                              (isinstance(patterns[p], bool) or 
                               (isinstance(patterns[p], dict) and patterns[p].get('detected'))))
        confidence += min(detected_advanced / 5.0, 1.0) * 0.10

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

    def _classify_max_ara(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            rv = latest.get('RVOL', 0)
            rsi = latest.get('RSI', 0)
            vwap = latest.get('VWAP', None)
            obv_slope = latest.get('OBV_SLOPE_5D', 0)
            ma20 = latest.get('MA20', 0)
            ma50 = latest.get('MA50', 0)
            ma200 = latest.get('MA200', 0)
            ma20_slope = latest.get('MA20_SLOPE', 0)
            ma20_pct5 = latest.get('MA20_PCT_CHANGE_5D', 0)
            atr_pct = latest.get('ATR_PCT', 0) * 100 if latest.get('ATR_PCT', None) is not None else None
            high20 = latest.get('HIGH_20D', None)
            close = latest.get('Close', 0)
            std10 = latest.get('STDDEV_10D', None)
            std20 = latest.get('STDDEV_20D', None)
            # Basic ratios
            if close > ma20 and close > ma50 and close > ma200:
                score += 15
                reasons.append('MA structure')
            if high20 and close >= high20:
                score += 15
                reasons.append('Breakout High20D')
            if rv >= 4:
                score += 20
                reasons.append('RVOL>=4')
            if vol20 and latest['Volume'] / vol20 >= 4:
                score += 10
                reasons.append('Vol/Avg20>=4')
            if vwap and close > vwap:
                score += 8
                reasons.append('Close>VWAP')
            if 55 <= rsi <= 75:
                score += 6
                reasons.append('RSI zone')
            if obv_slope and obv_slope > 0:
                score += 6
                reasons.append('OBV rising')
            if ma20_pct5 and ma20_pct5 >= 8:
                score += 10
                reasons.append('MA20 pct_change 5d>=8%')
            if atr_pct is not None and atr_pct <= 8:
                score += 5
                reasons.append('ATR%<=8')
            # Ratios vs ratios
            if ma50 and ma20 / ma50 >= 1.02:
                score += 2
            if ma200 and ma20 / ma200 >= 1.05:
                score += 2
            if high20 and close / high20 >= 1.0:
                score += 2
            if vol5 and latest['Volume'] / vol5 >= 2:
                score += 2
            if std10 and std20 and std10 / std20 <= 0.8:
                score += 2
            # Optional fundamentals
            if fundamentals:
                mc = fundamentals.get('market_cap')
                if mc and mc >= 500_000_000_000:
                    score += 5
                    reasons.append('MarketCap>=500B')
            cls = 'Very High' if score >= 85 else 'High' if score >= 70 else 'Medium' if score >= 55 else 'Low'
            return {'score': score, 'class': cls, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'class': 'Low', 'reasons': []}

    def _classify_bpjs(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            vol1 = latest.get('Volume', 0)
            pct_today = latest.get('PCT_CHANGE_TODAY', 0)
            vwap = latest.get('VWAP', None)
            atr_pct = latest.get('ATR_PCT', 0) * 100 if latest.get('ATR_PCT', None) is not None else None
            ma5 = df['Close'].rolling(5).mean().iloc[-1]
            ma10 = df['Close'].rolling(10).mean().iloc[-1]
            ma20 = latest.get('MA20', 0)
            close = latest.get('Close', 0)
            openp = latest.get('Open', None)
            high = latest.get('High', 0)
            low = latest.get('Low', 0)
            hl_range = (high - low) / close if close else None

            # Enhanced BPJS criteria with more sophisticated rules
            # 1. Intraday momentum - strong opening with continuation
            if openp is not None and openp < close:
                score += 10
                reasons.append('Open<Close (bullish intraday)')

            # 2. Strong intraday performance
            if pct_today >= 1.0:  # Increased from 0.8 to 1.0 for better quality
                score += 15
                reasons.append(f'PctToday>={pct_today:.1f}%')
            elif pct_today >= 0.5:
                score += 8
                reasons.append(f'PctToday>={pct_today:.1f}%')

            # 3. Volume surge patterns - more aggressive for BPJS
            if vol5 and latest['Volume'] / vol5 >= 4:  # Increased from 3 to 4
                score += 20
                reasons.append('Vol/Avg5>=4 (strong surge)')
            elif vol5 and latest['Volume'] / vol5 >= 2.5:
                score += 12
                reasons.append('Vol/Avg5>=2.5')

            if vol20 and latest['Volume'] / vol20 >= 3:  # Increased from 2 to 3
                score += 15
                reasons.append('Vol/Avg20>=3')
            elif vol20 and latest['Volume'] / vol20 >= 1.8:
                score += 8
                reasons.append('Vol/Avg20>=1.8')

            # 4. Tight intraday range for better risk management
            if hl_range is not None and hl_range <= 0.03:  # Tighter from 0.04 to 0.03
                score += 15
                reasons.append('Range<=3% (tight intraday)')
            elif hl_range is not None and hl_range <= 0.05:
                score += 8
                reasons.append('Range<=5%')

            # 5. VWAP position - critical for BPJS
            if vwap and close > vwap * 1.02:  # 2% above VWAP
                score += 20
                reasons.append('Close>VWAP+2% (strong position)')
            elif vwap and close > vwap:
                score += 10
                reasons.append('Close>VWAP')

            # 6. Short-term MA structure for intraday momentum
            if ma5 and ma10 and ma5 > ma10:
                score += 15
                reasons.append('MA5>MA10 (short-term momentum)')
            if ma10 and ma20 and ma10 > ma20:
                score += 10
                reasons.append('MA10>MA20 (medium-term momentum)')

            # 7. Low volatility for better risk/reward
            if atr_pct is not None and atr_pct <= 4:  # Tighter from 5 to 4
                score += 20
                reasons.append('ATR%<=4 (low volatility)')
            elif atr_pct is not None and atr_pct <= 6:
                score += 10
                reasons.append('ATR%<=6')

            # 8. Price near recent highs (breakout potential)
            high5 = df['High'].tail(5).max()
            if close >= high5 * 0.98:  # Within 2% of recent high
                score += 12
                reasons.append('Close near 5-day high')

            # 9. Fundamentals for liquidity
            if fundamentals:
                spread = fundamentals.get('bid_ask_spread_pct')
                if spread is not None and spread <= 1.0:  # Tighter from 1.5 to 1.0
                    score += 8
                    reasons.append('BidAskSpread<=1.0% (good liquidity)')
                elif spread is not None and spread <= 1.5:
                    score += 5
                    reasons.append('BidAskSpread<=1.5%')

                # Market cap filter for BPJS
                mc = fundamentals.get('market_cap')
                if mc and mc >= 100_000_000_000:  # 100B minimum for liquidity
                    score += 5
                    reasons.append('MarketCap>=100B (good liquidity)')

            # 10. Intraday volume pattern - increasing volume
            if len(df) >= 3:
                vol_trend = df['Volume'].tail(3).iloc[-1] > df['Volume'].tail(3).iloc[0]
                if vol_trend:
                    score += 8
                    reasons.append('Volume increasing intraday')

            # Enhanced advice with specific BPJS strategy
            advice = {
                'enter': '30-60m after open if Close>VWAP+1% & volume surge confirmed',
                'exit': 'before 15:30 or TP 2-4% / trailing 0.8-1.2%',
                'stop': 'Close<VWAP or SL=0.6*ATR or -1.5% from entry',
                'risk_management': 'Max 2% portfolio risk per trade, 5% daily loss limit',
                'best_time': '09:30-11:00 for entry, 14:30-15:30 for exit',
            }

            # Determine candidate status with tiered approach
            candidate = False
            strong_candidate = False
            if score >= 80:
                candidate = True
                strong_candidate = True
                reasons.append('STRONG_BPJS_CANDIDATE')
            elif score >= 65:
                candidate = True
                reasons.append('BPJS_CANDIDATE')
            elif score >= 50:
                candidate = True
                reasons.append('POTENTIAL_BPJS_CANDIDATE')

            return {
                'score': score,
                'reasons': reasons,
                'advice': advice,
                'candidate': candidate,
                'strong_candidate': strong_candidate,
                'suitability': 'HIGH' if score >= 80 else 'MEDIUM' if score >= 65 else 'LOW' if score >= 50 else 'NONE'
            }
        except Exception:
            return {'score': 0, 'reasons': [], 'advice': {}, 'candidate': False, 'strong_candidate': False, 'suitability': 'NONE'}

    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}
    def _classify_gorengan(self, df: pd.DataFrame, latest: pd.Series, fundamentals: Dict = None) -> Dict:
        try:
            score = 0
            reasons = []
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()
            pct1d = ((latest['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            close = latest.get('Close', 0)
            high20 = latest.get('HIGH_20D', None)
            if vol20 and latest['Volume'] / vol20 >= 6:
                score += 25
                reasons.append('Vol/Avg20>=6')
            if pct1d >= 15:
                score += 25
                reasons.append('Pct1D>=15%')
            if high20 and close > high20:
                score += 20
                reasons.append('Close>High20D')
            if vol5 and latest['Volume'] / vol5 >= 8:
                score += 15
                reasons.append('Vol/Avg5>=8')
            std5 = df['Close'].rolling(5).std().iloc[-1] if len(df) >= 5 else None
            std20 = df['Close'].rolling(20).std().iloc[-1] if len(df) >= 20 else None
            if std5 and std20 and std5 / std20 >= 1.8:
                score += 15
                reasons.append('Std5/Std20>=1.8')
            if fundamentals:
                mc = fundamentals.get('market_cap')
                spread = fundamentals.get('bid_ask_spread_pct')
                if mc and mc < 200_000_000_000:
                    score += 10
                    reasons.append('SmallCap<200B')
                if spread and spread >= 3:
                    score += 10
                    reasons.append('WideSpread>=3%')
            risk = 'High' if score >= 70 else 'Medium' if score >= 50 else 'Low'
            return {'score': score, 'risk': risk, 'reasons': reasons}
        except Exception:
            return {'score': 0, 'risk': 'Low', 'reasons': []}

    def _get_fundamentals(self, ticker: str) -> Dict:
        try:
            return self.fin_fetcher.get(ticker) or {}
        except Exception:
            return {}


    def _calculate_bpjs_combo_score(self, df: pd.DataFrame, latest: pd.Series, patterns: Dict, bpjs_classification: Dict) -> Dict:
        """
        Calculate combo score specifically for BPJS strategy
        Combines bullish/ara detection with BPJS suitability

        Args:
            df: DataFrame with indicators
            latest: Latest data point
            patterns: Detected patterns
            bpjs_classification: BPJS classification result

        Returns:
            Dictionary with combo score and analysis
        """
        try:
            combo_score = 0
            combo_reasons = []
            combo_factors = {}

            # Get current classifications
            bpjs_score = bpjs_classification.get('score', 0)
            bpjs_suitability = bpjs_classification.get('suitability', 'NONE')

            # 1. Bullish/ARA factors (40% weight)
            bullish_factors = 0

            # RVOL - key for momentum
            rvol = latest.get('RVOL', 0)
            if rvol >= 3.0:
                bullish_factors += 15
                combo_reasons.append(f"RVOL {rvol:.1f}x (strong momentum)")
            elif rvol >= 2.0:
                bullish_factors += 8
                combo_reasons.append(f"RVOL {rvol:.1f}x (moderate momentum)")

            # MA Structure - bullish trend
            ma20 = latest.get('MA20', 0)
            ma50 = latest.get('MA50', 0)
            ma200 = latest.get('MA200', 0)
            if ma20 > ma50 > ma200:
                bullish_factors += 12
                combo_reasons.append("Bullish MA Structure (20>50>200)")
            elif ma20 > ma50:
                bullish_factors += 8
                combo_reasons.append("Bullish MA Structure (20>50)")

            # RSI - momentum zone
            rsi = latest.get('RSI', 50)
            if 60 <= rsi <= 75:
                bullish_factors += 10
                combo_reasons.append(f"RSI {rsi:.0f} (strong momentum zone)")
            elif 55 <= rsi <= 80:
                bullish_factors += 6
                combo_reasons.append(f"RSI {rsi:.0f} (momentum zone)")

            # Patterns - bullish patterns
            bullish_patterns = 0
            if patterns.get('parabolic'):
                bullish_patterns += 1
                combo_reasons.append("Parabolic pattern detected")
            if patterns.get('vcp'):
                bullish_patterns += 1
                combo_reasons.append("VCP pattern detected")
            if patterns.get('money_flow'):
                bullish_patterns += 1
                combo_reasons.append("Strong money flow detected")
            if patterns.get('candlestick_bullish'):
                bullish_patterns += 1
                combo_reasons.append("Bullish candlestick pattern")

            bullish_factors += bullish_patterns * 8
            combo_factors['bullish_patterns'] = bullish_patterns

            # 2. BPJS-specific factors (50% weight)
            bpjs_factors = bpjs_score  # Use the BPJS score directly

            # 3. Volume confirmation (10% weight)
            volume_factors = 0
            vol20 = df['Volume'].tail(20).mean()
            vol5 = df['Volume'].tail(5).mean()

            if vol5 and latest['Volume'] / vol5 >= 3.0:
                volume_factors += 8
                combo_reasons.append("Strong volume surge (Vol/Avg5>=3)")
            elif vol5 and latest['Volume'] / vol5 >= 2.0:
                volume_factors += 5
                combo_reasons.append("Volume surge (Vol/Avg5>=2)")

            # 4. Risk management factors (10% weight)
            risk_factors = 0
            atr_pct = latest.get('ATR_PCT', 0) * 100 if latest.get('ATR_PCT', None) is not None else None

            if atr_pct is not None and atr_pct <= 5.0:
                risk_factors += 8
                combo_reasons.append(f"Low volatility (ATR%={atr_pct:.1f}%)")
            elif atr_pct is not None and atr_pct <= 8.0:
                risk_factors += 5
                combo_reasons.append(f"Moderate volatility (ATR%={atr_pct:.1f}%)")

            # Calculate weighted combo score
            combo_score = (
                bullish_factors * 0.40 +
                bpjs_factors * 0.50 +
                volume_factors * 0.10 +
                risk_factors * 0.10
            )

            # Determine combo suitability
            if combo_score >= 85:
                combo_suitability = 'VERY_HIGH'
                combo_reasons.append('VERY_HIGH_BPJS_COMBO_CANDIDATE')
            elif combo_score >= 75:
                combo_suitability = 'HIGH'
                combo_reasons.append('HIGH_BPJS_COMBO_CANDIDATE')
            elif combo_score >= 65:
                combo_suitability = 'MEDIUM'
                combo_reasons.append('MEDIUM_BPJS_COMBO_CANDIDATE')
            elif combo_score >= 55:
                combo_suitability = 'LOW'
                combo_reasons.append('LOW_BPJS_COMBO_CANDIDATE')
            else:
                combo_suitability = 'NONE'

            # Strategy recommendations based on combo
            strategy_advice = {
                'entry_strategy': 'Enter on pullback to VWAP with volume confirmation',
                'exit_strategy': 'Exit at 2-4% profit or before market close',
                'risk_management': 'Use tight stop loss (1-1.5%) and position size accordingly',
                'timeframe': 'Intraday (BPJS) - buy morning, sell afternoon',
                'volume_requirement': 'Wait for volume surge confirmation before entry'
            }

            if combo_score >= 80:
                strategy_advice['entry_strategy'] = 'Aggressive entry on breakout with volume'
                strategy_advice['exit_strategy'] = 'Take partial profits at 2%, full at 4%'
                strategy_advice['risk_management'] = 'Tight stop loss (0.8-1.2%), max 2% portfolio risk'

            return {
                'combo_score': combo_score,
                'bullish_factors': bullish_factors,
                'bpjs_factors': bpjs_factors,
                'volume_factors': volume_factors,
                'risk_factors': risk_factors,
                'combo_suitability': combo_suitability,
                'combo_reasons': combo_reasons,
                'strategy_advice': strategy_advice,
                'combo_factors': combo_factors,
                'is_combo_candidate': combo_score >= 65
            }
        except Exception as e:
            logger.error(f"Error calculating BPJS combo score: {e}")
            return {
                'combo_score': 0,
                'bullish_factors': 0,
                'bpjs_factors': 0,
                'volume_factors': 0,
                'risk_factors': 0,
                'combo_suitability': 'NONE',
                'combo_reasons': [],
                'strategy_advice': {},
                'combo_factors': {},
                'is_combo_candidate': False
            }
