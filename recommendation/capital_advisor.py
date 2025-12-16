"""
Capital Advisor - Personalized Stock Recommendations Based on Capital

This module provides:
- Position sizing calculations
- Capital-based stock recommendations
- Risk/Reward analysis
- Portfolio allocation suggestions

Author: ARA Bot V3 - Professional Quant Trader
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from config import CAPITAL_ADVISOR_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PositionSize:
    """Position sizing result"""
    ticker: str
    stock_price: float
    lot_size: int
    max_lots: int
    max_shares: int
    max_position_value: float
    position_pct: float
    risk_amount: float


@dataclass
class TradeRecommendation:
    """Individual trade recommendation"""
    ticker: str
    mode: str
    action: str             # 'BUY', 'WATCH', 'AVOID'
    entry_price: float
    entry_range: Dict[str, float]
    tp1: float
    tp2: float
    sl: float
    lots_recommended: int
    shares_recommended: int
    position_value: float
    risk_amount: float
    potential_profit_min: float
    potential_profit_max: float
    risk_reward_ratio: float
    score: int
    risk_level: str
    reasons: List[str]
    warnings: List[str]
    notes: str


@dataclass
class CapitalAllocation:
    """Capital allocation result"""
    total_capital: float
    allocated_capital: float
    remaining_capital: float
    num_positions: int
    recommendations: List[TradeRecommendation]
    summary: str


class CapitalAdvisor:
    """
    Generate personalized stock recommendations based on capital.
    
    Features:
    - Position sizing based on risk management
    - Multi-stock portfolio allocation
    - Mode-specific recommendations
    - Scalping-focused trade suggestions
    """
    
    def __init__(self, capital: float, risk_profile: str = "moderate"):
        """
        Initialize Capital Advisor
        
        Args:
            capital: Total trading capital in IDR
            risk_profile: 'conservative', 'moderate', 'aggressive'
        """
        self.capital = capital
        self.risk_profile = risk_profile
        self.config = CAPITAL_ADVISOR_CONFIG
        
        # Adjust parameters based on risk profile
        if risk_profile == "conservative":
            self.max_position_pct = 10
            self.risk_per_trade_pct = 1
            self.max_positions = 5
        elif risk_profile == "aggressive":
            self.max_position_pct = 30
            self.risk_per_trade_pct = 3
            self.max_positions = 3
        else:  # moderate
            self.max_position_pct = self.config["MAX_POSITION_PCT"]
            self.risk_per_trade_pct = self.config["RISK_PER_TRADE_PCT"]
            self.max_positions = 4
    
    def get_position_size(self, stock_price: float, ticker: str = "",
                          lot_size: int = None) -> PositionSize:
        """
        Calculate position size based on capital and risk management
        
        Args:
            stock_price: Current stock price
            ticker: Stock ticker
            lot_size: Lot size (default 100 for IDX)
        
        Returns:
            PositionSize with calculations
        """
        try:
            lot_size = lot_size or self.config["LOT_SIZE"]
            
            # Maximum position value (% of capital)
            max_position_value = self.capital * (self.max_position_pct / 100)
            
            # Maximum shares we can buy
            max_shares_by_capital = int(max_position_value / stock_price)
            
            # Round down to lot size
            max_lots = max_shares_by_capital // lot_size
            max_shares = max_lots * lot_size
            
            # Actual position value
            actual_position_value = max_shares * stock_price
            
            # Position as % of capital
            position_pct = (actual_position_value / self.capital) * 100 if self.capital > 0 else 0
            
            # Risk amount (based on risk per trade)
            risk_amount = self.capital * (self.risk_per_trade_pct / 100)
            
            return PositionSize(
                ticker=ticker,
                stock_price=stock_price,
                lot_size=lot_size,
                max_lots=max_lots,
                max_shares=max_shares,
                max_position_value=actual_position_value,
                position_pct=position_pct,
                risk_amount=risk_amount
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return PositionSize(
                ticker=ticker, stock_price=stock_price, lot_size=100,
                max_lots=0, max_shares=0, max_position_value=0,
                position_pct=0, risk_amount=0
            )
    
    def calculate_risk_reward(self, entry: float, tp: float, sl: float,
                              position_size: int) -> Dict[str, float]:
        """
        Calculate risk/reward metrics for a trade
        
        Args:
            entry: Entry price
            tp: Take profit price
            sl: Stop loss price
            position_size: Number of shares
        
        Returns:
            Dict with risk/reward calculations
        """
        try:
            # Risk and reward per share
            risk_per_share = entry - sl
            reward_per_share = tp - entry
            
            # Total risk and reward
            total_risk = risk_per_share * position_size
            total_reward = reward_per_share * position_size
            
            # R:R ratio
            rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
            
            # Percentages
            risk_pct = (risk_per_share / entry) * 100 if entry > 0 else 0
            reward_pct = (reward_per_share / entry) * 100 if entry > 0 else 0
            
            # Including commission and tax
            commission = self.config["COMMISSION_PCT"] / 100
            tax = self.config["TAX_SELL_PCT"] / 100
            
            # Buy cost
            buy_cost = entry * position_size * (1 + commission)
            
            # Sell revenue (TP)
            sell_revenue_tp = tp * position_size * (1 - commission - tax)
            
            # Sell revenue (SL)
            sell_revenue_sl = sl * position_size * (1 - commission - tax)
            
            # Net profit/loss
            net_profit_tp = sell_revenue_tp - buy_cost
            net_loss_sl = sell_revenue_sl - buy_cost
            
            return {
                'risk_per_share': risk_per_share,
                'reward_per_share': reward_per_share,
                'total_risk': total_risk,
                'total_reward': total_reward,
                'rr_ratio': rr_ratio,
                'risk_pct': risk_pct,
                'reward_pct': reward_pct,
                'buy_cost': buy_cost,
                'net_profit_tp': net_profit_tp,
                'net_loss_sl': net_loss_sl,
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward: {e}")
            return {}
    
    def recommend_stocks(self, results: List[Dict], mode: str = "scalping",
                         max_stocks: int = 5) -> List[TradeRecommendation]:
        """
        Generate stock recommendations based on capital and mode
        
        Args:
            results: List of screening results
            mode: Trading mode ('scalping', 'bpjs', 'ara', 'multibagger', 'all')
            max_stocks: Maximum number of recommendations
        
        Returns:
            List of TradeRecommendation
        """
        recommendations = []
        
        try:
            # Check minimum capital
            if mode == "scalping" and self.capital < self.config["MIN_CAPITAL_SCALPING"]:
                logger.warning(f"Modal {self.capital:,.0f} kurang untuk scalping (min {self.config['MIN_CAPITAL_SCALPING']:,.0f})")
            
            # Filter and sort results based on mode
            filtered_results = self._filter_results_by_mode(results, mode)
            
            # Sort by score
            filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Generate recommendations
            for result in filtered_results[:max_stocks]:
                rec = self._create_recommendation(result, mode)
                if rec and rec.action != "AVOID":
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _filter_results_by_mode(self, results: List[Dict], mode: str) -> List[Dict]:
        """Filter results based on trading mode"""
        filtered = []
        
        for result in results:
            if not result or result.get('signal') == 'NONE':
                continue
            
            # Get mode-specific result if available
            classifications = result.get('classifications', {})
            all_modes = classifications.get('all_modes', {})
            
            if mode == "all":
                filtered.append(result)
            elif mode == "scalping":
                # For scalping, prioritize high volume and momentum
                scalping_result = all_modes.get('scalping', {})
                if scalping_result.get('passed', False):
                    result['mode_result'] = scalping_result
                    filtered.append(result)
                elif result.get('confidence', 0) >= 0.6:
                    filtered.append(result)
            elif mode == "bpjs":
                bpjs_result = all_modes.get('bpjs', {})
                if bpjs_result.get('passed', False):
                    result['mode_result'] = bpjs_result
                    filtered.append(result)
            elif mode == "ara":
                ara_result = all_modes.get('ara', {})
                if ara_result.get('passed', False) and not ara_result.get('is_already_ara', False):
                    result['mode_result'] = ara_result
                    filtered.append(result)
            elif mode == "multibagger":
                mb_result = all_modes.get('multibagger', {})
                if mb_result.get('passed', False):
                    result['mode_result'] = mb_result
                    filtered.append(result)
        
        return filtered
    
    def _create_recommendation(self, result: Dict, mode: str) -> Optional[TradeRecommendation]:
        """Create a trade recommendation from screening result"""
        try:
            ticker = result.get('ticker', '')
            latest_price = result.get('latest_price', 0)
            
            if latest_price <= 0:
                return None
            
            # Get entry levels
            entry_levels = result.get('entry_levels', {})
            entry_low = entry_levels.get('entry_low', latest_price * 0.98)
            entry_high = entry_levels.get('entry_high', latest_price * 1.02)
            tp1 = entry_levels.get('take_profit_1', latest_price * 1.05)
            tp2 = entry_levels.get('take_profit_2', latest_price * 1.10)
            sl = entry_levels.get('stop_loss', latest_price * 0.95)
            
            # Adjust for scalping mode
            if mode == "scalping":
                tp1 = latest_price * 1.015  # 1.5%
                tp2 = latest_price * 1.03   # 3%
                sl = latest_price * 0.99    # 1%
            
            # Calculate position size
            position = self.get_position_size(latest_price, ticker)
            
            # Optimal lots based on risk
            # Risk amount / (entry - sl) = max shares
            risk_per_share = latest_price - sl
            if risk_per_share > 0:
                optimal_shares = int(position.risk_amount / risk_per_share)
                optimal_lots = optimal_shares // self.config["LOT_SIZE"]
                # Cap at max lots
                lots_recommended = min(optimal_lots, position.max_lots)
            else:
                lots_recommended = 1
            
            shares_recommended = lots_recommended * self.config["LOT_SIZE"]
            position_value = shares_recommended * latest_price
            
            # Calculate risk/reward
            rr = self.calculate_risk_reward(latest_price, tp1, sl, shares_recommended)
            
            # Determine action
            score = result.get('score', 0)
            confidence = result.get('confidence', 0)
            
            # Check gorengan warning
            gorengan = result.get('classifications', {}).get('enhanced_gorengan', {})
            is_gorengan = gorengan.get('is_gorengan', False)
            
            if score >= 0.7 and confidence >= 0.6 and not is_gorengan:
                action = "BUY"
            elif score >= 0.5 and confidence >= 0.5:
                action = "WATCH"
            else:
                action = "AVOID"
            
            # For gorengan, only scalp
            if is_gorengan:
                action = "SCALP ONLY" if mode == "scalping" else "AVOID"
            
            # Get reasons and warnings
            reasons = result.get('reasons', [])[:5]
            
            # Combine warnings
            warnings = []
            if is_gorengan:
                warnings.append("⚠️ GORENGAN - Scalp only, jangan hold!")
            gorengan_warnings = gorengan.get('warnings', [])
            warnings.extend(gorengan_warnings[:3])
            
            # Risk level
            risk_level = gorengan.get('risk_level', 'MEDIUM')
            if not risk_level or risk_level == 'LOW':
                risk_level = "MEDIUM" if score < 0.7 else "LOW"
            
            # Notes
            if mode == "scalping":
                notes = f"SCALPING: Entry Rp{latest_price:,.0f}, TP Rp{tp1:,.0f} ({(tp1/latest_price-1)*100:.1f}%), SL Rp{sl:,.0f}. Close before 15:50!"
            else:
                notes = f"Mode {mode.upper()}: Entry Rp{entry_low:,.0f}-{entry_high:,.0f}"
            
            return TradeRecommendation(
                ticker=ticker,
                mode=mode.upper(),
                action=action,
                entry_price=latest_price,
                entry_range={'low': entry_low, 'high': entry_high},
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                lots_recommended=lots_recommended,
                shares_recommended=shares_recommended,
                position_value=position_value,
                risk_amount=rr.get('total_risk', 0),
                potential_profit_min=rr.get('net_profit_tp', 0),
                potential_profit_max=rr.get('total_reward', 0) * 1.5,
                risk_reward_ratio=rr.get('rr_ratio', 0),
                score=int(score * 100) if score <= 1 else int(score),
                risk_level=risk_level,
                reasons=reasons,
                warnings=warnings,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Error creating recommendation: {e}")
            return None
    
    def allocate_capital(self, results: List[Dict], mode: str = "scalping") -> CapitalAllocation:
        """
        Allocate capital across multiple stocks
        
        Args:
            results: Screening results
            mode: Trading mode
        
        Returns:
            CapitalAllocation with recommendations
        """
        try:
            recommendations = self.recommend_stocks(results, mode, self.max_positions)
            
            # Calculate allocation
            allocated = sum(r.position_value for r in recommendations)
            remaining = self.capital - allocated
            
            # Summary
            if recommendations:
                tickers = ", ".join([r.ticker for r in recommendations])
                summary = f"Alokasi untuk {len(recommendations)} saham: {tickers}. Total posisi Rp{allocated:,.0f} ({allocated/self.capital*100:.1f}%)"
            else:
                summary = "Tidak ada rekomendasi yang cocok untuk modal dan profil risiko Anda"
            
            return CapitalAllocation(
                total_capital=self.capital,
                allocated_capital=allocated,
                remaining_capital=remaining,
                num_positions=len(recommendations),
                recommendations=recommendations,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error allocating capital: {e}")
            return CapitalAllocation(
                total_capital=self.capital,
                allocated_capital=0,
                remaining_capital=self.capital,
                num_positions=0,
                recommendations=[],
                summary=f"Error: {str(e)}"
            )
    
    def format_recommendation_text(self, rec: TradeRecommendation) -> str:
        """Format a single recommendation as text"""
        lines = [
            "=" * 50,
            f"📊 {rec.ticker}",
            "=" * 50,
            f"Mode: {rec.mode}",
            f"Action: {rec.action}",
            f"Trend: {'BULLISH' if rec.score >= 70 else 'NEUTRAL'}",
            "",
            f"Entry: Rp{rec.entry_price:,.0f}",
            f"TP1: Rp{rec.tp1:,.0f} ({(rec.tp1/rec.entry_price-1)*100:+.1f}%)",
            f"TP2: Rp{rec.tp2:,.0f} ({(rec.tp2/rec.entry_price-1)*100:+.1f}%)",
            f"SL: Rp{rec.sl:,.0f} ({(rec.sl/rec.entry_price-1)*100:+.1f}%)",
            "",
            f"Lot yang direkomendasikan: {rec.lots_recommended} lot ({rec.shares_recommended} lembar)",
            f"Nilai posisi: Rp{rec.position_value:,.0f}",
            f"Risk per trade: Rp{abs(rec.risk_amount):,.0f}",
            f"Risk/Reward: 1:{rec.risk_reward_ratio:.1f}",
            "",
            f"Score: {rec.score}",
            f"Risiko: {rec.risk_level}",
            "",
            "Alasan:",
        ]
        
        for reason in rec.reasons[:5]:
            lines.append(f"  • {reason}")
        
        if rec.warnings:
            lines.append("")
            lines.append("⚠️ Peringatan:")
            for warning in rec.warnings[:3]:
                lines.append(f"  {warning}")
        
        lines.append("")
        lines.append(f"📝 {rec.notes}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def format_allocation_text(self, allocation: CapitalAllocation) -> str:
        """Format complete allocation as text"""
        lines = [
            "",
            "🚀 REKOMENDASI TRADING BERDASARKAN MODAL ANDA",
            "=" * 60,
            "",
            f"💰 Modal: Rp{allocation.total_capital:,.0f}",
            f"📊 Mode: SCALPING",
            f"📈 Jumlah Rekomendasi: {allocation.num_positions} saham",
            f"💵 Total Alokasi: Rp{allocation.allocated_capital:,.0f} ({allocation.allocated_capital/allocation.total_capital*100:.1f}%)",
            f"💸 Sisa Modal: Rp{allocation.remaining_capital:,.0f}",
            "",
        ]
        
        for i, rec in enumerate(allocation.recommendations, 1):
            lines.append(f"\n{'='*60}")
            lines.append(f"📌 REKOMENDASI #{i}: {rec.ticker}")
            lines.append(f"{'='*60}")
            lines.append(f"Mode: {rec.mode} | Action: {rec.action} | Risk: {rec.risk_level}")
            lines.append(f"Entry: Rp{rec.entry_price:,.0f}")
            lines.append(f"TP: Rp{rec.tp1:,.0f} (+{(rec.tp1/rec.entry_price-1)*100:.1f}%) / Rp{rec.tp2:,.0f} (+{(rec.tp2/rec.entry_price-1)*100:.1f}%)")
            lines.append(f"SL: Rp{rec.sl:,.0f} ({(rec.sl/rec.entry_price-1)*100:.1f}%)")
            lines.append(f"Lot: {rec.lots_recommended} ({rec.shares_recommended} lembar) = Rp{rec.position_value:,.0f}")
            lines.append(f"R:R = 1:{rec.risk_reward_ratio:.1f} | Score: {rec.score}")
            
            if rec.warnings:
                lines.append(f"⚠️ {rec.warnings[0]}")
        
        lines.append("\n" + "=" * 60)
        lines.append("⚠️ DISCLAIMER: Ini bukan saran investasi. Trade at your own risk.")
        lines.append("💡 Tips: Selalu gunakan Stop Loss dan jangan serakah!")
        lines.append("=" * 60)
        
        return "\n".join(lines)
