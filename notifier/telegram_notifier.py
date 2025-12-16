"""Telegram notification with rich HTML formatting"""

import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from config import TELEGRAM_CONFIG
from utils.logger import setup_logger
from utils.helpers import format_currency, format_percentage
from datetime import datetime

logger = setup_logger(__name__)

class TelegramNotifier:
    """Send Telegram notifications with charts and formatted messages"""
    
    def __init__(self):
        """Initialize Telegram notifier"""
        self.bot_token = TELEGRAM_CONFIG["BOT_TOKEN"]
        self.chat_id = TELEGRAM_CONFIG["CHAT_ID"]
        self.enabled = TELEGRAM_CONFIG["ENABLED"] and bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send text message
        
        Args:
            text: Message text
            parse_mode: Parse mode (HTML or Markdown)
        
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Telegram message sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_photo(self, photo_path: Path, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Send photo with caption
        
        Args:
            photo_path: Path to image file
            caption: Caption text
            parse_mode: Parse mode
        
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": parse_mode,
                }
                
                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()
            
            logger.info(f"Telegram photo sent: {photo_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {e}")
            return False
    
    def format_signal_message(self, result: Dict) -> str:
        ticker = result.get('ticker', 'N/A')
        signal = result.get('signal', 'NONE')
        score = result.get('score', 0)
        price = result.get('latest_price', 0)
        entry_levels = result.get('entry_levels', {})
        reasons = result.get('reasons', [])
        patterns = result.get('patterns', {})
        confidence = result.get('confidence', 0)

        # 1. Header with Status
        emoji_map = {
            'SUPER_ALPHA': '🚀🚀', # Double Rocket for Super Setup
            'STRONG_AURA': '🔥',
            'WATCHLIST': '👀',
            'POTENTIAL': '💡',
            'NONE': '❌',
        }
        emoji = emoji_map.get(signal, '📊')
        
        message = f"<b>{emoji} {ticker} - {signal.replace('_', ' ')}</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"

        # 2. 🤖 AI Verdict (The "Analyst" View)
        # Construct dynamic verdict string
        verdict = "Observing..."
        if signal == 'SUPER_ALPHA':
            verdict = "<b>Strong Buy.</b> Confluence of Smart Money and Technical Breakout."
        elif signal == 'STRONG_AURA':
            verdict = "<b>Buy.</b> Strong momentum detected."
        elif signal == 'WATCHLIST':
            verdict = "<b>Watch.</b> Setup developing, wait for trigger."
        
        message += f"🤖 <b>AI Verdict:</b> {verdict}\n\n"

        # 3. Key Metrics
        message += f"💰 <b>Price:</b> {format_currency(price)}\n"
        message += f"📊 <b>Score:</b> {score:.0%} | 🎯 <b>Conf:</b> {confidence:.0%}\n"
        
        # 4. Risk Profile
        risk_level = "Medium"
        if confidence > 0.8: risk_level = "Low"
        elif confidence < 0.5: risk_level = "High"
        message += f"⚠️ <b>Risk Profile:</b> {risk_level}\n\n"

        # 5. Actionable Entry Plan
        message += "📝 <b>TRADING PLAN</b>\n"
        if entry_levels:
            entry_low = entry_levels.get('entry_low', 0)
            entry_high = entry_levels.get('entry_high', 0)
            sl = entry_levels.get('stop_loss', 0)
            tp1 = entry_levels.get('take_profit_1', 0)
            tp2 = entry_levels.get('take_profit_2', 0)
            
            # Smart formatting
            message += f"🔵 <b>BUY AREA:</b> {int(entry_low)} - {int(entry_high)}\n"
            message += f"🔴 <b>STOP LOSS:</b> {int(sl)} (Crucial)\n"
            message += f"🟢 <b>TARGET 1:</b> {int(tp1)}\n"
            message += f"🟢 <b>TARGET 2:</b> {int(tp2)}\n\n"

        # 6. Supporting Factors
        if reasons:
            message += "✅ <b>Analysis Factors:</b>\n"
            # Show top 5 reasons only to keep it clean
            for reason in reasons[:5]: 
                message += f"• {reason}\n"
            if len(reasons) > 5:
                message += f"• ...and {len(reasons)-5} more\n"
            message += "\n"

        # 7. Pattern Highlights
        detected_patterns = [k for k, v in patterns.items() if v and (isinstance(v, bool) or (isinstance(v, dict) and v.get('detected')))]
        if detected_patterns:
            message += "🧩 <b>Patterns Detected:</b> "
            clean_patterns = [p.replace('_', ' ').title() for p in detected_patterns]
            message += ", ".join(clean_patterns) + "\n\n"

        message += f"⏰ <i>Scanned at {datetime.now().strftime('%H:%M:%S')}</i>"

        return message
    
    def format_summary_message(self, results: List[Dict], top_n: int = 10) -> str:
        """
        Format summary message with top results
        
        Args:
            results: List of results
            top_n: Number of top results to include
        
        Returns:
            Formatted HTML message
        """
        top_results = results[:top_n]
        
        message = f"""
📊 <b>ARA BOT SCAN SUMMARY</b>
━━━━━━━━━━━━━━━━━━━━

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Top {len(top_results)} Results:</b>

"""
        
        for i, result in enumerate(top_results, 1):
            ticker = result.get('ticker', 'N/A')
            signal = result.get('signal', 'NONE')
            score = result.get('score', 0)
            confidence = result.get('confidence', 0)
            price = result.get('latest_price', 0)
            
            emoji_map = {
                'STRONG_AURA': '🔥',
                'WATCHLIST': '⭐',
                'POTENTIAL': '💡',
            }
            emoji = emoji_map.get(signal, '📊')
            
            message += f"{i}. {emoji} <b>{ticker}</b> - {signal}\n"
            message += f"   📊 Score: {score:.1%} | 🎯 Confidence: {confidence:.0%}\n"
            message += f"   💰 {format_currency(price)}\n\n"
        
        message += f"\n📈 Total scanned: {len(results)} tickers"
        
        return message
    
    def send_signal(self, result: Dict, chart_path: Optional[Path] = None) -> bool:
        """
        Send signal notification with chart
        
        Args:
            result: Screener result
            chart_path: Path to chart image
        
        Returns:
            True if successful
        """
        message = self.format_signal_message(result)
        
        if chart_path and chart_path.exists():
            return self.send_photo(chart_path, message)
        else:
            return self.send_message(message)
    
    def send_summary(self, results: List[Dict], top_n: int = None) -> bool:
        """
        Send summary notification
        
        Args:
            results: List of results
            top_n: Number of top results (uses config if None)
        
        Returns:
            True if successful
        """
        top_n = top_n or TELEGRAM_CONFIG["TOP_N_SUMMARY"]
        message = self.format_summary_message(results, top_n)
        return self.send_message(message)
    
    def format_technical_analysis(self, analysis) -> str:
        """
        Format multi-timeframe technical analysis for Telegram
        
        Args:
            analysis: MultiTFAnalysis object
        
        Returns:
            Formatted HTML message
        """
        ticker = analysis.ticker
        
        # Header
        message = f"📊 <b>TECHNICAL ANALYSIS: {ticker}</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Multi-TF Summary
        message += "🔍 <b>MULTI-TIMEFRAME SUMMARY</b>\n"
        
        tf_order = ["1D", "4h", "1h", "15m", "5m"]
        for tf in tf_order:
            if tf in analysis.timeframes:
                tf_analysis = analysis.timeframes[tf]
                message += f"├─ {tf}: {tf_analysis.trend_emoji} {tf_analysis.trend}"
                message += f" ({tf_analysis.summary[:25]}...)\n" if len(tf_analysis.summary) > 25 else f" ({tf_analysis.summary})\n"
        
        message += "\n"
        
        # Primary Analysis
        trend_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}.get(analysis.primary_trend, "⚪")
        conf_emoji = {"HIGH": "🎯", "MEDIUM": "⚡", "LOW": "⚠️"}.get(analysis.confluence, "❓")
        
        message += f"📈 <b>PRIMARY TREND:</b> {trend_emoji} {analysis.primary_trend}\n"
        message += f"🎯 <b>TRADING BIAS:</b> {analysis.bias}\n"
        message += f"📊 <b>CONFLUENCE:</b> {conf_emoji} {analysis.confluence} ({analysis.confidence:.0f}%)\n\n"
        
        # Key Levels
        message += "🔒 <b>KEY LEVELS</b>\n"
        message += f"├─ Resistance: {format_currency(analysis.key_resistance)}\n"
        message += f"├─ <b>Current:</b> {format_currency(analysis.current_price)}\n"
        message += f"└─ Support: {format_currency(analysis.key_support)}\n\n"
        
        # Trading Plan (if available)
        if analysis.trading_plan and analysis.bias != "AVOID":
            plan = analysis.trading_plan
            message += "📝 <b>TRADING PLAN</b>\n"
            message += f"🔵 <b>Entry:</b> {format_currency(plan.entry_low)} - {format_currency(plan.entry_high)}\n"
            message += f"🟢 <b>TP1:</b> {format_currency(plan.tp1)} (+{plan.tp1_pct:.1f}%)\n"
            message += f"🟢 <b>TP2:</b> {format_currency(plan.tp2)} (+{plan.tp2_pct:.1f}%)\n"
            message += f"🔴 <b>SL:</b> {format_currency(plan.sl)} (-{plan.sl_pct:.1f}%)\n"
            message += f"📈 <b>R:R:</b> 1:{plan.risk_reward:.1f}\n\n"
        
        # Warnings
        if analysis.warnings:
            message += "⚠️ <b>WARNINGS</b>\n"
            for warning in analysis.warnings:
                message += f"• {warning}\n"
            message += "\n"
        
        # Footer
        message += f"⏰ <i>Analyzed at {analysis.timestamp.strftime('%H:%M:%S')}</i>\n"
        message += "⚠️ <i>This is not financial advice. Trade at your own risk.</i>"
        
        return message
    
    def send_technical_analysis(self, analysis, chart_path: Optional[Path] = None) -> bool:
        """
        Send technical analysis notification with chart
        
        Args:
            analysis: MultiTFAnalysis object
            chart_path: Path to chart image
        
        Returns:
            True if successful
        """
        message = self.format_technical_analysis(analysis)
        
        if chart_path and chart_path.exists():
            return self.send_photo(chart_path, message)
        else:
            return self.send_message(message)
    
    def format_recommendation(self, recommendation: Dict) -> str:
        """
        Format capital-based recommendation for Telegram
        
        Args:
            recommendation: Recommendation dictionary
        
        Returns:
            Formatted HTML message
        """
        ticker = recommendation.get('ticker', 'N/A')
        mode = recommendation.get('mode', 'N/A')
        action = recommendation.get('action', 'N/A')
        risk = recommendation.get('risk', 'N/A')
        entry = recommendation.get('entry', 0)
        tp1 = recommendation.get('tp1', 0)
        tp1_pct = recommendation.get('tp1_pct', 0)
        tp2 = recommendation.get('tp2', 0)
        tp2_pct = recommendation.get('tp2_pct', 0)
        sl = recommendation.get('sl', 0)
        sl_pct = recommendation.get('sl_pct', 0)
        lot = recommendation.get('lot', 0)
        value = recommendation.get('value', 0)
        rr = recommendation.get('risk_reward', 0)
        bandar_timing = recommendation.get('bandar_timing', '')
        
        # Action emoji
        action_emoji = {"BUY": "🟢", "SCALP ONLY": "⚡", "AVOID": "❌"}.get(action, "📊")
        
        # Risk color
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "ACTIVE_GORENGAN": "⚠️"}.get(risk, "⚪")
        
        message = f"🚀 <b>{ticker}</b> - {mode.upper()}\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        message += f"📊 <b>Action:</b> {action_emoji} {action}\n"
        message += f"⚠️ <b>Risk Level:</b> {risk_emoji} {risk}\n\n"
        
        message += f"🔵 <b>Entry:</b> {format_currency(entry)}\n"
        message += f"🟢 <b>TP1:</b> {format_currency(tp1)} (+{tp1_pct:.1f}%)\n"
        message += f"🟢 <b>TP2:</b> {format_currency(tp2)} (+{tp2_pct:.1f}%)\n"
        message += f"🔴 <b>SL:</b> {format_currency(sl)} (-{sl_pct:.1f}%)\n\n"
        
        message += f"💰 <b>Position:</b> {lot} lot = {format_currency(value)}\n"
        message += f"📈 <b>R:R:</b> 1:{rr:.1f}\n"
        
        if bandar_timing:
            message += f"🎯 <b>Timing:</b> {bandar_timing}\n"
        
        return message
    
    def send_recommendation(self, recommendation: Dict, chart_path: Optional[Path] = None) -> bool:
        """
        Send recommendation notification
        
        Args:
            recommendation: Recommendation dictionary
            chart_path: Path to chart image
        
        Returns:
            True if successful
        """
        message = self.format_recommendation(recommendation)
        
        if chart_path and chart_path.exists():
            return self.send_photo(chart_path, message)
        else:
            return self.send_message(message)

