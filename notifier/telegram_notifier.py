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
        """
        Format signal message with HTML
        
        Args:
            result: Screener result
        
        Returns:
            Formatted HTML message
        """
        ticker = result.get('ticker', 'N/A')
        signal = result.get('signal', 'NONE')
        score = result.get('score', 0)
        price = result.get('latest_price', 0)
        entry_levels = result.get('entry_levels', {})
        reasons = result.get('reasons', [])
        patterns = result.get('patterns', {})
        ml_prob = result.get('ml_probability')
        
        # Signal emoji
        emoji_map = {
            'STRONG_AURA': '🔥',
            'WATCHLIST': '⭐',
            'POTENTIAL': '💡',
            'NONE': '❌',
        }
        emoji = emoji_map.get(signal, '📊')
        
        confidence = result.get('confidence', 0)
        parameter_count = result.get('parameter_count', 0)
        data_quality = result.get('data_quality', {})
        quality_score = data_quality.get('quality_score', 1.0)
        
        message = f"""
{emoji} <b>{ticker}</b> - {signal}
━━━━━━━━━━━━━━━━━━━━

💰 <b>Price:</b> {format_currency(price)}
📊 <b>Score:</b> {score:.2%}
🎯 <b>Confidence:</b> {confidence:.1%}
✅ <b>Parameters:</b> {parameter_count}/7
📈 <b>Data Quality:</b> {quality_score:.0%}
"""
        
        if ml_prob:
            message += f"🤖 <b>ML Probability:</b> {ml_prob:.2%}\n"
        
        # Data validation info
        if data_quality.get('issues'):
            message += f"\n⚠️ <b>Data Notes:</b>\n"
            for issue in data_quality['issues'][:2]:  # Max 2 issues
                message += f"• {issue}\n"
        
        message += "\n📈 <b>Entry Levels:</b>\n"
        if entry_levels:
            message += f"Entry: {format_currency(entry_levels.get('entry_low', 0))} - {format_currency(entry_levels.get('entry_high', 0))}\n"
            message += f"SL: {format_currency(entry_levels.get('stop_loss', 0))}\n"
            message += f"TP1: {format_currency(entry_levels.get('take_profit_1', 0))} (R:R {entry_levels.get('reward_risk_1', 0):.2f})\n"
            message += f"TP2: {format_currency(entry_levels.get('take_profit_2', 0))} (R:R {entry_levels.get('reward_risk_2', 0):.2f})\n"
        
        if reasons:
            message += "\n✅ <b>Reasons:</b>\n"
            for reason in reasons:
                message += f"• {reason}\n"
        
        # Patterns
        detected_patterns = [k for k, v in patterns.items() if v and (isinstance(v, bool) or (isinstance(v, dict) and v.get('detected')))]
        if detected_patterns:
            message += "\n🎯 <b>Patterns:</b>\n"
            for pattern in detected_patterns:
                message += f"• {pattern.replace('_', ' ').title()}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
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

