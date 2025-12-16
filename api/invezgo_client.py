"""
Invezgo API Client for Indonesian Stock Market Data

This module provides integration with Invezgo API for:
- Real-time stock data
- Order book data
- Intraday data
- Top Gainer/Loser
- Foreign flow (accumulation/distribution)
- Bandarmologi data
- Technical indicators
- Company information

API Documentation: https://invezgo.com (requires subscription)
"""

import os
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class InvezgoConfig:
    """Configuration for Invezgo API"""
    api_key: str
    base_url: str = "https://api.invezgo.com"
    timeout: int = 30
    enabled: bool = True


class InvezgoClient:
    """
    Client for Invezgo API - Indonesian Stock Market Data Provider
    
    Features:
    - Stock list and company information
    - Real-time order book
    - Intraday data (OHLCV)
    - Historical chart data
    - Top Gainer/Loser
    - Foreign flow analysis
    - Bandarmologi (bandar accumulation/distribution)
    - Technical indicators
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Invezgo API Client
        
        Args:
            api_key: Invezgo API Key (from env INVEZGO_API_KEY if not provided)
        """
        self.api_key = api_key or os.getenv("INVEZGO_API_KEY", "")
        self.base_url = "https://api.invezgo.com"
        self.timeout = 30
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            logger.warning("Invezgo API key not set. API features disabled.")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authorization"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """
        Make API request
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
        
        Returns:
            Response data or None on error
        """
        if not self.enabled:
            logger.warning("Invezgo API is not enabled. Set INVEZGO_API_KEY.")
            return None
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 204:
                logger.info(f"No data available for {endpoint}")
                return None
            elif response.status_code == 401:
                logger.error("Invezgo API: Authentication failed. Check API key.")
                return None
            elif response.status_code == 402:
                logger.error("Invezgo API: Payment required. Upgrade subscription.")
                return None
            elif response.status_code == 429:
                logger.error("Invezgo API: Rate limited. Wait and retry.")
                return None
            else:
                logger.error(f"Invezgo API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Invezgo API timeout for {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Invezgo API request error: {e}")
            return None
    
    # =========================================================================
    # STOCK DATA ENDPOINTS
    # =========================================================================
    
    def get_stock_list(self) -> Optional[List[Dict]]:
        """
        Get complete list of stocks listed on BEI (IDX)
        
        Returns:
            List of stocks with code, name, and logo
        """
        return self._make_request("/analysis/list/stock")
    
    def get_broker_list(self) -> Optional[List[Dict]]:
        """
        Get complete list of brokers/securities on BEI
        
        Returns:
            List of brokers with code and name
        """
        return self._make_request("/analysis/list/broker")
    
    def get_company_info(self, code: str) -> Optional[Dict]:
        """
        Get complete company information
        
        Args:
            code: Stock code (e.g., BBCA)
        
        Returns:
            Company information dict
        """
        return self._make_request(f"/analysis/information/{code}")
    
    # =========================================================================
    # MARKET DATA ENDPOINTS
    # =========================================================================
    
    def get_top_gainer_loser(self, date: str = None, limit: int = 10) -> Optional[Dict]:
        """
        Get top gainer and loser stocks for a date
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            limit: Number of results
        
        Returns:
            Dict with 'gain' and 'loss' lists
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        return self._make_request("/analysis/top/change", {"date": date, "limit": limit})
    
    def get_top_foreign_flow(self, date: str = None, limit: int = 10) -> Optional[Dict]:
        """
        Get top foreign accumulation and distribution
        
        Args:
            date: Date in YYYY-MM-DD format
            limit: Number of results
        
        Returns:
            Dict with 'accum' and 'dist' lists
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        return self._make_request("/analysis/top/foreign", {"date": date, "limit": limit})
    
    def get_top_bandarmologi(self, date: str = None, limit: int = 10) -> Optional[Dict]:
        """
        Get top bandar accumulation and distribution
        
        Args:
            date: Date in YYYY-MM-DD format
            limit: Number of results
        
        Returns:
            Dict with 'accum' and 'dist' lists
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        return self._make_request("/analysis/top/accumulation", {"date": date, "limit": limit})
    
    # =========================================================================
    # REAL-TIME & INTRADAY DATA
    # =========================================================================
    
    def get_order_book(self, code: str, market: str = "RG") -> Optional[Dict]:
        """
        Get order book (bid/offer) for a stock
        
        Args:
            code: Stock code (e.g., BBCA)
            market: Market type - RG (Regular), NG (Negotiated), TN (Tunai)
        
        Returns:
            Dict with 'code', 'bid', and 'offer' lists
        """
        return self._make_request(f"/analysis/order-book/{code}", {"market": market})
    
    def get_intraday_chart(self, code: str, market: str = "RG") -> Optional[List[Dict]]:
        """
        Get intraday chart data (OHLCV per minute)
        
        Args:
            code: Stock code
            market: Market type
        
        Returns:
            List of OHLCV data points
        """
        return self._make_request(f"/analysis/intraday/{code}", {"market": market})
    
    def get_intraday_data(self, code: str, market: str = "RG") -> Optional[Dict]:
        """
        Get current intraday summary data
        
        Args:
            code: Stock code
            market: Market type
        
        Returns:
            Dict with OHLCV, bid/offer, and summary data
        """
        return self._make_request(f"/analysis/intraday-data/{code}", {"market": market})
    
    # =========================================================================
    # HISTORICAL DATA
    # =========================================================================
    
    def get_ohlcv_chart(self, code: str, from_date: str = None, 
                        to_date: str = None) -> Optional[List[Dict]]:
        """
        Get historical OHLCV chart data
        
        Args:
            code: Stock code
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            List of OHLCV data
        """
        if not from_date:
            from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        return self._make_request(
            f"/analysis/chart/stock/{code}",
            {"from": from_date, "to": to_date}
        )
    
    def get_indicator_chart(self, code: str, indicator: str = "bdm",
                           from_date: str = None, to_date: str = None) -> Optional[List[Dict]]:
        """
        Get indicator chart data
        
        Args:
            code: Stock code
            indicator: Indicator type (bdm, rsi, macd, etc.)
            from_date: Start date
            to_date: End date
        
        Returns:
            List of indicator values
        """
        if not from_date:
            from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        return self._make_request(
            f"/analysis/chart/stock/{indicator}/{code}",
            {"from": from_date, "to": to_date}
        )
    
    # =========================================================================
    # SHAREHOLDER DATA
    # =========================================================================
    
    def get_shareholder(self, code: str) -> Optional[List[Dict]]:
        """
        Get shareholder composition
        
        Args:
            code: Stock code
        
        Returns:
            List of shareholders with percentage
        """
        return self._make_request(f"/analysis/shareholder/{code}")
    
    def get_shareholder_ksei(self, code: str, range_months: int = 6) -> Optional[List[Dict]]:
        """
        Get KSEI shareholder classification (foreign vs domestic)
        
        Args:
            code: Stock code
            range_months: Number of months (max 12)
        
        Returns:
            List of shareholder data by investor type
        """
        return self._make_request(
            f"/analysis/shareholder/ksei/{code}",
            {"range": range_months}
        )
    
    # =========================================================================
    # HELPER METHODS FOR BOT INTEGRATION
    # =========================================================================
    
    def get_stock_summary(self, code: str) -> Optional[Dict]:
        """
        Get comprehensive stock summary for screening
        
        Combines: intraday data, order book, company info
        
        Args:
            code: Stock code (without .JK suffix)
        
        Returns:
            Comprehensive stock data dict
        """
        # Remove .JK suffix if present
        code = code.replace(".JK", "").upper()
        
        summary = {
            'code': code,
            'intraday': None,
            'order_book': None,
            'info': None,
            'error': None
        }
        
        try:
            # Get intraday data
            summary['intraday'] = self.get_intraday_data(code)
            
            # Get order book
            summary['order_book'] = self.get_order_book(code)
            
            # Get company info (cached separately)
            summary['info'] = self.get_company_info(code)
            
        except Exception as e:
            summary['error'] = str(e)
            logger.error(f"Error getting stock summary for {code}: {e}")
        
        return summary
    
    def analyze_order_book(self, code: str) -> Optional[Dict]:
        """
        Analyze order book to detect buying/selling pressure
        
        Args:
            code: Stock code
        
        Returns:
            Analysis with bid/offer pressure
        """
        order_book = self.get_order_book(code)
        if not order_book:
            return None
        
        bids = order_book.get('bid', [])
        offers = order_book.get('offer', [])
        
        if not bids or not offers:
            return None
        
        # Calculate total bid/offer volume
        total_bid_lot = sum(b.get('bid1lot', 0) for b in bids)
        total_offer_lot = sum(o.get('offer1lot', 0) for o in offers)
        
        # Bid/offer ratio
        ratio = total_bid_lot / total_offer_lot if total_offer_lot > 0 else 0
        
        # Determine pressure
        if ratio > 1.5:
            pressure = "STRONG_BUY"
        elif ratio > 1.1:
            pressure = "BUY"
        elif ratio < 0.5:
            pressure = "STRONG_SELL"
        elif ratio < 0.9:
            pressure = "SELL"
        else:
            pressure = "NEUTRAL"
        
        # Best bid/offer
        best_bid = bids[0] if bids else {}
        best_offer = offers[0] if offers else {}
        
        return {
            'code': code,
            'best_bid': best_bid.get('bid1price', 0),
            'best_offer': best_offer.get('offer1price', 0),
            'spread': best_offer.get('offer1price', 0) - best_bid.get('bid1price', 0),
            'total_bid_lot': total_bid_lot,
            'total_offer_lot': total_offer_lot,
            'bid_offer_ratio': ratio,
            'pressure': pressure
        }
    
    def get_market_movers(self, date: str = None) -> Optional[Dict]:
        """
        Get market movers combining gainer, loser, foreign flow, and bandarmologi
        
        Args:
            date: Date in YYYY-MM-DD
        
        Returns:
            Combined market movers data
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        
        result = {
            'date': date,
            'gainers': [],
            'losers': [],
            'foreign_accum': [],
            'foreign_dist': [],
            'bandar_accum': [],
            'bandar_dist': []
        }
        
        # Get top gainer/loser
        top_change = self.get_top_gainer_loser(date, limit=20)
        if top_change:
            result['gainers'] = top_change.get('gain', [])
            result['losers'] = top_change.get('loss', [])
        
        # Get foreign flow
        foreign = self.get_top_foreign_flow(date, limit=20)
        if foreign:
            result['foreign_accum'] = foreign.get('accum', [])
            result['foreign_dist'] = foreign.get('dist', [])
        
        # Get bandarmologi
        bandar = self.get_top_bandarmologi(date, limit=20)
        if bandar:
            result['bandar_accum'] = bandar.get('accum', [])
            result['bandar_dist'] = bandar.get('dist', [])
        
        return result
    
    def detect_unusual_activity(self, code: str) -> Optional[Dict]:
        """
        Detect unusual activity on a stock
        
        Combines: order book, intraday, and bandarmologi signals
        
        Args:
            code: Stock code
        
        Returns:
            Unusual activity indicators
        """
        code = code.replace(".JK", "").upper()
        
        result = {
            'code': code,
            'is_unusual': False,
            'signals': [],
            'order_book_pressure': None,
            'intraday_data': None
        }
        
        # Get intraday data
        intraday = self.get_intraday_data(code)
        if intraday:
            result['intraday_data'] = intraday
            
            # Check for high volume
            volume = intraday.get('volume', 0)
            freq = intraday.get('freq', 0)
            
            if freq > 1000:
                result['signals'].append(f"High frequency: {freq} transactions")
                result['is_unusual'] = True
        
        # Analyze order book
        ob_analysis = self.analyze_order_book(code)
        if ob_analysis:
            result['order_book_pressure'] = ob_analysis
            
            if ob_analysis['pressure'] in ['STRONG_BUY', 'STRONG_SELL']:
                result['signals'].append(f"Order book: {ob_analysis['pressure']}")
                result['is_unusual'] = True
        
        return result


# Singleton instance
_invezgo_client = None

def get_invezgo_client() -> InvezgoClient:
    """Get or create Invezgo client singleton"""
    global _invezgo_client
    if _invezgo_client is None:
        _invezgo_client = InvezgoClient()
    return _invezgo_client
