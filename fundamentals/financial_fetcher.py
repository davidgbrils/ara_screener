import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from config import FUNDAMENTALS_CONFIG, DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FinancialFetcher:
    def __init__(self):
        self.enabled = FUNDAMENTALS_CONFIG.get("ENABLED", False)
        self.source = FUNDAMENTALS_CONFIG.get("SOURCE", "CSV")
        self.csv_path = Path(FUNDAMENTALS_CONFIG.get("CSV_PATH", DATA_DIR / "fundamentals.csv"))
        self.cache: Dict[str, Dict] = {}
        self._loaded = False

    def _load_csv(self):
        if self._loaded:
            return
        if not self.csv_path.exists():
            logger.warning(f"Fundamentals CSV not found: {self.csv_path}")
            self._loaded = True
            return
        try:
            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                ticker = str(row.get('ticker') or row.get('symbol') or '').strip()
                if not ticker:
                    continue
                data = {
                    'per': self._to_float(row.get('per')),
                    'pbv': self._to_float(row.get('pbv')),
                    'roe': self._to_float(row.get('roe')),
                    'market_cap': self._to_float(row.get('market_cap')),
                    'bid_ask_spread_pct': self._to_float(row.get('bid_ask_spread_pct')),
                    'number_of_trades': self._to_float(row.get('number_of_trades')),
                    'free_float_pct': self._to_float(row.get('free_float_pct')),
                }
                self.cache[ticker] = data
            self._loaded = True
            logger.info(f"Loaded fundamentals for {len(self.cache)} tickers")
        except Exception as e:
            logger.error(f"Error loading fundamentals CSV: {e}")
            self._loaded = True

    def _to_float(self, v):
        try:
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    def get(self, ticker: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        self._load_csv()
        return self.cache.get(ticker)

