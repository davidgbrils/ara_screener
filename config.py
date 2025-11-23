"""
ARA FULL BOT V2 - Configuration Module
Centralized configuration for all modules
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import timedelta

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use environment variables directly
    pass

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = BASE_DIR / "charts"
HISTORY_DIR = BASE_DIR / "history"
CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, RESULTS_DIR, CHARTS_DIR, HISTORY_DIR, CACHE_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True)

# Database
DB_PATH = DATA_DIR / "ara_bot.db"

# Yahoo Finance settings
YAHOO_FINANCE_SETTINGS = {
    "period": "1y",  # 1 year of data
    "interval": "1d",  # Daily candles
    "retry_count": 3,
    "retry_delay": 2,  # seconds
    "timeout": 30,
}

# IDX ticker list (Indonesian stocks)
IDX_TICKER_SUFFIX = ".JK"
IDX_TICKER_LIST = [
    "BBCA", "BBRI", "BMRI", "BNII", "BBNI", "BSDE", "BJBR", "BJTM", "BNGA",
    "BTPN", "INDF", "ICBP", "INTP", "KLBF", "UNVR", "TLKM", "EXCL", "FREN",
    "ISAT", "TLKM", "ASII", "AUTO", "GGRM", "HMSP", "INDF", "INTP", "JPFA",
    "JSMR", "KLBF", "LSIP", "MYOR", "PGAS", "PTBA", "SMGR", "SRIL", "TKIM",
    "UNVR", "WIKA", "WSKT", "ADRO", "ANTM", "BUMI", "BYAN", "GOLD", "HRUM",
    "INCO", "MDKA", "PGEO", "PTBA", "SMCB", "TINS", "TOBA", "UNSP", "WINS",
    # Add more tickers as needed
]

# Technical Indicator Settings
INDICATOR_CONFIG = {
    "MA_PERIODS": [20, 50, 200],
    "RSI_PERIOD": 14,
    "BB_PERIOD": 20,
    "BB_STD": 2,
    "ATR_PERIOD": 14,
    "OBV_ENABLED": True,
    "VWAP_ENABLED": True,
    "MIN_HISTORY_DAYS": 200,  # Minimum days for valid indicators
}

# Screener Settings
SCREENER_CONFIG = {
    "RVOL_THRESHOLD": 3.0,
    "RSI_MIN": 55,
    "RSI_MAX": 80,
    "MA_SLOPE_MIN": 0.05,
    "MIN_PRICE": 50,  # Minimum price in IDR (optional filter)
    "MAX_PRICE": 50000,  # Maximum price in IDR (optional filter)
    "MIN_VOLUME": 1000000,  # Minimum daily volume
}

# Scoring Weights
SCORING_WEIGHTS = {
    "RVOL": 0.25,
    "BOLLINGER_BREAKOUT": 0.20,
    "MA_STRUCTURE": 0.15,
    "RSI_MOMENTUM": 0.15,
    "OBV_RISING": 0.10,
    "VWAP_POSITION": 0.10,
    "MA_SLOPE": 0.05,
}

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    "STRONG_AURA": 0.75,
    "WATCHLIST": 0.60,
    "POTENTIAL": 0.45,
    "NONE": 0.0,
}

# Entry/Exit Engine Settings
ENTRY_CONFIG = {
    "ATR_MULTIPLIER_SL": 2.0,
    "ATR_MULTIPLIER_TP1": 3.0,
    "ATR_MULTIPLIER_TP2": 5.0,
    "ENTRY_ZONE_PCT": 0.02,  # 2% entry zone
    "SUPPORT_RESISTANCE_WINDOW": 20,  # Days to look for S/R
}

# Multiprocessing Settings
MULTIPROCESSING_CONFIG = {
    "MAX_WORKERS": 16,  # Increased for 900+ tickers
    "CHUNK_SIZE": 100,  # Larger chunks for efficiency
    "USE_ASYNC": True,
    "BATCH_SIZE": 20,  # Larger batches
    "PROGRESS_UPDATE_INTERVAL": 50,  # Update progress every N tickers
    "SAVE_INTERMEDIATE_RESULTS": True,  # Save results incrementally
    "RESUME_CAPABILITY": True,  # Enable resume from checkpoint
}

# Chart Settings
CHART_CONFIG = {
    "FIG_SIZE": (14, 10),
    "DPI": 100,
    "STYLE": "yahoo",
    "SHOW_VOLUME": True,
    "SHOW_INDICATORS": True,
    "ANNOTATE_BREAKOUTS": True,
    "ANNOTATE_VOLUME": True,
}

# Telegram Settings
TELEGRAM_CONFIG = {
    "BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
    "ENABLED": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
    "SEND_ON_SIGNAL_CHANGE": False,  # Disabled - only send top 10
    "SEND_SUMMARY": True,
    "TOP_N_SUMMARY": 10,  # Only send top 10
    "SEND_CHARTS": True,  # Send charts for top 10
    "MIN_CONFIDENCE": 0.65,  # Minimum confidence for sending
    "HTML_FORMAT": True,
}

# Cache Settings
CACHE_CONFIG = {
    "ENABLED": True,
    "TTL_HOURS": 24,
    "USE_SQLITE": True,
    "WARM_FETCH": True,  # Use cache for faster scans
}

# API Server Settings
API_CONFIG = {
    "HOST": "0.0.0.0",
    "PORT": 8000,
    "DEBUG": False,
    "CORS_ORIGINS": ["*"],
}

# ML Model Settings (Optional)
ML_CONFIG = {
    "ENABLED": False,
    "MODEL_PATH": DATA_DIR / "ml_models" / "ara_predictor.pkl",
    "FEATURES": [
        "rvol",
        "atr_expansion",
        "ma_structure_score",
        "bollinger_breakout",
        "obv_divergence",
        "vwap_distance",
        "rsi_zone",
    ],
    "RETRAIN_INTERVAL_DAYS": 30,
}

# Logging Settings
LOGGING_CONFIG = {
    "LEVEL": "INFO",
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "FILE": LOG_DIR / "ara_bot_v2.log",
    "MAX_BYTES": 10 * 1024 * 1024,  # 10MB
    "BACKUP_COUNT": 5,
}

# Advanced Pattern Detection
PATTERN_CONFIG = {
    "PARABOLIC_DETECTION": True,
    "VOLUME_CLIMAX": True,
    "VCP_DETECTION": True,
    "DARVAS_BOX": True,
    "POCKET_PIVOT": True,
    "GAP_UP_DETECTION": True,
    "REACCUMULATION_BASE": True,
}

# Fallback API Settings
FALLBACK_APIS = {
    "IDX_API": {
        "ENABLED": False,
        "BASE_URL": "https://api.idx.co.id",
    },
    "EODHD": {
        "ENABLED": False,
        "API_KEY": os.getenv("EODHD_API_KEY", ""),
        "BASE_URL": "https://eodhistoricaldata.com/api",
    },
    "TRADINGVIEW": {
        "ENABLED": False,
        "BASE_URL": "https://scanner.tradingview.com",
    },
}

@dataclass
class AppConfig:
    """Application configuration dataclass"""
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    charts_dir: Path = CHARTS_DIR
    history_dir: Path = HISTORY_DIR
    cache_dir: Path = CACHE_DIR
    log_dir: Path = LOG_DIR
    db_path: Path = DB_PATH
    
    def __post_init__(self):
        """Ensure all directories exist"""
        for dir_path in [
            self.data_dir, self.results_dir, self.charts_dir,
            self.history_dir, self.cache_dir, self.log_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Global config instance
app_config = AppConfig()

