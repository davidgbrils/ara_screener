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
    "ATR_PCT_RANGE": (0.015, 0.06),
}

# Relaxed screening adjustments (optional)
RELAXED_SCREENING = {
    "SCREENER_OVERRIDES": {
        "RVOL_THRESHOLD": 2.5,
        "RSI_MIN": 50,
        "RSI_MAX": 85,
        "MA_SLOPE_MIN": 0.03,
        "MIN_PRICE": 20,
        "MIN_VOLUME": 700000,
    },
    "SIGNAL_THRESHOLDS": {
        "STRONG_AURA": 0.70,
        "WATCHLIST": 0.55,
        "POTENTIAL": 0.40,
    },
    "MIN_CONFIDENCE": 0.55,
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
        "macd_hist",
        "atr_pct",
        "ma200_slope",
        "dist_52w_high",
        "bb_width",
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
    "VOLUME_DRY_UP": True,
    "MONEY_FLOW": True,
    "MARKET_REGIME": True,
}

# =============================================================================
# GORENGAN & UMA & INSIDER ACTIVITY DETECTION CONFIG
# =============================================================================

# Gorengan Detection Configuration
GORENGAN_CONFIG = {
    # Filter Awal - Ciri Saham Gorengan
    "MAX_PRICE_GORENGAN": 500,           # Harga < Rp 500
    "MAX_MARKET_CAP": 1_000_000_000_000, # Market Cap < 1T (1 Triliun)
    "MAX_FREE_FLOAT_PCT": 30,            # Free Float < 30% (requires external data)
    
    # Volume Ratio Thresholds
    "VR_ABNORMAL": 5,                    # Volume Ratio >= 5 = tidak normal
    "VR_SUSPICIOUS": 10,                 # Volume Ratio >= 10 = sangat mencurigakan
    
    # Price Spike Thresholds (in %)
    "PRICE_SPIKE_GORENGAN": 10,          # >= 10% = indikasi gorengan
    "PRICE_SPIKE_UMA": 20,               # >= 20% = kandidat UMA
    
    # UMA Detection
    "BODY_RATIO_THRESHOLD": 0.7,         # Body Ratio >= 0.7 (candle tidak wajar)
    "MULTI_DAY_PUMP_PCT": 25,            # >= 25% dalam N hari
    "MULTI_DAY_PUMP_DAYS": 3,            # Periode hari untuk multi-day pump
    
    # Insider Activity Detection
    "BROKER_CONCENTRATION_MIN": 30,      # >= 30% = bandar mulai kerja
    "BROKER_CONCENTRATION_HIGH": 50,     # >= 50% = insider/big player
    
    # Smart Money Detection (High Volume + Low Price Change)
    "HIGH_VOL_LOW_ATR_VOL_MULT": 2.0,    # Volume > 2x average
    "HIGH_VOL_LOW_ATR_RANGE_MAX": 2.0,   # Price Range % < 2%
    
    # Distribution Detection
    "RSI_DIVERGENCE_LOOKBACK": 14,       # Lookback period for RSI divergence
    "FAKE_BREAKOUT_TOLERANCE": 0.02,     # 2% tolerance for fake breakout
}

# Gorengan Scoring Weights (based on user specification)
GORENGAN_SCORING = {
    "VR_5": 2,              # Volume Ratio >= 5: +2 skor
    "VR_10": 4,             # Volume Ratio >= 10: +4 skor (replaces VR_5)
    "PRICE_SPIKE_10": 2,    # Price Spike >= 10%: +2 skor
    "PRICE_SPIKE_20": 4,    # Price Spike >= 20%: +4 skor (replaces PRICE_SPIKE_10)
    "BROKER_CONC_40": 3,    # Broker Concentration >= 40%: +3 skor
    "SIDEWAYS_NETBUY": 3,   # Sideways + Net Buy: +3 skor
    "RSI_DIVERGENCE": 2,    # RSI Divergence: +2 skor
    "BODY_RATIO_HIGH": 2,   # Body Ratio >= 0.7: +2 skor
    "MULTI_DAY_PUMP": 3,    # Multi-Day Pump >= 25%: +3 skor
    "HIGH_VOL_LOW_ATR": 2,  # Smart Money signal: +2 skor
    "LOW_PRICE": 1,         # Harga < 500: +1 skor
    "LOW_MARKET_CAP": 1,    # Market Cap < 1T: +1 skor
    "FAKE_BREAKOUT": 2,     # Fake Breakout detected: +2 skor
}

# Gorengan Risk Level Interpretation
GORENGAN_RISK_LEVELS = {
    "ACTIVE_GORENGAN": 8,    # Skor >= 8 → Saham Gorengan Aktif
    "HIGH_UMA_RISK": 12,     # Skor >= 12 → Risiko UMA tinggi
    "INSIDER_STRONG": 15,    # Skor >= 15 → Insider/Bandar kuat
}

# =============================================================================
# MULTI-MODE SCREENING CONFIGURATION
# =============================================================================

# Mode A - BPJS / Saham Sehat (Aman & Stabil)
MODE_BPJS_CONFIG = {
    "MARKET_CAP_MIN": 5_000_000_000_000,  # > 5T
    "AVG_VOL_20D_MIN": 10_000_000,         # > 10M
    "FREE_FLOAT_MIN": 35,                   # > 35% (estimated)
    "RSI_MIN": 50,
    "RSI_MAX": 65,
    "ATR_STABLE": True,
    "EMA_STRUCTURE": True,  # EMA5 > EMA20
    "PRICE_ABOVE_MA20": True,
}

# Mode B - Potensi ARA (Fast Move)
MODE_ARA_CONFIG = {
    "PRICE_MAX": 500,
    "VOLUME_RATIO_MIN": 7,         # Vol / Avg20 >= 7
    "GAP_UP_MIN": 3,               # Gap Up >= 3%
    "CLOSE_NEAR_HIGH_PCT": 90,     # Close >= 90% dari range
    "BODY_CANDLE_MIN": 70,         # Body >= 70%
    "ATR_RISING": True,
    "BREAK_RESISTANCE_20D": True,
    "NOT_ARA_TODAY": True,         # Belum ARA hari ini
    "NO_UMA": True,                # Tidak ada UMA resmi
}

# Mode C - Multi-Bagger Awal (Belum Naik Jauh)
MODE_MULTIBAGGER_CONFIG = {
    "PRICE_GAIN_FROM_BASE_MAX": 50,  # < 50% dari base
    "VOLUME_GRADUAL_INCREASE": True,  # Volume naik bertahap
    "RSI_MIN": 45,
    "RSI_MAX": 60,
    "EPS_GROWTH_MIN": 0,           # Positive
    "DER_MAX": 1.5,
    "ROE_MIN": 10,                 # > 10%
    "NOT_FREQUENT_ARA": True,      # Tidak sering ARA
}

# Mode D - Scalping Intraday (Cuan Cepat)
MODE_SCALPING_CONFIG = {
    "VOLUME_INTRADAY_MULT": 2,     # > 2x jam sebelumnya (proxy: daily surge)
    "VWAP_BREAK_RETEST": True,
    "EMA_STRUCTURE": True,          # EMA5 > EMA8 > EMA20
    "RSI_MIN": 55,
    "RSI_MAX": 70,
    "SPREAD_THIN": True,            # Spread tipis
    "TP_PCT_MIN": 1,
    "TP_PCT_MAX": 3,
    "SL_PCT_MIN": 0.5,
    "SL_PCT_MAX": 1,
}

# Mode E - Gorengan & UMA Filter (Already implemented in GORENGAN_CONFIG)
# Uses GORENGAN_CONFIG defined above

# =============================================================================
# ENHANCED SCORING SYSTEM (WAJIB ADA)
# =============================================================================

ENHANCED_SCORING = {
    # Positive factors
    "VOLUME_RATIO_5": +2,           # Volume Ratio >= 5
    "VOLUME_RATIO_10": +4,          # Volume Ratio >= 10
    "BROKER_DOMINANT_40": +3,       # Broker > 40% volume
    "BREAK_RESISTANCE_VALID": +2,   # Break resistance valid
    "RSI_HEALTHY": +1,              # RSI dalam zona sehat
    "EMA_STRUCTURE_BULLISH": +2,    # EMA5 > EMA8 > EMA20
    "ABOVE_VWAP": +1,               # Close > VWAP
    "BODY_CANDLE_STRONG": +1,       # Body > 70%
    "CLOSE_NEAR_HIGH": +1,          # Close >= 90% range
    "GAP_UP": +1,                   # Gap up
    
    # Negative factors
    "DIVERGENCE": -2,               # RSI Divergence
    "DISTRIBUTION": -3,             # Distribusi broker
    "FAKE_BREAKOUT": -2,            # Fake breakout
    "ALREADY_ARA": -5,              # Sudah ARA hari ini
    "UMA_ACTIVE": -3,               # Ada UMA resmi
    "OVERBOUGHT": -1,               # RSI > 80
    "OVERSOLD_MOMENTUM_LOSS": -2,   # RSI < 30 dan turun
}

SCORE_INTERPRETATION = {
    "HIGH_OPPORTUNITY": 10,    # Skor >= 10 → High Opportunity
    "STRONG_MOMENTUM": 14,     # Skor >= 14 → Strong Momentum
    "AVOID_THRESHOLD": 5,      # Skor < 5 → Avoid
}

# =============================================================================
# CAPITAL ADVISOR CONFIGURATION
# =============================================================================

CAPITAL_ADVISOR_CONFIG = {
    "LOT_SIZE": 100,               # Standard lot size IDX
    "MAX_POSITION_PCT": 20,        # Max 20% per posisi
    "RISK_PER_TRADE_PCT": 2,       # Risk 2% per trade
    "MIN_CAPITAL_SCALPING": 5_000_000,    # Min 5jt untuk scalping
    "MIN_CAPITAL_SWING": 10_000_000,      # Min 10jt untuk swing
    "COMMISSION_PCT": 0.15,         # Komisi 0.15%
    "TAX_SELL_PCT": 0.1,           # Pajak jual 0.1%
}

# Fundamentals settings (optional)
FUNDAMENTALS_CONFIG = {
    "ENABLED": False,
    "SOURCE": "CSV",
    "CSV_PATH": DATA_DIR / "fundamentals.csv",
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

