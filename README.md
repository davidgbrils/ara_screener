# ARA FULL BOT V2 - Advanced Multi-Bagger Scanner

Sistem Python profesional untuk mengidentifikasi saham Indonesia yang berpotensi menjadi **multi-bagger** dengan sinyal **ARA (Ara Brutal)**.

## 🎯 Fitur Utama

### 1. **Data Fetch & Normalization**
- Fetch OHLCV data dari Yahoo Finance untuk semua ticker .JK
- Auto-fix multi-index columns dari yfinance
- Retry mechanism untuk DNS errors
- Data caching dengan SQLite untuk performa cepat
- Fallback API support (IDX API, EODHD, TradingView)
- Warm fetch mode untuk scan cepat menggunakan cache

### 2. **Technical Indicators**
- **Moving Averages**: MA20, MA50, MA200
- **RSI**: RSI(14) untuk momentum
- **Bollinger Bands**: Upper, Mid, Lower dengan std dev 2
- **OBV**: On-Balance Volume untuk konfirmasi volume
- **VWAP**: Volume Weighted Average Price
- **ATR**: Average True Range untuk volatilitas
- **RVOL**: Relative Volume (volume vs rata-rata)
- **1-week percentage change**
- **MA20 slope acceleration**

### 3. **Advanced Screener & Scoring**
Sistem scoring yang mengidentifikasi saham dengan potensi multi-bagger:

**Kriteria Utama:**
- ✅ RVOL >= 3x (volume surge)
- ✅ Breakout di atas Bollinger upper band
- ✅ Bullish MA structure (MA20 > MA50 > MA200)
- ✅ RSI momentum 55-80
- ✅ OBV rising (akumulasi)
- ✅ Close > VWAP
- ✅ MA20 slope > 0.05 (akselerasi)

**Signal Levels:**
- 🔥 **STRONG_AURA**: Score >= 75%
- ⭐ **WATCHLIST**: Score >= 60%
- 💡 **POTENTIAL**: Score >= 45%
- ❌ **NONE**: Di bawah threshold

### 4. **Advanced Pattern Detection**
Deteksi pola-pola canggih untuk multi-bagger:

- **Parabolic Curve Detection**: Deteksi pergerakan parabolic
- **Volume Climax**: Deteksi volume exhaustion
- **VCP (Volatility Contraction Pattern)**: Pola kontraksi volatilitas
- **Darvas Box**: Pola breakout dari box
- **Pocket Pivot**: Volume surge pada hari naik
- **Gap-Up Detection**: Deteksi gap-up setup
- **Reaccumulation Base**: Deteksi base pattern

### 5. **Entry/Exit Engine**
Automatic calculation untuk:
- **Entry Zone**: entry_low & entry_high (2% zone)
- **Stop Loss**: Berdasarkan ATR atau support level
- **Take Profit 1 & 2**: Berdasarkan ATR multiplier
- **Reward-to-Risk Ratio**: Auto-calculated

### 6. **Professional Charting**
- Candlestick charts dengan mplfinance
- Volume bars
- Moving averages overlay
- Bollinger Bands overlay
- Entry/SL/TP markers
- Highlight breakout candles
- Auto-annotate patterns

### 7. **Telegram Notifications**
- Rich HTML formatted messages
- Chart images
- Entry/SL/TP levels
- Pattern detection results
- Summary report dengan top 10
- Hanya kirim saat signal berubah (anti-spam)

### 8. **State Management**
- Track signal changes
- Prevent duplicate notifications
- Historical state storage

### 9. **Multiprocessing Engine**
- Parallel processing untuk ratusan ticker
- ThreadPoolExecutor untuk I/O bound operations
- Async support untuk batch processing
- Memory-safe pandas operations

### 10. **FastAPI Web Server**
- RESTful API endpoints
- Web dashboard
- Real-time scan results
- Chart serving

### 11. **ML Scoring (Optional)**
- LightGBM model untuk prediksi ARA probability
- Feature engineering: RVOL, ATR expansion, MA structure, dll
- Combined score: Technical + ML probability

## 📁 Struktur Folder

```
ara-bot/
│
├── main.py                 # Main entry point (auto-run)
├── config.py              # Centralized configuration
├── requirements.txt       # Dependencies
├── README.md              # Documentation
│
├── data_fetch/            # Data fetching module
│   ├── __init__.py
│   ├── yahoo_fetcher.py
│   ├── fallback_fetcher.py
│   └── data_fetcher.py
│
├── data_cleaner/          # Data normalization
│   ├── __init__.py
│   └── normalizer.py
│
├── indicators/            # Technical indicators
│   ├── __init__.py
│   └── indicator_engine.py
│
├── screener/              # Screener & pattern detection
│   ├── __init__.py
│   ├── screener_engine.py
│   └── pattern_detector.py
│
├── ranking/               # Ranking & ML scoring
│   ├── __init__.py
│   ├── ranking_engine.py
│   └── ml_scorer.py
│
├── charts/                # Chart generation
│   ├── __init__.py
│   └── chart_engine.py
│
├── notifier/              # Telegram notifications
│   ├── __init__.py
│   └── telegram_notifier.py
│
├── state_manager/         # State management
│   ├── __init__.py
│   └── state_manager.py
│
├── cache_layer/           # Caching system
│   ├── __init__.py
│   ├── sqlite_cache.py
│   └── cache_manager.py
│
├── processing/            # Multiprocessing engine
│   ├── __init__.py
│   └── multiprocessing_engine.py
│
├── api/                   # FastAPI server
│   ├── __init__.py
│   └── api_server.py
│
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── logger.py
│   └── helpers.py
│
├── data/                  # Data storage
│   └── ara_bot.db        # SQLite database
│
├── results/               # Scan results
│   ├── ara_latest.json
│   ├── ara_latest.csv
│   └── prev_signals.json
│
├── charts/                # Generated charts
│   └── *.png
│
├── history/               # Historical scans
│   └── ara_YYYYMMDD.json
│
└── logs/                  # Log files
    └── ara_bot_v2.log
```

## 🚀 Instalasi

### 1. Clone atau Download Project

```bash
cd fixlaah
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables (Optional)

Buat file `.env` di root directory:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EODHD_API_KEY=your_eodhd_key_here  # Optional
```

**📱 Untuk setup Telegram bot lengkap, lihat [TELEGRAM_SETUP.md](TELEGRAM_SETUP.md)**

### 4. Konfigurasi

Edit `config.py` sesuai kebutuhan:
- Ticker list (IDX_TICKER_LIST)
- Screener thresholds
- Telegram settings
- Cache settings
- dll

## 💻 Cara Penggunaan

### Mode Auto-Run (Recommended)

```bash
python main.py
```

Bot akan:
1. Fetch data untuk semua ticker IDX
2. Calculate indicators
3. Screen dan score
4. Generate charts untuk signals
5. Save results
6. Send Telegram notifications (jika dikonfigurasi)

### Mode API Server

```bash
uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```

Akses:
- API: http://localhost:8000
- Dashboard: http://localhost:8000/dashboard
- Docs: http://localhost:8000/docs

### Endpoints API

- `POST /scan` - Scan tickers
- `GET /scan/ticker/{ticker}` - Scan single ticker
- `GET /results/latest` - Get latest results
- `GET /charts/{ticker}` - Get chart for ticker
- `GET /dashboard` - Web dashboard

### Programmatic Usage

```python
from main import ARABot

bot = ARABot()

# Scan all tickers
results = bot.run()

# Scan specific tickers
results = bot.run(tickers=['BBCA.JK', 'BBRI.JK'])

# Process single ticker
result = bot.process_ticker('BBCA.JK')
```

## 📊 Output

### JSON Results
`results/ara_latest.json`:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "count": 25,
  "results": [
    {
      "ticker": "BBCA.JK",
      "signal": "STRONG_AURA",
      "score": 0.85,
      "ml_probability": 0.78,
      "latest_price": 9500,
      "reasons": ["RVOL 3.5x", "Bollinger Breakout", ...],
      "patterns": {"parabolic": true, "volume_climax": true},
      "entry_levels": {
        "entry_low": 9300,
        "entry_high": 9700,
        "stop_loss": 9000,
        "take_profit_1": 10000,
        "take_profit_2": 11000,
        "reward_risk_1": 2.0,
        "reward_risk_2": 3.0
      }
    }
  ]
}
```

### CSV Results
`results/ara_latest.csv` - Spreadsheet-friendly format

### Charts
`charts/{TICKER}_{SIGNAL}.png` - Professional candlestick charts

### History
`history/ara_YYYYMMDD.json` - Daily historical scans

## ⚙️ Konfigurasi Lanjutan

### Screener Thresholds
Edit di `config.py`:
```python
SCREENER_CONFIG = {
    "RVOL_THRESHOLD": 3.0,
    "RSI_MIN": 55,
    "RSI_MAX": 80,
    "MA_SLOPE_MIN": 0.05,
    ...
}
```

### Scoring Weights
```python
SCORING_WEIGHTS = {
    "RVOL": 0.25,
    "BOLLINGER_BREAKOUT": 0.20,
    "MA_STRUCTURE": 0.15,
    ...
}
```

### Entry/Exit Levels
```python
ENTRY_CONFIG = {
    "ATR_MULTIPLIER_SL": 2.0,
    "ATR_MULTIPLIER_TP1": 3.0,
    "ATR_MULTIPLIER_TP2": 5.0,
    ...
}
```

## 🔧 Troubleshooting

### Yahoo Finance Errors
- Bot akan otomatis retry dengan delay
- Gunakan cache untuk mengurangi API calls
- Aktifkan fallback APIs jika tersedia

### Telegram Notifications Tidak Terkirim
- Pastikan `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID` sudah di-set
- Cek log file untuk error details

### Performance Issues
- Kurangi `MAX_WORKERS` di `MULTIPROCESSING_CONFIG`
- Aktifkan cache untuk warm fetch
- Filter ticker list untuk mengurangi jumlah scan

### Chart Generation Errors
- Pastikan mplfinance terinstall: `pip install mplfinance`
- Cek disk space untuk chart storage

## 📈 Fitur Tambahan yang Disarankan

### 1. Regime Filter
- Deteksi bull/bear market
- Adjust screener berdasarkan regime

### 2. Day-of-Week Pattern
- Analisis performa berdasarkan hari
- Filter signals berdasarkan hari optimal

### 3. Volatility Clusters
- Deteksi cluster volatilitas tinggi
- Identifikasi periode breakout potential

### 4. RS Line Indicator
- Relative Strength vs market
- Filter saham dengan RS tinggi

### 5. Backtesting Engine
- Test strategy historical
- Calculate win rate, avg return
- Optimize parameters

### 6. Database Integration
- PostgreSQL untuk production
- Real-time data streaming
- Advanced queries

## 🤖 ML Model Training (Optional)

Untuk mengaktifkan ML scoring:

1. Collect historical data dengan labels
2. Train model:
```python
from sklearn.ensemble import LightGBMClassifier
# ... training code
```
3. Save model ke `data/ml_models/ara_predictor.pkl`
4. Enable di `config.py`: `ML_CONFIG["ENABLED"] = True`

## 📝 Logging

Logs disimpan di `logs/ara_bot_v2.log` dengan rotation:
- Max size: 10MB
- Backup count: 5 files

## 🔒 Security

- Jangan commit `.env` file
- Jangan expose API key di code
- Gunakan environment variables
- Rate limiting untuk API endpoints

## 📄 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the project
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📧 Support

Untuk pertanyaan atau issues, buka GitHub Issues.

---

**Disclaimer**: Sistem ini adalah tool analisis teknis. Bukan saran investasi. Selalu lakukan riset sendiri dan konsultasi dengan financial advisor sebelum investasi.

