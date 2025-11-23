# ARA BOT V2 - Quick Start Guide

## 🚀 Quick Start (5 Menit)

> **💡 Untuk 900+ ticker, lihat [SCALING_GUIDE.md](SCALING_GUIDE.md) untuk optimasi**

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Bot
```bash
python main.py
```

Bot akan otomatis:
- Scan semua ticker IDX
- Generate signals
- Save results ke `results/`
- Generate charts ke `charts/`

### 3. Lihat Results
```bash
# JSON format
cat results/ara_latest.json

# CSV format (buka dengan Excel)
# results/ara_latest.csv
```

## 📱 Setup Telegram (Optional)

### 1. Buat Bot di Telegram
1. Chat dengan [@BotFather](https://t.me/botfather)
2. Kirim `/newbot`
3. Ikuti instruksi
4. Copy **Bot Token**

### 2. Dapatkan Chat ID
1. Chat dengan [@userinfobot](https://t.me/userinfobot)
2. Copy **Your Id**

### 3. Setup Environment
Buat file `.env`:
```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

### 4. Restart Bot
```bash
python main.py
```

## 🎯 Cara Kerja

### Flow Singkat:
```
1. Fetch Data → 2. Calculate Indicators → 3. Screen & Score
     ↓
4. Rank Results → 5. Generate Charts → 6. Send Notifications
```

### Signal Levels:
- 🔥 **STRONG_AURA**: Score >= 75% (Sangat Kuat)
- ⭐ **WATCHLIST**: Score >= 60% (Perlu Diperhatikan)
- 💡 **POTENTIAL**: Score >= 45% (Berpotensi)
- ❌ **NONE**: Di bawah threshold

## 📊 Interpretasi Results

### Score Components:
- **RVOL**: Relative Volume (volume vs rata-rata)
- **Bollinger Breakout**: Harga di atas upper band
- **MA Structure**: MA20 > MA50 > MA200 (trend bullish)
- **RSI**: Momentum 55-80 (strong but not overbought)
- **OBV Rising**: On-Balance Volume naik (akumulasi)
- **Above VWAP**: Harga di atas VWAP (bullish)
- **MA20 Slope**: MA20 naik cepat (akselerasi)

### Entry Levels:
- **Entry Zone**: Range harga untuk masuk
- **Stop Loss**: Batas kerugian maksimal
- **TP1 & TP2**: Target profit level 1 & 2
- **R:R**: Reward-to-Risk ratio

## 🔧 Customization

### Ubah Ticker List
Edit `config.py`:
```python
IDX_TICKER_LIST = [
    "BBCA", "BBRI", "BMRI",  # Tambah ticker di sini
    # ...
]
```

### Ubah Thresholds
Edit `config.py`:
```python
SCREENER_CONFIG = {
    "RVOL_THRESHOLD": 3.0,  # Ubah threshold
    "RSI_MIN": 55,
    "RSI_MAX": 80,
    # ...
}
```

### Filter Harga
Edit `config.py`:
```python
SCREENER_CONFIG = {
    "MIN_PRICE": 100,      # Minimum harga
    "MAX_PRICE": 10000,    # Maximum harga
    # ...
}
```

## 🌐 API Server

### Start Server:
```bash
uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```

### Endpoints:
- `GET /` - API info
- `POST /scan` - Scan tickers
- `GET /scan/ticker/{ticker}` - Scan single ticker
- `GET /results/latest` - Latest results
- `GET /charts/{ticker}` - Get chart
- `GET /dashboard` - Web dashboard

### Contoh:
```bash
# Get latest results
curl http://localhost:8000/results/latest

# Scan single ticker
curl http://localhost:8000/scan/ticker/BBCA.JK
```

## 🐛 Troubleshooting

### Error: "No module named 'yfinance'"
```bash
pip install yfinance
```

### Error: "Yahoo Finance timeout"
- Bot akan otomatis retry
- Gunakan cache untuk mengurangi API calls
- Cek koneksi internet

### Telegram tidak terkirim
- Pastikan `.env` file ada
- Cek `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID`
- Test dengan chat bot langsung

### Chart tidak ter-generate
- Pastikan `mplfinance` terinstall
- Cek disk space
- Cek log file untuk error details

## 📈 Tips Penggunaan

### 1. Scan Rutin
Jalankan bot setiap hari setelah market close:
```bash
# Linux/Mac (cron)
0 16 * * * cd /path/to/ara-bot && python main.py

# Windows (Task Scheduler)
# Buat scheduled task untuk run python main.py
```

### 2. Filter Results
Gunakan CSV untuk filter di Excel:
- Sort by score
- Filter by signal
- Filter by price range

### 3. Monitor Changes
Bot hanya kirim Telegram saat signal berubah
- Track signal transitions
- Avoid spam notifications

### 4. Use Cache
Aktifkan cache untuk scan cepat:
```python
# config.py
CACHE_CONFIG = {
    "ENABLED": True,
    "WARM_FETCH": True,  # Gunakan cache
}
```

## 🎓 Next Steps

1. **Baca README.md** untuk dokumentasi lengkap
2. **Baca ARCHITECTURE.md** untuk memahami struktur
3. **Baca FEATURES.md** untuk fitur tambahan
4. **Customize config.py** sesuai kebutuhan
5. **Monitor results** dan adjust thresholds

## 💡 Best Practices

1. **Backtest Strategy**: Test di historical data
2. **Risk Management**: Selalu gunakan stop loss
3. **Diversification**: Jangan all-in satu saham
4. **Research**: Lakukan riset fundamental juga
5. **Monitor**: Track performance secara rutin

---

**Happy Trading! 📈**

