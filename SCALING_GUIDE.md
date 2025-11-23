# 📈 Scaling Guide - Handle 900+ Tickers

Panduan untuk mengoptimalkan ARA Bot untuk menangani 900+ ticker dengan efisien.

## 🚀 Optimasi yang Sudah Diterapkan

### 1. **Progress Tracking**
- Progress bar real-time
- ETA calculation
- Update setiap 50 ticker (configurable)

### 2. **Resume Capability**
- Checkpoint system
- Resume dari checkpoint jika terhenti
- Intermediate results saving

### 3. **Parallel Processing**
- 16 workers (default, bisa diubah)
- ThreadPoolExecutor untuk I/O bound
- Batch processing

### 4. **Incremental Saving**
- Save results setiap 100 ticker
- Prevent data loss jika crash
- Intermediate JSON file

### 5. **Caching Strategy**
- Warm fetch menggunakan cache
- Reduce API calls
- TTL-based invalidation

## ⚙️ Konfigurasi untuk 900+ Ticker

### Edit `config.py`:

```python
# Multiprocessing Settings
MULTIPROCESSING_CONFIG = {
    "MAX_WORKERS": 16,  # Increase untuk lebih banyak parallel
    "CHUNK_SIZE": 100,
    "USE_ASYNC": True,
    "BATCH_SIZE": 20,
    "PROGRESS_UPDATE_INTERVAL": 50,  # Update setiap 50 ticker
    "SAVE_INTERMEDIATE_RESULTS": True,  # Penting untuk 900+ ticker
    "RESUME_CAPABILITY": True,  # Enable resume
}
```

### Cache Settings:
```python
CACHE_CONFIG = {
    "ENABLED": True,
    "TTL_HOURS": 24,
    "USE_SQLITE": True,
    "WARM_FETCH": True,  # Gunakan cache untuk speed
}
```

## 💻 Cara Penggunaan

### 1. Scan Normal (900+ ticker)
```bash
python main.py
```

Bot akan:
- Tampilkan progress bar
- Save intermediate results setiap 100 ticker
- Estimasi waktu selesai

### 2. Resume dari Checkpoint
Jika scan terhenti:
```bash
python main.py --resume
```

Bot akan:
- Load checkpoint terakhir
- Lanjutkan dari ticker yang belum diproses
- Combine dengan hasil sebelumnya

### 3. Scan Ticker Tertentu
```bash
python main.py --tickers BBCA.JK BBRI.JK BMRI.JK
```

## 📊 Estimasi Waktu

Dengan konfigurasi default:
- **900 ticker** dengan cache: ~15-30 menit
- **900 ticker** tanpa cache: ~45-90 menit
- **Dengan 16 workers**: ~2-3 ticker/detik

### Faktor yang Mempengaruhi:
1. **Cache hit rate** - Semakin tinggi, semakin cepat
2. **Network speed** - Yahoo Finance API response
3. **CPU cores** - Lebih banyak core = lebih cepat
4. **Memory** - Pastikan cukup untuk pandas operations

## 🔧 Optimasi Lanjutan

### 1. Increase Workers
Jika CPU kuat:
```python
MULTIPROCESSING_CONFIG = {
    "MAX_WORKERS": 32,  # Double workers
}
```

### 2. Reduce Progress Updates
Untuk performa lebih baik:
```python
MULTIPROCESSING_CONFIG = {
    "PROGRESS_UPDATE_INTERVAL": 100,  # Update setiap 100 ticker
}
```

### 3. Disable Charts untuk Speed
Edit `main.py` - comment chart generation:
```python
# Generate chart only for STRONG_AURA
if result.get('signal') == 'STRONG_AURA':
    chart_path = self.chart_engine.generate_chart(...)
```

### 4. Filter Ticker List
Hanya scan ticker aktif:
```python
# Di config.py, filter ticker list
IDX_TICKER_LIST = [
    # Hanya ticker liquid/aktif
    "BBCA", "BBRI", "BMRI", ...
]
```

### 5. Use Database untuk Cache
Upgrade ke PostgreSQL untuk cache lebih baik:
```python
# Install: pip install psycopg2-binary
# Setup PostgreSQL connection
```

## 📈 Monitoring Performance

### Check Progress:
- Progress bar menunjukkan real-time status
- ETA calculation untuk estimasi waktu
- Log file: `logs/ara_bot_v2.log`

### Check Intermediate Results:
```bash
# Lihat hasil sementara
cat results/ara_intermediate.json
```

### Check Checkpoint:
```bash
# Lihat checkpoint
cat cache/scan_checkpoint.json
```

## 🐛 Troubleshooting

### Problem: "Out of Memory"
**Solusi:**
1. Kurangi `MAX_WORKERS` ke 8
2. Increase `CHUNK_SIZE` untuk batch lebih besar
3. Disable chart generation sementara
4. Close aplikasi lain

### Problem: "Yahoo Finance Rate Limit"
**Solusi:**
1. Aktifkan cache (`WARM_FETCH = True`)
2. Tambah delay di `yahoo_fetcher.py`
3. Gunakan fallback APIs
4. Scan di waktu off-peak

### Problem: "Scan Terhenti"
**Solusi:**
1. Gunakan `--resume` untuk lanjutkan
2. Check log file untuk error
3. Clear checkpoint jika perlu restart:
   ```bash
   rm cache/scan_checkpoint.json
   ```

### Problem: "Terlalu Lama"
**Solusi:**
1. Pastikan cache enabled
2. Increase workers (jika CPU kuat)
3. Filter ticker list (hanya yang aktif)
4. Scan incremental (bagi jadi beberapa batch)

## 🎯 Best Practices untuk 900+ Ticker

### 1. **Run di Waktu Off-Peak**
- Malam hari atau weekend
- Kurangi load network

### 2. **Use Cache Aggressively**
- First scan: Full fetch (lama)
- Subsequent scans: Fast (pakai cache)

### 3. **Monitor Resources**
- CPU usage
- Memory usage
- Network bandwidth

### 4. **Incremental Scanning**
Scan per sektor:
```python
# Banking
banking = ["BBCA", "BBRI", "BMRI", ...]

# Mining
mining = ["ADRO", "ANTM", "BUMI", ...]
```

### 5. **Schedule Regular Scans**
```bash
# Cron job (Linux/Mac)
0 16 * * * cd /path/to/ara-bot && python main.py

# Task Scheduler (Windows)
# Set daily at 4 PM
```

## 📊 Expected Output

### Progress Bar:
```
Scanning tickers: [████████████████████░░░░░░░░░░░░░░░░░░░░] 45.0% (405/900) | ETA: 0:12:34
```

### Summary:
```
==================================================
SCAN SUMMARY
==================================================
Total scanned: 900
STRONG_AURA: 12
WATCHLIST: 45
POTENTIAL: 78
==================================================
```

## 🔄 Workflow untuk 900+ Ticker

1. **First Run** (Full Scan):
   ```bash
   python main.py
   ```
   - Fetch semua data
   - Build cache
   - ~45-90 menit

2. **Subsequent Runs** (Fast Scan):
   ```bash
   python main.py
   ```
   - Gunakan cache
   - Update hanya yang perlu
   - ~15-30 menit

3. **If Interrupted**:
   ```bash
   python main.py --resume
   ```
   - Resume dari checkpoint
   - Tidak perlu restart dari awal

## 💡 Tips Tambahan

1. **SSD Storage** - Lebih cepat untuk database cache
2. **Stable Internet** - Penting untuk API calls
3. **Dedicated Machine** - Untuk production use
4. **Monitoring** - Track performance metrics
5. **Backup** - Backup cache database regularly

---

**Dengan optimasi ini, 900+ ticker bisa di-handle dengan efisien! 🚀**

