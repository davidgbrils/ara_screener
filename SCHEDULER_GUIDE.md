# ⏰ Auto-Scheduler Guide - Market Open Detection

Panduan lengkap untuk memastikan bot scan di waktu yang sama setiap hari dan menggunakan data terbaru.

## ✅ Fitur yang Sudah Diterapkan

### 1. **Timezone-Aware Scheduler**
- ✅ Run otomatis jam 9:30 AM Jakarta time
- ✅ Skip weekend (Sabtu-Minggu)
- ✅ Validasi market open (9:30-16:00 WIB)
- ✅ Timezone conversion otomatis

### 2. **Data Freshness Validation**
- ✅ Force refresh data saat market open
- ✅ Validasi data hari ini tersedia
- ✅ Skip cache jika data tidak fresh
- ✅ Warning jika data tidak include hari ini

### 3. **Market Open Detection**
- ✅ Cek apakah market sedang buka
- ✅ Validasi hari kerja (Senin-Jumat)
- ✅ Validasi jam trading (9:30-16:00)

## 🚀 Cara Setup

### 1. Install Dependencies
```bash
pip install schedule pytz
```

### 2. Run Scheduler
```bash
python scheduler.py
```

Bot akan:
- ✅ Run otomatis setiap hari jam 9:30 WIB
- ✅ Force refresh data (mendapat data hari ini)
- ✅ Scan semua ticker dengan data fresh
- ✅ Kirim top 10 ke Telegram

### 3. Test Run Sekarang
```bash
python scheduler.py --now
```

## ⚙️ Cara Kerja

### Scheduler Flow:

```
1. Scheduler check waktu (setiap 30 detik)
   ↓
2. Jika jam 9:30 WIB dan market open
   ↓
3. Run bot dengan force_refresh=True
   ↓
4. Fetch fresh data dari Yahoo Finance
   ↓
5. Validasi data include hari ini
   ↓
6. Scan semua ticker
   ↓
7. Kirim top 10 ke Telegram
```

### Data Freshness Flow:

```
1. Bot dimulai dengan force_refresh=True
   ↓
2. Skip cache (tidak pakai cache)
   ↓
3. Fetch langsung dari Yahoo Finance
   ↓
4. Validasi last_date >= today
   ↓
5. Jika data fresh, gunakan
   ↓
6. Jika tidak fresh, warning di log
```

## 📊 Validasi Data

Bot akan memvalidasi:

1. **Data Date Check:**
   - Last date di data >= today
   - Jika tidak, warning di log

2. **Cache Freshness:**
   - Cek apakah cache punya data hari ini
   - Jika tidak, force fetch fresh

3. **Market Open Check:**
   - Validasi hari kerja
   - Validasi jam trading

## ⚠️ Important Notes

### 1. System Timezone
Scheduler menggunakan system timezone. Pastikan:
- System timezone sudah benar
- Atau gunakan timezone conversion manual

### 2. Yahoo Finance Data
- Data biasanya tersedia setelah market close
- Untuk data real-time, perlu API premium
- Bot akan warning jika data tidak fresh

### 3. Market Hours
- Market buka: 9:30-16:00 WIB
- Hari kerja: Senin-Jumat
- Bot akan skip weekend

## 🔧 Troubleshooting

### Problem: "Bot tidak run di jam 9:30"
**Solusi:**
1. Cek system timezone
2. Cek log: `logs/ara_bot_v2.log`
3. Test dengan `python scheduler.py --now`

### Problem: "Data tidak fresh"
**Solusi:**
1. Yahoo Finance mungkin belum update
2. Cek last_date di log
3. Bot akan tetap scan dengan data terbaru yang tersedia

### Problem: "Scheduler tidak jalan"
**Solusi:**
1. Pastikan `schedule` dan `pytz` terinstall
2. Cek apakah process masih running
3. Gunakan Task Scheduler (Windows) atau cron (Linux)

## 📝 Alternative Setup

### Windows Task Scheduler

1. Buka Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 9:30 AM
4. Action: Start program
5. Program: `python`
6. Arguments: `scheduler.py`
7. Start in: `C:\path\to\ara-bot`

### Linux/Mac Cron

```bash
# Edit crontab
crontab -e

# Add line (9:30 AM Jakarta = 2:30 AM UTC)
30 2 * * 1-5 cd /path/to/ara-bot && /usr/bin/python3 scheduler.py
```

## 🎯 Best Practices

1. **Run di Market Open:**
   - 9:30 WIB = Market buka
   - Data akan lebih fresh

2. **Monitor Logs:**
   - Cek log setiap hari
   - Pastikan bot run sesuai jadwal

3. **Data Validation:**
   - Bot akan warning jika data tidak fresh
   - Monitor warning di log

4. **Backup Plan:**
   - Setup Task Scheduler sebagai backup
   - Atau cron job untuk reliability

## 📈 Expected Behavior

### Setiap Hari (Senin-Jumat):

```
09:30 WIB → Scheduler trigger
  ↓
Bot start dengan force_refresh
  ↓
Fetch fresh data dari Yahoo
  ↓
Validasi data include hari ini
  ↓
Scan semua ticker
  ↓
Rank & filter top 10
  ↓
Kirim ke Telegram
  ↓
Selesai ~10-15 menit
```

### Log Output:

```
============================================================
AUTO-SCHEDULED SCAN STARTED
Time: 2024-01-15 09:30:00 WIB
Market Open: True
============================================================
Force Refresh: True
Starting scan of 900 tickers...
...
============================================================
AUTO-SCHEDULED SCAN COMPLETED
Results: 10 signals found
Time: 2024-01-15 09:45:00 WIB
============================================================
```

---

**Dengan setup ini, bot akan scan di waktu yang sama setiap hari dengan data terbaru! 🚀**

