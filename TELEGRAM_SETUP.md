# 📱 Setup Telegram Bot untuk ARA Bot

Panduan lengkap untuk setup Telegram bot agar ARA Bot bisa mengirim notifikasi.

## 🎯 Langkah 1: Buat Bot di Telegram

### 1.1 Buka BotFather
1. Buka aplikasi Telegram
2. Cari **@BotFather** di search
3. Klik dan mulai chat

### 1.2 Buat Bot Baru
1. Kirim command: `/newbot`
2. BotFather akan minta **nama bot** (contoh: "ARA Stock Scanner")
3. BotFather akan minta **username bot** (harus berakhiran `bot`, contoh: `ara_stock_scanner_bot`)
4. BotFather akan memberikan **Bot Token** seperti ini:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz1234567890
   ```
5. **SIMPAN TOKEN INI** - Anda akan membutuhkannya!

### 1.3 (Optional) Setup Bot Info
- `/setdescription` - Set deskripsi bot
- `/setabouttext` - Set about text
- `/setuserpic` - Upload foto profil bot

## 🆔 Langkah 2: Dapatkan Chat ID

Ada beberapa cara untuk mendapatkan Chat ID:

### Cara 1: Menggunakan @userinfobot (Paling Mudah)
1. Cari **@userinfobot** di Telegram
2. Mulai chat dengan bot tersebut
3. Bot akan mengirim informasi Anda termasuk **Your Id**
4. **SIMPAN ID INI** (contoh: `123456789`)

### Cara 2: Menggunakan @getidsbot
1. Cari **@getidsbot** di Telegram
2. Forward pesan dari chat Anda ke bot tersebut
3. Bot akan mengirim Chat ID

### Cara 3: Menggunakan API (Untuk Group Chat)
Jika ingin mengirim ke group:
1. Tambahkan bot ke group
2. Kirim pesan di group
3. Buka browser: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Cari `"chat":{"id":-123456789}` - ID negatif untuk group

## ⚙️ Langkah 3: Setup Environment Variables

### 3.1 Buat File .env
Buat file `.env` di root directory project (sama level dengan `main.py`):

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz1234567890
TELEGRAM_CHAT_ID=123456789
```

**PENTING:**
- Ganti dengan token dan chat ID yang Anda dapatkan
- Jangan ada spasi sebelum/sesudah `=`
- Jangan pakai tanda kutip

### 3.2 Install python-dotenv (Jika belum)
```bash
pip install python-dotenv
```

### 3.3 Update config.py untuk Load .env
Pastikan `config.py` sudah load environment variables. Cek apakah ada:

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file
```

Jika belum ada, tambahkan di bagian atas `config.py`.

## 🔧 Langkah 4: Update config.py

Pastikan `TELEGRAM_CONFIG` di `config.py` sudah benar:

```python
TELEGRAM_CONFIG = {
    "BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
    "ENABLED": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
    "SEND_ON_SIGNAL_CHANGE": True,
    "SEND_SUMMARY": True,
    "TOP_N_SUMMARY": 10,
    "HTML_FORMAT": True,
}
```

## ✅ Langkah 5: Test Bot

### 5.1 Test Manual
Buat file test sederhana `test_telegram.py`:

```python
import os
from dotenv import load_dotenv
from notifier.telegram_notifier import TelegramNotifier

load_dotenv()

notifier = TelegramNotifier()

if notifier.enabled:
    success = notifier.send_message("🧪 Test message dari ARA Bot!")
    if success:
        print("✅ Telegram bot berhasil dikonfigurasi!")
    else:
        print("❌ Gagal mengirim pesan. Cek token dan chat ID.")
else:
    print("❌ Telegram bot tidak enabled. Cek .env file.")
```

Jalankan:
```bash
python test_telegram.py
```

Jika berhasil, Anda akan menerima pesan di Telegram!

### 5.2 Test dengan ARA Bot
Jalankan ARA Bot:
```bash
python main.py
```

Bot akan otomatis mengirim notifikasi jika ada signal baru.

## 🎨 Format Notifikasi

ARA Bot akan mengirim notifikasi dengan format:

### Signal Notification
```
🔥 BBCA.JK - STRONG_AURA
━━━━━━━━━━━━━━━━━━━━

💰 Price: Rp 9,500
📊 Score: 85.00%
🤖 ML Probability: 78.00%

📈 Entry Levels:
Entry: Rp 9,300 - Rp 9,700
SL: Rp 9,000
TP1: Rp 10,000 (R:R 2.00)
TP2: Rp 11,000 (R:R 3.00)

✅ Reasons:
• RVOL 3.5x
• Bollinger Breakout
• Bullish MA Structure
• RSI 65.5
• OBV Rising
• Above VWAP

🎯 Patterns:
• Parabolic
• Volume Climax

⏰ 2024-01-15 10:30:00
```

### Summary Notification
```
📊 ARA BOT SCAN SUMMARY
━━━━━━━━━━━━━━━━━━━━

⏰ 2024-01-15 10:30:00

Top 10 Results:

1. 🔥 BBCA.JK - STRONG_AURA (85.0%)
   💰 Rp 9,500

2. ⭐ BBRI.JK - WATCHLIST (72.0%)
   💰 Rp 4,200

...
```

## 🔔 Pengaturan Notifikasi

### Hanya Kirim Saat Signal Berubah
Default: `SEND_ON_SIGNAL_CHANGE = True`
- Bot hanya kirim notifikasi saat signal berubah
- Mencegah spam
- Track perubahan dari scan sebelumnya

### Kirim Summary Setiap Scan
Default: `SEND_SUMMARY = True`
- Kirim summary dengan top 10 results
- Setiap kali scan selesai

### Ubah Jumlah Top Results
Edit di `config.py`:
```python
TELEGRAM_CONFIG = {
    "TOP_N_SUMMARY": 20,  # Ubah dari 10 ke 20
}
```

## 🐛 Troubleshooting

### Problem: "Telegram bot tidak enabled"
**Solusi:**
1. Cek file `.env` ada di root directory
2. Cek format `.env` benar (tidak ada spasi, tidak pakai kutip)
3. Cek `python-dotenv` terinstall: `pip install python-dotenv`
4. Restart Python setelah edit `.env`

### Problem: "Gagal mengirim pesan"
**Solusi:**
1. Cek Bot Token benar (copy-paste dari BotFather)
2. Cek Chat ID benar (gunakan @userinfobot)
3. Pastikan bot sudah di-start (kirim `/start` ke bot)
4. Cek koneksi internet
5. Cek log file untuk error details

### Problem: "Bot tidak merespon"
**Solusi:**
1. Pastikan bot sudah di-start: kirim `/start` ke bot
2. Cek bot tidak di-block
3. Cek Chat ID benar (untuk group, pastikan bot sudah ditambahkan)

### Problem: "Notifikasi tidak terkirim saat scan"
**Solusi:**
1. Cek `TELEGRAM_CONFIG["ENABLED"]` = True
2. Cek ada signal baru (bot hanya kirim saat signal berubah)
3. Cek `SEND_ON_SIGNAL_CHANGE` = True
4. Test dengan `test_telegram.py` dulu

## 📝 Checklist Setup

- [ ] Bot dibuat di BotFather
- [ ] Bot Token disimpan
- [ ] Chat ID didapatkan (dari @userinfobot)
- [ ] File `.env` dibuat dengan token dan chat ID
- [ ] `python-dotenv` terinstall
- [ ] `config.py` sudah load `.env`
- [ ] Test dengan `test_telegram.py` berhasil
- [ ] ARA Bot bisa kirim notifikasi

## 🎯 Tips

1. **Gunakan Private Chat** untuk testing (lebih mudah)
2. **Gunakan Group Chat** untuk production (bisa share dengan team)
3. **Backup Token** - Simpan di tempat aman
4. **Jangan Commit .env** - Sudah ada di `.gitignore`
5. **Test Dulu** - Pastikan bot bekerja sebelum production

## 🔒 Security

1. **Jangan share Bot Token** - Siapa pun bisa pakai token Anda
2. **Jangan commit .env** - File sudah di `.gitignore`
3. **Rotate Token** - Jika token ter-expose, buat bot baru
4. **Gunakan Environment Variables** - Jangan hardcode di code

## 📞 Support

Jika masih ada masalah:
1. Cek log file: `logs/ara_bot_v2.log`
2. Test dengan `test_telegram.py`
3. Cek Telegram Bot API: https://core.telegram.org/bots/api

---

**Selamat! Bot Telegram Anda sudah siap digunakan! 🎉**

