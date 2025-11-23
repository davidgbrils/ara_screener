# 📋 Ticker Setup Guide

Panduan lengkap untuk mengelola ticker list menggunakan file `ticker.txt`.

## 🎯 Quick Start

### 1. Edit File `ticker.txt`

Buka file `ticker.txt` di root directory dan tambahkan ticker Anda:

```
BBCA
BBRI
BMRI
BNII
BBNI
```

Atau copy-paste langsung dari sumber lain (Excel, website, dll).

### 2. Format yang Didukung

#### Format 1: Satu Ticker Per Baris
```
BBCA
BBRI
BMRI
```

#### Format 2: Comma-Separated
```
BBCA, BBRI, BMRI, BNII, BBNI
```

#### Format 3: Space-Separated
```
BBCA BBRI BMRI BNII BBNI
```

#### Format 4: Campuran
```
# Banking
BBCA, BBRI, BMRI

# Mining
ADRO ANTM BUMI
```

### 3. Run Bot

```bash
python main.py
```

Bot akan otomatis membaca ticker dari `ticker.txt`!

## 📝 Format File

### Aturan Format:

1. **Satu ticker per baris** - Paling mudah
2. **Comma-separated** - Untuk multiple ticker dalam satu baris
3. **Space-separated** - Juga didukung
4. **Dengan atau tanpa .JK** - Otomatis dinormalisasi
5. **Comments dengan #** - Baris yang dimulai dengan # diabaikan
6. **Empty lines** - Baris kosong diabaikan

### Contoh File Lengkap:

```txt
# ARA BOT V2 - Ticker List
# Format: Satu ticker per baris, atau dipisahkan koma

# Banking Sector
BBCA
BBRI
BMRI, BNII, BBNI

# Consumer Goods
INDF ICBP INTP
KLBF
UNVR

# Mining
ADRO
ANTM
BUMI BYAN GOLD

# Telekomunikasi
TLKM
EXCL
FREN
ISAT
```

## 🛠️ Ticker Manager Tool

Gunakan `ticker_manager.py` untuk manage ticker dengan mudah:

### List Semua Ticker
```bash
python ticker_manager.py list
```

### Tambah Ticker
```bash
python ticker_manager.py add BBCA
python ticker_manager.py add BBRI.JK
```

### Hapus Ticker
```bash
python ticker_manager.py remove BBCA
```

### Clear Semua Ticker
```bash
python ticker_manager.py clear
```

### Validate Format
```bash
python ticker_manager.py validate
```

### Help
```bash
python ticker_manager.py help
```

## 📋 Copy-Paste dari Excel/Website

### Dari Excel:
1. Copy kolom ticker dari Excel
2. Paste langsung ke `ticker.txt`
3. Bot akan otomatis parse formatnya

### Dari Website:
1. Copy list ticker dari website
2. Paste ke `ticker.txt`
3. Bot akan handle berbagai format

### Contoh Copy-Paste:

**Dari Excel (tab-separated):**
```
BBCA	BBRI	BMRI	BNII	BBNI
```
→ Paste langsung, bot akan parse

**Dari Website (comma-separated):**
```
BBCA, BBRI, BMRI, BNII, BBNI
```
→ Paste langsung, bot akan parse

**Dari List (one per line):**
```
BBCA
BBRI
BMRI
```
→ Paste langsung, bot akan parse

## 🔄 Workflow

### 1. Setup Awal
```bash
# Edit ticker.txt dengan ticker Anda
# Atau gunakan ticker_manager.py
python ticker_manager.py add BBCA
python ticker_manager.py add BBRI
```

### 2. Validate
```bash
python ticker_manager.py validate
```

### 3. Run Bot
```bash
python main.py
```

### 4. Update Ticker
```bash
# Tambah ticker baru
python ticker_manager.py add NEWTICKER

# Hapus ticker
python ticker_manager.py remove OLDTICKER

# List semua
python ticker_manager.py list
```

## 💡 Tips

### 1. Organize dengan Comments
```txt
# Banking
BBCA
BBRI
BMRI

# Mining
ADRO
ANTM
```

### 2. Group Related Tickers
```txt
# Blue Chips
BBCA, BBRI, BMRI, BNII, BBNI

# Second Liners
BJBR, BJTM, BNGA
```

### 3. Backup Ticker List
```bash
# Copy file
cp ticker.txt ticker_backup.txt
```

### 4. Bulk Add dari File
```bash
# Jika punya file dengan banyak ticker
cat other_ticker_list.txt >> ticker.txt
```

## 🐛 Troubleshooting

### Problem: "Ticker file not found"
**Solusi:**
- File `ticker.txt` akan otomatis dibuat jika tidak ada
- Atau buat manual di root directory

### Problem: "No tickers loaded"
**Solusi:**
1. Cek format file `ticker.txt`
2. Pastikan ada ticker yang valid
3. Run: `python ticker_manager.py validate`

### Problem: "Duplicate tickers"
**Solusi:**
- Bot otomatis remove duplicates
- Atau gunakan: `python ticker_manager.py validate` untuk cek

### Problem: "Ticker format error"
**Solusi:**
- Pastikan ticker valid (huruf/angka)
- Bot akan otomatis normalize (tambah .JK)
- Contoh: `BBCA` → `BBCA.JK`

## 📊 Contoh Use Cases

### Use Case 1: Copy dari IDX Website
1. Buka https://www.idx.co.id
2. Copy list ticker
3. Paste ke `ticker.txt`
4. Run bot

### Use Case 2: Update dari Excel
1. Export ticker dari Excel ke CSV
2. Copy kolom ticker
3. Paste ke `ticker.txt`
4. Run bot

### Use Case 3: Manual Entry
1. Buka `ticker.txt`
2. Tambah ticker satu per satu
3. Save file
4. Run bot

### Use Case 4: Programmatic
```python
from utils.ticker_loader import TickerLoader

loader = TickerLoader()
loader.add_ticker("BBCA")
loader.add_ticker("BBRI")
tickers = loader.load_tickers()
```

## 🔧 Advanced Usage

### Load Tickers Programmatically
```python
from utils.ticker_loader import TickerLoader

loader = TickerLoader()
tickers = loader.load_tickers()
print(f"Loaded {len(tickers)} tickers")
```

### Custom Ticker File
```python
from pathlib import Path
from utils.ticker_loader import TickerLoader

custom_file = Path("my_tickers.txt")
loader = TickerLoader(ticker_file=custom_file)
tickers = loader.load_tickers()
```

### Save Tickers
```python
from utils.ticker_loader import TickerLoader

loader = TickerLoader()
tickers = ["BBCA.JK", "BBRI.JK", "BMRI.JK"]
loader.save_tickers(tickers)
```

## 📝 Best Practices

1. **Backup Regularly** - Backup `ticker.txt` sebelum edit besar
2. **Use Comments** - Organize dengan comments untuk mudah dibaca
3. **Validate First** - Selalu validate sebelum run bot
4. **Keep It Clean** - Remove ticker yang tidak aktif
5. **Version Control** - Commit `ticker.txt` ke git (optional)

---

**Dengan sistem ini, Anda bisa dengan mudah copy-paste ticker dari mana saja! 🚀**

