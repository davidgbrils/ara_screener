"""
Test script untuk Telegram bot setup
Jalankan: python test_telegram.py
"""

import os
from pathlib import Path

# Load .env if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv tidak terinstall. Install dengan: pip install python-dotenv")
    print("   Menggunakan environment variables langsung...")

from notifier.telegram_notifier import TelegramNotifier

def test_telegram_setup():
    """Test Telegram bot configuration"""
    
    print("=" * 50)
    print("🧪 TEST TELEGRAM BOT SETUP")
    print("=" * 50)
    print()
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ File .env ditemukan")
    else:
        print("⚠️  File .env tidak ditemukan")
        print("   Buat file .env dengan format:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")
        print("   TELEGRAM_CHAT_ID=your_chat_id_here")
        print()
    
    # Check environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN tidak ditemukan")
        print("   Set di .env file atau environment variable")
    else:
        print(f"✅ TELEGRAM_BOT_TOKEN ditemukan: {bot_token[:10]}...")
    
    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID tidak ditemukan")
        print("   Set di .env file atau environment variable")
    else:
        print(f"✅ TELEGRAM_CHAT_ID ditemukan: {chat_id}")
    
    print()
    
    # Initialize notifier
    notifier = TelegramNotifier()
    
    if not notifier.enabled:
        print("❌ Telegram bot TIDAK ENABLED")
        print()
        print("🔧 CARA PERBAIKI:")
        print("1. Buat file .env di root directory")
        print("2. Tambahkan:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        print("3. Install python-dotenv: pip install python-dotenv")
        print("4. Restart script ini")
        return False
    
    print("✅ Telegram bot ENABLED")
    print()
    print("📤 Mengirim test message...")
    
    # Send test message
    test_message = """
🧪 <b>Test Message dari ARA Bot</b>

✅ Jika Anda melihat pesan ini, berarti:
   • Bot Token benar
   • Chat ID benar
   • Bot sudah dikonfigurasi dengan benar

🎉 Setup Telegram bot berhasil!
    """
    
    success = notifier.send_message(test_message)
    
    if success:
        print("✅ Test message berhasil dikirim!")
        print("   Cek Telegram Anda untuk melihat pesan.")
        print()
        print("🎉 SETUP BERHASIL!")
        return True
    else:
        print("❌ Gagal mengirim test message")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("1. Pastikan Bot Token benar (dari @BotFather)")
        print("2. Pastikan Chat ID benar (dari @userinfobot)")
        print("3. Pastikan bot sudah di-start (kirim /start ke bot)")
        print("4. Cek koneksi internet")
        print("5. Cek log file untuk error details")
        return False

if __name__ == "__main__":
    test_telegram_setup()

