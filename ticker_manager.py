"""
Ticker Manager - Utility script untuk manage ticker list
Usage:
    python ticker_manager.py list          # List semua ticker
    python ticker_manager.py add BBCA     # Tambah ticker
    python ticker_manager.py remove BBCA   # Hapus ticker
    python ticker_manager.py clear         # Clear semua ticker
    python ticker_manager.py validate     # Validate format
"""

import sys
from pathlib import Path
from utils.ticker_loader import TickerLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

def list_tickers():
    """List all tickers"""
    loader = TickerLoader()
    tickers = loader.load_tickers()
    
    if not tickers:
        print("❌ No tickers found in ticker.txt")
        print("   Add tickers using: python ticker_manager.py add TICKER")
        return
    
    print(f"\n📋 Found {len(tickers)} tickers:\n")
    for i, ticker in enumerate(tickers, 1):
        print(f"  {i:3d}. {ticker}")
    print()

def add_ticker(ticker: str):
    """Add ticker"""
    loader = TickerLoader()
    normalized = loader._normalize_ticker(ticker)
    
    # Check if already exists
    existing = loader.load_tickers()
    if normalized in existing:
        print(f"⚠️  Ticker {normalized} already exists")
        return
    
    loader.add_ticker(ticker)
    print(f"✅ Added ticker: {normalized}")

def remove_ticker(ticker: str):
    """Remove ticker"""
    loader = TickerLoader()
    normalized = loader._normalize_ticker(ticker)
    
    # Check if exists
    existing = loader.load_tickers()
    if normalized not in existing:
        print(f"⚠️  Ticker {normalized} not found")
        return
    
    loader.remove_ticker(ticker)
    print(f"✅ Removed ticker: {normalized}")

def clear_tickers():
    """Clear all tickers"""
    loader = TickerLoader()
    ticker_file = loader.ticker_file
    
    if not ticker_file.exists():
        print("❌ ticker.txt file not found")
        return
    
    confirm = input("⚠️  Are you sure you want to clear all tickers? (yes/no): ")
    if confirm.lower() == 'yes':
        # Create default file
        loader._create_default_file()
        print("✅ Cleared all tickers (default file created)")
    else:
        print("❌ Cancelled")

def validate_tickers():
    """Validate ticker format"""
    loader = TickerLoader()
    tickers = loader.load_tickers()
    
    if not tickers:
        print("❌ No tickers found")
        return
    
    print(f"\n🔍 Validating {len(tickers)} tickers...\n")
    
    valid = []
    invalid = []
    
    for ticker in tickers:
        if ticker.endswith('.JK') and len(ticker) > 3:
            valid.append(ticker)
        else:
            invalid.append(ticker)
    
    if invalid:
        print(f"❌ Invalid tickers ({len(invalid)}):")
        for ticker in invalid:
            print(f"   - {ticker}")
        print()
    
    if valid:
        print(f"✅ Valid tickers ({len(valid)}):")
        for ticker in valid[:10]:  # Show first 10
            print(f"   - {ticker}")
        if len(valid) > 10:
            print(f"   ... and {len(valid) - 10} more")
        print()
    
    print(f"📊 Summary: {len(valid)} valid, {len(invalid)} invalid")

def show_help():
    """Show help message"""
    help_text = """
📋 Ticker Manager - Manage ticker list for ARA Bot

Usage:
    python ticker_manager.py <command> [args]

Commands:
    list                    List all tickers
    add <TICKER>            Add ticker (e.g., BBCA or BBCA.JK)
    remove <TICKER>         Remove ticker
    clear                   Clear all tickers (with confirmation)
    validate                 Validate ticker format
    help                    Show this help message

Examples:
    python ticker_manager.py list
    python ticker_manager.py add BBCA
    python ticker_manager.py add BBRI.JK
    python ticker_manager.py remove BBCA
    python ticker_manager.py validate

File Format (ticker.txt):
    - One ticker per line
    - Or comma-separated: BBCA, BBRI, BMRI
    - Or space-separated: BBCA BBRI BMRI
    - With or without .JK suffix
    - Lines starting with # are comments
    - Empty lines are ignored
"""
    print(help_text)

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_tickers()
    
    elif command == 'add':
        if len(sys.argv) < 3:
            print("❌ Error: Please provide ticker to add")
            print("   Usage: python ticker_manager.py add <TICKER>")
            return
        add_ticker(sys.argv[2])
    
    elif command == 'remove':
        if len(sys.argv) < 3:
            print("❌ Error: Please provide ticker to remove")
            print("   Usage: python ticker_manager.py remove <TICKER>")
            return
        remove_ticker(sys.argv[2])
    
    elif command == 'clear':
        clear_tickers()
    
    elif command == 'validate':
        validate_tickers()
    
    elif command == 'help':
        show_help()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("   Use 'python ticker_manager.py help' for usage information")

if __name__ == "__main__":
    main()

