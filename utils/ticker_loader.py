"""Ticker loader from text file"""

import re
from pathlib import Path
from typing import List, Set
from config import IDX_TICKER_SUFFIX, BASE_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TickerLoader:
    """Load tickers from text file with flexible format support"""
    
    def __init__(self, ticker_file: Path = None):
        """
        Initialize ticker loader
        
        Args:
            ticker_file: Path to ticker file (defaults to ticker.txt in root)
        """
        self.ticker_file = ticker_file or BASE_DIR / "ticker.txt"
        self.suffix = IDX_TICKER_SUFFIX
    
    def load_tickers(self) -> List[str]:
        """
        Load tickers from file
        
        Supports multiple formats:
        - One ticker per line: BBCA
        - Comma-separated: BBCA, BBRI, BMRI
        - Space-separated: BBCA BBRI BMRI
        - With or without .JK suffix
        - Comments with # prefix
        - Empty lines ignored
        
        Returns:
            List of normalized ticker symbols
        """
        if not self.ticker_file.exists():
            logger.warning(f"Ticker file not found: {self.ticker_file}")
            logger.info("Creating default ticker.txt file...")
            self._create_default_file()
            return []
        
        try:
            with open(self.ticker_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tickers = self._parse_content(content)
            
            # Normalize tickers
            normalized = [self._normalize_ticker(t) for t in tickers]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_tickers = []
            for ticker in normalized:
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    unique_tickers.append(ticker)
            
            logger.info(f"Loaded {len(unique_tickers)} unique tickers from {self.ticker_file}")
            
            return unique_tickers
            
        except Exception as e:
            logger.error(f"Error loading tickers from {self.ticker_file}: {e}")
            return []
    
    def _parse_content(self, content: str) -> List[str]:
        """
        Parse content and extract tickers
        
        Args:
            content: File content
        
        Returns:
            List of ticker strings (not normalized)
        """
        tickers = []
        lines = content.split('\n')
        
        for line in lines:
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Try comma-separated first
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                tickers.extend([p for p in parts if p])
            # Try space-separated (multiple tickers in one line)
            elif ' ' in line and not line.endswith('.JK'):
                parts = line.split()
                tickers.extend([p.strip() for p in parts if p])
            # Single ticker
            else:
                tickers.append(line)
        
        return tickers
    
    def _normalize_ticker(self, ticker: str) -> str:
        """
        Normalize ticker to standard format
        
        Args:
            ticker: Raw ticker string
        
        Returns:
            Normalized ticker (e.g., "BBCA.JK")
        """
        if not ticker:
            return ""
        
        ticker = ticker.strip().upper()
        
        # Remove .JK if present, then add it back for consistency
        if ticker.endswith(self.suffix):
            ticker = ticker[:-len(self.suffix)]
        
        # Add suffix
        return f"{ticker}{self.suffix}"
    
    def _create_default_file(self):
        """Create default ticker.txt file"""
        default_content = """# ARA BOT V2 - Ticker List
# Format: Satu ticker per baris, atau dipisahkan koma
# Ticker bisa dengan atau tanpa .JK suffix
# Baris yang dimulai dengan # akan diabaikan (comment)
# Baris kosong akan diabaikan

# Contoh format:
# BBCA
# BBRI.JK
# BMRI, BNII, BBNI
# BBCA.JK BBRI.JK BMRI.JK

# Tambahkan ticker Anda di bawah ini:
BBCA
BBRI
BMRI
"""
        try:
            with open(self.ticker_file, 'w', encoding='utf-8') as f:
                f.write(default_content)
            logger.info(f"Created default ticker file: {self.ticker_file}")
        except Exception as e:
            logger.error(f"Error creating default ticker file: {e}")
    
    def save_tickers(self, tickers: List[str], append: bool = False):
        """
        Save tickers to file
        
        Args:
            tickers: List of tickers to save
            append: Append to existing file instead of overwrite
        """
        try:
            mode = 'a' if append else 'w'
            with open(self.ticker_file, mode, encoding='utf-8') as f:
                if not append:
                    f.write("# ARA BOT V2 - Ticker List\n")
                    f.write("# Format: Satu ticker per baris\n\n")
                
                for ticker in tickers:
                    # Remove .JK for cleaner format
                    clean_ticker = ticker.replace(self.suffix, '')
                    f.write(f"{clean_ticker}\n")
            
            logger.info(f"Saved {len(tickers)} tickers to {self.ticker_file}")
            
        except Exception as e:
            logger.error(f"Error saving tickers: {e}")
    
    def add_ticker(self, ticker: str):
        """
        Add single ticker to file
        
        Args:
            ticker: Ticker to add
        """
        normalized = self._normalize_ticker(ticker)
        clean_ticker = normalized.replace(self.suffix, '')
        
        try:
            with open(self.ticker_file, 'a', encoding='utf-8') as f:
                f.write(f"{clean_ticker}\n")
            logger.info(f"Added ticker: {normalized}")
        except Exception as e:
            logger.error(f"Error adding ticker: {e}")
    
    def remove_ticker(self, ticker: str):
        """
        Remove ticker from file
        
        Args:
            ticker: Ticker to remove
        """
        normalized = self._normalize_ticker(ticker)
        clean_ticker = normalized.replace(self.suffix, '')
        
        try:
            # Read all lines
            with open(self.ticker_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter out the ticker
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                # Check if line contains the ticker (with or without .JK)
                if stripped and not stripped.startswith('#'):
                    # Parse line to get all tickers
                    if ',' in stripped:
                        parts = [p.strip() for p in stripped.split(',')]
                        if clean_ticker not in parts and normalized not in parts:
                            filtered_lines.append(line)
                    elif stripped.upper() not in [clean_ticker.upper(), normalized.upper()]:
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            
            # Write back
            with open(self.ticker_file, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)
            
            logger.info(f"Removed ticker: {normalized}")
            
        except Exception as e:
            logger.error(f"Error removing ticker: {e}")

