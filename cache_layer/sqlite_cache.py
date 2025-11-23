"""SQLite-based cache for OHLCV data"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import json
from config import CACHE_CONFIG, DB_PATH
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SQLiteCache:
    """SQLite cache for storing and retrieving OHLCV data"""
    
    def __init__(self, db_path: Path = DB_PATH):
        """
        Initialize SQLite cache
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OHLCV data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                ticker TEXT PRIMARY KEY,
                last_updated TIMESTAMP,
                data_count INTEGER,
                is_valid BOOLEAN DEFAULT 1
            )
        """)
        
        # Indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON ohlcv_data(ticker, date DESC)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def store_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Store OHLCV data for a ticker
        
        Args:
            ticker: Ticker symbol
            df: DataFrame with OHLCV data
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Clean old data for this ticker
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ohlcv_data WHERE ticker = ?", (ticker,))
            
            # Insert new data
            df = df.copy()
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date']).dt.date
            elif df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                df['date'] = df.index.date if hasattr(df.index, 'date') else df.index
            
            df['ticker'] = ticker
            df_to_insert = df[['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df_to_insert.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            
            df_to_insert.to_sql('ohlcv_data', conn, if_exists='append', index=False)
            
            # Update metadata
            cursor.execute("""
                INSERT OR REPLACE INTO cache_metadata 
                (ticker, last_updated, data_count, is_valid)
                VALUES (?, ?, ?, 1)
            """, (ticker, datetime.now(), len(df_to_insert)))
            
            conn.commit()
            conn.close()
            logger.debug(f"Cached data for {ticker}: {len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing data for {ticker}: {e}")
            return False
    
    def get_data(self, ticker: str, days: int = None) -> Optional[pd.DataFrame]:
        """
        Retrieve OHLCV data for a ticker
        
        Args:
            ticker: Ticker symbol
            days: Number of days to retrieve (None for all)
        
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT date, open, high, low, close, volume FROM ohlcv_data WHERE ticker = ?"
            params = [ticker]
            
            if days:
                cutoff_date = (datetime.now() - timedelta(days=days)).date()
                query += " AND date >= ?"
                params.append(cutoff_date)
            
            query += " ORDER BY date ASC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            logger.debug(f"Retrieved cached data for {ticker}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {ticker}: {e}")
            return None
    
    def is_valid(self, ticker: str, ttl_hours: int = None) -> bool:
        """
        Check if cached data is still valid
        
        Args:
            ticker: Ticker symbol
            ttl_hours: Time-to-live in hours (uses config if None)
        
        Returns:
            True if cache is valid
        """
        try:
            ttl_hours = ttl_hours or CACHE_CONFIG["TTL_HOURS"]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT last_updated, is_valid 
                FROM cache_metadata 
                WHERE ticker = ?
            """, (ticker,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return False
            
            last_updated, is_valid = result
            if not is_valid:
                return False
            
            last_updated = datetime.fromisoformat(last_updated)
            age = datetime.now() - last_updated
            return age.total_seconds() < (ttl_hours * 3600)
            
        except Exception as e:
            logger.error(f"Error checking cache validity for {ticker}: {e}")
            return False
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all cached tickers"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM cache_metadata WHERE is_valid = 1")
            tickers = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tickers
        except Exception as e:
            logger.error(f"Error getting cached tickers: {e}")
            return []
    
    def invalidate(self, ticker: str):
        """Invalidate cache for a ticker"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE cache_metadata 
                SET is_valid = 0 
                WHERE ticker = ?
            """, (ticker,))
            conn.commit()
            conn.close()
            logger.debug(f"Invalidated cache for {ticker}")
        except Exception as e:
            logger.error(f"Error invalidating cache for {ticker}: {e}")

