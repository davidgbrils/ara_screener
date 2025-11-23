"""Multiprocessing engine with async support for fast scanning"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Optional
from config import MULTIPROCESSING_CONFIG, IDX_TICKER_LIST
from utils.logger import setup_logger
from utils.ticker_loader import TickerLoader
from utils.progress import ProgressTracker

logger = setup_logger(__name__)

class MultiprocessingEngine:
    """Multiprocessing engine for parallel ticker processing"""
    
    def __init__(self):
        """Initialize multiprocessing engine"""
        self.max_workers = MULTIPROCESSING_CONFIG["MAX_WORKERS"]
        self.chunk_size = MULTIPROCESSING_CONFIG["CHUNK_SIZE"]
        self.use_async = MULTIPROCESSING_CONFIG["USE_ASYNC"]
        self.batch_size = MULTIPROCESSING_CONFIG["BATCH_SIZE"]
    
    def process_tickers(
        self,
        tickers: List[str],
        process_func: Callable,
        use_threads: bool = False,
        show_progress: bool = True,
        progress_callback: Callable = None
    ) -> List[Dict]:
        """
        Process multiple tickers in parallel
        
        Args:
            tickers: List of ticker symbols
            process_func: Function to process each ticker
            use_threads: Use threads instead of processes
            show_progress: Show progress bar
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of results
        """
        results = []
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        # Initialize progress tracker
        progress = None
        if show_progress and len(tickers) > 10:
            progress = ProgressTracker(len(tickers), "Scanning tickers")
        
        update_interval = MULTIPROCESSING_CONFIG.get("PROGRESS_UPDATE_INTERVAL", 50)
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(process_func, ticker): ticker
                for ticker in tickers
            }
            
            completed = 0
            # Collect results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    completed += 1
                    
                    # Update progress
                    if progress and completed % update_interval == 0:
                        progress.update(update_interval)
                    elif progress and completed == len(tickers):
                        progress.update(completed - progress.current)
                    
                    # Callback
                    if progress_callback:
                        progress_callback(completed, len(tickers), ticker, result)
                        
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    completed += 1
                    if progress:
                        progress.update(1)
        
        if progress:
            progress.finish()
        
        return results
    
    async def process_tickers_async(
        self,
        tickers: List[str],
        process_func: Callable,
        batch_size: int = None
    ) -> List[Dict]:
        """
        Process tickers asynchronously
        
        Args:
            tickers: List of ticker symbols
            process_func: Async function to process each ticker
            batch_size: Batch size for processing
        
        Returns:
            List of results
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            tasks = [process_func(ticker) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in async processing: {result}")
                elif result:
                    results.append(result)
        
        return results
    
    def chunk_tickers(self, tickers: List[str], chunk_size: int = None) -> List[List[str]]:
        """
        Split tickers into chunks
        
        Args:
            tickers: List of tickers
            chunk_size: Size of each chunk
        
        Returns:
            List of ticker chunks
        """
        chunk_size = chunk_size or self.chunk_size
        return [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    def get_all_tickers(self, use_file: bool = True) -> List[str]:
        """
        Get all tickers from file or config
        
        Args:
            use_file: Use ticker.txt file if available, otherwise use config
        
        Returns:
            List of ticker symbols
        """
        if use_file:
            loader = TickerLoader()
            file_tickers = loader.load_tickers()
            if file_tickers:
                return file_tickers
        
        # Fallback to config
        return [f"{ticker}.JK" if not ticker.endswith(".JK") else ticker for ticker in IDX_TICKER_LIST]

