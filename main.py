"""
ARA FULL BOT V2 - Main Entry Point
Auto-run mode: python main.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from config import (
    RESULTS_DIR, HISTORY_DIR, CHARTS_DIR,
    IDX_TICKER_LIST, IDX_TICKER_SUFFIX
)
from data_fetch.data_fetcher import DataFetcher
from data_cleaner.normalizer import DataNormalizer
from indicators.indicator_engine import IndicatorEngine
from screener.screener_engine import ScreenerEngine
from ranking.ranking_engine import RankingEngine
from charts.chart_engine import ChartEngine
from notifier.telegram_notifier import TelegramNotifier
from state_manager.state_manager import StateManager
from processing.multiprocessing_engine import MultiprocessingEngine
from config import TELEGRAM_CONFIG, MULTIPROCESSING_CONFIG
from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager
from utils.ticker_loader import TickerLoader

logger = setup_logger(__name__)

class ARABot:
    """Main ARA Bot orchestrator"""
    
    def __init__(self):
        """Initialize ARA Bot"""
        self.data_fetcher = DataFetcher()
        self.normalizer = DataNormalizer()
        self.indicator_engine = IndicatorEngine()
        self.screener = ScreenerEngine()
        self.ranker = RankingEngine()
        self.chart_engine = ChartEngine()
        self.notifier = TelegramNotifier()
        self.state_manager = StateManager()
        self.processing_engine = MultiprocessingEngine()
    
    def process_ticker(self, ticker: str) -> Dict:
        """
        Process single ticker through full pipeline
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            Screener result or None
        """
        try:
            # Fetch data
            df = self.data_fetcher.fetch(ticker)
            if df is None:
                return None
            
            # Normalize
            df = self.normalizer.normalize(df)
            if df is None:
                return None
            
            # Calculate indicators
            df = self.indicator_engine.calculate_all(df)
            
            # Screen
            result = self.screener.screen(df, ticker)
            
            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()
            
            # Generate chart if signal (will generate for top 10 later)
            if result.get('signal') != 'NONE':
                chart_path = self.chart_engine.generate_chart(
                    df,
                    ticker,
                    result.get('signal'),
                    result.get('entry_levels'),
                    result.get('patterns')
                )
                result['chart_path'] = str(chart_path) if chart_path else None
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None
    
    def scan_all(
        self, 
        tickers: List[str] = None, 
        use_cache: bool = True,
        resume: bool = False,
        save_intermediate: bool = None
    ) -> List[Dict]:
        """
        Scan all tickers with progress tracking and resume capability
        
        Args:
            tickers: List of tickers (uses all IDX if None)
            use_cache: Whether to use cache
            resume: Resume from checkpoint if available
            save_intermediate: Save results incrementally (uses config if None)
        
        Returns:
            List of results
        """
        if tickers is None:
            # Try to load from ticker.txt file first
            loader = TickerLoader()
            file_tickers = loader.load_tickers()
            if file_tickers:
                tickers = file_tickers
                logger.info(f"Using tickers from ticker.txt: {len(tickers)} tickers")
            else:
                # Fallback to config
                tickers = self.processing_engine.get_all_tickers(use_file=False)
                logger.info(f"Using tickers from config: {len(tickers)} tickers")
        
        save_intermediate = save_intermediate or MULTIPROCESSING_CONFIG.get("SAVE_INTERMEDIATE_RESULTS", False)
        checkpoint_manager = CheckpointManager()
        
        # Handle resume
        if resume and MULTIPROCESSING_CONFIG.get("RESUME_CAPABILITY", False):
            processed_tickers, existing_results = checkpoint_manager.load_checkpoint()
            remaining_tickers = checkpoint_manager.get_remaining_tickers(tickers)
            
            if remaining_tickers:
                logger.info(f"Resuming scan: {len(processed_tickers)} already processed, {len(remaining_tickers)} remaining")
                tickers = remaining_tickers
                results = existing_results
            else:
                logger.info("No remaining tickers to process")
                return self.ranker.rank(existing_results)
        else:
            results = []
            if resume:
                checkpoint_manager.clear_checkpoint()
        
        logger.info(f"Starting scan of {len(tickers)} tickers...")
        
        # Progress callback for intermediate saving
        processed_tickers_set = set()
        
        def progress_callback(completed, total, ticker, result):
            if result and result.get('ticker'):
                processed_tickers_set.add(result.get('ticker'))
            
            if save_intermediate and completed % 100 == 0:
                # Save intermediate results every 100 tickers
                try:
                    # Get all results so far (will be updated in loop)
                    all_results = [r for r in results if r is not None]
                    if result:
                        all_results.append(result)
                    
                    self._save_intermediate_results(all_results)
                    checkpoint_manager.save_checkpoint(
                        processed_tickers_set,
                        all_results
                    )
                except Exception as e:
                    logger.error(f"Error saving intermediate results: {e}")
        
        # Process in parallel
        new_results = self.processing_engine.process_tickers(
            tickers,
            self.process_ticker,
            use_threads=True,  # Use threads for I/O bound operations
            show_progress=True,
            progress_callback=progress_callback if save_intermediate else None
        )
        
        # Combine results
        results.extend([r for r in new_results if r is not None])
        
        # Final save
        if save_intermediate:
            checkpoint_manager.save_checkpoint(set(tickers), results)
        
        # Clear checkpoint on success
        checkpoint_manager.clear_checkpoint()
        
        # Rank results - only top 10 with high confidence
        ranked_results = self.ranker.rank(results, top_n=10, min_confidence=0.65)
        
        logger.info(f"Scan complete: {len(ranked_results)} high-confidence signals found from {len(results)} processed")
        
        return ranked_results
    
    def _save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results"""
        try:
            json_file = RESULTS_DIR / "ara_intermediate.json"
            with open(json_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(results),
                    'results': results
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def save_results(self, results: List[Dict]):
        """
        Save results to files
        
        Args:
            results: List of results
        """
        try:
            # Save JSON
            json_file = RESULTS_DIR / "ara_latest.json"
            with open(json_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(results),
                    'results': results
                }, f, indent=2, default=str)
            
            # Save CSV
            if results:
                csv_file = RESULTS_DIR / "ara_latest.csv"
                df = pd.DataFrame(results)
                df.to_csv(csv_file, index=False)
            
            # Save to history
            history_file = HISTORY_DIR / f"ara_{datetime.now().strftime('%Y%m%d')}.json"
            with open(history_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(results),
                    'results': results
                }, f, indent=2, default=str)
            
            logger.info(f"Results saved: {len(results)} signals")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def send_notifications(self, results: List[Dict]):
        """
        Send Telegram notifications for top 10 with charts
        
        Args:
            results: List of results (already filtered to top 10)
        """
        if not self.notifier.enabled:
            return
        
        try:
            # Only send top 10 with high confidence
            top_10 = results[:10]
            
            if not top_10:
                logger.info("No high-confidence signals to send")
                return
            
            # Generate charts for top 10 if not already generated
            logger.info(f"Generating charts for top {len(top_10)} signals...")
            for result in top_10:
                if not result.get('chart_path'):
                    # Need to regenerate chart - fetch data again
                    ticker = result.get('ticker')
                    if ticker:
                        try:
                            df = self.data_fetcher.fetch(ticker)
                            if df is not None:
                                df = self.normalizer.normalize(df)
                                if df is not None:
                                    df = self.indicator_engine.calculate_all(df)
                                    chart_path = self.chart_engine.generate_chart(
                                        df,
                                        ticker,
                                        result.get('signal'),
                                        result.get('entry_levels'),
                                        result.get('patterns')
                                    )
                                    if chart_path:
                                        result['chart_path'] = str(chart_path)
                        except Exception as e:
                            logger.warning(f"Could not generate chart for {ticker}: {e}")
            
            # Send individual signals with charts for top 10
            logger.info(f"Sending {len(top_10)} top signals with charts...")
            
            for i, result in enumerate(top_10, 1):
                chart_path = None
                if result.get('chart_path'):
                    chart_path = Path(result['chart_path'])
                    if not chart_path.exists():
                        chart_path = None
                
                # Send signal with chart
                success = self.notifier.send_signal(result, chart_path)
                if success:
                    logger.info(f"✅ Sent signal {i}/10: {result.get('ticker')} (Confidence: {result.get('confidence', 0):.1%})")
                else:
                    logger.warning(f"❌ Failed to send signal: {result.get('ticker')}")
            
            # Send summary message
            if TELEGRAM_CONFIG.get("SEND_SUMMARY", True):
                self.notifier.send_summary(top_10, top_n=10)
            
            logger.info(f"✅ Notifications sent: {len(top_10)} top signals with charts")
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def run(
        self, 
        tickers: List[str] = None,
        resume: bool = False
    ):
        """
        Run full scan workflow
        
        Args:
            tickers: Optional list of tickers to scan
            resume: Resume from checkpoint if available
        """
        logger.info("=" * 50)
        logger.info("ARA BOT V2 - Starting Full Scan")
        logger.info("=" * 50)
        
        # Scan
        results = self.scan_all(tickers, resume=resume)
        
        # Save results
        self.save_results(results)
        
        # Update state
        self.state_manager.save_state(results)
        
        # Send notifications
        self.send_notifications(results)
        
        # Print summary
        summary = self.ranker.get_summary_stats(results)
        logger.info("=" * 50)
        logger.info("SCAN SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total scanned: {summary.get('total', 0)}")
        logger.info(f"STRONG_AURA: {summary.get('strong_aura', 0)}")
        logger.info(f"WATCHLIST: {summary.get('watchlist', 0)}")
        logger.info(f"POTENTIAL: {summary.get('potential', 0)}")
        logger.info("=" * 50)
        
        return results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARA Bot V2 - Multi-Bagger Scanner")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Specific tickers to scan (e.g., BBCA.JK BBRI.JK)"
    )
    
    args = parser.parse_args()
    
    bot = ARABot()
    results = bot.run(tickers=args.tickers, resume=args.resume)
    
    # Print top 10
    print("\n" + "=" * 50)
    print("TOP 10 RESULTS")
    print("=" * 50)
    for i, result in enumerate(results[:10], 1):
        print(f"{i}. {result.get('ticker')} - {result.get('signal')} ({result.get('score', 0):.1%})")

if __name__ == "__main__":
    main()

