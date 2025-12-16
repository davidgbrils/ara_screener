"""
ARA FULL BOT V2 - Main Entry Point
Auto-run mode: python main.py
"""

import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
    
    def process_ticker(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Process single ticker through full pipeline
        
        Args:
            ticker: Ticker symbol
            force_refresh: Force refresh data (get today's data)
        
        Returns:
            Screener result or None
        """
        try:
            # Fetch data (with force refresh if needed)
            df = self.data_fetcher.fetch(ticker, force_refresh=force_refresh)
            if df is None:
                return None
            
            # Validate data freshness
            if len(df) > 0:
                from datetime import datetime, timezone
                last_date = df.index[-1]
                today = datetime.now(timezone.utc).date()
                
                if hasattr(last_date, 'date'):
                    last_date_only = last_date.date()
                else:
                    last_date_only = last_date
                
                # Log if data is not from today
                if last_date_only < today:
                    logger.debug(f"{ticker}: Using data from {last_date_only} (today: {today})")
            
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
            
            # Generate chart ONLY for SUPER_ALPHA and STRONG_AURA
            if result.get('signal') in ['SUPER_ALPHA', 'STRONG_AURA']:
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
        
        # If force refresh, don't use cache
        if not use_cache:
            logger.info("Force refresh enabled: Will fetch fresh data for all tickers")
        
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
        logger.info(f"Using cache: {use_cache}, Force refresh: {not use_cache}")
        
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
        
        # Create wrapper function for process_ticker with force_refresh
        def process_ticker_wrapper(ticker: str):
            return self.process_ticker(ticker, force_refresh=not use_cache)
        
        # Process in parallel
        new_results = self.processing_engine.process_tickers(
            tickers,
            process_ticker_wrapper,
            use_threads=True,  # Use threads for I/O bound operations
            show_progress=True,
            progress_callback=progress_callback if save_intermediate else None
        )
        
        # Combine results - filter out None and ensure all have required fields
        for r in new_results:
            if r is not None and r.get('ticker'):
                # Ensure confidence exists
                if 'confidence' not in r:
                    r['confidence'] = 0.0
                # Ensure data_quality is dict
                if 'data_quality' in r and isinstance(r['data_quality'], str):
                    try:
                        import ast
                        r['data_quality'] = ast.literal_eval(r['data_quality'])
                    except:
                        r['data_quality'] = {'is_valid': True, 'issues': [], 'quality_score': 1.0}
                results.append(r)
        
        # Final save
        if save_intermediate:
            checkpoint_manager.save_checkpoint(set(tickers), results)
        
        # Clear checkpoint on success
        checkpoint_manager.clear_checkpoint()
        
        # Log before ranking
        signals_before = [r for r in results if r and r.get('signal') != 'NONE']
        logger.info(f"Before ranking: {len(signals_before)} signals found from {len(results)} processed")
        
        if signals_before:
            # Show confidence distribution
            confidences = [r.get('confidence', 0) for r in signals_before if r.get('confidence')]
            if confidences:
                logger.info(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}, mean: {sum(confidences)/len(confidences):.3f}")
        
        # Rank results - top 5 with high confidence
        ranked_results = self.ranker.rank(results, top_n=5, min_confidence=0.70)
        
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
        Send Telegram notifications for Top 5 SUPER_ALPHA and Top 5 STRONG_AURA
        
        Args:
            results: List of results
        """
        if not self.notifier.enabled:
            return
        
        try:
            # Filter specifically for the requested signals
            super_alpha = [r for r in results if r.get('signal') == 'SUPER_ALPHA']
            strong_aura = [r for r in results if r.get('signal') == 'STRONG_AURA']
            
            # Sort by score descending
            super_alpha.sort(key=lambda x: x.get('score', 0), reverse=True)
            strong_aura.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Take top 5 of each
            top_super = super_alpha[:5]
            top_strong = strong_aura[:5]
            
            final_list = top_super + top_strong
            
            if not final_list:
                logger.info("No SUPER_ALPHA or STRONG_AURA signals found to send.")
                return

            logger.info(f"Preparing notifications for {len(top_super)} Super Alpha and {len(top_strong)} Strong Aura signals...")
            
            # Generate charts ONLY for these final candidates
            for result in final_list:
                ticker = result.get('ticker')
                if not result.get('chart_path') and ticker:
                    try:
                        # Regenerate df for chart
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
            
            # Send individual signals with charts
            for i, result in enumerate(final_list, 1):
                chart_path = None
                if result.get('chart_path'):
                    chart_path = Path(result['chart_path'])
                    if not chart_path.exists():
                        chart_path = None
                
                # Send signal with chart
                success = self.notifier.send_signal(result, chart_path)
                if success:
                    logger.info(f"[SENT] {result.get('signal')} {i}: {result.get('ticker')}")
            
            # Send summary message (only these top picks)
            if TELEGRAM_CONFIG.get("SEND_SUMMARY", True):
                self.notifier.send_summary(final_list, top_n=10)
            
            logger.info(f"[DONE] Sent {len(final_list)} priority notifications")
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run(
        self, 
        tickers: List[str] = None,
        resume: bool = False,
        force_refresh: bool = False
    ):
        """
        Run full scan workflow
        
        Args:
            tickers: Optional list of tickers to scan
            resume: Resume from checkpoint if available
            force_refresh: Force refresh data (get today's data)
        """
        from datetime import datetime
        
        logger.info("=" * 50)
        logger.info("ARA BOT V2 - Starting Full Scan")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Force Refresh: {force_refresh}")
        logger.info("=" * 50)
        
        # Cleanup old charts before starting
        self.chart_engine.cleanup_charts()
        
        # Scan with fresh data if requested
        if force_refresh:
            results = self.scan_all(tickers, resume=resume, use_cache=False)
        else:
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

def interactive_mode():
    """
    Interactive mode with capital input for personalized recommendations
    """
    from recommendation.capital_advisor import CapitalAdvisor
    
    print("\n" + "=" * 60)
    print("🚀 ARA BOT V3 - Professional Quant Screening System")
    print("=" * 60)
    print("\nSelamat datang! Bot ini akan membantu Anda menemukan")
    print("peluang trading berdasarkan modal dan gaya trading Anda.")
    print()
    
    # Get capital input
    while True:
        try:
            capital_input = input("💰 Masukkan modal Anda (Rp): ").replace(",", "").replace(".", "").strip()
            capital = float(capital_input)
            if capital < 1_000_000:
                print("⚠️ Modal minimal Rp 1.000.000")
                continue
            break
        except ValueError:
            print("❌ Input tidak valid. Masukkan angka saja (contoh: 10000000)")
    
    # Get risk profile
    print("\n📊 Pilih Profil Risiko:")
    print("  1. Conservative (Aman, risiko rendah)")
    print("  2. Moderate (Seimbang)")
    print("  3. Aggressive (Berani, risiko tinggi)")
    
    risk_choice = input("Pilihan (1/2/3): ").strip()
    risk_profiles = {"1": "conservative", "2": "moderate", "3": "aggressive"}
    risk_profile = risk_profiles.get(risk_choice, "moderate")
    
    # Get mode preference
    print("\n🎯 Pilih Mode Screening:")
    print("  A. BPJS (Saham Sehat - Swing/Scalping Aman)")
    print("  B. ARA (Potensi Auto Rejection Atas)")
    print("  C. Multi-Bagger (Saham Undervalue)")
    print("  D. Scalping (Intraday Quick Trade)")
    print("  E. Gorengan/UMA (Filter Risiko)")
    print("  F. Semua Mode (Rekomendasi Terbaik)")
    
    mode_choice = input("Pilihan (A/B/C/D/E/F): ").strip().upper()
    mode_map = {
        "A": "bpjs", "B": "ara", "C": "multibagger",
        "D": "scalping", "E": "gorengan", "F": "all"
    }
    selected_mode = mode_map.get(mode_choice, "scalping")
    
    print(f"\n✅ Modal: Rp{capital:,.0f}")
    print(f"✅ Profil Risiko: {risk_profile.upper()}")
    print(f"✅ Mode: {selected_mode.upper()}")
    print("\n🔍 Memulai scanning... (ini mungkin memakan waktu beberapa menit)")
    print("-" * 60)
    
    # Run bot
    bot = ARABot()
    results = bot.run(force_refresh=False)
    
    # Generate capital-based recommendations
    print("\n" + "=" * 60)
    print("📈 GENERATING PERSONALIZED RECOMMENDATIONS...")
    print("=" * 60)
    
    advisor = CapitalAdvisor(capital, risk_profile)
    allocation = advisor.allocate_capital(results, selected_mode)
    
    # Print recommendations
    print(advisor.format_allocation_text(allocation))
    
    # Send to Telegram if available
    from config import TELEGRAM_CONFIG
    if TELEGRAM_CONFIG.get("ENABLED"):
        from notifier.telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier()
        
        # Send capital-based summary
        summary_text = advisor.format_allocation_text(allocation)
        notifier.send_message(summary_text[:4000])  # Telegram limit
        print("\n✅ Rekomendasi terkirim ke Telegram!")
    
    return results, allocation


def analyze_ticker(ticker: str, notify: bool = False) -> None:
    """
    Perform multi-timeframe technical analysis on a ticker
    
    Args:
        ticker: Stock ticker
        notify: Send to Telegram
    """
    print(f"\n{'='*60}")
    print(f"📊 TECHNICAL ANALYSIS: {ticker}")
    print(f"{'='*60}\n")
    
    # Initialize components
    from analysis import TechnicalAnalyzer
    from data_fetch.data_fetcher import DataFetcher
    from charts.chart_engine import ChartEngine
    from notifier.telegram_notifier import TelegramNotifier
    
    analyzer = TechnicalAnalyzer()
    fetcher = DataFetcher()
    chart_engine = ChartEngine()
    notifier = TelegramNotifier()
    
    # Normalize ticker
    if not ticker.endswith('.JK'):
        ticker = f"{ticker}.JK"
    
    # Fetch data for each timeframe
    print("📥 Fetching data...")
    data = {}
    
    # Daily data
    df_daily = fetcher.fetch(ticker, force_refresh=True)
    if df_daily is not None:
        data["1D"] = df_daily
        print(f"  ✓ 1D: {len(df_daily)} candles")
    
    # For intraday we simulate from daily data (Yahoo limitation)
    # In production, use intraday API
    if "1D" in data:
        # Simulate 4h from daily
        data["4h"] = data["1D"].copy()
        data["1h"] = data["1D"].copy()
        print(f"  ✓ 4h, 1h: Using daily data (simulated)")
    
    if not data:
        print("❌ No data available for analysis")
        return
    
    # Perform analysis
    print("\n🔍 Analyzing...")
    analysis = analyzer.analyze(ticker, data)
    
    if not analysis:
        print("❌ Analysis failed")
        return
    
    # Print results
    print(f"\n{'='*50}")
    print("🔍 MULTI-TIMEFRAME SUMMARY")
    print(f"{'='*50}")
    
    tf_order = ["1D", "4h", "1h", "15m", "5m"]
    for tf in tf_order:
        if tf in analysis.timeframes:
            tf_a = analysis.timeframes[tf]
            print(f"  {tf}: {tf_a.trend_emoji} {tf_a.trend} ({tf_a.summary})")
    
    print(f"\n📈 PRIMARY TREND: {analysis.primary_trend}")
    print(f"🎯 TRADING BIAS: {analysis.bias}")
    print(f"📊 CONFLUENCE: {analysis.confluence} ({analysis.confidence:.0f}%)")
    
    print(f"\n🔒 KEY LEVELS")
    print(f"  Resistance: Rp{analysis.key_resistance:,.0f}")
    print(f"  Current:    Rp{analysis.current_price:,.0f}")
    print(f"  Support:    Rp{analysis.key_support:,.0f}")
    
    if analysis.trading_plan and analysis.bias != "AVOID":
        plan = analysis.trading_plan
        print(f"\n📝 TRADING PLAN")
        print(f"  Entry: Rp{plan.entry_low:,.0f} - Rp{plan.entry_high:,.0f}")
        print(f"  TP1:   Rp{plan.tp1:,.0f} (+{plan.tp1_pct:.1f}%)")
        print(f"  TP2:   Rp{plan.tp2:,.0f} (+{plan.tp2_pct:.1f}%)")
        print(f"  SL:    Rp{plan.sl:,.0f} (-{plan.sl_pct:.1f}%)")
        print(f"  R:R:   1:{plan.risk_reward:.1f}")
    
    if analysis.warnings:
        print(f"\n⚠️ WARNINGS")
        for w in analysis.warnings:
            print(f"  • {w}")
    
    # Generate chart
    print(f"\n📈 Generating chart...")
    if "1D" in data:
        chart_path = chart_engine.generate_chart(
            data["1D"],
            ticker,
            f"TA_{analysis.primary_trend}",
            analysis.trading_plan.__dict__ if analysis.trading_plan else {},
            {}
        )
        if chart_path:
            print(f"  ✓ Chart saved: {chart_path}")
    else:
        chart_path = None
    
    # Send to Telegram
    if notify:
        print(f"\n📤 Sending to Telegram...")
        success = notifier.send_technical_analysis(analysis, chart_path)
        if success:
            print("  ✓ Sent successfully!")
        else:
            print("  ❌ Failed to send")
    
    print(f"\n{'='*60}")
    print("⚠️ DISCLAIMER: This is not financial advice. Trade at your own risk.")
    print(f"{'='*60}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARA Bot V3 - Professional Quant Screener")
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
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with capital input"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Trading capital in IDR (e.g., 10000000)"
    )
    parser.add_argument(
        "--mode",
        choices=["bpjs", "ara", "multibagger", "scalping", "gorengan", "all"],
        default="scalping",
        help="Screening mode (default: scalping)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh data"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        help="Perform technical analysis on a ticker (e.g., --analyze BBCA)"
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send analysis to Telegram"
    )
    
    args = parser.parse_args()
    
    # Technical Analysis mode
    if args.analyze:
        analyze_ticker(args.analyze, notify=args.notify)
        return None
    
    # Interactive mode
    if args.interactive:
        results, allocation = interactive_mode()
        return results
    
    # Standard mode with optional capital
    bot = ARABot()
    results = bot.run(tickers=args.tickers, resume=args.resume, force_refresh=args.refresh)
    
    # If capital provided, generate recommendations
    if args.capital:
        from recommendation.capital_advisor import CapitalAdvisor
        
        print("\n" + "=" * 60)
        print(f"📊 REKOMENDASI UNTUK MODAL Rp{args.capital:,.0f}")
        print("=" * 60)
        
        advisor = CapitalAdvisor(args.capital)
        allocation = advisor.allocate_capital(results, args.mode)
        print(advisor.format_allocation_text(allocation))
    else:
        # Print top results (original behavior)
        print("\n" + "=" * 50)
        print("TOP 5 RESULTS")
        print("=" * 50)
        for i, result in enumerate(results[:5], 1):
            ticker = result.get('ticker', 'N/A')
            signal = result.get('signal', 'N/A')
            score = result.get('score', 0)
            
            # Get mode info if available
            all_modes = result.get('classifications', {}).get('all_modes', {})
            best_mode = all_modes.get('best_mode', '')
            bandar_timing = all_modes.get('bandar_timing', '')
            
            print(f"{i}. {ticker} - {signal} ({score:.1%})")
            if best_mode:
                print(f"   Mode: {best_mode} | Timing: {bandar_timing[:30]}...")
    
    return results


if __name__ == "__main__":
    main()

