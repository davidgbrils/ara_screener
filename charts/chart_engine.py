"""Professional candlestick chart generator"""

import matplotlib
matplotlib.use('Agg') # Fix for main thread error
import mplfinance as mpf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from config import CHART_CONFIG, CHARTS_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ChartEngine:
    """Generate professional candlestick charts"""
    
    def __init__(self):
        """Initialize chart engine"""
        self.config = CHART_CONFIG
        self.charts_dir = CHARTS_DIR
        self.charts_dir.mkdir(exist_ok=True)

    def cleanup_charts(self):
        """Delete all existing charts in the charts directory"""
        try:
            logger.info(f"Cleaning up charts directory: {self.charts_dir}")
            for chart_file in self.charts_dir.glob("*.png"):
                try:
                    chart_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {chart_file}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning charts directory: {e}")
    
    def generate_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        signal: str,
        entry_levels: Dict = None,
        highlight_patterns: Dict = None
    ) -> Optional[Path]:
        """
        Generate candlestich chart with indicators and signal markers
        
        Args:
            df: DataFrame with OHLCV and indicators
            ticker: Ticker symbol
            signal: Signal type
            entry_levels: Entry/SL/TP levels
            highlight_patterns: Patterns to highlight
        
        Returns:
            Path to saved chart or None
        """
        if df is None or df.empty:
            return None
        
        try:
            # Prepare data (last 120 days for better context)
            chart_df = df.tail(120).copy()
            
            # Create custom Professional Dark style
            mc = mpf.make_marketcolors(
                up='#26a69a',      # TradingView Green
                down='#ef5350',    # TradingView Red
                edge='inherit',
                wick='inherit',
                volume={'up': '#1b5e20', 'down': '#b71c1c'},
                ohlc='inherit'
            )
            
            style = mpf.make_mpf_style(
                base_mpf_style='nightclouds',
                marketcolors=mc,
                gridstyle=':',
                gridcolor='#2a2e39',
                facecolor='#131722',  # Dark background
                edgecolor='#2a2e39',
                rc={'figure.facecolor': '#131722', 'axes.labelcolor': '#b2b5be', 'xtick.color': '#b2b5be', 'ytick.color': '#b2b5be'}
            )
            
            # Prepare additional plots (Indicators & Signals)
            addplot = self._prepare_addplot(chart_df, entry_levels, highlight_patterns)
            
            # Title with signal info
            title = f"{ticker} - {signal.replace('_', ' ')} Setup"
            
            # Generate chart
            filename = f"{ticker}_{signal}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.png"
            filepath = self.charts_dir / filename
            
            mpf.plot(
                chart_df,
                type='candle',
                style=style,
                volume=True if self.config["SHOW_VOLUME"] else False,
                addplot=addplot,
                title=dict(title=title, color='#d1d4dc', size=14),
                figsize=self.config["FIG_SIZE"],
                savefig=dict(fname=filepath, dpi=self.config["DPI"], bbox_inches='tight', facecolor='#131722'),
                show_nontrading=False,
                tight_layout=True,
                panel_ratios=(6, 2), # Larger price panel
                datetime_format='%b %d',
                xrotation=0,
                scale_width_adjustment=dict(candle=1.1, volume=0.7),
            )
            
            logger.info(f"Generated professional chart: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating chart for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _prepare_addplot(self, df: pd.DataFrame, entry_levels: Dict = None, patterns: Dict = None) -> list:
        """Prepare additional plots (MAs, entry levels, signals)"""
        addplot = []
        
        # 1. Moving Averages (Smooth lines)
        if self.config["SHOW_INDICATORS"]:
            for period, color in [(20, '#2962ff'), (50, '#ff9800'), (200, '#f50057')]: # Blue, Orange, Pink
                ma_col = f'MA{period}'
                if ma_col in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df[ma_col],
                            color=color,
                            width=1.5 if period == 20 else 1.2,
                            alpha=0.9
                        )
                    )
            
            # Bollinger Bands (Subtle fill)
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                addplot.append(
                    mpf.make_addplot(df['BB_Upper'], color='#3179f5', width=0.8, alpha=0.3)
                )
                addplot.append(
                    mpf.make_addplot(df['BB_Lower'], color='#3179f5', width=0.8, alpha=0.3)
                )
                # Note: fill_between is not easily supported in simple addplot list, skipping for simplicity

            # VWAP (Institutional benchmark)
            if 'VWAP' in df.columns:
                addplot.append(
                    mpf.make_addplot(df['VWAP'], color='#e040fb', width=1.2, linestyle='-.', alpha=0.8)
                )

        # 2. Buy Signals & Volume Spikes (Markers)
        # Pocket Pivot / Volume Spike Marker
        if 'Volume' in df.columns:
            vol_ma = df['Volume'].rolling(20).mean()
            vol_spike_mask = (df['Volume'] > vol_ma * 2.5) & (df['Close'] > df['Open'])
            
            # Create a series of NaNs, populate only where condition is True
            spike_marker = np.full(len(df), np.nan)
            spike_marker[vol_spike_mask] = df['Low'][vol_spike_mask] * 0.98 # Place below candle
            
            if np.any(~np.isnan(spike_marker)):
                addplot.append(
                    mpf.make_addplot(
                        spike_marker,
                        type='scatter',
                        markersize=50,
                        marker='^',
                        color='#00e676', # Bright green
                        panel=0
                    )
                )

        # 3. Entry Levels (Actionable Zones)
        if entry_levels:
            entry_low = entry_levels.get('entry_low')
            entry_high = entry_levels.get('entry_high')
            stop_loss = entry_levels.get('stop_loss')
            tp1 = entry_levels.get('take_profit_1')
            
            # We plot these as horizontal lines across the whole chart for clarity
            if entry_high and entry_low:
                # Use mean for single line visualization of entry
                entry_avg = (entry_low + entry_high) / 2
                addplot.append(
                    mpf.make_addplot(
                        [entry_avg] * len(df),
                        color='#00e676',
                        linestyle='--',
                        width=1.0,
                        alpha=0.8
                    )
                )
            
            if stop_loss:
                addplot.append(
                    mpf.make_addplot(
                        [stop_loss] * len(df),
                        color='#ff1744',
                        linestyle='--',
                        width=1.0,
                        alpha=0.8
                    )
                )
            
            if tp1:
                addplot.append(
                    mpf.make_addplot(
                        [tp1] * len(df),
                        color='#00b0ff',
                        linestyle='--',
                        width=1.0,
                        alpha=0.8
                    )
                )
        
        return addplot

    def generate_comparison_chart(self, tickers: list, dfs: dict) -> Optional[Path]:
        """Placeholder for comparison chart"""
        return None
