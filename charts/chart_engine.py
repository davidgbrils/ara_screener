"""Professional candlestick chart generator"""

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
    
    def generate_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        signal: str,
        entry_levels: Dict = None,
        highlight_patterns: Dict = None
    ) -> Optional[Path]:
        """
        Generate candlestick chart with indicators
        
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
            # Prepare data (last 100 days for clarity)
            chart_df = df.tail(100).copy()
            
            # Create custom style
            mc = mpf.make_marketcolors(
                up='#00ff00',
                down='#ff0000',
                edge='inherit',
                wick={'upcolor': '#00ff00', 'downcolor': '#ff0000'},
                volume='in'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#333333',
                y_on_right=False
            )
            
            # Prepare additional plots
            addplot = self._prepare_addplot(chart_df, entry_levels)
            
            # Generate chart
            filename = f"{ticker}_{signal}.png"
            filepath = self.charts_dir / filename
            
            mpf.plot(
                chart_df,
                type='candle',
                style=style,
                volume=True if self.config["SHOW_VOLUME"] else False,
                addplot=addplot,
                figsize=self.config["FIG_SIZE"],
                savefig=dict(fname=filepath, dpi=self.config["DPI"], bbox_inches='tight'),
                show_nontrading=False,
                tight_layout=True,
            )
            
            logger.info(f"Generated chart: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating chart for {ticker}: {e}")
            return None
    
    def _prepare_addplot(self, df: pd.DataFrame, entry_levels: Dict = None) -> list:
        """Prepare additional plots (MAs, entry levels, etc.)"""
        addplot = []
        
        # Moving Averages
        if self.config["SHOW_INDICATORS"]:
            for period in [20, 50, 200]:
                ma_col = f'MA{period}'
                if ma_col in df.columns:
                    addplot.append(
                        mpf.make_addplot(
                            df[ma_col],
                            color='blue' if period == 20 else 'orange' if period == 50 else 'red',
                            width=1,
                            alpha=0.7
                        )
                    )
            
            # Bollinger Bands
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                addplot.append(
                    mpf.make_addplot(df['BB_Upper'], color='gray', width=1, alpha=0.5)
                )
                addplot.append(
                    mpf.make_addplot(df['BB_Lower'], color='gray', width=1, alpha=0.5)
                )

            # VWAP line
            if 'VWAP' in df.columns:
                addplot.append(
                    mpf.make_addplot(df['VWAP'], color='#6A5ACD', width=1, alpha=0.7)
                )

            # 52W High and 20D High lines (last value as horizontal reference)
            if 'HIGH_52W' in df.columns and not df['HIGH_52W'].isna().all():
                h52 = df['HIGH_52W'].iloc[-1]
                addplot.append(
                    mpf.make_addplot([h52] * len(df), color='#00CED1', linestyle=':', width=1, alpha=0.8)
                )
            if 'HIGH_20D' in df.columns and not df['HIGH_20D'].isna().all():
                h20 = df['HIGH_20D'].iloc[-1]
                addplot.append(
                    mpf.make_addplot([h20] * len(df), color='#FFD700', linestyle=':', width=1, alpha=0.8)
                )
        
        # Entry levels (horizontal lines)
        if entry_levels:
            entry_low = entry_levels.get('entry_low')
            entry_high = entry_levels.get('entry_high')
            stop_loss = entry_levels.get('stop_loss')
            tp1 = entry_levels.get('take_profit_1')
            tp2 = entry_levels.get('take_profit_2')
            
            if entry_low:
                addplot.append(
                    mpf.make_addplot(
                        [entry_low] * len(df),
                        color='green',
                        linestyle='--',
                        width=1,
                        alpha=0.7
                    )
                )
            
            if entry_high:
                addplot.append(
                    mpf.make_addplot(
                        [entry_high] * len(df),
                        color='green',
                        linestyle='--',
                        width=1,
                        alpha=0.7
                    )
                )
            
            if stop_loss:
                addplot.append(
                    mpf.make_addplot(
                        [stop_loss] * len(df),
                        color='red',
                        linestyle='--',
                        width=1,
                        alpha=0.7
                    )
                )
            
            if tp1:
                addplot.append(
                    mpf.make_addplot(
                        [tp1] * len(df),
                        color='blue',
                        linestyle='--',
                        width=1,
                        alpha=0.7
                    )
                )
            
            if tp2:
                addplot.append(
                    mpf.make_addplot(
                        [tp2] * len(df),
                        color='purple',
                        linestyle='--',
                        width=1,
                        alpha=0.7
                    )
                )
        
        return addplot
    
    def generate_comparison_chart(self, tickers: list, dfs: dict) -> Optional[Path]:
        """
        Generate comparison chart for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            dfs: Dictionary mapping ticker to DataFrame
        
        Returns:
            Path to saved chart or None
        """
        # This would require more complex plotting
        # For now, return None
        return None

