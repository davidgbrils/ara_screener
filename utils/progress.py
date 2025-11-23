"""Progress tracking utilities"""

import sys
from typing import Optional
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ProgressTracker:
    """Track and display progress for large operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker
        
        Args:
            total: Total number of items
            description: Description of operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.last_update = 0
    
    def update(self, increment: int = 1, show_progress: bool = True):
        """
        Update progress
        
        Args:
            increment: Number of items processed
            show_progress: Whether to display progress
        """
        self.current += increment
        
        if show_progress:
            self._display()
    
    def _display(self):
        """Display progress"""
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        # Calculate ETA
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            rate = self.current / elapsed
            remaining = self.total - self.current
            eta_seconds = remaining / rate if rate > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta)
        else:
            eta_str = "Calculating..."
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Display
        sys.stdout.write(
            f"\r{self.description}: [{bar}] {percentage:.1f}% "
            f"({self.current}/{self.total}) | ETA: {eta_str}"
        )
        sys.stdout.flush()
    
    def finish(self):
        """Finish and display final stats"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        
        sys.stdout.write(
            f"\r{self.description}: [{'█' * 40}] 100.0% "
            f"({self.current}/{self.total}) | Completed in {elapsed:.1f}s "
            f"({rate:.1f} items/sec)\n"
        )
        sys.stdout.flush()
        
        logger.info(
            f"{self.description} completed: {self.current}/{self.total} "
            f"in {elapsed:.1f}s ({rate:.1f} items/sec)"
        )

