"""Checkpoint system for resume capability"""

import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from datetime import datetime
from config import CACHE_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CheckpointManager:
    """Manage checkpoints for resuming scans"""
    
    def __init__(self, checkpoint_file: Path = None):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = checkpoint_file or CACHE_DIR / "scan_checkpoint.json"
        self.checkpoint_file.parent.mkdir(exist_ok=True)
    
    def save_checkpoint(self, processed_tickers: Set[str], results: List[Dict] = None):
        """
        Save checkpoint
        
        Args:
            processed_tickers: Set of processed tickers
            results: Optional results to save
        """
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "processed_tickers": list(processed_tickers),
                "count": len(processed_tickers),
                "results": results or [],
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.debug(f"Checkpoint saved: {len(processed_tickers)} tickers")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self) -> Tuple[Set[str], List[Dict]]:
        """
        Load checkpoint
        
        Returns:
            Tuple of (processed_tickers set, results list)
        """
        try:
            if not self.checkpoint_file.exists():
                return set(), []
            
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            processed = set(data.get("processed_tickers", []))
            results = data.get("results", [])
            
            logger.info(f"Checkpoint loaded: {len(processed)} tickers already processed")
            return processed, results
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return set(), []
    
    def clear_checkpoint(self):
        """Clear checkpoint"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            logger.debug("Checkpoint cleared")
        except Exception as e:
            logger.error(f"Error clearing checkpoint: {e}")
    
    def get_remaining_tickers(self, all_tickers: List[str]) -> List[str]:
        """
        Get list of remaining tickers to process
        
        Args:
            all_tickers: List of all tickers
        
        Returns:
            List of remaining tickers
        """
        processed, _ = self.load_checkpoint()
        remaining = [t for t in all_tickers if t not in processed]
        return remaining

