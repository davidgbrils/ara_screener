"""State manager for tracking signal changes"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from config import RESULTS_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class StateManager:
    """Manage state to detect signal changes"""
    
    def __init__(self, state_file: Path = None):
        """
        Initialize state manager
        
        Args:
            state_file: Path to state file
        """
        self.state_file = state_file or RESULTS_DIR / "prev_signals.json"
        self.state_file.parent.mkdir(exist_ok=True)
        self.current_state = {}
        self.previous_state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load previous state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state: {e}")
        
        return {}
    
    def save_state(self, results: List[Dict]):
        """
        Save current state
        
        Args:
            results: List of current results
        """
        try:
            # Create state dictionary
            state = {}
            for result in results:
                ticker = result.get('ticker')
                if ticker:
                    state[ticker] = {
                        'signal': result.get('signal', 'NONE'),
                        'score': result.get('score', 0),
                        'timestamp': result.get('timestamp', ''),
                    }
            
            self.current_state = state
            
            # Save to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"State saved: {len(state)} tickers")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_signal_changes(self, results: List[Dict]) -> List[Dict]:
        """
        Get results with changed signals
        
        Args:
            results: List of current results
        
        Returns:
            List of results with changed signals
        """
        changes = []
        
        for result in results:
            ticker = result.get('ticker')
            current_signal = result.get('signal', 'NONE')
            
            prev_data = self.previous_state.get(ticker, {})
            prev_signal = prev_data.get('signal', 'NONE')
            
            if current_signal != prev_signal and current_signal != 'NONE':
                result['previous_signal'] = prev_signal
                result['signal_changed'] = True
                changes.append(result)
        
        return changes
    
    def has_changed(self, ticker: str, current_signal: str) -> bool:
        """
        Check if signal has changed for a ticker
        
        Args:
            ticker: Ticker symbol
            current_signal: Current signal
        
        Returns:
            True if changed
        """
        prev_data = self.previous_state.get(ticker, {})
        prev_signal = prev_data.get('signal', 'NONE')
        return current_signal != prev_signal

