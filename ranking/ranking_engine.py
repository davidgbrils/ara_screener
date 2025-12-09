"""Ranking engine for sorting and filtering results"""

import pandas as pd
from typing import List, Dict
from .ml_scorer import MLScorer
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RankingEngine:
    """Rank and sort screener results"""
    
    def __init__(self):
        """Initialize ranking engine"""
        self.ml_scorer = MLScorer()
    
    def rank(self, results: List[Dict], top_n: int = None, min_confidence: float = 0.4) -> List[Dict]:
        """
        Rank results by score, confidence, and ML probability
        
        Args:
            results: List of screener results
            top_n: Number of top results to return
            min_confidence: Minimum confidence score (default 0.7 for high quality)
        
        Returns:
            Sorted list of results
        """
        # Add ML probability if enabled
        for result in results:
            if result.get('signal') != 'NONE':
                # Get indicator values for ML
                indicator_values = {
                    'close': result.get('latest_price', 0),
                    'rvol': result.get('patterns', {}).get('volume_climax', 0),
                    'rsi': 0,  # Would need to pass from indicators
                    'atr': 0,
                    'ma20': 0,
                    'ma50': 0,
                    'ma200': 0,
                    'vwap': 0,
                }
                
                ml_prob = self.ml_scorer.predict(result, indicator_values)
                if ml_prob is not None:
                    result['ml_probability'] = ml_prob
                    # Combine score with ML probability
                    result['combined_score'] = (result.get('score', 0) * 0.6) + (ml_prob * 0.4)
                else:
                    result['combined_score'] = result.get('score', 0)
            else:
                result['combined_score'] = 0.0
        
        # Filter by signal and data quality
        filtered = []
        for r in results:
            if r.get('signal') != 'NONE':
                # Check data quality (handle both dict and string)
                data_quality = r.get('data_quality', {})
                if isinstance(data_quality, str):
                    # Try to parse string representation
                    try:
                        import ast
                        data_quality = ast.literal_eval(data_quality)
                    except:
                        # If parsing fails, assume valid
                        data_quality = {'is_valid': True}
                
                is_valid = data_quality.get('is_valid', True) if isinstance(data_quality, dict) else True
                
                if is_valid:
                    # Check confidence
                    confidence = r.get('confidence', 0)
                    if confidence >= min_confidence:
                        filtered.append(r)
                    else:
                        logger.debug(f"Filtered out {r.get('ticker')}: confidence {confidence:.3f} < {min_confidence}")
                else:
                    logger.debug(f"Filtered out {r.get('ticker')}: data quality invalid")
        
        # Sort by combined score and confidence
        sorted_results = sorted(
            filtered,
            key=lambda x: (
                x.get('combined_score', x.get('score', 0)) * 0.6 + 
                x.get('confidence', 0) * 0.4
            ),
            reverse=True
        )
        
        logger.info(f"Ranking: {len(results)} total results, {len(filtered)} passed filters (confidence >= {min_confidence})")
        
        if top_n:
            result = sorted_results[:top_n]
            logger.info(f"Returning top {len(result)} results")
            return result
        
        return sorted_results
    
    def filter_by_signal(self, results: List[Dict], signal: str) -> List[Dict]:
        """
        Filter results by signal type
        
        Args:
            results: List of results
            signal: Signal type to filter
        
        Returns:
            Filtered results
        """
        return [r for r in results if r.get('signal') == signal]
    
    def get_summary_stats(self, results: List[Dict]) -> Dict:
        """
        Get summary statistics
        
        Args:
            results: List of results
        
        Returns:
            Summary statistics
        """
        total = len(results)
        signals = {}
        
        for result in results:
            signal = result.get('signal', 'NONE')
            signals[signal] = signals.get(signal, 0) + 1
        
        return {
            'total': total,
            'signals': signals,
            'strong_aura': signals.get('STRONG_AURA', 0),
            'watchlist': signals.get('WATCHLIST', 0),
            'potential': signals.get('POTENTIAL', 0),
        }

