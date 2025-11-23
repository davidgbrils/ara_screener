"""ML-based scoring for ARA probability prediction"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from config import ML_CONFIG, DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MLScorer:
    """ML model for predicting ARA probability"""
    
    def __init__(self):
        """Initialize ML scorer"""
        self.enabled = ML_CONFIG["ENABLED"]
        self.model_path = ML_CONFIG["MODEL_PATH"]
        self.model = None
        self.features = ML_CONFIG["FEATURES"]
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load ML model"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded ML model from {self.model_path}")
            else:
                logger.warning(f"ML model not found at {self.model_path}. ML scoring disabled.")
                self.enabled = False
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.enabled = False
    
    def predict(self, screener_result: Dict, indicator_values: Dict) -> Optional[float]:
        """
        Predict ARA probability
        
        Args:
            screener_result: Result from screener engine
            indicator_values: Latest indicator values
        
        Returns:
            Probability score (0-1) or None
        """
        if not self.enabled or self.model is None:
            return None
        
        try:
            # Extract features
            features = self._extract_features(screener_result, indicator_values)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([features])[0]
                return float(proba[1])  # Probability of positive class
            else:
                prediction = self.model.predict([features])[0]
                return float(prediction)
                
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return None
    
    def _extract_features(self, screener_result: Dict, indicator_values: Dict) -> list:
        """Extract feature vector"""
        features = []
        
        feature_map = {
            'rvol': indicator_values.get('rvol', 0),
            'atr_expansion': self._calculate_atr_expansion(indicator_values),
            'ma_structure_score': self._calculate_ma_structure_score(indicator_values),
            'bollinger_breakout': 1 if screener_result.get('signal') != 'NONE' else 0,
            'obv_divergence': self._calculate_obv_divergence(screener_result),
            'vwap_distance': self._calculate_vwap_distance(indicator_values),
            'rsi_zone': self._calculate_rsi_zone(indicator_values.get('rsi', 50)),
        }
        
        for feat_name in self.features:
            features.append(feature_map.get(feat_name, 0))
        
        return features
    
    def _calculate_atr_expansion(self, indicator_values: Dict) -> float:
        """Calculate ATR expansion ratio"""
        atr = indicator_values.get('atr', 0)
        close = indicator_values.get('close', 1)
        if close > 0:
            return atr / close
        return 0
    
    def _calculate_ma_structure_score(self, indicator_values: Dict) -> float:
        """Calculate MA structure score"""
        ma20 = indicator_values.get('ma20', 0)
        ma50 = indicator_values.get('ma50', 0)
        ma200 = indicator_values.get('ma200', 0)
        close = indicator_values.get('close', 0)
        
        score = 0.0
        if close > ma20 > ma50 > ma200:
            score = 1.0
        elif close > ma20 > ma50:
            score = 0.5
        
        return score
    
    def _calculate_obv_divergence(self, screener_result: Dict) -> float:
        """Calculate OBV divergence score"""
        # Simplified: check if OBV is rising
        patterns = screener_result.get('patterns', {})
        if patterns.get('pocket_pivot'):
            return 1.0
        return 0.5
    
    def _calculate_vwap_distance(self, indicator_values: Dict) -> float:
        """Calculate distance from VWAP"""
        close = indicator_values.get('close', 0)
        vwap = indicator_values.get('vwap', 0)
        
        if vwap > 0:
            return (close - vwap) / vwap
        return 0
    
    def _calculate_rsi_zone(self, rsi: float) -> float:
        """Calculate RSI zone score"""
        if 55 <= rsi <= 80:
            return 1.0
        elif 50 <= rsi < 55 or 80 < rsi <= 85:
            return 0.5
        return 0.0

