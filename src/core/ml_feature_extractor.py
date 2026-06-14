#!/usr/bin/env python3
"""
ML Feature Extractor (Tier 3, Item 12)
=====================================
Converts market intelligence and signal data into 
numerical features for XGBoost/LightGBM training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class MLFeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, intel: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, float]:
        """
        Flattens complex objects into a flat feature vector.
        """
        features = {}

        # ── Regime Features ─────────────────────────────────────
        features['er'] = intel.get('efficiency_ratio', 0.5)
        features['compression'] = intel.get('compression_score', 0.0)
        features['volatility_percentile'] = intel.get('volatility_percentile', 50.0)

        # ── Structure Features ──────────────────────────────────
        struct = intel.get('structure')
        if struct:
            features['trend_val'] = 1.0 if struct.trend == 'BULLISH' else (-1.0 if struct.trend == 'BEARISH' else 0.0)
            features['is_bos'] = 1.0 if struct.last_bos != 'NONE' else 0.0
            features['is_choch'] = 1.0 if struct.last_choch != 'NONE' else 0.0

        # ── Liquidity Features ──────────────────────────────────
        sweep = intel.get('liquidity_sweeps', 'NONE')
        features['is_sweep'] = 1.0 if sweep != 'NONE' else 0.0
        
        # ── Context Features ────────────────────────────────────
        features['hour'] = pd.to_datetime(signal.get('timestamp')).hour
        features['confidence'] = signal.get('confidence', 0)
        
        # ── Volume Features ─────────────────────────────────────
        features['is_vdb'] = 1.0 if intel.get('vdb_breakout') else 0.0
        features['is_climax'] = 1.0 if intel.get('volume_climax') != 'NONE' else 0.0

        return features
