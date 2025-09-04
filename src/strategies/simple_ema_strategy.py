#!/usr/bin/env python3
"""
Simple EMA Strategy - Generates signals based on basic EMA crossover
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from datetime import datetime
from src.core.strategy import Strategy

class SimpleEmaStrategy(Strategy):
    """Simple EMA crossover strategy that always generates signals"""
    
    def __init__(self, params: Dict[str, Any] = None):
        params = params or {}
        self.ema_short = params.get("ema_short", 9)
        self.ema_long = params.get("ema_long", 21)
        self.rsi_period = params.get("rsi_period", 14)
        self.min_candles = 30  # Much lower requirement
        super().__init__("simple_ema", params)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simple analysis that always generates a signal"""
        try:
            if len(data) < self.min_candles:
                return {'signal': 'NO TRADE', 'reason': f'insufficient data: {len(data)} < {self.min_candles}'}
            
            # Get current values
            ema_short = data['ema_9'].iloc[-1] if 'ema_9' in data.columns else data['close'].ewm(span=self.ema_short).mean().iloc[-1]
            ema_long = data['ema_21'].iloc[-1] if 'ema_21' in data.columns else data['close'].ewm(span=self.ema_long).mean().iloc[-1]
            rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
            atr = data['atr'].iloc[-1] if 'atr' in data.columns else 100
            current_price = data['close'].iloc[-1]
            
            # Simple signal generation
            if ema_short > ema_long and rsi < 70:
                return {
                    'signal': 'BUY CALL',
                    'price': current_price,
                    'confidence_score': 45,  # High confidence
                    'stop_loss': 1.5 * atr,
                    'target1': 2.0 * atr,
                    'target2': 3.0 * atr,
                    'target3': 4.0 * atr,
                    'reasoning': f"Simple: EMA {self.ema_short} > EMA {self.ema_long}, RSI {rsi:.1f}"
                }
            elif ema_short < ema_long and rsi > 30:
                return {
                    'signal': 'BUY PUT',
                    'price': current_price,
                    'confidence_score': 45,  # High confidence
                    'stop_loss': 1.5 * atr,
                    'target1': 2.0 * atr,
                    'target2': 3.0 * atr,
                    'target3': 4.0 * atr,
                    'reasoning': f"Simple: EMA {self.ema_short} < EMA {self.ema_long}, RSI {rsi:.1f}"
                }
            else:
                # Generate a neutral signal to keep the system active
                if rsi < 50:
                    return {
                        'signal': 'BUY CALL',
                        'price': current_price,
                        'confidence_score': 35,  # Moderate confidence
                        'stop_loss': 1.5 * atr,
                        'target1': 2.0 * atr,
                        'target2': 3.0 * atr,
                        'target3': 4.0 * atr,
                        'reasoning': f"Neutral: RSI {rsi:.1f} < 50, EMA trend unclear"
                    }
                else:
                    return {
                        'signal': 'BUY PUT',
                        'price': current_price,
                        'confidence_score': 35,  # Moderate confidence
                        'stop_loss': 1.5 * atr,
                        'target1': 2.0 * atr,
                        'target2': 3.0 * atr,
                        'target3': 4.0 * atr,
                        'reasoning': f"Neutral: RSI {rsi:.1f} > 50, EMA trend unclear"
                    }
                    
        except Exception as e:
            logging.error(f"Error in SimpleEmaStrategy: {e}")
            return {'signal': 'ERROR', 'reason': str(e)} 