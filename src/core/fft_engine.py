#!/usr/bin/env python3
"""
Failed Follow-Through (FFT) Engine
==================================
Detects "Trapped Participation" by identifying high-volume breakouts 
that immediately fail to hold their new levels.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict

class FFTEngine:
    def __init__(self, failure_threshold: float = 0.8):
        self.failure_threshold = failure_threshold

    def detect_trap(self, df: pd.DataFrame, bos_level: float, trend: str) -> Optional[str]:
        """
        Returns 'BUY CALL' (Bullish Trap) or 'BUY PUT' (Bearish Trap) if a trap is found.
        Logic: 
        1. Current candle is a BOS.
        2. Next candle immediately closes back inside the level.
        """
        if len(df) < 5: return None
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Bullish Breakout Failure (Bearish Trap)
        if trend == "BULLISH":
            # If prev candle broke HIGH but last candle closed back BELOW the level
            if prev_candle['close'] > bos_level and last_candle['close'] < bos_level:
                return "BUY PUT"
                
        # Bearish Breakout Failure (Bullish Trap)
        elif trend == "BEARISH":
            # If prev candle broke LOW but last candle closed back ABOVE the level
            if prev_candle['close'] < bos_level and last_candle['close'] > bos_level:
                return "BUY CALL"
                
        return None
