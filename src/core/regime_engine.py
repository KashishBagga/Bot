#!/usr/bin/env python3
"""
Market Regime Detection Engine (Priority 4)
===========================================
Classifies market conditions: TREND_UP, TREND_DOWN, RANGE, HIGH_VOL, LOW_VOL, etc.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RegimeEngine:
    """Classifies the current market regime for intelligence gathering."""
    
    def __init__(self, vol_window: int = 20, trend_window: int = 50):
        self.vol_window = vol_window
        self.trend_window = trend_window

    def detect_regime(self, df: pd.DataFrame) -> str:
        """Classify the current market regime."""
        if df is None or len(df) < self.trend_window:
            return "UNKNOWN"
            
        # 1. Volatility Classification
        closes = df['close']
        returns = closes.pct_change().dropna()
        vol = returns.tail(self.vol_window).std() * np.sqrt(252)
        avg_vol = returns.tail(100).std() * np.sqrt(252) if len(returns) > 100 else vol
        
        is_high_vol = vol > avg_vol * 1.5
        is_low_vol = vol < avg_vol * 0.7
        
        # 2. Trend Classification (using EMA and slope)
        ema_short = closes.ewm(span=20).mean()
        ema_long = closes.ewm(span=50).mean()
        
        last_short = ema_short.iloc[-1]
        last_long = ema_long.iloc[-1]
        prev_short = ema_short.iloc[-5]
        
        is_trending_up = (last_short > last_long) and (last_short > prev_short)
        is_trending_down = (last_short < last_long) and (last_short < prev_short)
        
        # 3. Range Detection (ADX or simple consolidation check)
        # Using a simple range check: is price oscillating between a tight band?
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        range_pct = (recent_high - recent_low) / recent_low
        
        is_ranging = range_pct < 0.01 # 1% range
        
        # 4. Final Classification
        if is_ranging:
            regime = "RANGE"
        elif is_trending_up:
            regime = "TREND_UP"
        elif is_trending_down:
            regime = "TREND_DOWN"
        else:
            regime = "NEUTRAL"
            
        if is_high_vol:
            regime += "_HIGH_VOL"
        elif is_low_vol:
            regime += "_LOW_VOL"
            
        return regime

    def get_day_type(self, df: pd.DataFrame) -> str:
        """
        Classifies the day type:
        - OPENING_DRIVE: Strong move in first 30 mins.
        - TREND_DAY: Continuous move with little pullbacks.
        - RANGE_DAY: Price stays within initial balance.
        - DOUBLE_DISTRIBUTION: Two separate value areas.
        """
        if len(df) < 10: return "NORMAL"
        
        # Simple heuristics for now
        high = df['high'].max()
        low = df['low'].max()
        open_price = df['open'].iloc[0]
        close_price = df['close'].iloc[-1]
        
        move_pct = abs(close_price - open_price) / open_price
        range_pct = (high - low) / low
        
        if move_pct > 0.015 and range_pct > 0.02:
            return "TREND_DAY"
        elif range_pct < 0.008:
            return "RANGE_DAY"
        else:
            return "NORMAL_DAY"

    def get_session_type(self, timestamp: datetime) -> str:
        """Categorize session into OPEN, MID, or CLOSE."""
        # Indian market hours: 09:15 to 15:30
        hour = timestamp.hour
        minute = timestamp.minute
        
        if hour == 9 or (hour == 10 and minute < 30):
            return "OPEN"
        elif hour >= 14:
            return "CLOSE"
        else:
            return "MID"
