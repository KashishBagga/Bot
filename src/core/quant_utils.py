#!/usr/bin/env python3
"""
Quant Utilities (Structural Version 3.1)
========================================
Implements:
- Fractal Structural Bias (Daily HH/HL)
- Wick-to-Body Determinism
- ToD RVOL Percentiles
- Move Efficiency (Multi-Dimensional)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

class QuantUtils:
    
    @staticmethod
    def get_structural_bias(df: pd.DataFrame, timeframe: str = '1d') -> str:
        """
        Pure Structural Bias:
        - BULLISH: 2 consecutive Higher Lows + 2 Higher Highs.
        - BEARISH: 2 consecutive Lower Highs + 2 Lower Lows.
        Uses a fractal window of 5 candles to identify pivots.
        """
        if len(df) < 20: return "NEUTRAL"
        
        # Identify Swing Points (Low/High)
        df = df.copy()
        df['is_low'] = (df['low'] == df['low'].rolling(5, center=True).min())
        df['is_high'] = (df['high'] == df['high'].rolling(5, center=True).max())
        
        lows = df[df['is_low']]['low'].tail(3).values
        highs = df[df['is_high']]['high'].tail(3).values
        
        if len(lows) < 2 or len(highs) < 2: return "NEUTRAL"
        
        # Bullish: HH + HL
        is_bullish = (lows[-1] > lows[-2]) and (highs[-1] > highs[-2])
        # Bearish: LH + LL
        is_bearish = (lows[-1] < lows[-2]) and (highs[-1] < highs[-2])
        
        if is_bullish: return "BULLISH"
        if is_bearish: return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def is_strong_rejection(df: pd.DataFrame, candle_idx: int = -1) -> bool:
        """
        Deterministic Rejection:
        - Wick >= 2x Body
        - Close in extreme 25% of the candle range
        """
        candle = df.iloc[candle_idx]
        high, low, close, open_ = candle['high'], candle['low'], candle['close'], candle['open']
        body = abs(close - open_)
        candle_range = high - low
        
        if candle_range == 0: return False
        
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        
        # 1. Wick-to-Body Ratio
        wick_dominant = (upper_wick >= 2 * body) or (lower_wick >= 2 * body)
        
        # 2. Extreme Close
        close_pos = (close - low) / candle_range
        extreme_close = (close_pos <= 0.25) or (close_pos >= 0.75)
        
        return wick_dominant and extreme_close

    @staticmethod
    def calculate_move_efficiency(df: pd.DataFrame, lookback: int = 10) -> float:
        """
        Multi-Dimensional Move Efficiency:
        Net Move / Total Path Traveled.
        """
        if len(df) < lookback: return 0.5
        segment = df.tail(lookback)
        
        net_move = abs(segment['close'].iloc[-1] - segment['close'].iloc[0])
        path_traveled = abs(segment['close'].diff()).sum()
        
        return net_move / path_traveled if path_traveled > 0 else 0.0

    @staticmethod
    def calculate_wickiness(df: pd.DataFrame, lookback: int = 5) -> float:
        """
        Calculates the ratio of wick size to total candle range.
        High wickiness (> 0.5) indicates an unstable, struggling move.
        """
        segment = df.tail(lookback)
        total_range = (segment['high'] - segment['low']).sum()
        if total_range == 0: return 0.0
        
        bodies = abs(segment['close'] - segment['open']).sum()
        total_wicks = total_range - bodies
        
        return total_wicks / total_range

    @staticmethod
    def get_adaptive_rvol_rank(df: pd.DataFrame, candle_idx: int = -1, lookback: int = 50) -> float:
        """Returns percentile rank of volume."""
        if len(df) < lookback: return 0.5
        volumes = df['volume'].iloc[-(lookback+1):].values
        current_vol = df['volume'].iloc[candle_idx]
        return (volumes < current_vol).mean()
