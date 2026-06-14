#!/usr/bin/env python3
"""
Candle Pattern Engine
=====================
Implements precise price action candle triggers:
- Pin Bar (Hammer/Shooting Star) with Institutional Ratios
- Engulfing Candles with Volume Confirmation
- Morning/Evening Star Reversals
"""

import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class CandlePatternEngine:
    def __init__(self):
        pass

    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Detects all supported price action candle patterns."""
        if len(df) < 5:
            return {}

        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        vol = df['volume'].iloc[-1]
        
        bull_pin, bear_pin = self._detect_pin_bar(o, h, l, c)
        bull_eng, bear_eng = self._detect_engulfing(df)
        morning_star = self._detect_morning_star(df)
        evening_star = self._detect_evening_star(df)

        return {
            'bull_pin': bull_pin,
            'bear_pin': bear_pin,
            'bull_eng': bull_eng,
            'bear_eng': bear_eng,
            'morning_star': morning_star,
            'evening_star': evening_star
        }

    def _detect_pin_bar(self, o: float, h: float, l: float, c: float, rvol: float = 1.0) -> Tuple[bool, bool]:
        """
        RUTHLESSLY OBJECTIVE PIN BAR (Rule #13):
        - Wick >= 2.5x Body
        - RVOL > 1.5
        - Close near extreme (within 10% of range)
        """
        body = abs(c - o)
        full_range = h - l
        if full_range == 0: return False, False
        
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        # Bullish: Long lower wick, close in top 10%
        bull_pin = (lower_wick >= body * 2.5 
                   and (h - c) / full_range < 0.10
                   and rvol > 1.5)
        
        # Bearish: Long upper wick, close in bottom 10%
        bear_pin = (upper_wick >= body * 2.5 
                   and (c - l) / full_range < 0.10
                   and rvol > 1.5)
        
        return bull_pin, bear_pin

    def _detect_engulfing(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Engulfing: Covers previous body entirely + volume expansion.
        """
        c1_o, c1_c = df['open'].iloc[-2], df['close'].iloc[-2]
        c2_o, c2_c = df['open'].iloc[-1], df['close'].iloc[-1]
        v1, v2 = df['volume'].iloc[-2], df['volume'].iloc[-1]
        
        prev_body = abs(c1_c - c1_o)
        curr_body = abs(c2_c - c2_o)
        
        # Bullish engulfing: green covers red
        bull_eng = (c1_c < c1_o and c2_c > c2_o
                   and c2_o <= c1_c and c2_c >= c1_o
                   and curr_body > prev_body * 1.1
                   and v2 > v1 * 1.3)
        
        # Bearish engulfing: red covers green
        bear_eng = (c1_c > c1_o and c2_c < c2_o
                   and c2_o >= c1_c and c2_c <= c1_o
                   and curr_body > prev_body * 1.1
                   and v2 > v1 * 1.3)
        
        return bull_eng, bear_eng

    def _detect_morning_star(self, df: pd.DataFrame) -> bool:
        """3-Candle bullish reversal."""
        c1_o, c1_c = df['open'].iloc[-3], df['close'].iloc[-3]
        c2_o, c2_c = df['open'].iloc[-2], df['close'].iloc[-2]
        c3_o, c3_c = df['open'].iloc[-1], df['close'].iloc[-1]
        
        c1_red = c1_c < c1_o
        c2_small = abs(c2_c - c2_o) < abs(c1_c - c1_o) * 0.3
        c3_green = c3_c > c3_o and c3_c > (c1_o + c1_c) / 2
        
        return c1_red and c2_small and c3_green

    def _detect_evening_star(self, df: pd.DataFrame) -> bool:
        """3-Candle bearish reversal."""
        c1_o, c1_c = df['open'].iloc[-3], df['close'].iloc[-3]
        c2_o, c2_c = df['open'].iloc[-2], df['close'].iloc[-2]
        c3_o, c3_c = df['open'].iloc[-1], df['close'].iloc[-1]
        
        c1_green = c1_c > c1_o
        c2_small = abs(c2_c - c2_o) < abs(c1_c - c1_o) * 0.3
        c3_red = c3_c < c3_o and c3_c < (c1_o + c1_c) / 2
        
        return c1_green and c2_small and c3_red
