#!/usr/bin/env python3
"""
Market Structure Engine (Phase 1 Foundation)
============================================
Handles: HH/HL, LH/LL, BOS, CHOCH, and Compression.
This replaces indicator-heavy logic with pure price action context.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StructureReport:
    trend: str                  # BULLISH, BEARISH, NEUTRAL
    market_phase: str           # EXPANSION, RETRACEMENT, COMPRESSION
    bos_count: int              # Strength of trend
    choch_detected: bool        # Early trend shift
    last_swing_high: float
    last_swing_low: float
    is_compressed: bool         # Tightening price action
    quality_score: float        # 0-100 based on structural health

class StructureEngine:
    """Foundational Market Structure analysis."""

    def __init__(self, pivot_window: int = 3):
        self.w = pivot_window

    def analyze(self, df: pd.DataFrame) -> StructureReport:
        if df is None or len(df) < 20:
            return StructureReport("NEUTRAL", "NEUTRAL", 0, False, 0, 0, False, 0)

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # ── 1. Identify Swing Points ───────────────────────────────
        sw_highs, sw_lows = self._find_swings(highs, lows)
        
        if len(sw_highs) < 2 or len(sw_lows) < 2:
            return StructureReport("NEUTRAL", "NEUTRAL", 0, False, 0, 0, False, 0)

        # ── 2. Classify HH/HL / LH/LL ────────────────────────────
        trend, phase = self._classify_trend(sw_highs, sw_lows, closes[-1])

        # ── 3. Detect BOS (Break of Structure) ───────────────────
        # Trend continuation signal
        bos_count, last_bos_idx = self._detect_bos(df, sw_highs, sw_lows, trend)

        # ── 4. Detect CHOCH (Change of Character) ────────────────
        # First sign of trend reversal
        choch_detected = self._detect_choch(closes[-1], sw_highs, sw_lows, trend)

        # ── 5. Detect Compression ────────────────────────────────
        # Narrowing range / Tightening action
        is_compressed = self._detect_compression(df)

        # ── 6. Structural Health Scoring ─────────────────────────
        score = self._calculate_quality(trend, bos_count, choch_detected, is_compressed)

        return StructureReport(
            trend=trend,
            market_phase=phase,
            bos_count=bos_count,
            choch_detected=choch_detected,
            last_swing_high=sw_highs[-1][1],
            last_swing_low=sw_lows[-1][1],
            is_compressed=is_compressed,
            quality_score=score
        )

    def _find_swings(self, highs, lows) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find fractal pivots using window w."""
        sw_highs, sw_lows = [], []
        n = len(highs)
        for i in range(self.w, n - self.w):
            if all(highs[i] >= highs[i-j] for j in range(1, self.w+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, self.w+1)):
                sw_highs.append((i, float(highs[i])))
            if all(lows[i] <= lows[i-j] for j in range(1, self.w+1)) and \
               all(lows[i] < lows[i+j] for j in range(1, self.w+1)):
                sw_lows.append((i, float(lows[i])))
        return sw_highs, sw_lows

    def _classify_trend(self, sw_highs, sw_lows, current_close) -> Tuple[str, str]:
        last_h, prev_h = sw_highs[-1][1], sw_highs[-2][1]
        last_l, prev_l = sw_lows[-1][1], sw_lows[-2][1]

        trend = "NEUTRAL"
        phase = "NEUTRAL"

        if last_h > prev_h and last_l > prev_l:
            trend = "BULLISH"
        elif last_h < prev_h and last_l < prev_l:
            trend = "BEARISH"
            
        # Determine Phase
        if trend == "BULLISH":
            phase = "EXPANSION" if current_close > last_h else "RETRACEMENT"
        elif trend == "BEARISH":
            phase = "EXPANSION" if current_close < last_l else "RETRACEMENT"
            
        return trend, phase

    def _detect_bos(self, df, sw_highs, sw_lows, trend) -> Tuple[int, int]:
        """Counts consecutive breaks of trend-continuation swings."""
        bos_count = 0
        last_idx = -1
        
        if trend == "BULLISH":
            # Count how many times high was broken by a close
            for i in range(len(sw_highs) - 1, 0, -1):
                h_idx, h_val = sw_highs[i-1]
                # Look for a close above this high between this high and the next high
                next_h_idx = sw_highs[i][0]
                segment = df['close'].iloc[h_idx:next_h_idx]
                if not segment.empty and any(segment > h_val):
                    bos_count += 1
                    last_idx = h_idx
                else:
                    break # Trend broken or didn't continue here
        elif trend == "BEARISH":
            for i in range(len(sw_lows) - 1, 0, -1):
                l_idx, l_val = sw_lows[i-1]
                next_l_idx = sw_lows[i][0]
                segment = df['close'].iloc[l_idx:next_l_idx]
                if not segment.empty and any(segment < l_val):
                    bos_count += 1
                    last_idx = l_idx
                else:
                    break
                    
        return bos_count, last_idx

    def _detect_choch(self, current_close, sw_highs, sw_lows, trend) -> bool:
        """CHOCH is the first counter-trend break of a significant swing."""
        if trend == "BULLISH":
            # Break below the last HL (Higher Low)
            return current_close < sw_lows[-1][1]
        elif trend == "BEARISH":
            # Break above the last LH (Lower High)
            return current_close > sw_highs[-1][1]
        return False

    def _detect_compression(self, df: pd.DataFrame) -> bool:
        """
        Detects tightening price action using ATR contraction 
        and Average Bar Size reduction.
        """
        closes = df['close'].values
        # Simple measure: is the standard deviation of the last 5 closes 
        # less than the standard deviation of the previous 15?
        if len(closes) < 20: return False
        
        std_short = np.std(closes[-5:])
        std_long = np.std(closes[-20:])
        
        # Also check for Inside Bar clusters
        ranges = df['high'] - df['low']
        avg_range_short = ranges.tail(5).mean()
        avg_range_long = ranges.tail(20).mean()
        
        return (std_short < std_long * 0.6) and (avg_range_short < avg_range_long * 0.8)

    def _calculate_quality(self, trend, bos, choch, compression) -> float:
        if trend == "NEUTRAL": return 0.0
        score = 50.0
        score += min(bos * 10, 30) # More BOS = stronger trend
        if choch: score -= 40      # CHOCH = trend dying
        if compression: score += 20 # Compression = potential explosive move
        return max(0, min(100, score))
