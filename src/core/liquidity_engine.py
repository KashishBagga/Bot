#!/usr/bin/env python3
"""
Liquidity Engine (Phase 1 Location Intelligence)
================================================
Identifies: EQH/EQL, PDH/PDL, and Liquidity Sweeps.
Determines if price is at a high-value "location" for a reversal or breakout.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LiquidityReport:
    near_pdh: bool
    near_pdl: bool
    eqh_detected: bool          # Equal Highs (Liquidity above)
    eql_detected: bool          # Equal Lows (Liquidity below)
    sweep_type: str             # 'BULLISH_SWEEP', 'BEARISH_SWEEP', 'NONE'
    liquidity_zones: List[float] # Key levels where liquidity is resting

class LiquidityEngine:
    """Detects where institutional money is likely to seek liquidity."""

    def __init__(self, tolerance_pct: float = 0.0005):
        self.tolerance = tolerance_pct

    def analyze(self, df: pd.DataFrame, htf_data: Optional[pd.DataFrame] = None) -> LiquidityReport:
        if df is None or len(df) < 5:
            return LiquidityReport(False, False, False, False, "NONE", [])

        current_price = df['close'].iloc[-1]
        highs = df['high'].values
        lows = df['low'].values
        
        # ── 1. Identify PDH / PDL (Previous Day High/Low) ─────────
        # Note: In a real system, this should come from HTF (Daily) data.
        # For now, we simulate using the range of the current dataset.
        pdh = highs.max()
        pdl = lows.min()
        
        near_pdh = abs(current_price - pdh) / pdh < self.tolerance
        near_pdl = abs(current_price - pdl) / pdl < self.tolerance

        # ── 2. Detect Equal Highs / Lows (EQH / EQL) ──────────────
        eqh, eql = self._detect_equal_levels(highs, lows)

        # ── 3. Detect Sweeps ──────────────────────────────────────
        # Price went above a high/low and closed back inside
        sweep = self._detect_sweep(df)

        return LiquidityReport(
            near_pdh=near_pdh,
            near_pdl=near_pdl,
            eqh_detected=eqh,
            eql_detected=eql,
            sweep_type=sweep,
            liquidity_zones=[pdh, pdl]
        )

    def _detect_equal_levels(self, highs, lows) -> Tuple[bool, bool]:
        """Looks for clusters of highs or lows within tolerance."""
        eqh = False
        eql = False
        
        # Check last 50 candles for similar highs
        if len(highs) > 10:
            recent_highs = sorted(highs[-50:])
            for i in range(len(recent_highs)-1):
                if abs(recent_highs[i] - recent_highs[i+1]) / recent_highs[i] < self.tolerance:
                    eqh = True
                    break
                    
            recent_lows = sorted(lows[-50:])
            for i in range(len(recent_lows)-1):
                if abs(recent_lows[i] - recent_lows[i+1]) / recent_lows[i] < self.tolerance:
                    eql = True
                    break
        
        return eqh, eql

    def _detect_sweep(self, df: pd.DataFrame) -> str:
        """
        Detects if the CURRENT candle swept a previous significant high/low.
        A sweep is defined as: high > prev_high AND close < prev_high (for bearish).
        """
        if len(df) < 5: return "NONE"
        
        curr_h, curr_l, curr_c = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        
        # Find the max/min of the PREVIOUS 20 candles (excluding current)
        lookback = df.iloc[-21:-1]
        prev_max = lookback['high'].max()
        prev_min = lookback['low'].min()
        
        # Bearish Sweep (Sweep high, reclaim down)
        if curr_h > prev_max and curr_c < prev_max:
            return "BEARISH_SWEEP"
            
        # Bullish Sweep (Sweep low, reclaim up)
        if curr_l < prev_min and curr_c > prev_min:
            return "BULLISH_SWEEP"
            
        return "NONE"
