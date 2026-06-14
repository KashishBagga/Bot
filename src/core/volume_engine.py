#!/usr/bin/env python3
"""
Institutional Volume Engine (Phase 3 - Truth)
==============================================
Implements Time-of-Day (ToD) Normalized RVOL.
Compares current volume to the historical average for this EXACT time slot.
Eliminates the "9:15 AM Bias" and provides a true participation metric.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VolumeReport:
    rvol_tod: float            # Time-of-Day Normalized RVOL
    is_high_participation: bool # rvol_tod > 2.0 (or 80th percentile)
    volume_trend: str          # RISING, FALLING, FLAT
    tod_avg_vol: float         # Historical avg for this time slot

class VolumeEngine:
    """Analyzes volume with Time-of-Day normalization."""

    def __init__(self, historical_days: int = 20):
        self.historical_days = historical_days
        self._tod_cache = {} # (symbol, time) -> avg_volume

    def analyze(self, df: pd.DataFrame, symbol: str) -> VolumeReport:
        if df is None or len(df) < 50:
            return VolumeReport(1.0, False, "FLAT", 0.0)

        current_candle = df.iloc[-1]
        current_time = current_candle.name.time() # Extract HH:MM:SS
        current_vol = current_candle['volume']
        
        # ── 1. Calculate ToD Average ──────────────────────────────
        # Filter all previous candles at this EXACT time
        same_time_candles = df[df.index.time == current_time]
        
        if len(same_time_candles) < 2:
            # Fallback to simple rolling mean if we don't have enough history
            tod_avg = df['volume'].tail(20).mean()
        else:
            # Avg volume for this exact minute/5-minute slot over the lookback
            tod_avg = same_time_candles['volume'].tail(self.historical_days).mean()

        # ── 2. Calculate ToD RVOL ─────────────────────────────────
        rvol_tod = current_vol / tod_avg if tod_avg > 0 else 1.0
        
        # ── 3. Detect Participation ──────────────────────────────
        # In a real institutional setup, this would use a rolling 80th percentile 
        # of the ToD RVOLs, but for Phase 3 we start with a strict 2.0 threshold.
        is_high_participation = rvol_tod >= 2.0

        # ── 4. Volume Trend ──────────────────────────────────────
        prev_vols = df['volume'].tail(5).values
        if current_vol > np.mean(prev_vols):
            trend = "RISING"
        elif current_vol < np.mean(prev_vols) * 0.8:
            trend = "FALLING"
        else:
            trend = "FLAT"

        return VolumeReport(
            rvol_tod=round(rvol_tod, 2),
            is_high_participation=is_high_participation,
            volume_trend=trend,
            tod_avg_vol=round(tod_avg, 2)
        )
