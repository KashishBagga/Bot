#!/usr/bin/env python3
"""
Pivot Engine (MKE Stage 1 Context)
==================================
Statelessly extracts local fractal price pivots (SwingPoints).
Computes multi-factor strength components and enforces session insulation.
"""

import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.core.market_knowledge import SwingPoint, SwingStatus
from src.core.time_of_day_engine import TimeOfDayEngine

logger = logging.getLogger(__name__)


class PivotEngine:
    """Foundational pivot detection engine."""
    required_history = 80

    def __init__(self, pivot_window: int = 3):
        self.w = pivot_window
        self.tod_engine = TimeOfDayEngine()

    def detect_pivots(self, df: pd.DataFrame, symbol: str, timeframe: str = "m5") -> List[SwingPoint]:
        """
        Scans a dataframe to locate fractal pivots.
        Enforces session insulation (fractal windows cannot cross overnight gap boundaries).
        """
        if df is None or len(df) < (self.w * 2 + 1):
            return []

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        timestamps = df.index
        
        # Precompute rolling ATR for normalization
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean().fillna(df['high'] - df['low'])
        atr_values = atr_series.values

        swings: List[SwingPoint] = []
        n = len(df)
        clean_symbol = symbol.replace(":", "_").replace("-", "_")

        for i in range(self.w, n - self.w):
            date_i = timestamps[i].date()
            
            # Enforce Session Insulation: all window bars must belong to the SAME trading day
            is_insulated = True
            for j in range(1, self.w + 1):
                if timestamps[i - j].date() != date_i or timestamps[i + j].date() != date_i:
                    is_insulated = False
                    break
            
            if not is_insulated:
                continue

            atr = max(atr_values[i], 1.0)
            vol = float(volumes[i])
            price_high = float(highs[i])
            price_low = float(lows[i])
            ts = timestamps[i].to_pydatetime()
            ts_str = ts.strftime("%Y%m%d_%H%M%S")

            # Check High Pivot
            is_high = all(highs[i] >= highs[i - j] for j in range(1, self.w + 1)) and \
                      all(highs[i] > highs[i + j] for j in range(1, self.w + 1))

            if is_high:
                # Geometry Strength: peak prominence over neighbors
                neighbors_max = max(max(highs[i - self.w : i]), max(highs[i + 1 : i + self.w + 1]))
                prominence = max(0.0, price_high - neighbors_max)
                geom_score = min(1.0, prominence / atr)

                # Participation Strength: volume vs local rolling average, normalized by TimeOfDay Profile
                avg_vol = float(np.mean(volumes[max(0, i - 10) : i]))
                avg_vol = max(avg_vol, 1.0)
                profile = self.tod_engine.lookup(ts)
                participation_score = min(1.0, (vol / avg_vol) * profile.avg_volume_factor)

                # Reaction Strength: velocity of reversal in subsequent bars
                reversal_dist = price_high - float(closes[i + self.w])
                reaction_score = min(1.0, max(0.0, reversal_dist / (atr * 1.5)))

                # Combine sub-components
                components = {
                    "geometry": round(geom_score, 3),
                    "participation": round(participation_score, 3),
                    "reaction": round(reaction_score, 3),
                    "persistence": 0.5  # Base value, upgraded by StructureEngine on retests
                }
                strength = round(0.4 * geom_score + 0.3 * participation_score + 0.2 * reaction_score + 0.1 * 0.5, 3)

                swing_id = f"sw_{clean_symbol}_{timeframe}_high_{ts_str}"
                swings.append(SwingPoint(
                    id=swing_id,
                    timestamp=ts,
                    price=price_high,
                    type="HIGH",
                    status=SwingStatus.ACTIVE,
                    confidence=1.0,
                    strength=strength,
                    strength_components=components,
                    provenance={
                        "engine": "PivotEngine",
                        "version": "v2.0",
                        "window": self.w
                    }
                ))

            # Check Low Pivot
            is_low = all(lows[i] <= lows[i - j] for j in range(1, self.w + 1)) and \
                     all(lows[i] < lows[i + j] for j in range(1, self.w + 1))

            if is_low:
                # Geometry Strength
                neighbors_min = min(min(lows[i - self.w : i]), min(lows[i + 1 : i + self.w + 1]))
                prominence = max(0.0, neighbors_min - price_low)
                geom_score = min(1.0, prominence / atr)

                # Participation Strength
                avg_vol = float(np.mean(volumes[max(0, i - 10) : i]))
                avg_vol = max(avg_vol, 1.0)
                profile = self.tod_engine.lookup(ts)
                participation_score = min(1.0, (vol / avg_vol) * profile.avg_volume_factor)

                # Reaction Strength
                reversal_dist = float(closes[i + self.w]) - price_low
                reaction_score = min(1.0, max(0.0, reversal_dist / (atr * 1.5)))

                components = {
                    "geometry": round(geom_score, 3),
                    "participation": round(participation_score, 3),
                    "reaction": round(reaction_score, 3),
                    "persistence": 0.5
                }
                strength = round(0.4 * geom_score + 0.3 * participation_score + 0.2 * reaction_score + 0.1 * 0.5, 3)

                swing_id = f"sw_{clean_symbol}_{timeframe}_low_{ts_str}"
                swings.append(SwingPoint(
                    id=swing_id,
                    timestamp=ts,
                    price=price_low,
                    type="LOW",
                    status=SwingStatus.ACTIVE,
                    confidence=1.0,
                    strength=strength,
                    strength_components=components,
                    provenance={
                        "engine": "PivotEngine",
                        "version": "v2.0",
                        "window": self.w
                    }
                ))

        return swings
