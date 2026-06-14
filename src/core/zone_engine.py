#!/usr/bin/env python3
"""
Zone Quality Engine (Tier 1, Priority 1)
========================================
Replaces primitive swing clustering with institutional-grade zone scoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)

@dataclass
class Zone:
    """A single S/R zone with quality metrics."""
    level: float
    zone_type: str          # 'SUPPLY' or 'DEMAND'
    touches: List[int]      # indices of candles that touched this zone
    score: float = 0.0
    rejection_count: int = 0
    avg_rvol_at_touch: float = 0.0
    max_impulse_pct: float = 0.0
    freshness: float = 0.0
    precision: float = 0.0

class ZoneEngine:
    """Detects, scores, and invalidates S/R zones."""

    WEIGHTS = {
        'rejection_count':   30,
        'rvol_at_rejection': 30,
        'impulse_away':      30,
        'freshness':         10,
    }
    MIN_ZONE_SCORE = 60

    def __init__(self, cluster_pct: float = 0.003, swing_window: int = 5):
        self.cluster_pct = cluster_pct
        self.swing_window = swing_window

    def detect_zones(self, df: pd.DataFrame) -> List[Zone]:
        if df is None or len(df) < 30: return []

        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        w = self.swing_window

        # ── Step 1: Find all swing pivots ──────────────────────────
        swing_points = []
        for i in range(w, n - w):
            if highs[i] == max(highs[i - w: i + w + 1]):
                swing_points.append((i, float(highs[i]), 'SUPPLY'))
            if lows[i] == min(lows[i - w: i + w + 1]):
                swing_points.append((i, float(lows[i]), 'DEMAND'))

        if not swing_points: return []

        # ── Step 2: Cluster into zones (±cluster_pct) ─────────────
        swing_points.sort(key=lambda x: x[1])
        raw_zones: List[Dict] = []
        cluster = [swing_points[0]]

        for sp in swing_points[1:]:
            cluster_mean = np.mean([c[1] for c in cluster])
            if abs(sp[1] - cluster_mean) / cluster_mean < self.cluster_pct:
                cluster.append(sp)
            else:
                raw_zones.append(self._build_zone(cluster))
                cluster = [sp]
        raw_zones.append(self._build_zone(cluster))

        # ── Step 3: Score each zone ───────────────────────────────
        avg_vol = df['volume'].mean()
        scored_zones = []

        for rz in raw_zones:
            # ── Step 4: Invalidation Check ────────────────────────
            if self._is_zone_invalid(rz, df): continue
                
            zone = self._score_zone(rz, df, avg_vol, n)
            if zone.score >= self.MIN_ZONE_SCORE:
                scored_zones.append(zone)

        scored_zones.sort(key=lambda z: z.score, reverse=True)
        return scored_zones[:8]

    def _is_zone_invalid(self, rz: Dict, df: pd.DataFrame) -> bool:
        """Invalidates a zone if price has closed through it twice."""
        level = rz['level']
        closes = df['close'].tail(50).values
        wrong_side_closes = 0
        for c in closes:
            if rz['zone_type'] == 'SUPPLY' and c > level: wrong_side_closes += 1
            elif rz['zone_type'] == 'DEMAND' and c < level: wrong_side_closes += 1
        return wrong_side_closes >= 2

    def _build_zone(self, cluster: List[tuple]) -> Dict:
        prices = [c[1] for c in cluster]
        types = [c[2] for c in cluster]
        zone_type = 'SUPPLY' if types.count('SUPPLY') >= types.count('DEMAND') else 'DEMAND'
        return {
            'level': float(np.mean(prices)),
            'zone_type': zone_type,
            'touch_indices': [c[0] for c in cluster],
            'touch_prices': prices,
        }

    def _score_zone(self, rz: Dict, df: pd.DataFrame, avg_vol: float, n: int) -> Zone:
        level = rz['level']
        touches = rz['touch_indices']
        volumes = df['volume'].values

        rej_count = len(touches)
        # Inverted Logic: 1st touch (100), 2nd (80), 3rd (50), 4th+ (20 - Breakout Risk)
        if rej_count == 1: rej_score = 100
        elif rej_count == 2: rej_score = 80
        elif rej_count == 3: rej_score = 50
        else: rej_score = 20 # Liquidity consumed, breakout likely

        rvols = [QuantUtils.get_adaptive_rvol_rank(df, idx) for idx in touches if idx < len(volumes)]
        avg_rvol_rank = float(np.mean(rvols)) if rvols else 0.5
        rvol_score = avg_rvol_rank * 100

        efficiencies = []
        for idx in touches:
            if idx + 10 < n:
                eff = QuantUtils.calculate_move_efficiency(df.iloc[idx : idx+10])
                efficiencies.append(eff)
        avg_efficiency = float(np.mean(efficiencies)) if efficiencies else 0.0
        impulse_score = 100 if avg_efficiency >= 0.7 else (60 if avg_efficiency >= 0.4 else 20)

        candles_ago = n - 1 - max(touches)
        fresh_score = max(0, 100 - (candles_ago * 2))

        total = (rej_score * self.WEIGHTS['rejection_count'] +
                 rvol_score * self.WEIGHTS['rvol_at_rejection'] +
                 impulse_score * self.WEIGHTS['impulse_away'] +
                 fresh_score * self.WEIGHTS['freshness']) / 100

        return Zone(level=level, zone_type=rz['zone_type'], touches=touches,
                    score=round(total, 1), rejection_count=rej_count,
                    avg_rvol_at_touch=round(avg_rvol_rank, 2),
                    max_impulse_pct=round(avg_efficiency * 100, 2),
                    freshness=round(fresh_score, 1), precision=0.0)

    def is_price_at_zone(self, price: float, zones: List[Zone], tolerance: float = 0.003) -> tuple:
        for zone in zones:
            dist = abs(price - zone.level) / zone.level
            if dist < tolerance: return True, zone, round(dist * 100, 3)
        return False, None, 0.0
