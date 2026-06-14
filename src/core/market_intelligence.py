#!/usr/bin/env python3
"""
Market Intelligence Engine v2
==============================
Orchestrates all analysis sub-engines into a single intel report.
Uses ZoneEngine + StructureEngine (not primitive clustering).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from src.core.zone_engine import ZoneEngine
from src.core.structure_engine import StructureEngine
from src.core.breakout_engine import BreakoutEngine

logger = logging.getLogger(__name__)


class MarketIntelligence:
    def __init__(self, fractal_window: int = 3):
        self.zone_engine = ZoneEngine(cluster_pct=0.003, swing_window=5)
        self.struct_engine = StructureEngine(fractal_window=fractal_window)
        self.breakout_engine = BreakoutEngine()

    def analyze(self, data: pd.DataFrame, current_price: float = 0) -> Dict[str, Any]:
        """Full intelligence pass. Returns a flat dict consumed by ProbabilityEngine."""
        if data is None or len(data) < 50:
            return {}

        df = data.copy()

        # 1. Regime
        er = self._efficiency_ratio(df)
        compression = self._compression(df)

        # 2. Structure (with health metrics)
        structure = self.struct_engine.analyze(df)

        # 3. Zones (with quality scoring)
        zones = self.zone_engine.detect_zones(df)
        price = current_price or df['close'].iloc[-1]
        zone_info = self.zone_engine.is_price_at_zone(price, zones)

        # 4. Liquidity sweep
        sweep = self._detect_sweep(df)

        # 5. Breakout Analysis (High ROI)
        # Use detected zones + PDH/PDL
        recent_zones = sorted(zones, key=lambda z: abs(z.level - price))[:2]
        levels = {
            'resistance': recent_zones[0].level if recent_zones else 0,
            'support': recent_zones[1].level if len(recent_zones) > 1 else 0
        }
        # Optionally add PDH/PDL if available in data
        if 'pdh' in df.columns: levels['pdh'] = df['pdh'].iloc[-1]
        if 'pdl' in df.columns: levels['pdl'] = df['pdl'].iloc[-1]
        
        breakout_intel = self.breakout_engine.analyze(df, levels)

        # 6. Volume intelligence
        climax = self._volume_climax(df)

        return {
            'efficiency_ratio': er,
            'compression_score': compression,
            'structure': structure,
            'zones': zones,
            'zone_info': zone_info,        # (at_zone, Zone, dist_pct)
            'liquidity_sweeps': sweep,
            'volume_climax': climax,
            'breakout': breakout_intel,
            'volatility_percentile': self._vol_percentile(df),
        }

    # ── Regime ────────────────────────────────────────────────────
    def _efficiency_ratio(self, df: pd.DataFrame, n: int = 14) -> float:
        direction = abs(df['close'].iloc[-1] - df['close'].iloc[-n])
        noise = df['close'].diff().abs().rolling(n).sum().iloc[-1]
        return float(direction / noise) if noise > 0 else 0

    def _compression(self, df: pd.DataFrame) -> float:
        if 'bb_upper' not in df.columns:
            return 0.5
        width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        percentile = float((width > width.iloc[-1]).mean())
        last_7 = (df['high'] - df['low']).tail(7)
        nr7 = 1.0 if last_7.iloc[-1] == last_7.min() else 0.0
        return (percentile + nr7) / 2

    def _vol_percentile(self, df: pd.DataFrame) -> float:
        if 'atr' not in df.columns:
            return 50.0
        s = df['atr'].tail(100)
        return float((s < s.iloc[-1]).mean() * 100)

    # ── Sweeps ────────────────────────────────────────────────────
    def _detect_sweep(self, df: pd.DataFrame) -> str:
        ph = df['high'].shift(1).iloc[-1]
        pl = df['low'].shift(1).iloc[-1]
        ch, cl, cc = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        if ch > ph and cc < ph:
            return 'BEARISH_SWEEP'
        if cl < pl and cc > pl:
            return 'BULLISH_SWEEP'
        return 'NONE'

    # ── Volume ────────────────────────────────────────────────────
    def _volume_climax(self, df: pd.DataFrame) -> str:
        avg = df['volume'].rolling(20).mean().iloc[-1]
        curr = df['volume'].iloc[-1]
        spike = curr > avg * 2.5
        move = abs(df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        if spike and move > 0.008:
            return 'BEARISH_CLIMAX' if df['close'].iloc[-1] < df['close'].iloc[-2] else 'BULLISH_CLIMAX'
        return 'NONE'

    def _vdb_breakout(self, df: pd.DataFrame) -> bool:
        w = 8
        vols = df['volume'].tail(w)
        avg20 = df['volume'].rolling(20).mean().iloc[-1]
        declining = vols.iloc[:-1].mean() < avg20 * 0.7
        expanding = df['volume'].iloc[-1] > vols.iloc[:-1].mean() * 2.0
        breakout = df['close'].iloc[-1] > df['high'].tail(w).max()
        return bool(declining and expanding and breakout)
