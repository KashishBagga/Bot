#!/usr/bin/env python3
"""
IndicatorPipeline — Compute all indicators once per symbol per candle.
======================================================================
Receives raw OHLCV DataFrames from the data provider and produces a
fully decorated MarketSnapshot for all strategies to consume.

No API calls. No strategy logic. Pure computation.

Staged internally for unit testability:
    _stage_structure()  → daily bias, H1 structure, H1 zones
    _stage_volume()     → ToD-normalized RVOL
    _stage_regime()     → market regime classification
    _stage_features()   → FeatureStore (atr, ema20, ema50, ...)

Adding a new indicator: one line in _stage_features(). Nothing else changes.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.core.feature_store import FeatureStore
from src.core.market_snapshot import MarketSnapshot
from src.core.structure_engine import StructureEngine
from src.core.zone_engine import ZoneEngine
from src.core.volume_engine import VolumeEngine
from src.core.regime_engine import RegimeEngine
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)


class IndicatorPipeline:
    """
    Single-pass indicator computation. Instantiate once, call compute() each candle.
    All engine instances are reused — no re-initialisation per symbol.
    """

    def __init__(
        self,
        pivot_window: int = 3,
        zone_cluster_pct: float = 0.002,
        min_zone_score: float = 50.0,
        historical_days: int = 20,
    ):
        self.structure_engine = StructureEngine(pivot_window=pivot_window)
        self.zone_engine = ZoneEngine(cluster_pct=zone_cluster_pct)
        self.zone_engine.MIN_ZONE_SCORE = min_zone_score
        self.volume_engine = VolumeEngine(historical_days=historical_days)
        self.regime_engine = RegimeEngine()

    # ── Public interface ───────────────────────────────────────────────────

    def compute(
        self,
        symbol: str,
        price: float,
        d1: Optional[pd.DataFrame],
        h1: Optional[pd.DataFrame],
        m5: Optional[pd.DataFrame],
        timestamp: datetime,
    ) -> Optional[MarketSnapshot]:
        """
        Run all indicator stages and return a fully decorated MarketSnapshot.
        Returns None if data is insufficient (e.g. API returned empty frames).
        """
        if h1 is None or m5 is None:
            logger.warning(f"[Pipeline] Insufficient data for {symbol} — skipping snapshot.")
            return None

        try:
            # Stage 1: Market structure (structural bias, zones)
            daily_bias, h1_structure, h1_zones = self._stage_structure(d1, h1)

            # Stage 2: Volume participation
            volume_report = self._stage_volume(m5, symbol)

            # Stage 3: Market regime
            market_regime = self._stage_regime(m5)

            # Stage 4: Feature store (ATR, EMAs, derived metrics)
            features = self._stage_features(m5)

            return MarketSnapshot(
                symbol=symbol,
                current_price=price,
                timestamp=timestamp,
                d1=d1,
                h1=h1,
                m5=m5,
                daily_bias=daily_bias,
                h1_structure=h1_structure,
                h1_zones=h1_zones,
                market_regime=market_regime,
                volume_report=volume_report,
                features=features,
            )

        except Exception as e:
            logger.error(f"[Pipeline] Snapshot computation failed for {symbol}: {e}")
            return None

    # ── Stage 1: Structure ────────────────────────────────────────────────

    def _stage_structure(self, d1, h1):
        """Daily bias, H1 structure report, H1 supply/demand zones."""
        daily_bias = QuantUtils.get_structural_bias(d1) if d1 is not None else "NEUTRAL"
        h1_structure = self.structure_engine.analyze(h1)
        h1_zones = self.zone_engine.detect_zones(h1)
        return daily_bias, h1_structure, h1_zones

    # ── Stage 2: Volume ───────────────────────────────────────────────────

    def _stage_volume(self, m5, symbol: str):
        """Time-of-Day normalized RVOL."""
        return self.volume_engine.analyze(m5, symbol)

    # ── Stage 3: Regime ───────────────────────────────────────────────────

    def _stage_regime(self, m5) -> str:
        """Market regime classification (TREND_UP, RANGE, etc.)."""
        return self.regime_engine.detect_regime(m5)

    # ── Stage 4: Features ─────────────────────────────────────────────────

    def _stage_features(self, m5: pd.DataFrame) -> FeatureStore:
        """
        Compute all indicator values into a FeatureStore.
        ─────────────────────────────────────────────────
        To add a new indicator (e.g. RSI, VWAP, ADX):
            1. Compute it here.
            2. Add it to the `data` dict with a clear key name.
            3. Nothing else changes — strategies read via features.get_float("rsi14").
        """
        # ATR (simple range-based approximation)
        atr = float(m5["high"].tail(14).mean() - m5["low"].tail(14).mean())

        # EMA values on close
        ema20 = float(m5["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(m5["close"].ewm(span=50, adjust=False).mean().iloc[-1])

        # Move quality metrics (used by StructuralStrategy filters)
        move_efficiency = float(QuantUtils.calculate_move_efficiency(m5, lookback=10))
        wickiness = float(QuantUtils.calculate_wickiness(m5, lookback=5))

        # EMA cross direction (convenience boolean for EMA strategy)
        ema_bullish = ema20 > ema50

        data = {
            "atr":             round(atr, 2),
            "ema20":           round(ema20, 2),
            "ema50":           round(ema50, 2),
            "move_efficiency": round(move_efficiency, 4),
            "wickiness":       round(wickiness, 4),
            "ema_bullish":     ema_bullish,
        }

        return FeatureStore(data)
