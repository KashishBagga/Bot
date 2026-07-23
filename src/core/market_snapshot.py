#!/usr/bin/env python3
"""
MarketSnapshot — Single shared market state object for one symbol, one candle.
==============================================================================
Produced once per symbol per 5-minute cycle by IndicatorPipeline.compute().
Consumed read-only by all strategies in ExperimentRegistry.run().

Design principle: compute once, share everywhere.
  - One Fyers API call
  - One pass through each indicator engine
  - Unlimited strategies read from the same snapshot
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from src.core.feature_store import FeatureStore


# These are imported lazily to avoid circular imports in type hints.
# The actual StructureReport / VolumeReport types live in their respective engines.

@dataclass
class MarketSnapshot:
    """
    Immutable (by convention) market state for one symbol at one point in time.
    Created by IndicatorPipeline.compute() and never modified after construction.

    Structural fields are typed explicitly because the StructuralStrategy
    needs direct, typed access to zones, structure, and bias.

    All other indicators live in the extensible FeatureStore.
    Adding a new indicator = one line in IndicatorPipeline._stage_features().
    Zero changes to this dataclass or any strategy.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    symbol: str
    current_price: float
    timestamp: datetime

    # ── Raw OHLCV DataFrames ───────────────────────────────────────────────
    d1: Optional[pd.DataFrame]   # Daily — 40 days of history
    h1: Optional[pd.DataFrame]   # Hourly — 10 days of history
    m5: Optional[pd.DataFrame]   # 5-minute — 5 days of history

    # ── Structural layer (typed — StructuralStrategy reads these directly) ─
    daily_bias: str              # "BULLISH" | "BEARISH" | "NEUTRAL"
    h1_structure: object         # StructureReport (typed in engine, duck-typed here)
    h1_zones: list               # List of Zone objects from ZoneEngine
    market_regime: str           # e.g. "TREND_UP_HIGH_VOL", "RANGE"
    volume_report: object        # VolumeReport (rvol_tod, is_high_participation, etc.)

    # ── Extensible feature store ───────────────────────────────────────────
    # Strategies access via snapshot.features.get_float("atr") etc.
    # New indicators: add to IndicatorPipeline._stage_features(), nothing else.
    features: FeatureStore
    market: object  # MarketContext object containing MKE structure, trend, etc.

    # ── Shared confluence layer (optional) ─────────────────────────────────
    # MarketView aggregates chart patterns + bias + regime + RVOL. Strategies
    # read it to boost/dampen their own confidence (composability). Optional so
    # non-pipeline construction paths (tests, legacy engine) still work.
    market_view: object = None  # src.core.market_view.MarketView | None

    def __repr__(self) -> str:
        return (
            f"MarketSnapshot("
            f"symbol={self.symbol}, "
            f"price={self.current_price}, "
            f"bias={self.daily_bias}, "
            f"regime={self.market_regime}, "
            f"features={self.features}"
            f")"
        )
