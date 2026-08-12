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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.feature_store import FeatureStore


# ── Option snapshot typed row (replaces raw Dict in options strategies) ────────

@dataclass(frozen=True)
class OptionSnapshotRow:
    """
    One row from option_snapshots table, typed for safe strategy consumption.
    Used by OptionsScalpingStrategy via db.get_atm_oi_series().
    """
    time: datetime
    strike: float
    option_type: str        # "CE" | "PE"
    ltp: float
    bid: float
    ask: float
    oi: int
    volume: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_ratio(self) -> float:
        """Relative bid-ask spread. inf when mid is zero."""
        m = self.mid
        return (self.ask - self.bid) / m if m > 0 else float('inf')


# ── Pre-market data structures ─────────────────────────────────────────────────

@dataclass(frozen=True)
class PreMarketData:
    """
    Collected by PreMarketCollector, polling 9:00–9:14 IST.
    Snapshot frozen at 9:15 IST (last available indicative price before open).
    is_available=False when the collector failed the entire window.
    """
    gap_pct: float                # (preopen_price - prev_close) / prev_close × 100
    gap_direction: str            # "UP" | "DOWN" | "FLAT"
    gap_magnitude: str            # "SMALL" (<0.5%) | "MEDIUM" (0.5-1%) | "LARGE" (>1%)
    pdh: float                    # previous day high
    pdl: float                    # previous day low
    prev_close: float
    preopen_price: float          # latest indicative price before 9:15
    captured_at: Optional[datetime] = None
    is_available: bool = True

    @property
    def overnight_range(self) -> float:
        return self.pdh - self.pdl


@dataclass(frozen=True)
class OpeningData:
    """
    Collected at 9:20 IST after the first 5-minute bar closes.
    NEVER used in signals before 9:20 — leakage guard enforced in GapStrategy.
    is_available=False before 9:20 or if first bar data is unavailable.
    """
    first_5m_direction: str       # "UP" | "DOWN" | "FLAT"
    first_5m_rvol: float          # first bar RVOL vs 5-day avg first bar
    opening_volume_ratio: float   # first_5m_volume / 5-day avg first_5m_volume
    is_available: bool = True


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
    h1_zones: list               # List of Zone objects from ZoneEngine (h1) — unchanged, existing consumers
    m5_zones: list               # List of Zone objects from ZoneEngine (m5) — additive, MTF confluence
    d1_zones: list               # List of Zone objects from ZoneEngine (d1) — additive, MTF confluence
    market_regime: str           # e.g. "STRONG_TREND_UP_HIGH_VOL", "RANGE_NORMAL"
    volume_report: object        # VolumeReport (rvol_tod, is_high_participation, etc.)
    regime_detail: object        # RegimeLabel — structured breakdown (adx, atr_pct, gap_pct, session)
                                 # Optional: None when constructed by older code paths.

    # ── Extensible feature store ───────────────────────────────────────────
    # Strategies access via snapshot.features.get_float("atr") etc.
    # New indicators: add to IndicatorPipeline._stage_features(), nothing else.
    features: FeatureStore
    market: object  # MarketContext object containing MKE structure, trend, etc.

    # ── New optional sub-snapshots (None until populated by respective collectors) ─
    premarket: Optional[PreMarketData] = None      # None before 9:15 or on collector failure
    opening: Optional[OpeningData] = None          # None before 9:20
    atm_chain: Optional[List[OptionSnapshotRow]] = None  # Last 10 min ATM CE+PE rows

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
