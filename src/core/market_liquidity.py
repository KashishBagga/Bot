#!/usr/bin/env python3
"""
Market Liquidity Data Entities (MKE Stage 7 Context)
===================================================
Defines enums, imbalances, liquidity pools, sweeps, directional pressure map,
transition events, and the LiquidityContext container.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Iterator

from src.core.market_knowledge import ResearchEvent
from src.core.market_patterns import PatternDirection


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity Enums
# ─────────────────────────────────────────────────────────────────────────────

class ImbalanceType(Enum):
    FVG = "FVG"
    OPENING_GAP = "OPENING_GAP"
    VOLUME_VOID = "VOLUME_VOID"
    SINGLE_PRINT = "SINGLE_PRINT"


class SweptLevelType(Enum):
    SWING = "SWING"
    PDH = "PDH"
    PDL = "PDL"
    PSH = "PSH"
    PSL = "PSL"
    EQH = "EQH"
    EQL = "EQL"
    POOL = "POOL"


class SweepType(Enum):
    BULLISH = "BULLISH"  # Swept low (liquidity below), rejecting up
    BEARISH = "BEARISH"  # Swept high (liquidity above), rejecting down


class SweepState(Enum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    REVERSAL = "REVERSAL"
    CONTINUATION = "CONTINUATION"
    TIMEOUT = "TIMEOUT"
    ARCHIVED = "ARCHIVED"


class LiquidityPressureType(Enum):
    ABSORBING = "ABSORBING"
    SEEKING_HIGHER = "SEEKING_HIGHER"
    SEEKING_LOWER = "SEEKING_LOWER"
    VACUUM = "VACUUM"
    BALANCED = "BALANCED"


# ─────────────────────────────────────────────────────────────────────────────
# Imbalances
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImbalanceConfidenceComponents:
    gap_size_score: float
    displacement_score: float
    volume_score: float
    structure_score: float
    freshness_score: float


@dataclass(frozen=True)
class Imbalance:
    id: str
    type: ImbalanceType
    direction: PatternDirection                      # LONG (bullish) | SHORT (bearish)
    top: float
    bottom: float
    creation_bar: int
    creation_time: datetime
    last_seen_bar: int
    
    # Stateful partial mitigation details
    is_mitigated: bool
    mitigated_bar: Optional[int]
    mitigated_time: Optional[datetime]
    mitigated_price: Optional[float]
    is_fully_filled: bool
    fill_percentage: float                           # 0.0 to 1.0 representation
    remaining_gap: float
    deepest_fill: float
    
    # Decomposable confidence
    confidence: float
    confidence_components: ImbalanceConfidenceComponents
    raw_components: Dict[str, float]
    explanation: str


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity Pools
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LiquidityPool:
    id: str
    type: str                                        # "EQH" | "EQL" | "PDH" | "PDL" | "SWING"
    center_price: float
    width: float                                     # price width of the stop cluster
    touches: int
    last_interaction: datetime
    
    # Decomposed scoring components
    type_score: float
    touch_score: float
    volume_score: float
    confluence_score: float
    pattern_score: float
    estimated_stop_density: float                    # Scored via pluggable StopDensityScorer
    
    confidence: float
    member_swing_ids: Tuple[str, ...]


# ─────────────────────────────────────────────────────────────────────────────
# Sweeps
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SweepConfidenceComponents:
    wick_length_score: float
    reclaim_score: float
    volume_score: float
    liquidity_score: float
    structure_score: float
    pattern_score: float


@dataclass(frozen=True)
class LiquiditySweep:
    id: str
    type: SweepType
    state: SweepState
    level_swept: float
    level_type: SweptLevelType
    swept_object_id: str                             # Reference to the actual swept entity
    swept_object_type: str                           # "LIQUIDITY_POOL", "TRENDLINE", etc.
    object_confidence: float
    
    bar_index: int
    timestamp: datetime
    
    # Raw metrics
    volume_multiplier: float
    wick_size_atr: float
    rejection_wick_size: float
    
    # Decomposable confidence
    confidence: float
    confidence_components: SweepConfidenceComponents
    raw_components: Dict[str, float]
    
    # Context snapshots
    pattern_ids: Tuple[str, ...]
    geometry_ids: Tuple[str, ...]
    confluence_zone_id: Optional[str]
    
    # Outcome tracking metrics
    outcome: SweepState                              # REVERSAL | CONTINUATION | TIMEOUT
    bars_until_resolution: Optional[int] = None
    max_excursion: float = 0.0                       # Highest price reached in sweep direction
    max_adverse_excursion: float = 0.0               # Most adverse price hit during lifetime


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity Map
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LiquidityMap:
    nearest_bullish_imbalance: Optional[Imbalance]
    nearest_bearish_imbalance: Optional[Imbalance]
    nearest_liquidity_above: Optional[LiquidityPool]
    nearest_liquidity_below: Optional[LiquidityPool]
    active_sweep: Optional[LiquiditySweep]
    bullish_pressure: float                          # Magnet/Support strength (0.0 to 1.0)
    bearish_pressure: float                          # Magnet/Resistance strength (0.0 to 1.0)
    pressure_state: LiquidityPressureType


# ─────────────────────────────────────────────────────────────────────────────
# Transition Events & Context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LiquidityTransitionEvent:
    event_id: str
    event_type: str                                  # "IMBALANCE_CREATED", "IMBALANCE_MITIGATED", "IMBALANCE_FILLED", "SWEEP_DETECTED", "SWEEP_OUTCOME_RESOLVED"
    symbol: str
    bar: int
    timestamp: datetime
    payload: Dict[str, Any]
    research_id: Optional[str] = None

    def to_research_event(self, now: datetime) -> ResearchEvent:
        """Converts to a central ResearchEvent container for SQL persistence."""
        return ResearchEvent(
            event_id=self.event_id,
            timestamp=now,
            occurrence_timestamp=self.timestamp,
            symbol=self.symbol,
            event_type=self.event_type,
            engine_version="v2C",
            payload=self.payload,
            research_id=self.research_id
        )


@dataclass(frozen=True)
class LiquidityContext:
    symbol: str
    timestamp: datetime
    current_bar: int
    active_imbalances: Tuple[Imbalance, ...] = ()
    mitigated_imbalances: Tuple[Imbalance, ...] = ()
    pools: Tuple[LiquidityPool, ...] = ()
    sweeps: Tuple[LiquiditySweep, ...] = ()
    liq_map: Optional[LiquidityMap] = None
    transition_events: Tuple[LiquidityTransitionEvent, ...] = ()


# ─────────────────────────────────────────────────────────────────────────────
# ID Builder Helpers
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityIDBuilder:
    @staticmethod
    def make_imbalance_id(symbol: str, creation_bar: int, im_type: ImbalanceType) -> str:
        key = f"imb|{symbol}|{creation_bar}|{im_type.value}"
        return "imb_" + hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def make_sweep_id(symbol: str, bar_index: int, level_price: float, sweep_type: SweepType) -> str:
        key = f"swp|{symbol}|{bar_index}|{round(level_price, 2)}|{sweep_type.value}"
        return "swp_" + hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def make_pool_id(symbol: str, pool_type: str, price: float, key_ts: datetime) -> str:
        ts_str = key_ts.strftime("%Y%m%d_%H%M%S")
        key = f"pool|{symbol}|{pool_type}|{round(price, 2)}|{ts_str}"
        return "pool_" + hashlib.sha256(key.encode()).hexdigest()[:16]
