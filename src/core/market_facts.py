#!/usr/bin/env python3
"""
Market Facts Data Entity (MKE Stage 1–4 Sealed primitives)
===========================================================
Holds all facts computed during Stages 1–4 that are needed by geometry, patterns,
and other downstream engines. This serves as the shared read-only primitive context.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from src.core.market_knowledge import (
    SwingPoint, CompletedLeg, DevelopingLeg, LiquidityCluster, SwingRelationship
)


@dataclass(frozen=True)
class MarketFacts:
    """
    Immutable container of Stage 1–4 primitive facts.
    Sealed after Stage 4 and passed to all downstream engines.
    """
    symbol: str
    current_price: float
    current_bar: int
    atr: float
    tick_size: float
    timestamp: datetime
    session: str                    # "OPEN" | "MID" | "CLOSE"
    is_open_blackout: bool
    is_close_blackout: bool
    swings: Tuple[SwingPoint, ...]
    completed_legs: Tuple[CompletedLeg, ...]
    developing_leg: Optional[DevelopingLeg]
    clusters: Tuple[LiquidityCluster, ...]
    relationships: Dict[str, SwingRelationship]
    rvol_tod: float
    atr_percentile: float
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    is_compressed: bool
