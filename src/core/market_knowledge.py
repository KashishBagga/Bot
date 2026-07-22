#!/usr/bin/env python3
"""
Market Knowledge Engine (MKE) Data Entities
===========================================
Defines the core data entities, enums, and structures for the layered Market Context model.
All entities are immutable (frozen) to ensure safety across concurrent execution threads.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union


class SwingStatus(Enum):
    ACTIVE = "ACTIVE"          # Confirmed swing, not yet breached by price
    BREACHED = "BREACHED"      # Price has closed beyond the swing price
    RETESTED = "RETESTED"      # Price has returned to test the swing level
    ARCHIVED = "ARCHIVED"      # Swing is historical/stale


class AnchorType(Enum):
    SESSION = "SESSION"
    SWING = "SWING"
    SYNTHETIC = "SYNTHETIC"
    STRUCTURAL = "STRUCTURAL"


@dataclass(frozen=True)
class SessionAnchor:
    id: str
    timestamp: datetime
    price: float
    type: AnchorType


@dataclass(frozen=True)
class SwingPoint:
    id: str
    timestamp: datetime
    price: float
    type: str                  # "HIGH" or "LOW"
    status: SwingStatus
    confidence: float          # Decays over time, refreshes on test
    strength: float            # Aggregate weighted score (0.0 to 1.0)
    strength_components: Dict[str, float]  # geometry, participation, reaction, persistence
    provenance: Dict[str, Any]  # engine, settings, left_window, right_window

    def __float__(self) -> float:
        return self.price

    def __int__(self) -> int:
        return int(self.price)

    def __add__(self, other):
        if hasattr(other, 'price'):
            return self.price + other.price
        return self.price + other

    def __radd__(self, other):
        return self.price + other

    def __sub__(self, other):
        if hasattr(other, 'price'):
            return self.price - other.price
        return self.price - other

    def __rsub__(self, other):
        return other - self.price

    def __mul__(self, other):
        if hasattr(other, 'price'):
            return self.price * other.price
        return self.price * other

    def __rmul__(self, other):
        return self.price * other

    def __truediv__(self, other):
        if hasattr(other, 'price'):
            return self.price / other.price
        return self.price / other

    def __rtruediv__(self, other):
        return other / self.price

    def __eq__(self, other):
        if hasattr(other, 'id'):
            return self.id == other.id
        if hasattr(other, 'price'):
            return self.price == other.price
        return self.price == other

    def __lt__(self, other):
        if hasattr(other, 'price'):
            return self.price < other.price
        return self.price < other

    def __le__(self, other):
        if hasattr(other, 'price'):
            return self.price <= other.price
        return self.price <= other

    def __gt__(self, other):
        if hasattr(other, 'price'):
            return self.price > other.price
        return self.price > other

    def __ge__(self, other):
        if hasattr(other, 'price'):
            return self.price >= other.price
        return self.price >= other

    def __format__(self, format_spec):
        return format(self.price, format_spec)


@dataclass(frozen=True)
class SwingRelationship:
    swing_id: str
    label: str                 # "HH", "HL", "LH", "LL", "NEUTRAL"


@dataclass(frozen=True)
class LiquidityCluster:
    id: str
    price: float
    type: str                  # "EQH" or "EQL"
    member_swing_ids: List[str]
    strength: float
    last_touched: datetime


@dataclass(frozen=True)
class CompletedLeg:
    id: str
    start_anchor: Union[SwingPoint, SessionAnchor]
    end_pivot: SwingPoint
    type: str                  # "UP_LEG" or "DOWN_LEG"
    price_range: float
    bars_held: int


@dataclass(frozen=True)
class DevelopingLeg:
    start_anchor: Union[SwingPoint, SessionAnchor]
    current_price: float
    current_high: float
    current_low: float
    current_duration_bars: int
    current_extension_atr: float
    live_velocity: float


@dataclass(frozen=True)
class ResearchEvent:
    event_id: str
    timestamp: datetime        # Confirmation time
    occurrence_timestamp: datetime  # Event occurrence time
    symbol: str
    event_type: str            # "BOS_CONFIRMED", "CHOCH_CONFIRMED", "STRUCTURE_RESET"
    engine_version: str
    payload: Dict[str, Any]
    research_id: Optional[str] = None  # Added for M2B

    def to_dict(self) -> Dict[str, Any]:
        """Strict application-level schema contract serialization."""
        d = {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "occurrence_timestamp": self.occurrence_timestamp.isoformat() if isinstance(self.occurrence_timestamp, datetime) else self.occurrence_timestamp,
            "symbol": str(self.symbol),
            "event_type": str(self.event_type),
            "engine_version": str(self.engine_version),
            "payload": self.payload
        }
        if self.research_id is not None:
            d["research_id"] = str(self.research_id)
        return d



@dataclass(frozen=True)
class StructureState:
    swings: List[SwingPoint] = field(default_factory=list)
    relationships: Dict[str, SwingRelationship] = field(default_factory=dict)
    legs: List[CompletedLeg] = field(default_factory=list)
    clusters: List[LiquidityCluster] = field(default_factory=list)
    developing_leg: Optional[DevelopingLeg] = None
    candidate_swing: Optional[SwingPoint] = None
    last_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None
    is_compressed: bool = False

    @property
    def trend(self) -> str:
        high_swings = [s for s in self.swings if s.type == "HIGH"]
        low_swings = [s for s in self.swings if s.type == "LOW"]
        if len(high_swings) < 2 or len(low_swings) < 2:
            return "NEUTRAL"
        last_h, prev_h = high_swings[-1].price, high_swings[-2].price
        last_l, prev_l = low_swings[-1].price, low_swings[-2].price
        if last_h > prev_h and last_l > prev_l:
            return "BULLISH"
        elif last_h < prev_h and last_l < prev_l:
            return "BEARISH"
        return "NEUTRAL"

    @property
    def bos_count(self) -> int:
        return sum(1 for s in self.swings if s.status in (SwingStatus.BREACHED, SwingStatus.RETESTED))

    @property
    def choch_detected(self) -> bool:
        for s in self.swings:
            if s.status in (SwingStatus.BREACHED, SwingStatus.RETESTED):
                rel = self.relationships.get(s.id)
                if rel and rel.label in ["LH", "HL"]:
                    return True
        return False

    @property
    def market_phase(self) -> str:
        trend = self.trend
        if trend == "NEUTRAL" or not self.swings:
            return "NEUTRAL"
        if self.developing_leg:
            return self.developing_leg.type
        return "NEUTRAL"

    @property
    def quality_score(self) -> float:
        score = 50.0
        score += min(self.bos_count * 10, 30)
        if self.choch_detected:
            score -= 40
        if self.is_compressed:
            score += 20
        return max(0.0, min(100.0, score))


@dataclass(frozen=True)
class HTFStructure:
    confirmed: StructureState
    developing: StructureState


class MarketContext:
    """The unified container representing MKE state attached to MarketSnapshot."""
    def __init__(self):
        self.structure: Optional[StructureState] = None
        self.htf_structure: Dict[str, HTFStructure] = {}
        self.trend: Optional[Any] = None
        self.levels: Optional[Any] = None
        self.geometry: Optional[Any] = None
        self.patterns: Optional[Any] = None
        self.liquidity: Optional[Any] = None
        self.regime: Optional[Any] = None
