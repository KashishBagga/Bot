#!/usr/bin/env python3
"""
Market Geometry Data Entities (MKE Stage 5 Context)
====================================================
All geometry layer types: enums, raw level/trendline objects, composite levels,
confluence zones, narrative, and the GeometryContext view.

Design principles:
  - All entities are frozen=True (immutable, safe across threads)
  - Every engine only reads from lower layers and appends derived information
  - GeometryContext pre-computes all O(1) view properties at construction time
  - Narrative fields are typed enums; renderers decide display format
"""

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Dict, Any, List, Optional, Tuple, FrozenSet, Iterator


# ─────────────────────────────────────────────────────────────────────────────
# Geometry Lifecycle (consistent with SwingStatus)
# ─────────────────────────────────────────────────────────────────────────────

class GeometryStatus(Enum):
    CREATED  = "CREATED"    # First detected this session
    ACTIVE   = "ACTIVE"     # Confirmed and untested
    TESTED   = "TESTED"     # Price approached within proximity threshold
    BROKEN   = "BROKEN"     # Price closed through it
    RETESTED = "RETESTED"   # Broken level tested from the other side
    ARCHIVED = "ARCHIVED"   # Stale / outside relevant price range


# ─────────────────────────────────────────────────────────────────────────────
# Level Enums
# ─────────────────────────────────────────────────────────────────────────────

class LevelType(Enum):
    SUPPORT      = "SUPPORT"
    RESISTANCE   = "RESISTANCE"
    PDH          = "PDH"          # Previous Day High
    PDL          = "PDL"          # Previous Day Low
    PWH          = "PWH"          # Previous Week High
    PWL          = "PWL"          # Previous Week Low
    ROUND_NUMBER = "ROUND_NUMBER"
    VWAP         = "VWAP"
    EMA20        = "EMA20"
    EMA50        = "EMA50"
    OPEN_OF_DAY  = "OPEN_OF_DAY"


class LevelDirection(Enum):
    SUPPORT    = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class LevelPriority(IntEnum):
    """
    Determines ordering in narrative labels and report surfacing.
    Does NOT affect confluence scoring math.
    IntEnum so min() naturally selects the highest priority member.
    """
    INSTITUTIONAL = 1   # PDH, PDL, PWH, PWL, LiquidityCluster anchors
    STRUCTURAL    = 2   # Confirmed swing high/low (strength >= 0.45)
    TECHNICAL     = 3   # Trendline, VWAP, EMA, Round Number


class FormationReason(Enum):
    """Why a CompositeLevel was formed."""
    PRICE_PROXIMITY        = "PRICE_PROXIMITY"
    ROUND_NUMBER_CLUSTER   = "ROUND_NUMBER_CLUSTER"
    TRENDLINE_INTERSECTION = "TRENDLINE_INTERSECTION"
    ROLE_REVERSAL          = "ROLE_REVERSAL"
    LIQUIDITY_CLUSTER      = "LIQUIDITY_CLUSTER"


# ─────────────────────────────────────────────────────────────────────────────
# Trendline Enums
# ─────────────────────────────────────────────────────────────────────────────

class TrendlineDirection(Enum):
    ASCENDING  = "ASCENDING"
    DESCENDING = "DESCENDING"
    FLAT       = "FLAT"


class TrendlineRole(Enum):
    SUPPORT    = "SUPPORT"
    RESISTANCE = "RESISTANCE"


# ─────────────────────────────────────────────────────────────────────────────
# Narrative Enums — renderers decide display; engine stays pure
# ─────────────────────────────────────────────────────────────────────────────

class TrendBias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class StructurePhase(Enum):
    CONTINUATION = "CONTINUATION"
    PULLBACK     = "PULLBACK"
    COMPRESSION  = "COMPRESSION"
    BREAKOUT     = "BREAKOUT"
    REVERSAL     = "REVERSAL"


class VolatilityState(Enum):
    COMPRESSED = "COMPRESSED"
    EXPANDING  = "EXPANDING"
    NORMAL     = "NORMAL"


class LiquidityState(Enum):
    EQH_ABOVE  = "EQH_ABOVE"
    EQL_BELOW  = "EQL_BELOW"
    SWEPT_UP   = "SWEPT_UP"
    SWEPT_DOWN = "SWEPT_DOWN"
    CLEAN      = "CLEAN"


class NarrativeBias(Enum):
    CONTINUATION = "CONTINUATION"
    REVERSAL     = "REVERSAL"
    NEUTRAL      = "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Raw Level — one per source, no merging
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HorizontalLevel:
    """
    A single horizontal price level from one source.
    LevelEngine produces these. FusionEngine merges them into CompositeLevels.
    """
    id: str
    price: float
    type: LevelType
    direction: LevelDirection
    priority: LevelPriority
    status: GeometryStatus
    touches: int
    strength: float          # Additive: 0.30×touch + 0.30×participation + 0.25×impulse + 0.15×recency
    freshness: float         # 1.0=just formed; used for display/ordering, NOT in confluence mul
    role_reversal: bool
    confidence: float        # Decays per bar; resets on touch
    distance_pct: float      # |price - current_price| / current_price
    provenance: Dict[str, Any]
    importance: float = 0.5



# ─────────────────────────────────────────────────────────────────────────────
# Trendline — with decomposed confidence and stable 2-anchor identity
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrendlineConfidenceComponents:
    """Decomposed confidence. Allows post-hoc research: why did this line fail?"""
    geometry: float      # R² of OLS fit (structural quality)
    reaction: float      # Average move_efficiency after touch events
    participation: float # Average RVOL rank at touch bars (ToD-normalized)
    persistence: float   # Degrades if line identity keeps changing
    freshness: float     # 1.0 - (bars_since_last_touch / 75)


@dataclass(frozen=True)
class Trendline:
    """
    A fitted price trendline with stable identity.

    Identity rule: ID = SHA256(sorted([oldest_anchor_id, second_oldest_anchor_id]) + role)
    Adding a 3rd+ touch is metadata. The ID never changes once the line is born.
    This makes TRENDLINE_CREATED and TRENDLINE_BROKEN reference the same object.
    """
    id: str                                  # "tl_" + sha256(primary anchors + role)[:16]
    primary_anchor_ids: Tuple[str, str]      # Oldest two anchors — define identity
    all_anchor_ids: Tuple[str, ...]          # All confirmed touches (metadata, grows over time)
    anchor_bar_ids: Tuple[int, ...]          # TradingClock.trading_bar_id() per anchor
    anchor_prices: Tuple[float, ...]
    slope: float                             # Points per trading_bar_id unit (restart-proof)
    angle_degrees: float
    direction: TrendlineDirection
    role: TrendlineRole
    status: GeometryStatus
    touches: int
    age_bars: int
    r_squared: float
    confidence: float
    confidence_components: TrendlineConfidenceComponents
    price_at_now: float                      # Projected using TradingClock bar IDs
    distance_pct: float
    provenance: Dict[str, Any]
    importance: float = 0.5


    @staticmethod
    def make_id(anchor_id_a: str, anchor_id_b: str, role: str) -> str:
        """Build stable trendline ID from the two oldest (primary) anchor swing IDs."""
        key = "|".join(sorted([anchor_id_a, anchor_id_b])) + f"|{role}"
        return "tl_" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# CompositeLevel — FusionEngine output (the single merging point)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompositeLevel:
    """
    A merged cluster of nearby HorizontalLevels and/or Trendlines at a single price zone.
    Produced exclusively by FusionEngine. All downstream engines consume these.

    geometry_relations: lightweight graph — maps each member ID to its co-members.
    PatternEngine reads this in 2B to avoid re-discovering relationships.
    """
    id: str
    price: float                                 # Confidence-weighted average of member prices
    band_low: float
    band_high: float
    width: float                                 # band_high - band_low (points)
    direction: LevelDirection
    priority: LevelPriority                      # min(member priorities) — INSTITUTIONAL wins
    status: GeometryStatus
    confidence: float                            # max(member confidences)
    raw_levels: Tuple[HorizontalLevel, ...]
    raw_trendlines: Tuple[Trendline, ...]
    member_types: Tuple[str, ...]                # ("PDH", "ROUND_NUMBER", "EMA20") for labels
    formation_reasons: FrozenSet[FormationReason]
    geometry_relations: Dict[str, List[str]]     # member_id → [other_member_ids in this composite]
    provenance: Dict[str, Any]

    @staticmethod
    def make_id(member_ids: List[str]) -> str:
        key = "|".join(sorted(member_ids))
        return "cl_" + hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        """
        Returns a human-readable label surfacing members in priority order.
        e.g. "PDH + Round 22100 + EMA20"
        """
        # Sort by priority (INSTITUTIONAL first), then by type name
        priority_order = {
            "PDH": 1, "PDL": 1, "PWH": 1, "PWL": 1,
            "SUPPORT": 2, "RESISTANCE": 2,
            "TRENDLINE": 3,
            "VWAP": 4, "EMA20": 4, "EMA50": 4, "ROUND_NUMBER": 4,
            "OPEN_OF_DAY": 5,
        }
        types = sorted(set(self.member_types), key=lambda t: priority_order.get(t, 9))
        return " + ".join(types) if types else "Level"


# ─────────────────────────────────────────────────────────────────────────────
# ConfluenceZone — ConfluenceEngine output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfluenceComponent:
    source_id: str
    source_type: str          # "COMPOSITE_PDH+ROUND", "TRENDLINE_SUPPORT"
    contribution: float       # composite.confidence × distance_decay
    distance_pct: float
    explanation: str          # "PDH + Round @ 22100 (0.08% away, conf=0.91)"


@dataclass(frozen=True)
class ConfluenceZone:
    """
    A price band where multiple geometry objects agree.
    Score = 100 × tanh(raw_sum / MIDPOINT) — saturates naturally as new engines added.
    """
    price: float
    band_low: float
    band_high: float
    width: float
    total_score: float        # 0–100, tanh-normalized
    direction: LevelDirection
    components: Tuple[ConfluenceComponent, ...]
    explanation: str          # "Trendline + PDH + EMA20 @ 22151 (±5pts, score=74)"


# ─────────────────────────────────────────────────────────────────────────────
# MarketNarrative — NarrativeEngine output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NarrativeEvidence:
    """Traceable sources for one narrative field. Nothing synthesized silently."""
    sources: Tuple[str, ...]


@dataclass(frozen=True)
class MarketNarrative:
    """
    A synthesized market interpretation for one tick.
    All fields are typed enums. Renderers (UI, reports) decide display format.
    Every field has a corresponding NarrativeEvidence entry in `evidence`.
    """
    primary_trend: TrendBias
    secondary_structure: StructurePhase
    volatility_state: VolatilityState
    liquidity_state: LiquidityState
    nearest_support_label: str              # "Trendline + PDH @ 22150 (±4pts)"
    nearest_resistance_label: str
    support_confluence_score: float
    resistance_confluence_score: float
    dominant_pattern: Optional[str]         # None until 2B
    bias: NarrativeBias
    bias_confidence: float                  # 0.0–1.0
    evidence: Dict[str, NarrativeEvidence]  # field_name → traceable sources


# ─────────────────────────────────────────────────────────────────────────────
# GeometryQuery — typed fluent builder for research / advanced filtering
# ─────────────────────────────────────────────────────────────────────────────

class LevelQuery:
    """O(n) fluent filter builder. For research scripts and EOD reports — NOT hot loop."""

    def __init__(self, composites: Tuple["CompositeLevel", ...]):
        self._items: List[CompositeLevel] = list(composites)

    def support(self) -> "LevelQuery":
        self._items = [c for c in self._items if c.direction == LevelDirection.SUPPORT]
        return self

    def resistance(self) -> "LevelQuery":
        self._items = [c for c in self._items if c.direction == LevelDirection.RESISTANCE]
        return self

    def not_broken(self) -> "LevelQuery":
        self._items = [c for c in self._items if c.status != GeometryStatus.BROKEN]
        return self

    def min_confidence(self, v: float) -> "LevelQuery":
        if not isinstance(v, (int, float)):
            raise TypeError(f"min_confidence requires float, got {type(v)}")
        self._items = [c for c in self._items if c.confidence >= v]
        return self

    def min_priority(self, p: LevelPriority) -> "LevelQuery":
        self._items = [c for c in self._items if c.priority <= p]  # IntEnum: lower = higher priority
        return self

    def institutional(self) -> "LevelQuery":
        return self.min_priority(LevelPriority.INSTITUTIONAL)

    def within_distance(self, pct: float) -> "LevelQuery":
        if not isinstance(pct, (int, float)):
            raise TypeError(f"within_distance requires float, got {type(pct)}")
        self._items = [c for c in self._items if c.confidence > 0 and
                       min(abs(rl.distance_pct) for rl in c.raw_levels) <= pct
                       if c.raw_levels]
        return self

    def has_reason(self, reason: FormationReason) -> "LevelQuery":
        self._items = [c for c in self._items if reason in c.formation_reasons]
        return self

    def first(self) -> Optional[CompositeLevel]:
        return self._items[0] if self._items else None

    def top(self, n: int) -> List[CompositeLevel]:
        return self._items[:n]

    def all(self) -> List[CompositeLevel]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class TrendlineQuery:
    """O(n) fluent filter builder for trendlines."""

    def __init__(self, trendlines: Tuple["Trendline", ...]):
        self._items: List[Trendline] = list(trendlines)

    def support(self) -> "TrendlineQuery":
        self._items = [t for t in self._items if t.role == TrendlineRole.SUPPORT]
        return self

    def resistance(self) -> "TrendlineQuery":
        self._items = [t for t in self._items if t.role == TrendlineRole.RESISTANCE]
        return self

    def not_broken(self) -> "TrendlineQuery":
        self._items = [t for t in self._items if t.status != GeometryStatus.BROKEN]
        return self

    def min_touches(self, n: int) -> "TrendlineQuery":
        if not isinstance(n, int):
            raise TypeError(f"min_touches requires int, got {type(n)}")
        self._items = [t for t in self._items if t.touches >= n]
        return self

    def min_r2(self, v: float) -> "TrendlineQuery":
        if not isinstance(v, (int, float)):
            raise TypeError(f"min_r2 requires float, got {type(v)}")
        self._items = [t for t in self._items if t.r_squared >= v]
        return self

    def min_confidence(self, v: float) -> "TrendlineQuery":
        if not isinstance(v, (int, float)):
            raise TypeError(f"min_confidence requires float, got {type(v)}")
        self._items = [t for t in self._items if t.confidence >= v]
        return self

    def ascending(self) -> "TrendlineQuery":
        self._items = [t for t in self._items if t.direction == TrendlineDirection.ASCENDING]
        return self

    def descending(self) -> "TrendlineQuery":
        self._items = [t for t in self._items if t.direction == TrendlineDirection.DESCENDING]
        return self

    def within_distance(self, pct: float) -> "TrendlineQuery":
        if not isinstance(pct, (int, float)):
            raise TypeError(f"within_distance requires float, got {type(pct)}")
        self._items = [t for t in self._items if abs(t.distance_pct) <= pct]
        return self

    def first(self) -> Optional[Trendline]:
        return self._items[0] if self._items else None

    def top(self, n: int) -> List[Trendline]:
        return self._items[:n]

    def all(self) -> List[Trendline]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


# ─────────────────────────────────────────────────────────────────────────────
# Views — pre-computed O(1) properties, collection protocol
# ─────────────────────────────────────────────────────────────────────────────

class LevelsView:
    """
    Pre-computed, immutable view of all composite levels.
    All convenience methods are O(1) — computed at construction, never at call time.
    Use .query for complex research filtering (O(n)).
    Implements __iter__ and __len__ for collection protocol.
    """

    def __init__(self, composites: List[CompositeLevel], current_price: float):
        # Sort by distance from current price — O(n log n) once at construction
        _sorted = tuple(sorted(composites, key=lambda c: abs(c.price - current_price)))
        self._all: Tuple[CompositeLevel, ...] = _sorted

        # Pre-partition by direction
        self._support    = tuple(c for c in _sorted if c.direction == LevelDirection.SUPPORT)
        self._resistance = tuple(c for c in _sorted if c.direction == LevelDirection.RESISTANCE)
        self._above      = tuple(c for c in _sorted if c.price > current_price)
        self._below      = tuple(c for c in _sorted if c.price <= current_price)

        # Pre-compute named singletons
        self._nearest_support    = self._support[0] if self._support else None
        self._nearest_resistance = self._resistance[0] if self._resistance else None
        self._nearest            = _sorted[0] if _sorted else None

        # Named institutional levels — search raw_levels of all composites
        self._pdh  = self._find_raw_level(LevelType.PDH)
        self._pdl  = self._find_raw_level(LevelType.PDL)
        self._pwh  = self._find_raw_level(LevelType.PWH)
        self._pwl  = self._find_raw_level(LevelType.PWL)
        self._open = self._find_raw_level(LevelType.OPEN_OF_DAY)
        self._vwap = self._find_raw_level(LevelType.VWAP)

        # Round numbers
        self._rounds = tuple(
            c for c in _sorted
            if "ROUND_NUMBER" in c.member_types
        )

    def _find_raw_level(self, lt: LevelType) -> Optional[HorizontalLevel]:
        for composite in self._all:
            for rl in composite.raw_levels:
                if rl.type == lt:
                    return rl
        return None

    # ── O(1) convenience ────────────────────────────────────────────────────

    def nearest(self) -> Optional[CompositeLevel]:            return self._nearest
    def nearest_support(self) -> Optional[CompositeLevel]:    return self._nearest_support
    def nearest_resistance(self) -> Optional[CompositeLevel]: return self._nearest_resistance
    def above_price(self) -> Tuple[CompositeLevel, ...]:      return self._above
    def below_price(self) -> Tuple[CompositeLevel, ...]:      return self._below
    def round_numbers(self) -> Tuple[CompositeLevel, ...]:    return self._rounds

    def pdh(self) -> Optional[HorizontalLevel]:  return self._pdh
    def pdl(self) -> Optional[HorizontalLevel]:  return self._pdl
    def pwh(self) -> Optional[HorizontalLevel]:  return self._pwh
    def pwl(self) -> Optional[HorizontalLevel]:  return self._pwl
    def open_of_day(self) -> Optional[HorizontalLevel]: return self._open
    def vwap(self) -> Optional[HorizontalLevel]: return self._vwap

    # ── Collection protocol ─────────────────────────────────────────────────

    def __iter__(self) -> Iterator[CompositeLevel]: return iter(self._all)
    def __len__(self) -> int:                       return len(self._all)

    # ── Fluent query (O(n), for research only) ──────────────────────────────

    @property
    def query(self) -> LevelQuery:
        return LevelQuery(self._all)


class TrendlinesView:
    """
    Pre-computed, immutable view of all trendlines.
    O(1) convenience methods pre-computed at construction.
    Implements __iter__ and __len__.
    """

    def __init__(self, trendlines: List[Trendline], current_price: float):
        _sorted = tuple(sorted(trendlines, key=lambda t: abs(t.distance_pct)))
        self._all: Tuple[Trendline, ...] = _sorted

        self._support    = tuple(t for t in _sorted if t.role == TrendlineRole.SUPPORT)
        self._resistance = tuple(t for t in _sorted if t.role == TrendlineRole.RESISTANCE)
        self._active     = tuple(t for t in _sorted if t.status != GeometryStatus.BROKEN)

        self._nearest_support    = self._support[0] if self._support else None
        self._nearest_resistance = self._resistance[0] if self._resistance else None

    # ── O(1) convenience ────────────────────────────────────────────────────

    def nearest_support(self) -> Optional[Trendline]:    return self._nearest_support
    def nearest_resistance(self) -> Optional[Trendline]: return self._nearest_resistance
    def all_support(self) -> Tuple[Trendline, ...]:      return self._support
    def all_resistance(self) -> Tuple[Trendline, ...]:   return self._resistance
    def active(self) -> Tuple[Trendline, ...]:           return self._active

    # ── Collection protocol ─────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Trendline]: return iter(self._all)
    def __len__(self) -> int:                  return len(self._all)

    # ── Fluent query (O(n), for research only) ──────────────────────────────

    @property
    def query(self) -> TrendlineQuery:
        return TrendlineQuery(self._all)


# ─────────────────────────────────────────────────────────────────────────────
# GeometryContext — the immutable view attached to MarketContext
# ─────────────────────────────────────────────────────────────────────────────

class GeometryContext:
    """
    Immutable, pre-computed geometry view for one tick and one symbol.

    Constructed once by IndicatorPipeline._stage_geometry().
    All view properties are O(1) — computed at construction, not at call time.

    This IS the geometry cache: stateless, replay-safe, restart-proof.
    Cross-tick caching would add statefulness that violates the read-derive-append invariant.

    pending_events: geometry lifecycle events emitted this tick.
    Written to market_events hypertable by the pipeline after construction.
    """

    def __init__(
        self,
        composites: List[CompositeLevel],
        trendlines: List[Trendline],
        current_price: float,
        support_confluence: Optional[ConfluenceZone] = None,
        resistance_confluence: Optional[ConfluenceZone] = None,
        narrative: Optional[MarketNarrative] = None,
        pending_events: Optional[List[Any]] = None,  # List[ResearchEvent]
    ):
        self.levels    = LevelsView(composites, current_price)
        self.trendlines = TrendlinesView(trendlines, current_price)
        self.support_confluence    = support_confluence
        self.resistance_confluence = resistance_confluence
        self.narrative             = narrative
        self.pending_events        = tuple(pending_events or [])
