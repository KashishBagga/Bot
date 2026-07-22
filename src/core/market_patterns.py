#!/usr/bin/env python3
"""
Market Patterns Data Entities (MKE Stage 6 Context)
===================================================
Defines enums, anchors, evidence, confidence components, transition events,
deltas, pattern objects, and the PatternsContext container view.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Iterator

from src.core.market_knowledge import ResearchEvent


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Enums
# ─────────────────────────────────────────────────────────────────────────────

class PatternType(Enum):
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    RECTANGLE = "RECTANGLE"
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    INVERSE_HEAD_AND_SHOULDERS = "INVERSE_HEAD_AND_SHOULDERS"


class PatternState(Enum):
    CREATED = "CREATED"
    FORMING = "FORMING"
    ACTIVE = "ACTIVE"
    READY = "READY"
    BREAKOUT = "BREAKOUT"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class PatternDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BILATERAL = "BILATERAL"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Entities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatternAnchor:
    """An execution anchor for a pattern (representing structural pivots)."""
    id: str
    role: str                       # e.g., "LEFT_HIGH", "NECKLINE", "POLE_START"
    price: float
    bar_index: int
    timestamp: datetime             # Replay-safe timestamp
    source_swing_id: Optional[str] = None
    source_composite_id: Optional[str] = None
    source_trendline_id: Optional[str] = None


@dataclass(frozen=True)
class PatternEvidence:
    """Provenance and metrics supporting the pattern discovery."""
    swing_ids: Tuple[str, ...]
    trendline_ids: Tuple[str, ...]
    composite_ids: Tuple[str, ...]
    confluence_scores: Dict[str, float]
    explanation: Tuple[str, ...]
    metrics: Dict[str, Any]
    detector_name: str
    detector_version: str
    detector_parameters: Dict[str, Any]
    snapshot_uri: Optional[str] = None


@dataclass(frozen=True)
class PatternConfidenceComponents:
    """Confidence weights for evaluation."""
    anchor_quality: float
    geometry_confluence: float
    symmetry: float
    compression: float
    context_alignment: float


@dataclass(frozen=True)
class PatternDelta:
    """Transition metric changes compared to previous tick."""
    pattern_id: str
    state_changed: bool
    confidence_delta: float
    trigger_quality_delta: float
    completion_delta: float
    age_delta: int
    breakout_distance_delta: float


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Entity
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Pattern:
    """
    A single detected market chart pattern.
    Stateless and immutable; re-evaluated each tick.
    """
    id: str
    research_id: str
    type: PatternType
    state: PatternState
    direction: PatternDirection

    
    quality_score: float                     # static (geometry only)
    confidence: float                        # weighted confidence score
    confidence_components: PatternConfidenceComponents
    raw_components: Dict[str, float]          # unweighted components
    
    trigger_quality: float                   # proximity to breakout, dynamic
    
    breakout_level: float
    original_invalidation: float
    current_invalidation: float
    targets: Tuple[float, ...]
    target_labels: Tuple[str, ...]
    
    completion_pct: float
    age_bars: int
    retest_count: int
    last_seen_bar: int
    
    anchors: Tuple[PatternAnchor, ...]
    evidence: PatternEvidence
    explanation: str
    
    parent_pattern_id: Optional[str] = None
    child_pattern_ids: Tuple[str, ...] = ()


    @staticmethod
    def make_id(anchor_ids: List[str], pattern_type: str) -> str:
        key = "|".join(sorted(anchor_ids)) + f"|{pattern_type}"
        return "pat_" + hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def make_research_id(pattern_id: str, symbol: str, first_bar: int) -> str:
        key = f"{pattern_id}|{symbol}|{first_bar}"
        return "rs_" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Transition Events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatternTransitionEvent:
    """Describes a pattern lifecycle state transition or valuation update."""
    pattern_id: str
    research_id: str
    symbol: str
    from_state: Optional[PatternState]
    to_state: PatternState
    pattern: Pattern
    bar: int
    delta: Optional[PatternDelta] = None

    def to_research_event(self, now: datetime) -> ResearchEvent:
        """Converts to a ResearchEvent for database insertion."""
        payload = {
            "pattern_id": self.pattern_id,
            "research_id": self.research_id,
            "pattern_type": self.pattern.type.value,
            "direction": self.pattern.direction.value,
            "from_state": self.from_state.value if self.from_state else None,
            "to_state": self.to_state.value,
            "confidence": self.pattern.confidence,
            "quality_score": self.pattern.quality_score,
            "trigger_quality": self.pattern.trigger_quality,
            "breakout_level": self.pattern.breakout_level,
            "targets": list(self.pattern.targets),
            "target_labels": list(self.pattern.target_labels),
            "original_invalidation": self.pattern.original_invalidation,
            "current_invalidation": self.pattern.current_invalidation,
            "age_bars": self.pattern.age_bars,
            "completion_pct": self.pattern.completion_pct,
            "explanation": self.pattern.explanation,
            "metrics": self.pattern.evidence.metrics,
        }
        if self.delta:
            payload["delta"] = {
                "state_changed": self.delta.state_changed,
                "confidence_delta": round(self.delta.confidence_delta, 4),
                "trigger_quality_delta": round(self.delta.trigger_quality_delta, 4),
                "completion_delta": round(self.delta.completion_delta, 4),
                "age_delta": self.delta.age_delta,
                "breakout_distance_delta": round(self.delta.breakout_distance_delta, 4)
            }
        
        # Occurrence is anchor timestamp or now fallback
        occ_time = self.pattern.anchors[-1].timestamp if self.pattern.anchors else now

        return ResearchEvent(
            event_id=f"ev_pat_{self.pattern_id}_{self.to_state.value.lower()}_{self.bar}",
            timestamp=now,
            occurrence_timestamp=occ_time,
            symbol=self.symbol,
            event_type=f"PATTERN_{self.to_state.value}",
            engine_version=self.pattern.evidence.detector_version,
            payload=payload,
            research_id=self.research_id
        )



# ─────────────────────────────────────────────────────────────────────────────
# Context View
# ─────────────────────────────────────────────────────────────────────────────

class PatternsContext:
    """
    Immutable container of active patterns and transition events on a tick.
    Provides fast O(1) properties for strategy lookup.
    """

    def __init__(self, patterns: List[Pattern], transition_events: List[PatternTransitionEvent]):
        self.patterns = tuple(patterns)
        self.transition_events = tuple(transition_events)

        # Pre-compute lookups
        self._by_type: Dict[PatternType, List[Pattern]] = {}
        for p in self.patterns:
            self._by_type.setdefault(p.type, []).append(p)

    def double_top(self) -> Optional[Pattern]:
        items = self._by_type.get(PatternType.DOUBLE_TOP, [])
        return items[0] if items else None

    def double_bottom(self) -> Optional[Pattern]:
        items = self._by_type.get(PatternType.DOUBLE_BOTTOM, [])
        return items[0] if items else None

    def head_and_shoulders(self) -> Optional[Pattern]:
        items = self._by_type.get(PatternType.HEAD_AND_SHOULDERS, [])
        return items[0] if items else None

    def inv_head_and_shoulders(self) -> Optional[Pattern]:
        items = self._by_type.get(PatternType.INVERSE_HEAD_AND_SHOULDERS, [])
        return items[0] if items else None

    def bull_flag(self) -> Optional[Pattern]:
        items = self._by_type.get(PatternType.BULL_FLAG, [])
        return items[0] if items else None

    def bear_flag(self) -> Optional[Pattern]:
        items = self._by_type.get(PatternType.BEAR_FLAG, [])
        return items[0] if items else None

    def flags(self) -> Tuple[Pattern, ...]:
        return tuple(
            self._by_type.get(PatternType.BULL_FLAG, []) +
            self._by_type.get(PatternType.BEAR_FLAG, [])
        )

    def triangles(self) -> Tuple[Pattern, ...]:
        return tuple(
            self._by_type.get(PatternType.ASCENDING_TRIANGLE, []) +
            self._by_type.get(PatternType.DESCENDING_TRIANGLE, [])
        )

    def rectangles(self) -> Tuple[Pattern, ...]:
        return tuple(self._by_type.get(PatternType.RECTANGLE, []))

    def ready(self) -> Tuple[Pattern, ...]:
        return tuple(p for p in self.patterns if p.state == PatternState.READY)

    def most_confident(self) -> Optional[Pattern]:
        if not self.patterns:
            return None
        return max(self.patterns, key=lambda p: p.confidence)

    def top(self, n: int) -> Tuple[Pattern, ...]:
        sorted_pats = sorted(self.patterns, key=lambda p: p.confidence, reverse=True)
        return tuple(sorted_pats[:n])

    def by_quality(self, label: str) -> Tuple[Pattern, ...]:
        """
        Filter by presentation quality score labels:
        - "PERFECT": >= 0.85
        - "GOOD": >= 0.65 and < 0.85
        - "MESSY": >= 0.40 and < 0.65
        - "POOR": < 0.40
        """
        lbl = label.upper()
        res = []
        for p in self.patterns:
            q = p.quality_score
            if lbl == "PERFECT" and q >= 0.85:
                res.append(p)
            elif lbl == "GOOD" and 0.65 <= q < 0.85:
                res.append(p)
            elif lbl == "MESSY" and 0.40 <= q < 0.65:
                res.append(p)
            elif lbl == "POOR" and q < 0.40:
                res.append(p)
        return tuple(res)

    def __iter__(self) -> Iterator[Pattern]:
        return iter(self.patterns)

    def __len__(self) -> int:
        return len(self.patterns)
