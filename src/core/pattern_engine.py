#!/usr/bin/env python3
"""
Pattern Engine (MKE Stage 6)
=============================
Detects, validates, scores, and manages active chart patterns.
Implements the registry pattern, confidence scorer, hard rule gates,
soft scorers, measured move target projection, and caching with lifecycle transitions.
"""

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Iterable, ClassVar, Set

from src.core.market_facts import MarketFacts
from src.core.market_geometry import CompositeLevel, Trendline, ConfluenceZone, GeometryStatus
from src.core.market_patterns import (
    Pattern, PatternType, PatternState, PatternDirection, PatternAnchor,
    PatternEvidence, PatternConfidenceComponents, PatternDelta, PatternTransitionEvent,
    PatternsContext
)
from src.core.market_knowledge import SwingPoint, ResearchEvent
from src.core.hs_geometry import NecklineBuilder, ShoulderSymmetry, HeadProminence, NecklineBreak

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternCandidate:
    """A pattern candidate emitted by a detector before validation."""
    type: PatternType
    direction: PatternDirection
    anchors: List[PatternAnchor]
    raw_breakout_level: float
    raw_invalidation: float
    raw_symmetry: float
    raw_completion_pct: float
    source_swings: List[SwingPoint]
    source_composites: List[CompositeLevel]
    source_trendlines: List[Trendline]
    metrics: Dict[str, Any]
    detector_name: str


# ─────────────────────────────────────────────────────────────────────────────
# Plugin Abstract Base Class
# ─────────────────────────────────────────────────────────────────────────────

class PatternPlugin(ABC):
    """Abstract Base Class for all chart pattern detectors."""
    pattern_type: ClassVar[PatternType]
    required_geometry: ClassVar[bool] = True
    required_structure: ClassVar[bool] = True
    min_swings: ClassVar[int] = 4

    @abstractmethod
    def detect(
        self,
        facts: MarketFacts,
        composites: List[CompositeLevel],
        trendlines: List[Trendline],
        support_conf: Optional[ConfluenceZone],
        resistance_conf: Optional[ConfluenceZone],
    ) -> Iterable[PatternCandidate]:
        """Detect potential pattern candidates on the current candle."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Hard Rule Checker
# ─────────────────────────────────────────────────────────────────────────────

class HardRuleChecker:
    """Performs binary checks to determine if a candidate represents a valid pattern."""

    def check(self, candidate: PatternCandidate, atr: float) -> Tuple[bool, str]:
        if atr <= 0:
            atr = 1.0
        
        # Rule 1: Min swings check
        if len(candidate.source_swings) < 2:
            return False, "Insufficient source swings"

        # Rule 2: Minimum anchor count
        if not candidate.anchors:
            return False, "No anchors defined"

        # Rule 3: Valid breakout and invalidation levels
        if candidate.raw_breakout_level <= 0 or candidate.raw_invalidation <= 0:
            return False, "Invalid breakout or invalidation price level"

        # Rule 4: Completion check
        if candidate.raw_completion_pct <= 0.0:
            return False, "Pattern completion is 0% or negative"

        # Rule 5: Risk/Reward check (Target vs Invalidation)
        # We calculate target projection in validator, but here we can check basic structure
        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Soft Scorer
# ─────────────────────────────────────────────────────────────────────────────

class SoftScorer:
    """Evaluates qualitative factors of a candidate to yield confidence components."""

    def score(
        self,
        candidate: PatternCandidate,
        composites: List[CompositeLevel],
        confluence_zones: Tuple[Optional[ConfluenceZone], Optional[ConfluenceZone]],
        atr: float,
        facts: MarketFacts
    ) -> PatternConfidenceComponents:
        if atr <= 0:
            atr = 1.0

        support_conf, resistance_conf = confluence_zones

        # 1. Anchor quality: average strength of the source swings
        if candidate.source_swings:
            anchor_quality = sum(s.strength for s in candidate.source_swings) / len(candidate.source_swings)
        else:
            anchor_quality = 0.5

        # 2. Geometry confluence: check if breakout level is near any composite level or confluence zone
        geom_confluence = 0.5
        all_zones = []
        if support_conf:
            all_zones.append(support_conf)
        if resistance_conf:
            all_zones.append(resistance_conf)

        if all_zones:
            min_dist = min(abs(z.price - candidate.raw_breakout_level) for z in all_zones)
            geom_confluence = max(0.0, min(1.0, 1.0 - (min_dist / (2.0 * atr))))
        elif composites:
            min_dist = min(abs(c.price - candidate.raw_breakout_level) for c in composites)
            geom_confluence = max(0.0, min(1.0, 1.0 - (min_dist / (2.0 * atr))))

        # 3. Symmetry: from detector metrics
        symmetry = max(0.0, min(1.0, candidate.raw_symmetry))

        # 4. Compression: volatility percentile (using ATR percentile)
        compression = facts.atr_percentile

        # 5. Context alignment: trend alignment (default neutral or from facts)
        context_alignment = 0.5
        # If structure indicates compression or trend, align
        if facts.last_swing_high and facts.last_swing_low:
            # simple context check
            if candidate.direction == PatternDirection.LONG and facts.current_price > facts.last_swing_low.price:
                context_alignment = 0.8
            elif candidate.direction == PatternDirection.SHORT and facts.current_price < facts.last_swing_high.price:
                context_alignment = 0.8

        return PatternConfidenceComponents(
            anchor_quality=round(anchor_quality, 3),
            geometry_confluence=round(geom_confluence, 3),
            symmetry=round(symmetry, 3),
            compression=round(compression, 3),
            context_alignment=round(context_alignment, 3)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Scorer Protocol
# ─────────────────────────────────────────────────────────────────────────────

class WeightedSumScorer:
    """Applies standard weights to compute overall confidence score."""
    WEIGHTS = {
        "anchor_quality": 0.30,
        "geometry_confluence": 0.30,
        "symmetry": 0.20,
        "compression": 0.10,
        "context_alignment": 0.10
    }

    def score(self, c: PatternConfidenceComponents) -> float:
        val = (
            self.WEIGHTS["anchor_quality"] * c.anchor_quality +
            self.WEIGHTS["geometry_confluence"] * c.geometry_confluence +
            self.WEIGHTS["symmetry"] * c.symmetry +
            self.WEIGHTS["compression"] * c.compression +
            self.WEIGHTS["context_alignment"] * c.context_alignment
        )
        return round(val, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Measured Move Engine
# ─────────────────────────────────────────────────────────────────────────────

class MeasuredMoveEngine:
    """Computes price targets based on geometry measurements."""

    @staticmethod
    def calculate(
        candidate: PatternCandidate, atr: float
    ) -> Tuple[Tuple[float, ...], Tuple[str, ...]]:
        """
        Returns projected targets and labels.
        Targets are formatted as Tuple[float, ...] and labels as Tuple[str, ...].
        """
        targets = []
        labels = []
        b_lvl = candidate.raw_breakout_level

        # Compute typical pattern height
        prices = [a.price for a in candidate.anchors]
        if len(prices) >= 2:
            height = max(prices) - min(prices)
        else:
            height = 2.0 * atr

        if height <= 0:
            height = 2.0 * atr

        if candidate.direction == PatternDirection.LONG:
            # Target 1: Measured move (100% height projection)
            t1 = b_lvl + height
            targets.append(round(t1, 2))
            labels.append("measured_move")

            # Target 2: Double size or structural confluence target
            t2 = b_lvl + 1.618 * height
            targets.append(round(t2, 2))
            labels.append("extended_target")

        elif candidate.direction == PatternDirection.SHORT:
            # Target 1: Measured move
            t1 = b_lvl - height
            targets.append(round(t1, 2))
            labels.append("measured_move")

            t2 = b_lvl - 1.618 * height
            targets.append(round(t2, 2))
            labels.append("extended_target")

        else:
            # Bilateral breakout
            t1_up = b_lvl + height
            t1_down = b_lvl - height
            targets.extend([round(t1_up, 2), round(t1_down, 2)])
            labels.extend(["measured_move_up", "measured_move_down"])

        return tuple(targets), tuple(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Validator
# ─────────────────────────────────────────────────────────────────────────────

class PatternValidator:
    """Orchestrates candidate filtering, scoring, and normalization."""

    def __init__(
        self,
        hard_checker: HardRuleChecker,
        soft_scorer: SoftScorer,
        confidence_scorer: WeightedSumScorer
    ):
        self.hard_checker = hard_checker
        self.soft_scorer = soft_scorer
        self.confidence_scorer = confidence_scorer

    def validate(
        self,
        candidate: PatternCandidate,
        composites: List[CompositeLevel],
        confluence_zones: Tuple[Optional[ConfluenceZone], Optional[ConfluenceZone]],
        atr: float,
        facts: MarketFacts
    ) -> Optional[Pattern]:
        # 1. Hard Rule Checks
        passed, reason = self.hard_checker.check(candidate, atr)
        if not passed:
            logger.debug(f"Candidate rejected by hard rules: {reason}")
            return None

        # 2. Soft Scoring
        components = self.soft_scorer.score(
            candidate=candidate,
            composites=composites,
            confluence_zones=confluence_zones,
            atr=atr,
            facts=facts
        )

        # 3. Confidence Calculation
        confidence = self.confidence_scorer.score(components)

        # 4. Target Calculation
        targets, target_labels = MeasuredMoveEngine.calculate(candidate, atr)

        # 5. Build ID
        anchor_ids = [a.id for a in candidate.anchors]
        pat_id = Pattern.make_id(anchor_ids, candidate.type.value)
        res_id = Pattern.make_research_id(pat_id, facts.symbol, facts.current_bar)

        # 6. Quality score (derived only from anchor quality and symmetry)
        quality_score = round(0.5 * components.anchor_quality + 0.5 * components.symmetry, 3)

        # 7. Convert components to raw dict for MLM
        raw_components = {
            "anchor_quality": components.anchor_quality,
            "geometry_confluence": components.geometry_confluence,
            "symmetry": components.symmetry,
            "compression": components.compression,
            "context_alignment": components.context_alignment
        }

        # Determine pattern state based on completion
        state = PatternState.FORMING
        if candidate.raw_completion_pct >= 1.0:
            state = PatternState.BREAKOUT
        elif candidate.raw_completion_pct >= 0.85:
            state = PatternState.READY
        elif candidate.raw_completion_pct >= 0.5:
            state = PatternState.ACTIVE

        # Evidences
        evidence = PatternEvidence(
            swing_ids=tuple(s.id for s in candidate.source_swings),
            trendline_ids=tuple(t.id for t in candidate.source_trendlines),
            composite_ids=tuple(c.id for c in candidate.source_composites),
            confluence_scores={
                "support": confluence_zones[0].total_score if confluence_zones[0] else 0.0,
                "resistance": confluence_zones[1].total_score if confluence_zones[1] else 0.0,
            },
            explanation=tuple(candidate.metrics.get("explanation_lines", [f"Detected {candidate.type.value}"])),
            metrics=candidate.metrics,
            detector_name=candidate.detector_name,
            detector_version="v1.0",
            detector_parameters=candidate.metrics.get("parameters", {})
        )

        return Pattern(
            id=pat_id,
            research_id=res_id,
            type=candidate.type,
            state=state,
            direction=candidate.direction,
            quality_score=quality_score,
            confidence=confidence,
            confidence_components=components,
            raw_components=raw_components,
            trigger_quality=0.0,  # Computed fresh by PatternEngine
            breakout_level=candidate.raw_breakout_level,
            original_invalidation=candidate.raw_invalidation,
            current_invalidation=candidate.raw_invalidation,
            targets=targets,
            target_labels=target_labels,
            completion_pct=candidate.raw_completion_pct,
            age_bars=0,
            retest_count=0,
            last_seen_bar=facts.current_bar,
            anchors=tuple(candidate.anchors),
            evidence=evidence,
            explanation=f"Detected {candidate.type.value} pattern with confidence {confidence}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class PatternRegistry:
    """Manages active detector plugins."""

    def __init__(self):
        self._plugins: Dict[PatternType, PatternPlugin] = {}
        self._enabled: Set[PatternType] = set()

    def register(self, plugin: PatternPlugin):
        self._plugins[plugin.pattern_type] = plugin
        self._enabled.add(plugin.pattern_type)

    def enable(self, pattern_type: PatternType):
        if pattern_type in self._plugins:
            self._enabled.add(pattern_type)

    def disable(self, pattern_type: PatternType):
        self._enabled.discard(pattern_type)

    def enable_only(self, types: List[PatternType]):
        self._enabled = {t for t in types if t in self._plugins}

    def detect_all(
        self,
        facts: MarketFacts,
        composites: List[CompositeLevel],
        trendlines: List[Trendline],
        support_conf: Optional[ConfluenceZone],
        resistance_conf: Optional[ConfluenceZone]
    ) -> List[PatternCandidate]:
        candidates = []
        for t in self._enabled:
            plugin = self._plugins[t]
            try:
                candidates.extend(plugin.detect(facts, composites, trendlines, support_conf, resistance_conf))
            except Exception as e:
                logger.error(f"Detector {plugin.__class__.__name__} failed: {e}", exc_info=True)
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 9 Pattern Detector Plugins
# ─────────────────────────────────────────────────────────────────────────────

class DoubleTopDetector(PatternPlugin):
    pattern_type = PatternType.DOUBLE_TOP
    min_swings = 4

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        # Need at least 4 swings: High1, Low1, High2, Low2
        # Let's filter swings: High, Low, High
        swings = facts.swings
        n = len(swings)
        if n < 3:
            return
        
        # Scan recent swings for alternating structure: HIGH -> LOW -> HIGH
        for i in range(n - 2):
            s1 = swings[i]
            s2 = swings[i+1]
            s3 = swings[i+2]
            
            if s1.type == "HIGH" and s2.type == "LOW" and s3.type == "HIGH":
                # Check price similarity of highs
                diff = abs(s1.price - s3.price)
                max_delta = 0.35 * facts.atr
                if diff <= max_delta:
                    # Neckline is the intermediate low
                    neckline = s2.price
                    
                    # Proximity checks
                    high_price = max(s1.price, s3.price)
                    if high_price == neckline:
                        continue
                    
                    curr = facts.current_price
                    completion = 1.0 - max(0.0, (curr - neckline) / (high_price - neckline))
                    completion = max(0.0, min(1.0, completion))
                    
                    # If price is already way below neckline, it is a breakout
                    if curr < neckline:
                        completion = 1.0

                    anchors = [
                        PatternAnchor(id=f"{s1.id}_a1", role="LEFT_HIGH", price=s1.price, bar_index=s1.provenance.get("bar_index", 0), timestamp=s1.timestamp, source_swing_id=s1.id),
                        PatternAnchor(id=f"{s2.id}_a2", role="NECKLINE", price=s2.price, bar_index=s2.provenance.get("bar_index", 0), timestamp=s2.timestamp, source_swing_id=s2.id),
                        PatternAnchor(id=f"{s3.id}_a3", role="RIGHT_HIGH", price=s3.price, bar_index=s3.provenance.get("bar_index", 0), timestamp=s3.timestamp, source_swing_id=s3.id)
                    ]
                    
                    symmetry = 1.0 - (diff / max_delta) if max_delta > 0 else 1.0
                    
                    yield PatternCandidate(
                        type=PatternType.DOUBLE_TOP,
                        direction=PatternDirection.SHORT,
                        anchors=anchors,
                        raw_breakout_level=neckline,
                        raw_invalidation=high_price + 0.5 * facts.atr,
                        raw_symmetry=round(symmetry, 3),
                        raw_completion_pct=round(completion, 3),
                        source_swings=[s1, s2, s3],
                        source_composites=[],
                        source_trendlines=[],
                        metrics={
                            "neckline_price": neckline,
                            "equal_highs_delta_atr": round(diff / facts.atr, 3) if facts.atr > 0 else 0.0,
                            "left_high_bar": s1.provenance.get("bar_index", 0),
                            "right_high_bar": s3.provenance.get("bar_index", 0),
                            "explanation_lines": [
                                f"Double Top formed by peaks at {s1.price} and {s3.price}",
                                f"Neckline at {neckline}"
                            ]
                        },
                        detector_name="DoubleTopDetector"
                    )


class DoubleBottomDetector(PatternPlugin):
    pattern_type = PatternType.DOUBLE_BOTTOM
    min_swings = 4

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        swings = facts.swings
        n = len(swings)
        if n < 3:
            return
        
        for i in range(n - 2):
            s1 = swings[i]
            s2 = swings[i+1]
            s3 = swings[i+2]
            
            if s1.type == "LOW" and s2.type == "HIGH" and s3.type == "LOW":
                diff = abs(s1.price - s3.price)
                max_delta = 0.35 * facts.atr
                if diff <= max_delta:
                    neckline = s2.price
                    low_price = min(s1.price, s3.price)
                    if neckline == low_price:
                        continue
                    
                    curr = facts.current_price
                    completion = 1.0 - max(0.0, (neckline - curr) / (neckline - low_price))
                    completion = max(0.0, min(1.0, completion))
                    
                    if curr > neckline:
                        completion = 1.0

                    anchors = [
                        PatternAnchor(id=f"{s1.id}_a1", role="LEFT_LOW", price=s1.price, bar_index=s1.provenance.get("bar_index", 0), timestamp=s1.timestamp, source_swing_id=s1.id),
                        PatternAnchor(id=f"{s2.id}_a2", role="NECKLINE", price=s2.price, bar_index=s2.provenance.get("bar_index", 0), timestamp=s2.timestamp, source_swing_id=s2.id),
                        PatternAnchor(id=f"{s3.id}_a3", role="RIGHT_LOW", price=s3.price, bar_index=s3.provenance.get("bar_index", 0), timestamp=s3.timestamp, source_swing_id=s3.id)
                    ]
                    
                    symmetry = 1.0 - (diff / max_delta) if max_delta > 0 else 1.0
                    
                    yield PatternCandidate(
                        type=PatternType.DOUBLE_BOTTOM,
                        direction=PatternDirection.LONG,
                        anchors=anchors,
                        raw_breakout_level=neckline,
                        raw_invalidation=low_price - 0.5 * facts.atr,
                        raw_symmetry=round(symmetry, 3),
                        raw_completion_pct=round(completion, 3),
                        source_swings=[s1, s2, s3],
                        source_composites=[],
                        source_trendlines=[],
                        metrics={
                            "neckline_price": neckline,
                            "equal_lows_delta_atr": round(diff / facts.atr, 3) if facts.atr > 0 else 0.0,
                            "left_low_bar": s1.provenance.get("bar_index", 0),
                            "right_low_bar": s3.provenance.get("bar_index", 0),
                            "explanation_lines": [
                                f"Double Bottom formed by troughs at {s1.price} and {s3.price}",
                                f"Neckline at {neckline}"
                            ]
                        },
                        detector_name="DoubleBottomDetector"
                    )


class AscendingTriangleDetector(PatternPlugin):
    pattern_type = PatternType.ASCENDING_TRIANGLE
    min_swings = 4

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        # Needs flat resistance (equal highs) + ascending trendline support (higher lows)
        swings = facts.swings
        n = len(swings)
        if n < 4:
            return

        for i in range(n - 3):
            s1 = swings[i]   # HIGH
            s2 = swings[i+1] # LOW
            s3 = swings[i+2] # HIGH
            s4 = swings[i+3] # LOW

            if s1.type == "HIGH" and s2.type == "LOW" and s3.type == "HIGH" and s4.type == "LOW":
                # Flat resistance check: High1 ~ High3
                diff = abs(s1.price - s3.price)
                if diff <= 0.35 * facts.atr:
                    # Ascending support check: Low2 < Low4
                    if s4.price > s2.price:
                        resistance = max(s1.price, s3.price)
                        
                        # Find descending line
                        curr = facts.current_price
                        completion = max(0.0, min(1.0, (curr - s4.price) / (resistance - s4.price)))
                        if curr > resistance:
                            completion = 1.0

                        anchors = [
                            PatternAnchor(id=f"{s1.id}_a1", role="RESISTANCE_1", price=s1.price, bar_index=s1.provenance.get("bar_index", 0), timestamp=s1.timestamp, source_swing_id=s1.id),
                            PatternAnchor(id=f"{s2.id}_a2", role="SUPPORT_1", price=s2.price, bar_index=s2.provenance.get("bar_index", 0), timestamp=s2.timestamp, source_swing_id=s2.id),
                            PatternAnchor(id=f"{s3.id}_a3", role="RESISTANCE_2", price=s3.price, bar_index=s3.provenance.get("bar_index", 0), timestamp=s3.timestamp, source_swing_id=s3.id),
                            PatternAnchor(id=f"{s4.id}_a4", role="SUPPORT_2", price=s4.price, bar_index=s4.provenance.get("bar_index", 0), timestamp=s4.timestamp, source_swing_id=s4.id)
                        ]

                        yield PatternCandidate(
                            type=PatternType.ASCENDING_TRIANGLE,
                            direction=PatternDirection.LONG,
                            anchors=anchors,
                            raw_breakout_level=resistance,
                            raw_invalidation=s4.price - 0.2 * facts.atr,
                            raw_symmetry=0.8,
                            raw_completion_pct=round(completion, 3),
                            source_swings=[s1, s2, s3, s4],
                            source_composites=[],
                            source_trendlines=[],
                            metrics={
                                "resistance_price": resistance,
                                "lows_slope": (s4.price - s2.price) / max(1, s4.provenance.get("bar_index", 0) - s2.provenance.get("bar_index", 0)),
                                "explanation_lines": [
                                    f"Ascending Triangle: Flat resistance at {resistance}",
                                    f"Higher lows from {s2.price} to {s4.price}"
                                ]
                            },
                            detector_name="AscendingTriangleDetector"
                        )


class DescendingTriangleDetector(PatternPlugin):
    pattern_type = PatternType.DESCENDING_TRIANGLE
    min_swings = 4

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        swings = facts.swings
        n = len(swings)
        if n < 4:
            return

        for i in range(n - 3):
            s1 = swings[i]   # LOW
            s2 = swings[i+1] # HIGH
            s3 = swings[i+2] # LOW
            s4 = swings[i+3] # HIGH

            if s1.type == "LOW" and s2.type == "HIGH" and s3.type == "LOW" and s4.type == "HIGH":
                diff = abs(s1.price - s3.price)
                if diff <= 0.35 * facts.atr:
                    if s4.price < s2.price:
                        support = min(s1.price, s3.price)
                        curr = facts.current_price
                        completion = max(0.0, min(1.0, (s4.price - curr) / (s4.price - support)))
                        if curr < support:
                            completion = 1.0

                        anchors = [
                            PatternAnchor(id=f"{s1.id}_a1", role="SUPPORT_1", price=s1.price, bar_index=s1.provenance.get("bar_index", 0), timestamp=s1.timestamp, source_swing_id=s1.id),
                            PatternAnchor(id=f"{s2.id}_a2", role="RESISTANCE_1", price=s2.price, bar_index=s2.provenance.get("bar_index", 0), timestamp=s2.timestamp, source_swing_id=s2.id),
                            PatternAnchor(id=f"{s3.id}_a3", role="SUPPORT_2", price=s3.price, bar_index=s3.provenance.get("bar_index", 0), timestamp=s3.timestamp, source_swing_id=s3.id),
                            PatternAnchor(id=f"{s4.id}_a4", role="RESISTANCE_2", price=s4.price, bar_index=s4.provenance.get("bar_index", 0), timestamp=s4.timestamp, source_swing_id=s4.id)
                        ]

                        yield PatternCandidate(
                            type=PatternType.DESCENDING_TRIANGLE,
                            direction=PatternDirection.SHORT,
                            anchors=anchors,
                            raw_breakout_level=support,
                            raw_invalidation=s4.price + 0.2 * facts.atr,
                            raw_symmetry=0.8,
                            raw_completion_pct=round(completion, 3),
                            source_swings=[s1, s2, s3, s4],
                            source_composites=[],
                            source_trendlines=[],
                            metrics={
                                "support_price": support,
                                "highs_slope": (s4.price - s2.price) / max(1, s4.provenance.get("bar_index", 0) - s2.provenance.get("bar_index", 0)),
                                "explanation_lines": [
                                    f"Descending Triangle: Flat support at {support}",
                                    f"Lower highs from {s2.price} to {s4.price}"
                                ]
                            },
                            detector_name="DescendingTriangleDetector"
                        )


class BullFlagDetector(PatternPlugin):
    pattern_type = PatternType.BULL_FLAG
    min_swings = 2

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        # Pole = Completed UP leg (with length >= 2.0 * atr)
        # Flag = Developing DOWN leg (retracement between 10% and 50% of the pole range)
        legs = facts.completed_legs
        dev_leg = facts.developing_leg
        
        if not legs or not dev_leg:
            return
        
        # Check last completed leg
        last_leg = legs[-1]
        if last_leg.type == "UP_LEG" and last_leg.price_range >= 2.0 * facts.atr:
            # We have a candidate pole!
            # Now inspect the developing leg which should be a DOWN_LEG representing the pullback (flag)
            # Retracement check: pull back should be between 10% and 50% of the pole size
            retracement = last_leg.end_pivot.price - facts.current_price
            retracement_pct = retracement / last_leg.price_range if last_leg.price_range > 0 else 0.0
            
            if 0.10 <= retracement_pct <= 0.60:
                # Valid flag pull back
                curr = facts.current_price
                # Breakout level is the flag high (end of the pole)
                b_level = last_leg.end_pivot.price
                
                # Completion pct: increases as retracement completes/turns back
                completion = 1.0 - (b_level - curr) / (last_leg.price_range * 0.5)
                completion = max(0.1, min(1.0, completion))

                anchors = [
                    PatternAnchor(id=f"{last_leg.id}_pole_start", role="POLE_START", price=last_leg.start_anchor.price, bar_index=last_leg.start_anchor.provenance.get("bar_index", 0) if hasattr(last_leg.start_anchor, 'provenance') else 0, timestamp=last_leg.start_anchor.timestamp),
                    PatternAnchor(id=f"{last_leg.end_pivot.id}_pole_end", role="POLE_END", price=last_leg.end_pivot.price, bar_index=last_leg.end_pivot.provenance.get("bar_index", 0), timestamp=last_leg.end_pivot.timestamp, source_swing_id=last_leg.end_pivot.id)
                ]

                yield PatternCandidate(
                    type=PatternType.BULL_FLAG,
                    direction=PatternDirection.LONG,
                    anchors=anchors,
                    raw_breakout_level=b_level,
                    raw_invalidation=last_leg.start_anchor.price,
                    raw_symmetry=0.9,
                    raw_completion_pct=round(completion, 3),
                    source_swings=[last_leg.end_pivot],
                    source_composites=[],
                    source_trendlines=[],
                    metrics={
                        "pole_length_atr": round(last_leg.price_range / facts.atr, 3) if facts.atr > 0 else 0.0,
                        "retracement_pct": round(retracement_pct, 3),
                        "explanation_lines": [
                            f"Bull Flag: Pole UP of {last_leg.price_range} points",
                            f"Pullback retraced {round(retracement_pct*100, 1)}%"
                        ]
                    },
                    detector_name="BullFlagDetector"
                )


class BearFlagDetector(PatternPlugin):
    pattern_type = PatternType.BEAR_FLAG
    min_swings = 2

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        legs = facts.completed_legs
        dev_leg = facts.developing_leg
        
        if not legs or not dev_leg:
            return
        
        last_leg = legs[-1]
        if last_leg.type == "DOWN_LEG" and last_leg.price_range >= 2.0 * facts.atr:
            retracement = facts.current_price - last_leg.end_pivot.price
            retracement_pct = retracement / last_leg.price_range if last_leg.price_range > 0 else 0.0
            
            if 0.10 <= retracement_pct <= 0.60:
                curr = facts.current_price
                b_level = last_leg.end_pivot.price
                
                completion = 1.0 - (curr - b_level) / (last_leg.price_range * 0.5)
                completion = max(0.1, min(1.0, completion))

                anchors = [
                    PatternAnchor(id=f"{last_leg.id}_pole_start", role="POLE_START", price=last_leg.start_anchor.price, bar_index=last_leg.start_anchor.provenance.get("bar_index", 0) if hasattr(last_leg.start_anchor, 'provenance') else 0, timestamp=last_leg.start_anchor.timestamp),
                    PatternAnchor(id=f"{last_leg.end_pivot.id}_pole_end", role="POLE_END", price=last_leg.end_pivot.price, bar_index=last_leg.end_pivot.provenance.get("bar_index", 0), timestamp=last_leg.end_pivot.timestamp, source_swing_id=last_leg.end_pivot.id)
                ]

                yield PatternCandidate(
                    type=PatternType.BEAR_FLAG,
                    direction=PatternDirection.SHORT,
                    anchors=anchors,
                    raw_breakout_level=b_level,
                    raw_invalidation=last_leg.start_anchor.price,
                    raw_symmetry=0.9,
                    raw_completion_pct=round(completion, 3),
                    source_swings=[last_leg.end_pivot],
                    source_composites=[],
                    source_trendlines=[],
                    metrics={
                        "pole_length_atr": round(last_leg.price_range / facts.atr, 3) if facts.atr > 0 else 0.0,
                        "retracement_pct": round(retracement_pct, 3),
                        "explanation_lines": [
                            f"Bear Flag: Pole DOWN of {last_leg.price_range} points",
                            f"Pullback retraced {round(retracement_pct*100, 1)}%"
                        ]
                    },
                    detector_name="BearFlagDetector"
                )


class RectangleDetector(PatternPlugin):
    pattern_type = PatternType.RECTANGLE
    min_swings = 4

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        # Needs 4 alternating swings forming equal highs and equal lows (flat channel)
        swings = facts.swings
        n = len(swings)
        if n < 4:
            return

        for i in range(n - 3):
            s1 = swings[i]
            s2 = swings[i+1]
            s3 = swings[i+2]
            s4 = swings[i+3]

            # Alternating highs and lows
            if (s1.type == "HIGH" and s2.type == "LOW" and s3.type == "HIGH" and s4.type == "LOW") or \
               (s1.type == "LOW" and s2.type == "HIGH" and s3.type == "LOW" and s4.type == "HIGH"):
                
                highs = [s for s in [s1, s2, s3, s4] if s.type == "HIGH"]
                lows = [s for s in [s1, s2, s3, s4] if s.type == "LOW"]
                
                if len(highs) == 2 and len(lows) == 2:
                    h_diff = abs(highs[0].price - highs[1].price)
                    l_diff = abs(lows[0].price - lows[1].price)
                    
                    if h_diff <= 0.35 * facts.atr and l_diff <= 0.35 * facts.atr:
                        # Rectangle detected!
                        h_lvl = max(highs[0].price, highs[1].price)
                        l_lvl = min(lows[0].price, lows[1].price)
                        
                        curr = facts.current_price
                        # We project both breakout levels (bilateral)
                        completion = 0.8  # Default rectangle completion
                        if curr > h_lvl or curr < l_lvl:
                            completion = 1.0

                        anchors = [
                            PatternAnchor(id=f"{s1.id}_a1", role="RECT_1", price=s1.price, bar_index=s1.provenance.get("bar_index", 0), timestamp=s1.timestamp, source_swing_id=s1.id),
                            PatternAnchor(id=f"{s2.id}_a2", role="RECT_2", price=s2.price, bar_index=s2.provenance.get("bar_index", 0), timestamp=s2.timestamp, source_swing_id=s2.id),
                            PatternAnchor(id=f"{s3.id}_a3", role="RECT_3", price=s3.price, bar_index=s3.provenance.get("bar_index", 0), timestamp=s3.timestamp, source_swing_id=s3.id),
                            PatternAnchor(id=f"{s4.id}_a4", role="RECT_4", price=s4.price, bar_index=s4.provenance.get("bar_index", 0), timestamp=s4.timestamp, source_swing_id=s4.id)
                        ]

                        yield PatternCandidate(
                            type=PatternType.RECTANGLE,
                            direction=PatternDirection.BILATERAL,
                            anchors=anchors,
                            raw_breakout_level=h_lvl,  # Bilateral uses high boundary as main breakout level
                            raw_invalidation=l_lvl,
                            raw_symmetry=0.9,
                            raw_completion_pct=round(completion, 3),
                            source_swings=[s1, s2, s3, s4],
                            source_composites=[],
                            source_trendlines=[],
                            metrics={
                                "range_high": h_lvl,
                                "range_low": l_lvl,
                                "range_height": h_lvl - l_lvl,
                                "explanation_lines": [
                                    f"Rectangle/Channel formed between {l_lvl} and {h_lvl}"
                                ]
                            },
                            detector_name="RectangleDetector"
                        )


class HeadAndShouldersDetector(PatternPlugin):
    pattern_type = PatternType.HEAD_AND_SHOULDERS
    min_swings = 5

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        swings = facts.swings
        n = len(swings)
        if n < 5:
            return

        for i in range(n - 4):
            # Alternating pivots: LS(HIGH), T1(LOW), HEAD(HIGH), T2(LOW), RS(HIGH)
            s1 = swings[i]   # LS
            s2 = swings[i+1] # T1
            s3 = swings[i+2] # Head
            s4 = swings[i+3] # T2
            s5 = swings[i+4] # RS

            if s1.type == "HIGH" and s2.type == "LOW" and s3.type == "HIGH" and s4.type == "LOW" and s5.type == "HIGH":
                # Structural check: Head is higher than both shoulders
                if s3.price > s1.price and s3.price > s5.price:
                    # Symmetry check LS vs RS within 1.5%
                    sym_passed, sym_score = ShoulderSymmetry.check(s1.price, s5.price, s3.price, max_ratio=0.015)
                    if sym_passed:
                        # Head Prominence Check
                        prom_passed, prom_score = HeadProminence.check(s1.price, s5.price, s3.price, facts.atr, is_inverse=False)
                        if prom_passed:
                            # Neckline Build
                            m, c = NecklineBuilder.build_neckline(
                                s2.provenance.get("bar_index", 0), s2.price,
                                s4.provenance.get("bar_index", 0), s4.price
                            )
                            # Project neckline at RS bar index to get baseline neckline price
                            rs_bar = s5.provenance.get("bar_index", 0)
                            neckline_proj = NecklineBuilder.project(rs_bar, m, c)
                            
                            curr = facts.current_price
                            completion = 1.0 - max(0.0, (curr - neckline_proj) / (s3.price - neckline_proj)) if s3.price != neckline_proj else 0.5
                            completion = max(0.1, min(1.0, completion))
                            if curr < neckline_proj:
                                completion = 1.0

                            anchors = [
                                PatternAnchor(id=f"{s1.id}_ls", role="LEFT_SHOULDER", price=s1.price, bar_index=s1.provenance.get("bar_index", 0), timestamp=s1.timestamp, source_swing_id=s1.id),
                                PatternAnchor(id=f"{s2.id}_t1", role="NECKLINE_1", price=s2.price, bar_index=s2.provenance.get("bar_index", 0), timestamp=s2.timestamp, source_swing_id=s2.id),
                                PatternAnchor(id=f"{s3.id}_head", role="HEAD", price=s3.price, bar_index=s3.provenance.get("bar_index", 0), timestamp=s3.timestamp, source_swing_id=s3.id),
                                PatternAnchor(id=f"{s4.id}_t2", role="NECKLINE_2", price=s4.price, bar_index=s4.provenance.get("bar_index", 0), timestamp=s4.timestamp, source_swing_id=s4.id),
                                PatternAnchor(id=f"{s5.id}_rs", role="RIGHT_SHOULDER", price=s5.price, bar_index=rs_bar, timestamp=s5.timestamp, source_swing_id=s5.id)
                            ]

                            yield PatternCandidate(
                                type=PatternType.HEAD_AND_SHOULDERS,
                                direction=PatternDirection.SHORT,
                                anchors=anchors,
                                raw_breakout_level=neckline_proj,
                                raw_invalidation=s3.price,
                                raw_symmetry=sym_score,
                                raw_completion_pct=round(completion, 3),
                                source_swings=[s1, s2, s3, s4, s5],
                                source_composites=[],
                                source_trendlines=[],
                                metrics={
                                    "shoulder_symmetry": sym_score,
                                    "neckline_slope": round(m, 5),
                                    "head_height_atr": round((s3.price - neckline_proj) / facts.atr, 3) if facts.atr > 0 else 0.0,
                                    "ls_bar": s1.provenance.get("bar_index", 0),
                                    "rs_bar": rs_bar,
                                    "explanation_lines": [
                                        f"Head & Shoulders pattern detected: LS={s1.price}, Head={s3.price}, RS={s5.price}",
                                        f"Neckline projected at RS={round(neckline_proj, 2)}"
                                    ]
                                },
                                detector_name="HeadAndShouldersDetector"
                            )


class InverseHeadAndShouldersDetector(PatternPlugin):
    pattern_type = PatternType.INVERSE_HEAD_AND_SHOULDERS
    min_swings = 5

    def detect(self, facts: MarketFacts, composites, trendlines, support_conf, resistance_conf) -> Iterable[PatternCandidate]:
        swings = facts.swings
        n = len(swings)
        if n < 5:
            return

        for i in range(n - 4):
            # Alternating pivots: LS(LOW), P1(HIGH), HEAD(LOW), P2(HIGH), RS(LOW)
            s1 = swings[i]   # LS
            s2 = swings[i+1] # P1
            s3 = swings[i+2] # Head
            s4 = swings[i+3] # P2
            s5 = swings[i+4] # RS

            if s1.type == "LOW" and s2.type == "HIGH" and s3.type == "LOW" and s4.type == "HIGH" and s5.type == "LOW":
                if s3.price < s1.price and s3.price < s5.price:
                    # Symmetry
                    sym_passed, sym_score = ShoulderSymmetry.check(s1.price, s5.price, s3.price, max_ratio=0.015)
                    if sym_passed:
                        # Head Prominence
                        prom_passed, prom_score = HeadProminence.check(s1.price, s5.price, s3.price, facts.atr, is_inverse=True)
                        if prom_passed:
                            # Neckline Build
                            m, c = NecklineBuilder.build_neckline(
                                s2.provenance.get("bar_index", 0), s2.price,
                                s4.provenance.get("bar_index", 0), s4.price
                            )
                            rs_bar = s5.provenance.get("bar_index", 0)
                            neckline_proj = NecklineBuilder.project(rs_bar, m, c)
                            
                            curr = facts.current_price
                            completion = 1.0 - max(0.0, (neckline_proj - curr) / (neckline_proj - s3.price)) if s3.price != neckline_proj else 0.5
                            completion = max(0.1, min(1.0, completion))
                            if curr > neckline_proj:
                                completion = 1.0

                            anchors = [
                                PatternAnchor(id=f"{s1.id}_ls", role="LEFT_SHOULDER", price=s1.price, bar_index=s1.provenance.get("bar_index", 0), timestamp=s1.timestamp, source_swing_id=s1.id),
                                PatternAnchor(id=f"{s2.id}_p1", role="NECKLINE_1", price=s2.price, bar_index=s2.provenance.get("bar_index", 0), timestamp=s2.timestamp, source_swing_id=s2.id),
                                PatternAnchor(id=f"{s3.id}_head", role="HEAD", price=s3.price, bar_index=s3.provenance.get("bar_index", 0), timestamp=s3.timestamp, source_swing_id=s3.id),
                                PatternAnchor(id=f"{s4.id}_p2", role="NECKLINE_2", price=s4.price, bar_index=s4.provenance.get("bar_index", 0), timestamp=s4.timestamp, source_swing_id=s4.id),
                                PatternAnchor(id=f"{s5.id}_rs", role="RIGHT_SHOULDER", price=s5.price, bar_index=rs_bar, timestamp=s5.timestamp, source_swing_id=s5.id)
                            ]

                            yield PatternCandidate(
                                type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                                direction=PatternDirection.LONG,
                                anchors=anchors,
                                raw_breakout_level=neckline_proj,
                                raw_invalidation=s3.price,
                                raw_symmetry=sym_score,
                                raw_completion_pct=round(completion, 3),
                                source_swings=[s1, s2, s3, s4, s5],
                                source_composites=[],
                                source_trendlines=[],
                                metrics={
                                    "shoulder_symmetry": sym_score,
                                    "neckline_slope": round(m, 5),
                                    "head_height_atr": round((neckline_proj - s3.price) / facts.atr, 3) if facts.atr > 0 else 0.0,
                                    "ls_bar": s1.provenance.get("bar_index", 0),
                                    "rs_bar": rs_bar,
                                    "explanation_lines": [
                                        f"Inverse Head & Shoulders pattern detected: LS={s1.price}, Head={s3.price}, RS={s5.price}",
                                        f"Neckline projected at RS={round(neckline_proj, 2)}"
                                    ]
                                },
                                detector_name="InverseHeadAndShouldersDetector"
                            )


# ─────────────────────────────────────────────────────────────────────────────
# Detector Stats Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectorStats:
    candidate_count: int = 0
    hard_rule_rejects: int = 0
    soft_score_rejects: int = 0
    duplicate_rejects: int = 0
    cache_merges: int = 0
    expired: int = 0
    archived: int = 0
    patterns_created: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Engine orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PatternEngine:
    """Orchestrates pattern detection, conflict resolution, caching, and lifecycle management."""
    CACHE_EVICTION_BARS = 5
    MAX_AGE_BARS = {
        PatternState.FORMING: 20,
        PatternState.ACTIVE: 30,
        PatternState.READY: 40,
        PatternState.BREAKOUT: 10,
        PatternState.CONFIRMED: 40,
    }

    def __init__(self, scorer=None):
        self.registry = PatternRegistry()
        self.hard_checker = HardRuleChecker()
        self.soft_scorer = SoftScorer()
        self.confidence_scorer = scorer or WeightedSumScorer()
        self.validator = PatternValidator(
            hard_checker=self.hard_checker,
            soft_scorer=self.soft_scorer,
            confidence_scorer=self.confidence_scorer
        )

        # symbol -> {pat_id -> Pattern}
        self._cache: Dict[str, Dict[str, Pattern]] = {}
        # pat_id -> research_id (persists across ticks)
        self._research_ids: Dict[str, str] = {}
        # symbol -> {pattern_type_name -> DetectorStats}
        self._stats: Dict[str, Dict[str, DetectorStats]] = {}

        # Register standard 9 detectors
        self.registry.register(DoubleTopDetector())
        self.registry.register(DoubleBottomDetector())
        self.registry.register(AscendingTriangleDetector())
        self.registry.register(DescendingTriangleDetector())
        self.registry.register(BullFlagDetector())
        self.registry.register(BearFlagDetector())
        self.registry.register(RectangleDetector())
        self.registry.register(HeadAndShouldersDetector())
        self.registry.register(InverseHeadAndShouldersDetector())

    def get_stats(self, symbol: str) -> Dict[str, DetectorStats]:
        return self._stats.setdefault(symbol, {
            t.value: DetectorStats() for t in PatternType
        })

    def detect(
        self,
        facts: MarketFacts,
        composites: List[CompositeLevel],
        trendlines: List[Trendline],
        support_confluence: Optional[ConfluenceZone],
        resistance_confluence: Optional[ConfluenceZone]
    ) -> PatternsContext:
        symbol = facts.symbol
        current_bar = facts.current_bar
        current_price = facts.current_price
        atr = facts.atr

        # Initialize caches if needed
        symbol_cache = self._cache.setdefault(symbol, {})
        stats = self.get_stats(symbol)

        # 1. Run all registered detectors
        candidates = self.registry.detect_all(
            facts=facts,
            composites=composites,
            trendlines=trendlines,
            support_conf=support_confluence,
            resistance_conf=resistance_confluence
        )

        # Update stats candidate counts
        for cand in candidates:
            stats[cand.type.value].candidate_count += 1

        # 2. Validate candidates
        detected_patterns: List[Pattern] = []
        for cand in candidates:
            # Check hard rules
            passed_hard, reason = self.hard_checker.check(cand, atr)
            if not passed_hard:
                stats[cand.type.value].hard_rule_rejects += 1
                continue

            pat = self.validator.validate(
                candidate=cand,
                composites=composites,
                confluence_zones=(support_confluence, resistance_confluence),
                atr=atr,
                facts=facts
            )
            if pat is None:
                stats[cand.type.value].soft_score_rejects += 1
                continue

            detected_patterns.append(pat)

        # 3. Conflict resolution
        resolved = self._resolve_conflicts(detected_patterns, stats)

        # 4. Age increase on existing cached items + reconciliation
        active_ids = {p.id for p in resolved}
        transition_events: List[PatternTransitionEvent] = []
        updated_cache: Dict[str, Pattern] = {}

        # First, process newly detected/resolved patterns
        for new_pat in resolved:
            old_pat = symbol_cache.get(new_pat.id)
            if old_pat:
                # Cache merge / update
                stats[new_pat.type.value].cache_merges += 1
                
                # Retain original invalidation, research_id, and starting stats
                res_id = old_pat.research_id
                orig_inval = old_pat.original_invalidation
                age = old_pat.age_bars + 1
                
                # Check for state transition rules (e.g. BREAKOUT -> CONFIRMED after 2 bars)
                state = new_pat.state
                retest_cnt = old_pat.retest_count
                
                # State transition logic based on current price & validation levels
                if old_pat.state == PatternState.BREAKOUT:
                    # If it survived 2 bars past breakout_level, mark CONFIRMED
                    if old_pat.age_bars >= 2:
                        state = PatternState.CONFIRMED
                
                # Check failure/invalidation
                if new_pat.direction == PatternDirection.LONG:
                    if current_price < orig_inval:
                        state = PatternState.FAILED
                elif new_pat.direction == PatternDirection.SHORT:
                    if current_price > orig_inval:
                        state = PatternState.FAILED

                # Recompute trigger quality dynamically
                trigger_quality = self._compute_trigger_quality(new_pat, current_price, atr)

                # Tighten invalidation level as pattern matures
                curr_inval = old_pat.current_invalidation
                if state == PatternState.READY:
                    # tighten closer to swing lows
                    if new_pat.direction == PatternDirection.LONG:
                        curr_inval = max(orig_inval, new_pat.current_invalidation)
                    else:
                        curr_inval = min(orig_inval, new_pat.current_invalidation)

                # Re-construct updated pattern
                updated_pat = Pattern(
                    id=new_pat.id,
                    research_id=res_id,
                    parent_pattern_id=new_pat.parent_pattern_id,
                    child_pattern_ids=new_pat.child_pattern_ids,
                    type=new_pat.type,
                    state=state,
                    direction=new_pat.direction,
                    quality_score=new_pat.quality_score,
                    confidence=new_pat.confidence,
                    confidence_components=new_pat.confidence_components,
                    raw_components=new_pat.raw_components,
                    trigger_quality=trigger_quality,
                    breakout_level=new_pat.breakout_level,
                    original_invalidation=orig_inval,
                    current_invalidation=curr_inval,
                    targets=new_pat.targets,
                    target_labels=new_pat.target_labels,
                    completion_pct=new_pat.completion_pct,
                    age_bars=age,
                    retest_count=retest_cnt,
                    last_seen_bar=current_bar,
                    anchors=new_pat.anchors,
                    evidence=new_pat.evidence,
                    explanation=new_pat.explanation
                )

                # Compute delta and emit transition event if changed
                delta = self._create_delta(old_pat, updated_pat, current_price)
                if old_pat.state != updated_pat.state or abs(delta.confidence_delta) > 0.05 or abs(delta.trigger_quality_delta) > 0.05:
                    transition_events.append(
                        PatternTransitionEvent(
                            pattern_id=updated_pat.id,
                            research_id=res_id,
                            symbol=symbol,
                            from_state=old_pat.state,
                            to_state=updated_pat.state,
                            pattern=updated_pat,
                            bar=current_bar,
                            delta=delta
                        )
                    )
                updated_cache[updated_pat.id] = updated_pat
            else:
                # Brand new pattern!
                stats[new_pat.type.value].patterns_created += 1
                res_id = self._research_ids.setdefault(new_pat.id, new_pat.research_id)
                
                trigger_quality = self._compute_trigger_quality(new_pat, current_price, atr)
                
                updated_pat = Pattern(
                    id=new_pat.id,
                    research_id=res_id,
                    parent_pattern_id=new_pat.parent_pattern_id,
                    child_pattern_ids=new_pat.child_pattern_ids,
                    type=new_pat.type,
                    state=new_pat.state,
                    direction=new_pat.direction,
                    quality_score=new_pat.quality_score,
                    confidence=new_pat.confidence,
                    confidence_components=new_pat.confidence_components,
                    raw_components=new_pat.raw_components,
                    trigger_quality=trigger_quality,
                    breakout_level=new_pat.breakout_level,
                    original_invalidation=new_pat.original_invalidation,
                    current_invalidation=new_pat.current_invalidation,
                    targets=new_pat.targets,
                    target_labels=new_pat.target_labels,
                    completion_pct=new_pat.completion_pct,
                    age_bars=0,
                    retest_count=0,
                    last_seen_bar=current_bar,
                    anchors=new_pat.anchors,
                    evidence=new_pat.evidence,
                    explanation=new_pat.explanation
                )

                transition_events.append(
                    PatternTransitionEvent(
                        pattern_id=updated_pat.id,
                        research_id=res_id,
                        symbol=symbol,
                        from_state=None,
                        to_state=updated_pat.state,
                        pattern=updated_pat,
                        bar=current_bar,
                        delta=None
                    )
                )
                updated_cache[updated_pat.id] = updated_pat

        # Second, keep patterns from the cache that were NOT detected this tick (e.g. they are active and age is incrementing)
        for cached_id, cached_pat in symbol_cache.items():
            if cached_id not in active_ids:
                # If it's already FAILED or ARCHIVED, don't keep tracking
                if cached_pat.state in (PatternState.FAILED, PatternState.ARCHIVED):
                    continue

                age = cached_pat.age_bars + 1
                state = cached_pat.state
                
                # Expiry check by state age limit
                max_age = self.MAX_AGE_BARS.get(state, 30)
                expired = age >= max_age
                
                # Eviction bars check (not seen for 5 bars)
                evicted = (current_bar - cached_pat.last_seen_bar) >= self.CACHE_EVICTION_BARS

                if expired or evicted:
                    to_state = PatternState.ARCHIVED
                    stats[cached_pat.type.value].expired += 1
                    
                    # If price breached it while we weren't looking, it's failed
                    if cached_pat.direction == PatternDirection.LONG and current_price < cached_pat.original_invalidation:
                        to_state = PatternState.FAILED
                    elif cached_pat.direction == PatternDirection.SHORT and current_price > cached_pat.original_invalidation:
                        to_state = PatternState.FAILED

                    updated_pat = Pattern(
                        id=cached_pat.id,
                        research_id=cached_pat.research_id,
                        parent_pattern_id=cached_pat.parent_pattern_id,
                        child_pattern_ids=cached_pat.child_pattern_ids,
                        type=cached_pat.type,
                        state=to_state,
                        direction=cached_pat.direction,
                        quality_score=cached_pat.quality_score,
                        confidence=cached_pat.confidence,
                        confidence_components=cached_pat.confidence_components,
                        raw_components=cached_pat.raw_components,
                        trigger_quality=0.0,
                        breakout_level=cached_pat.breakout_level,
                        original_invalidation=cached_pat.original_invalidation,
                        current_invalidation=cached_pat.current_invalidation,
                        targets=cached_pat.targets,
                        target_labels=cached_pat.target_labels,
                        completion_pct=cached_pat.completion_pct,
                        age_bars=age,
                        retest_count=cached_pat.retest_count,
                        last_seen_bar=cached_pat.last_seen_bar,
                        anchors=cached_pat.anchors,
                        evidence=cached_pat.evidence,
                        explanation=cached_pat.explanation
                    )
                    
                    stats[cached_pat.type.value].archived += 1
                    transition_events.append(
                        PatternTransitionEvent(
                            pattern_id=updated_pat.id,
                            research_id=cached_pat.research_id,
                            symbol=symbol,
                            from_state=cached_pat.state,
                            to_state=to_state,
                            pattern=updated_pat,
                            bar=current_bar,
                            delta=None
                        )
                    )
                else:
                    # Keep it as-is, increment age
                    updated_pat = Pattern(
                        id=cached_pat.id,
                        research_id=cached_pat.research_id,
                        parent_pattern_id=cached_pat.parent_pattern_id,
                        child_pattern_ids=cached_pat.child_pattern_ids,
                        type=cached_pat.type,
                        state=state,
                        direction=cached_pat.direction,
                        quality_score=cached_pat.quality_score,
                        confidence=cached_pat.confidence,
                        confidence_components=cached_pat.confidence_components,
                        raw_components=cached_pat.raw_components,
                        trigger_quality=self._compute_trigger_quality(cached_pat, current_price, atr),
                        breakout_level=cached_pat.breakout_level,
                        original_invalidation=cached_pat.original_invalidation,
                        current_invalidation=cached_pat.current_invalidation,
                        targets=cached_pat.targets,
                        target_labels=cached_pat.target_labels,
                        completion_pct=cached_pat.completion_pct,
                        age_bars=age,
                        retest_count=cached_pat.retest_count,
                        last_seen_bar=cached_pat.last_seen_bar,
                        anchors=cached_pat.anchors,
                        evidence=cached_pat.evidence,
                        explanation=cached_pat.explanation
                    )
                    updated_cache[updated_pat.id] = updated_pat

        # Store back in symbol cache
        self._cache[symbol] = updated_cache

        # Return context
        return PatternsContext(
            patterns=list(updated_cache.values()),
            transition_events=transition_events
        )

    def _resolve_conflicts(self, patterns: List[Pattern], stats: Dict[str, DetectorStats]) -> List[Pattern]:
        """Resolves overlapping candidates by keeping the one with higher confidence."""
        if not patterns:
            return []

        # Sort by confidence descending
        sorted_pats = sorted(patterns, key=lambda p: p.confidence, reverse=True)
        resolved: List[Pattern] = []

        for p in sorted_pats:
            conflict = False
            p_anchors = {a.id for a in p.anchors if a.source_swing_id is not None}
            
            for acc in resolved:
                acc_anchors = {a.id for a in acc.anchors if a.source_swing_id is not None}
                # Check intersection size
                shared = p_anchors.intersection(acc_anchors)
                if len(shared) >= 2:
                    conflict = True
                    break
            
            if conflict:
                stats[p.type.value].duplicate_rejects += 1
            else:
                resolved.append(p)

        return resolved

    def _compute_trigger_quality(self, pattern: Pattern, price: float, atr: float) -> float:
        """
        Trigger quality: Price proximity to breakout_level.
        Formula: tanh(completion_pct * 2) * proximity_score
        where proximity_score = 1 - (dist_to_breakout / (2 * atr))
        """
        if atr <= 0:
            atr = 1.0
        
        dist = abs(price - pattern.breakout_level)
        prox = max(0.0, min(1.0, 1.0 - (dist / (2.0 * atr))))
        
        return round(math.tanh(pattern.completion_pct * 2.0) * prox, 3)

    def _create_delta(self, old: Pattern, new: Pattern, price: float) -> PatternDelta:
        """Creates the PatternDelta showing what changed between ticks."""
        old_dist = abs(price - old.breakout_level)
        new_dist = abs(price - new.breakout_level)
        
        return PatternDelta(
            pattern_id=new.id,
            state_changed=(old.state != new.state),
            confidence_delta=round(new.confidence - old.confidence, 3),
            trigger_quality_delta=round(new.trigger_quality - old.trigger_quality, 3),
            completion_delta=round(new.completion_pct - old.completion_pct, 3),
            age_delta=new.age_bars - old.age_bars,
            breakout_distance_delta=round(new_dist - old_dist, 3)
        )
