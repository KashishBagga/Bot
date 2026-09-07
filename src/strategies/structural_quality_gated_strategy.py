#!/usr/bin/env python3
"""
StructuralQualityGatedStrategy — StructuralStrategy + a SWEEP/TRAP quality gate.
=================================================================================
EnhancedStrategyEngine (frozen, v3.2) only gates move_efficiency/wickiness on
BREAKOUT setups — SWEEP and TRAP are exempt by design. Real losses across
2026-08-24, 08-31, 09-01 and 09-04 show this exemption reliably passing
through choppy, indecisive candles: wickiness up to 0.71 and move_efficiency
as low as 0.085 on SWEEP entries, several with fully NEUTRAL/NEUTRAL daily and
hourly bias.

This is a NEW strategy/experiment variant — not an edit to the frozen engine,
and not an edit to StructuralStrategy (whose class docstring guarantees
byte-for-byte parity with the engine's raw output for every experiment that
uses it unmodified). This subclass deliberately does NOT preserve that parity
for SWEEP/TRAP signals, and says so here rather than silently breaking that
promise in place.

Does NOT gate: BREAKOUT setups. The frozen engine already gates BREAKOUT on
these same two features, so re-applying the same thresholds here would be
redundant (not incorrect) — BREAKOUT signals pass through unchanged.
"""

import logging

from src.core.base_strategy import StrategyResult
from src.core.market_snapshot import MarketSnapshot
from src.strategies.structural_strategy import StructuralStrategy

logger = logging.getLogger(__name__)


class StructuralQualityGatedStrategy(StructuralStrategy):
    """StructuralStrategy, with SWEEP/TRAP additionally gated on move_efficiency/wickiness."""

    def __init__(
        self,
        min_zone_score: float = 50.0,
        rvol_threshold: float = 1.0,
        min_move_efficiency: float = 0.6,
        max_wickiness: float = 0.5,
    ):
        super().__init__(min_zone_score=min_zone_score, rvol_threshold=rvol_threshold)
        self.min_move_efficiency = min_move_efficiency
        self.max_wickiness = max_wickiness
        logger.info(
            f"🏛️ StructuralQualityGatedStrategy initialized "
            f"[rvol>={rvol_threshold}, zone_score>={min_zone_score}, "
            f"min_eff={min_move_efficiency}, max_wick={max_wickiness}]"
        )

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        result = super().evaluate(snapshot, experiment_name)
        for sig in result.signals:
            if sig.get("strategy") not in ("SWEEP", "TRAP"):
                continue
            features = sig.get("features") or {}
            move_efficiency = features.get("move_efficiency")
            wickiness = features.get("wickiness")
            extra_reasons = []
            if move_efficiency is not None and move_efficiency <= self.min_move_efficiency:
                extra_reasons.append("LOW_EFFICIENCY")
            if wickiness is not None and wickiness >= self.max_wickiness:
                extra_reasons.append("HIGH_WICKINESS")
            if extra_reasons:
                sig["rejection_reasons"] = list(sig.get("rejection_reasons") or []) + extra_reasons
                sig["accepted"] = False
        return result

    def __repr__(self) -> str:
        return (
            f"StructuralQualityGatedStrategy("
            f"rvol>={self.rvol_threshold}, zone_score>={self.min_zone_score}, "
            f"min_eff={self.min_move_efficiency}, max_wick={self.max_wickiness})"
        )
