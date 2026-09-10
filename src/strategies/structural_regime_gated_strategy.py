#!/usr/bin/env python3
"""
StructuralRegimeGatedStrategy — StructuralStrategy + a SWEEP/TRAP counter-trend gate.
======================================================================================
SWEEP and TRAP are reversal setups (liquidity-grab / failed-breakout) — by
construction they can fire in either direction, gated only on daily/h1 bias
resolving non-NEUTRAL. On 2026-09-07/08/09 (STRONG_TREND_DOWN_NORMAL both
days), 09-08 real trades show 11 of 13 SWEEP/TRAP/LIQUIDITY_SWEEP signals were
BUY CALL — long, against a day that closed near session lows — and all but one
lost. A reversal setup betting against a confirmed trend is fighting the same
force that keeps invalidating it; STRONG_TREND regimes are exactly where a
trend has enough force to run over a bounce/exhaustion read.

This is a NEW strategy/experiment variant — not an edit to the frozen engine,
and not an edit to StructuralStrategy (whose class docstring guarantees
byte-for-byte parity with the engine's raw output for every experiment that
uses it unmodified). This subclass deliberately does NOT preserve that parity
for SWEEP/TRAP signals fired counter to a STRONG_TREND regime, and says so
here rather than silently breaking that promise in place.

Distinct hypothesis from StructuralQualityGatedStrategy (candle-quality gate
on move_efficiency/wickiness) — kept as a separate experiment so
filter_attribution.py can tell which gate (if either) actually helps, rather
than conflating two unrelated filters in one A/B clone.

Does NOT gate: BREAKOUT setups (trend-continuation by construction, so a
regime-direction gate is redundant there) or SWEEP/TRAP fired in RANGE/GAP/
COMPRESSION/WEAK_TREND regimes — only STRONG_TREND_UP/STRONG_TREND_DOWN
opposite the signal's own direction are rejected.
"""

import logging

from src.core.base_strategy import StrategyResult
from src.core.market_snapshot import MarketSnapshot
from src.strategies.structural_strategy import StructuralStrategy

logger = logging.getLogger(__name__)

_LONG_SIGNALS = {"BUY CALL"}
_SHORT_SIGNALS = {"BUY PUT"}


class StructuralRegimeGatedStrategy(StructuralStrategy):
    """StructuralStrategy, with SWEEP/TRAP additionally gated against counter-trend regimes."""

    def __init__(
        self,
        min_zone_score: float = 50.0,
        rvol_threshold: float = 1.0,
    ):
        super().__init__(min_zone_score=min_zone_score, rvol_threshold=rvol_threshold)
        logger.info(
            f"🏛️ StructuralRegimeGatedStrategy initialized "
            f"[rvol>={rvol_threshold}, zone_score>={min_zone_score}]"
        )

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        result = super().evaluate(snapshot, experiment_name)
        regime_primary = getattr(snapshot.regime_detail, "primary", None)
        if regime_primary not in ("STRONG_TREND_UP", "STRONG_TREND_DOWN"):
            return result

        for sig in result.signals:
            if sig.get("strategy") not in ("SWEEP", "TRAP"):
                continue
            signal_dir = sig.get("signal")
            counter_trend = (
                (regime_primary == "STRONG_TREND_DOWN" and signal_dir in _LONG_SIGNALS)
                or (regime_primary == "STRONG_TREND_UP" and signal_dir in _SHORT_SIGNALS)
            )
            if counter_trend:
                sig["rejection_reasons"] = list(sig.get("rejection_reasons") or []) + ["COUNTER_TREND_REGIME"]
                sig["accepted"] = False
        return result

    def __repr__(self) -> str:
        return (
            f"StructuralRegimeGatedStrategy("
            f"rvol>={self.rvol_threshold}, zone_score>={self.min_zone_score})"
        )
