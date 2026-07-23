#!/usr/bin/env python3
"""
Continuation Pattern Strategy
=============================
Hypothesis: After a strong impulse, tight consolidations (flags) and converging
ranges (triangles) resolve in the direction of the prevailing move. Trades the
breakout of triangles and bull/bear flags.

Trigger (BUY CALL): TRIANGLE_ASC/SYM upside break, or BULL_FLAG breakout.
Trigger (BUY PUT):  TRIANGLE_DESC/SYM downside break, or BEAR_FLAG breakdown.
Filters: RVOL participation + move efficiency; bias alignment is required (hard)
because continuations should not fight the higher-timeframe trend.
"""

import logging
from typing import Any, Dict, List

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.chart_patterns import detect_triangle, detect_flag, BULLISH
from src.strategies._signal_builder import build_directional_signal

logger = logging.getLogger(__name__)


class ContinuationPatternStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        id="continuation_pattern",
        name="Continuation Pattern (Triangle / Flag)",
        hypothesis_id="continuation_breakout",
        hypothesis_family="Breakouts",
        hypothesis_text="Consolidations after an impulse resolve in the trend direction.",
        version="v1.0", maturity="PAPER", risk_profile="medium",
        tags=["continuation", "triangle", "flag", "breakout", "pattern"],
    )

    def __init__(self, min_confidence: float = 0.45, min_rvol: float = 1.1,
                 min_efficiency: float = 0.5):
        self.min_confidence = min_confidence
        self.min_rvol = min_rvol
        self.min_efficiency = min_efficiency

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        signals: List[Dict[str, Any]] = []
        try:
            m5 = snapshot.m5
            if m5 is None or len(m5) < 30:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])
            atr = snapshot.features.get_float("atr")
            price = snapshot.current_price
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 1.0
            eff = snapshot.features.get_float("move_efficiency")

            pattern = detect_flag(m5, atr) or detect_triangle(m5, atr)
            if pattern is None:
                return self._empty_result(experiment_name)

            side = "BUY CALL" if pattern.direction == BULLISH else "BUY PUT"
            # Stop at the far edge of the pattern (stop_hint), min 0.5 ATR.
            if side == "BUY CALL":
                sl = min(pattern.stop_hint, price - 0.5 * atr)
            else:
                sl = max(pattern.stop_hint, price + 0.5 * atr)

            reasons: List[str] = []
            if pattern.confidence < self.min_confidence:
                reasons.append("LOW_PATTERN_CONFIDENCE")
            if rvol < self.min_rvol:
                reasons.append("LOW_RVOL")
            if eff < self.min_efficiency:
                reasons.append("LOW_EFFICIENCY")

            sig = build_directional_signal(
                snapshot, side, pattern.name, entry=price, stop_loss=sl,
                base_confidence=pattern.confidence, rejection_reasons=reasons,
                diagnostics={"pattern": pattern.as_dict(), "rvol": round(rvol, 2),
                             "move_efficiency": round(eff, 3)},
                respect_bias="hard",  # continuations must align with the trend
            )
            if sig is not None:
                self._tag_signal(sig, experiment_name)
                signals.append(sig)
        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[ContinuationPatternStrategy] {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(experiment_name, self.id, self.version, signals, {}, errors, [])
