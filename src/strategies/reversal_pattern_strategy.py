#!/usr/bin/env python3
"""
Reversal Pattern Strategy
=========================
Hypothesis: Completed reversal patterns at the end of a move mark exhaustion and
a high-probability turn. Trades Head & Shoulders / Inverse H&S and Double
Top / Double Bottom, confirmed on the neckline break.

Trigger (BUY PUT):  HEAD_SHOULDERS or DOUBLE_TOP confirmed (close < neckline).
Trigger (BUY CALL): INVERSE_HEAD_SHOULDERS or DOUBLE_BOTTOM confirmed (close > neckline).
Stop: pattern invalidation (head / opposite peak).  Bias: soft (reversals fade bias).
"""

import logging
from typing import Any, Dict, List

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.chart_patterns import (
    detect_head_shoulders, detect_double_top_bottom, BULLISH,
)
from src.strategies._signal_builder import build_directional_signal

logger = logging.getLogger(__name__)


class ReversalPatternStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        id="reversal_pattern",
        name="Reversal Pattern (H&S / Double Top-Bottom)",
        hypothesis_id="reversal_exhaustion",
        hypothesis_family="Reversals",
        hypothesis_text="Confirmed reversal patterns mark trend exhaustion and a turn.",
        version="v1.0", maturity="PAPER", risk_profile="medium",
        tags=["reversal", "head_shoulders", "double_top", "double_bottom", "pattern"],
    )

    def __init__(self, min_confidence: float = 0.5, min_rvol: float = 1.0):
        self.min_confidence = min_confidence
        self.min_rvol = min_rvol

    def thesis_key(self, signal: dict) -> tuple:
        return (signal.get("symbol", ""), signal.get("strategy", ""), signal.get("signal", ""))

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

            # Prefer the higher-conviction H&S; fall back to double top/bottom.
            pattern = detect_head_shoulders(m5, atr) or detect_double_top_bottom(m5, atr)
            if pattern is None:
                return self._empty_result(experiment_name)

            side = "BUY CALL" if pattern.direction == BULLISH else "BUY PUT"
            # Stop at pattern invalidation, but never tighter than 0.5 ATR.
            if side == "BUY CALL":
                sl = min(pattern.stop_hint, price - 0.5 * atr)
            else:
                sl = max(pattern.stop_hint, price + 0.5 * atr)

            reasons: List[str] = []
            if pattern.confidence < self.min_confidence:
                reasons.append("LOW_PATTERN_CONFIDENCE")
            if rvol < self.min_rvol:
                reasons.append("LOW_RVOL")

            sig = build_directional_signal(
                snapshot, side, pattern.name, entry=price, stop_loss=sl,
                base_confidence=pattern.confidence, rejection_reasons=reasons,
                diagnostics={"pattern": pattern.as_dict(), "rvol": round(rvol, 2)},
                respect_bias="soft",  # reversals are allowed to fade the prevailing bias
            )
            if sig is not None:
                self._tag_signal(sig, experiment_name)
                signals.append(sig)
        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[ReversalPatternStrategy] {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(experiment_name, self.id, self.version, signals, {}, errors, [])
