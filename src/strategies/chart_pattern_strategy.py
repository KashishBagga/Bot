#!/usr/bin/env python3
"""
ChartPatternStrategy — trades off the existing (previously unused) pattern
engine, confirmed by a named candlestick pattern and cross-checked against
zone_engine support/resistance.
================================================================================
Hypothesis:
    A completed/breaking-out chart pattern (Double Top/Bottom, Head & Shoulders,
    triangles, flags) is a higher-probability structural thesis when (a) the
    trigger candle itself shows a real reversal/continuation candlestick
    pattern in the same direction, and (b) the pattern's breakout level sits
    at a genuine zone_engine supply/demand zone — not just the pattern
    engine's own internal geometry confluence.

Architecture: reads snapshot.market.patterns (PatternsContext, MKE Stage 6 —
    9 detectors: Double Top/Bottom, Ascending/Descending Triangle, Bull/Bear
    Flag, Rectangle, Head & Shoulders, Inverse Head & Shoulders) and
    snapshot.h1_zones (ZoneEngine supply/demand). Rectangle (BILATERAL
    direction) is skipped for now — the detector only tracks a single
    breakout_level (the upper boundary), so a downside breakdown wouldn't be
    represented correctly; tracked as a known simplification, not a bug.

Multiple targets, all already computed by MeasuredMoveEngine — nothing here
invents new target math:
    tp1              — 1.5R partial (same convention as every other strategy)
    tp_nearest_zone   — nearest opposing zone_engine zone beyond entry
    tp_measured_move  — 100% pattern-height projection from the breakout level
    tp_extended       — 161.8% pattern-height projection
    take_profit       — whichever of tp_nearest_zone / tp_measured_move is
                         closer to entry (the more conservative, structurally
                         grounded target), capped at tp_atr_cap × ATR like
                         every other strategy. This is the field the trader's
                         position manager actually trails/expands against;
                         the others are informational (surfaced in
                         diagnostics for the dashboard).

Rejection reasons are stored in signal['rejection_reasons']; accepted=False
signals are still emitted for counterfactual tracking, same as every other
strategy in this framework.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.market_patterns import PatternState, PatternDirection, Pattern
from src.core.candlestick_patterns import detect as detect_candlesticks, strongest_signal, CandleDirection

logger = logging.getLogger(__name__)


def _candidate_id(symbol: str, pattern_type: str, price: float, ts: datetime) -> str:
    safe = symbol.replace(":", "_").replace("-", "_")
    return f"cand_{safe}_{pattern_type}_{price:.2f}_{ts.strftime('%Y%m%d_%H%M%S')}"


class ChartPatternStrategy(BaseStrategy):
    """Chart-pattern strategy: pattern engine + candlestick confirmation + zone confluence."""

    metadata = StrategyMetadata(
        id="chart_pattern",
        name="Chart Pattern Strategy",
        hypothesis_id="chart_patterns_with_candle_and_zone_confirmation",
        hypothesis_family="ChartPattern",
        hypothesis_text=(
            "Classic chart patterns (double top/bottom, H&S, triangles, flags) "
            "resolve in their textbook direction more reliably when confirmed by "
            "a real candlestick reversal pattern at the trigger candle and backed "
            "by a genuine supply/demand zone at the breakout level."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["chart-pattern", "candlestick", "zone-confluence", "measured-move"],
    )

    def __init__(
        self,
        min_pattern_confidence: float = 0.55,
        min_candle_strength: float = 0.40,
        zone_tolerance_pct: float = 0.003,
        tp_atr_cap: float = 5.0,
        min_rr: float = 1.5,
    ):
        self.min_pattern_confidence = min_pattern_confidence
        self.min_candle_strength = min_candle_strength
        self.zone_tolerance_pct = zone_tolerance_pct
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        logger.info(
            f"📐 ChartPatternStrategy initialized [pattern_conf>={min_pattern_confidence}, "
            f"candle_strength>={min_candle_strength}, min_rr>={min_rr}]"
        )

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            patterns_ctx = getattr(snapshot.market, "patterns", None)
            if patterns_ctx is None or not patterns_ctx.patterns:
                return self._empty_result(
                    experiment_name, diagnostics={"ready_patterns": 0}
                )

            atr: float = snapshot.features.get_float("atr") or 0.0
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            price = snapshot.current_price
            ts = snapshot.timestamp
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 3:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA:m5"])

            zones = snapshot.h1_zones or []

            candidates = [
                p for p in patterns_ctx.patterns
                if p.state in (PatternState.READY, PatternState.BREAKOUT, PatternState.CONFIRMED)
                and p.direction in (PatternDirection.LONG, PatternDirection.SHORT)
            ]

            for pattern in candidates:
                sig = self._build_signal_for_pattern(
                    pattern, price, atr, m5_df, ts, snapshot, experiment_name, zones
                )
                if sig:
                    signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[ChartPatternStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={
                "min_pattern_confidence": self.min_pattern_confidence,
                "min_candle_strength": self.min_candle_strength,
                "min_rr": self.min_rr,
                "ready_patterns": len(patterns_ctx.patterns) if patterns_ctx else 0,
            },
            errors=errors,
            warnings=warnings,
        )

    def _build_signal_for_pattern(
        self,
        pattern: Pattern,
        price: float,
        atr: float,
        m5_df,
        ts: datetime,
        snapshot: MarketSnapshot,
        experiment_name: str,
        zones: list,
    ) -> Optional[Dict[str, Any]]:
        symbol = snapshot.symbol
        side = "BUY CALL" if pattern.direction == PatternDirection.LONG else "BUY PUT"
        rejection_reasons: List[str] = []

        # ── 1. Candlestick confirmation on the trigger candle ─────────────
        candle_signals = detect_candlesticks(m5_df, idx=-1)
        wanted_dir = CandleDirection.BULLISH if side == "BUY CALL" else CandleDirection.BEARISH
        confirming = strongest_signal(candle_signals, direction=wanted_dir)
        if confirming is None or confirming.strength < self.min_candle_strength:
            rejection_reasons.append("NO_CANDLESTICK_CONFIRMATION")

        # ── 2. Pattern engine's own confidence gate ───────────────────────
        if pattern.confidence < self.min_pattern_confidence:
            rejection_reasons.append(f"LOW_PATTERN_CONFIDENCE:{pattern.confidence}")

        # ── 3. Zone confluence — cross-check against zone_engine S/R, a
        # separate system from the pattern engine's own internal geometry
        # confluence (which only looks at CompositeLevel/ConfluenceZone).
        tolerance = price * self.zone_tolerance_pct
        near_zone = any(abs(z.level - pattern.breakout_level) <= tolerance for z in zones)
        if not near_zone:
            rejection_reasons.append("NO_ZONE_CONFLUENCE")

        # ── 4. SL from the pattern's own invalidation level ───────────────
        sl = pattern.current_invalidation
        if side == "BUY CALL":
            risk_dist = price - sl
        else:
            risk_dist = sl - price

        min_sl_dist = atr * 0.5
        if risk_dist < min_sl_dist:
            sl = (price - min_sl_dist) if side == "BUY CALL" else (price + min_sl_dist)
            risk_dist = min_sl_dist

        if risk_dist <= 0:
            rejection_reasons.append("ZERO_RISK")
            risk_dist = atr  # avoid div-by-zero below; signal is already rejected

        # ── 5. Multiple targets ────────────────────────────────────────────
        tp1 = price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5)

        measured = dict(zip(pattern.target_labels, pattern.targets))
        max_tp_dist = atr * self.tp_atr_cap

        def _cap(tp):
            if tp is None:
                return None
            if abs(tp - price) > max_tp_dist:
                return (price + max_tp_dist) if side == "BUY CALL" else (price - max_tp_dist)
            return tp

        tp_measured = _cap(measured.get("measured_move"))
        tp_extended = _cap(measured.get("extended_target"))

        opposing_type = "SUPPLY" if side == "BUY CALL" else "DEMAND"
        opposing = [
            z.level for z in zones
            if z.zone_type == opposing_type and (
                (side == "BUY CALL" and z.level > price) or (side == "BUY PUT" and z.level < price)
            )
        ]
        tp_zone = (min(opposing) if side == "BUY CALL" else max(opposing)) if opposing else None

        # Canonical take_profit: whichever of tp_zone / tp_measured is closer
        # to entry (more conservative, structurally grounded), falling back to
        # tp1 if neither is available.
        final_candidates = [t for t in (tp_zone, tp_measured) if t is not None]
        take_profit = min(final_candidates, key=lambda t: abs(t - price)) if final_candidates else tp1

        # tp1 and take_profit are computed from independent geometry (SL comes
        # from the pattern's invalidation level; the target comes from pattern
        # height or a zone level) — they aren't guaranteed to already be
        # ordered. tp1 is meant to be reached first, so the final target must
        # never sit closer to entry than it.
        if abs(take_profit - price) < abs(tp1 - price):
            take_profit = tp1

        rr_ratio = round(abs(take_profit - price) / risk_dist, 2) if risk_dist > 0 else 0.0
        if rr_ratio < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr_ratio}")

        accepted = len(rejection_reasons) == 0
        candidate_id = _candidate_id(symbol, pattern.type.value, price, ts)

        sig = {
            "symbol": symbol,
            "candidate_id": candidate_id,
            "signal": side,
            "price": round(price, 2),
            "stop_loss": round(sl, 2),
            "take_profit": round(take_profit, 2),
            "tp1": round(tp1, 2),
            "rr_ratio": rr_ratio,
            "strategy": pattern.type.value,
            "confidence": round(pattern.confidence * 100, 1),
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "timestamp": ts,
            "diagnostics": {
                "pattern_state": pattern.state.value,
                "pattern_quality_score": pattern.quality_score,
                "pattern_completion_pct": pattern.completion_pct,
                "pattern_trigger_quality": pattern.trigger_quality,
                "breakout_level": round(pattern.breakout_level, 2),
                "candlestick_pattern": confirming.name if confirming else None,
                "candlestick_strength": confirming.strength if confirming else None,
                "tp_nearest_zone": round(tp_zone, 2) if tp_zone is not None else None,
                "tp_measured_move": round(tp_measured, 2) if tp_measured is not None else None,
                "tp_extended_1618": round(tp_extended, 2) if tp_extended is not None else None,
                "pattern_explanation": pattern.explanation,
            },
        }
        return self._tag_signal(sig, experiment_name)
