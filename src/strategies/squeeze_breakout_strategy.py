#!/usr/bin/env python3
"""
Squeeze Breakout Strategy (volatility expansion, directional)
=============================================================
Hypothesis: Volatility is mean-reverting. After a Bollinger-in-Keltner squeeze
(compression), the market expands; the first decisive break of the compression
range sets the direction.

Trigger: a VOL_SQUEEZE is/was active AND the latest close breaks the recent
compression range → BUY CALL (upside) / BUY PUT (downside), with RVOL confirming.
This is the directional sibling of the non-directional straddle strategy — both
key off the same squeeze, but this one only fires once a direction is chosen.
"""

import logging
from typing import Any, Dict, List

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.chart_patterns import detect_squeeze
from src.strategies._signal_builder import build_directional_signal

logger = logging.getLogger(__name__)


class SqueezeBreakoutStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        id="squeeze_breakout",
        name="Squeeze Breakout",
        hypothesis_id="volatility_expansion",
        hypothesis_family="Breakouts",
        hypothesis_text="Compressed volatility expands; the first range break sets direction.",
        version="v1.0", maturity="PAPER", risk_profile="medium",
        tags=["squeeze", "volatility", "breakout", "bollinger", "keltner"],
    )

    def __init__(self, range_lookback: int = 20, min_rvol: float = 1.2):
        self.range_lookback = range_lookback
        self.min_rvol = min_rvol

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        signals: List[Dict[str, Any]] = []
        try:
            m5 = snapshot.m5
            if m5 is None or len(m5) < self.range_lookback + 5:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])
            atr = snapshot.features.get_float("atr")
            price = snapshot.current_price
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 1.0

            squeeze = detect_squeeze(m5, atr)
            if squeeze is None:
                return self._empty_result(experiment_name)  # not compressed → no setup

            # Range of the compression window, excluding the breakout (current) bar.
            window = m5.iloc[-(self.range_lookback + 1):-1]
            hi = float(window["high"].max())
            lo = float(window["low"].min())
            close = float(m5["close"].iloc[-1])

            side = sl = None
            if close > hi:
                side, sl = "BUY CALL", min(lo, price - 0.5 * atr)
            elif close < lo:
                side, sl = "BUY PUT", max(hi, price + 0.5 * atr)
            else:
                return self._empty_result(experiment_name)  # still inside the squeeze

            reasons: List[str] = []
            if rvol < self.min_rvol:
                reasons.append("LOW_RVOL")

            sig = build_directional_signal(
                snapshot, side, "SQUEEZE_BREAKOUT", entry=price, stop_loss=sl,
                base_confidence=squeeze.confidence, rejection_reasons=reasons,
                diagnostics={"squeeze": squeeze.as_dict(), "range_high": round(hi, 2),
                             "range_low": round(lo, 2), "rvol": round(rvol, 2)},
                respect_bias="off",  # expansion can start against the prior bias
            )
            if sig is not None:
                self._tag_signal(sig, experiment_name)
                signals.append(sig)
        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[SqueezeBreakoutStrategy] {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(experiment_name, self.id, self.version, signals, {}, errors, [])
