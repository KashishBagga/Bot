#!/usr/bin/env python3
"""
RSI Divergence Strategy (mean-reversion / exhaustion)
=====================================================
Hypothesis: When price makes a new extreme but RSI does not, momentum is fading
and a mean-reversion bounce is likely.

Trigger (BUY CALL): RSI_BULL_DIVERGENCE — price lower low, RSI higher low, RSI < 45.
Trigger (BUY PUT):  RSI_BEAR_DIVERGENCE — price higher high, RSI lower high, RSI > 55.
Stop: beyond the divergence extreme. Bias: soft (divergence trades counter-trend).
"""

import logging
from typing import Any, Dict, List

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.chart_patterns import detect_rsi_divergence, BULLISH
from src.strategies._signal_builder import build_directional_signal

logger = logging.getLogger(__name__)


class RsiDivergenceStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        id="rsi_divergence",
        name="RSI Divergence",
        hypothesis_id="momentum_divergence",
        hypothesis_family="MeanReversion",
        hypothesis_text="Price/RSI divergence signals momentum exhaustion and reversion.",
        version="v1.0", maturity="PAPER", risk_profile="medium",
        tags=["rsi", "divergence", "mean_reversion", "exhaustion"],
    )

    def __init__(self, min_confidence: float = 0.5, max_rvol: float = 3.0):
        self.min_confidence = min_confidence
        self.max_rvol = max_rvol  # extreme RVOL = climax; divergence less reliable mid-blowoff

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

            pattern = detect_rsi_divergence(m5, atr)
            if pattern is None:
                return self._empty_result(experiment_name)

            side = "BUY CALL" if pattern.direction == BULLISH else "BUY PUT"
            if side == "BUY CALL":
                sl = min(pattern.stop_hint, price - 0.5 * atr)
            else:
                sl = max(pattern.stop_hint, price + 0.5 * atr)

            reasons: List[str] = []
            if pattern.confidence < self.min_confidence:
                reasons.append("LOW_PATTERN_CONFIDENCE")
            if rvol > self.max_rvol:
                reasons.append("VOLUME_CLIMAX")

            sig = build_directional_signal(
                snapshot, side, pattern.name, entry=price, stop_loss=sl,
                base_confidence=pattern.confidence, rejection_reasons=reasons,
                diagnostics={"pattern": pattern.as_dict(), "rvol": round(rvol, 2),
                             "rsi14": snapshot.features.get_float("rsi14")},
                respect_bias="soft",
            )
            if sig is not None:
                self._tag_signal(sig, experiment_name)
                signals.append(sig)
        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[RsiDivergenceStrategy] {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(experiment_name, self.id, self.version, signals, {}, errors, [])
