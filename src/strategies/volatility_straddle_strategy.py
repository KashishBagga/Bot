#!/usr/bin/env python3
"""
Volatility Straddle / Strangle Strategy (non-directional, multi-leg)
====================================================================
Hypothesis: When volatility is compressed and cheap (deep squeeze, low ATR
percentile) a large move is likely but its direction is unclear. Buying BOTH a
call and a put profits from the expansion in either direction.

Emits a COMBO signal (not a directional one):
    signal = "STRADDLE"  → buy ATM call + ATM put   (deep compression, move imminent)
    signal = "STRANGLE"  → buy OTM call + OTM put   (cheaper, needs a bigger move)

The combo is managed in PREMIUM space by the trader (combined premium %-stop /
target / time-stop), because a straddle has no meaningful index-point stop.

Confluence: this strategy is the natural home when MarketView.dominant_direction
is VOLATILE. A strong directional read (|directional_score| large) is recorded as
a rejection so the counterfactual engine can measure whether fading it was right.
"""

import logging
from typing import Any, Dict, List

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.chart_patterns import detect_squeeze

logger = logging.getLogger(__name__)


class VolatilityStraddleStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        id="volatility_straddle",
        name="Volatility Straddle / Strangle",
        hypothesis_id="volatility_expansion_nondirectional",
        hypothesis_family="Volatility",
        hypothesis_text="Cheap compressed volatility expands; a two-leg combo profits either way.",
        version="v1.0", maturity="PAPER", risk_profile="high",
        tags=["straddle", "strangle", "volatility", "squeeze", "multi_leg"],
    )

    def __init__(
        self,
        straddle_atr_pct: float = 0.15,   # <= this ATR percentile → prefer ATM straddle
        max_atr_pct: float = 0.40,        # > this → vol too expensive, skip
        strangle_otm_pct: float = 0.01,   # OTM offset for strangle legs (1% of spot)
        premium_sl_pct: float = 0.40,     # exit combo if combined premium falls 40%
        premium_tp_pct: float = 0.60,     # exit combo at +60% combined premium
        max_bars: int = 24,               # time stop (~2h on 5m)
    ):
        self.straddle_atr_pct = straddle_atr_pct
        self.max_atr_pct = max_atr_pct
        self.strangle_otm_pct = strangle_otm_pct
        self.premium_sl_pct = premium_sl_pct
        self.premium_tp_pct = premium_tp_pct
        self.max_bars = max_bars

    def thesis_key(self, signal: dict) -> tuple:
        return (signal.get("symbol", ""), signal.get("signal", ""))  # STRADDLE/STRANGLE per symbol

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        signals: List[Dict[str, Any]] = []
        try:
            m5 = snapshot.m5
            if m5 is None or len(m5) < 25:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])
            atr = snapshot.features.get_float("atr")
            atr_pct = snapshot.features.get_float("atr_percentile", 0.5)
            price = snapshot.current_price
            view = snapshot.market_view

            squeeze = detect_squeeze(m5, atr)
            vol_score = view.vol_score if view is not None else 0.0
            directional = abs(view.directional_score) if view is not None else 0.0

            # Setup only when volatility is compressed (squeeze OR low ATR percentile).
            if squeeze is None and atr_pct > 0.30:
                return self._empty_result(experiment_name)

            # Choose combo type.
            if atr_pct <= self.straddle_atr_pct and vol_score >= 0.5:
                combo_type, otm_pct, setup = "STRADDLE", 0.0, "VOL_SQUEEZE_STRADDLE"
            else:
                combo_type, otm_pct, setup = "STRANGLE", self.strangle_otm_pct, "VOL_SQUEEZE_STRANGLE"

            reasons: List[str] = []
            if atr_pct > self.max_atr_pct:
                reasons.append("VOL_TOO_EXPENSIVE")
            if directional > 0.5:
                # A strong directional read argues against a non-directional combo.
                reasons.append("STRONG_DIRECTIONAL")

            base_conf = squeeze.confidence if squeeze is not None else (0.30 - atr_pct) * 2 + 0.4
            confidence = round(min(0.99, max(0.0, base_conf)) * 100, 1)

            ts = snapshot.timestamp
            sym_clean = snapshot.symbol.replace(":", "_").replace("-", "_")
            candidate_id = f"cand_{sym_clean}_{combo_type}_{price:.2f}_{ts.strftime('%Y%m%d_%H%M%S')}"

            sig = {
                "symbol": snapshot.symbol,
                "signal": combo_type,                 # STRADDLE | STRANGLE
                "strategy": setup,
                "price": price,                        # underlying reference
                "stop_loss": None,                     # combo is managed in premium space
                "take_profit": None,
                "rr_ratio": round(self.premium_tp_pct / self.premium_sl_pct, 2),
                "timestamp": ts.isoformat(),
                "accepted": len(reasons) == 0,
                "rejection_reasons": reasons,
                "features": snapshot.features.to_dict(),
                "candidate_id": candidate_id,
                "confidence": confidence,
                "diagnostics": {
                    "atr_percentile": round(atr_pct, 3),
                    "vol_score": round(vol_score, 3),
                    "directional_score": round(view.directional_score, 3) if view else 0.0,
                    "squeeze": squeeze.as_dict() if squeeze else None,
                    "market_view": view.summary if view is not None else "n/a",
                },
                # The multi-leg contract for the execution engine / trader.
                "combo": {
                    "type": combo_type,
                    "otm_pct": otm_pct,
                    "premium_sl_pct": self.premium_sl_pct,
                    "premium_tp_pct": self.premium_tp_pct,
                    "max_bars": self.max_bars,
                },
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)
        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[VolatilityStraddleStrategy] {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(experiment_name, self.id, self.version, signals, {}, errors, [])
