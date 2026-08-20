#!/usr/bin/env python3
"""
Straddle / Strangle Strategy — long volatility, direction-agnostic
====================================================================
Hypothesis: volatility compression (low ATR percentile — the same signal
atr_squeeze.py already uses for a directional breakout bet) tends to resolve
into expansion. Instead of betting on which direction the expansion goes,
buy both a call and a put and profit from the magnitude of the move,
regardless of direction.

IMPORTANT — this is a realized-volatility proxy, not implied volatility.
This codebase has no IV/IV-percentile/Greeks data anywhere (confirmed: Fyers
quotes give LTP/bid/ask only, and get_option_chain()'s OI fields are
hardcoded placeholders, not real data). A real vol desk would size this
entry on "IV is cheap relative to its own history"; this strategy instead
uses "realized price action has been unusually quiet" (atr_percentile) as
the nearest available real-data stand-in. That's a materially weaker signal
than true IV rank and should be treated as such in research review — it's
disclosed here and in diagnostics (`vol_signal_type: realized_vol_proxy`)
rather than presented as if it were a real volatility-surface read.

wing_strikes=0 -> LONG_STRADDLE (both legs ATM).
wing_strikes>0 -> LONG_STRANGLE (both legs that many strikes OTM — cheaper
premium, wider breakevens).
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class StraddleStrangleStrategy(BaseStrategy):
    """Long Straddle/Strangle on realized-volatility compression."""

    metadata = StrategyMetadata(
        id="straddle_strangle",
        name="Straddle/Strangle (Volatility Compression)",
        hypothesis_id="vol_compression_long_vol",
        hypothesis_family="Volatility",
        hypothesis_text=(
            "Realized-volatility compression (low ATR percentile) tends to "
            "resolve into expansion; a long straddle/strangle profits from the "
            "magnitude of that expansion regardless of direction. Uses realized "
            "vol as a proxy — this system has no implied-volatility data."
        ),
        version="v1.0",
        archetype="Volatility",
        exit_profile="PREMIUM_TARGET_R",
        maturity="RESEARCH",
        tags=["options", "straddle", "strangle", "combo", "volatility"],
    )

    def __init__(
        self,
        atr_percentile_threshold: float = 0.20,
        wing_strikes: int = 0,
        decision_cutoff_hour: int = 14,
        target_r: float = 1.2,
        stop_r: float = -0.5,
    ):
        self.atr_percentile_threshold = atr_percentile_threshold
        self.wing_strikes = wing_strikes
        # Long vol needs time for the expansion to develop — no new entries
        # this late, mirroring ORB's LATE_SESSION guard.
        self.decision_cutoff_hour = decision_cutoff_hour
        self.target_r = target_r
        self.stop_r = stop_r

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 10:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr_percentile = snapshot.features.get_float("atr_percentile")
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            current_time = snapshot.timestamp
            combo_type = "LONG_STRADDLE" if self.wing_strikes == 0 else "LONG_STRANGLE"

            rejection_reasons: List[str] = []
            if atr_percentile > self.atr_percentile_threshold:
                rejection_reasons.append(f"NO_COMPRESSION:{atr_percentile:.2f}")
            if current_time.hour >= self.decision_cutoff_hour:
                rejection_reasons.append("LATE_SESSION")

            combo_legs = [
                {"option_type": "CE", "side": "BUY", "strikes_away": self.wing_strikes},
                {"option_type": "PE", "side": "BUY", "strikes_away": -self.wing_strikes},
            ]

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_"
                f"{combo_type}_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            confidence = 0.5
            if accepted:
                # Deeper compression -> higher conviction the expansion is overdue.
                compression_factor = max(0.0, 1.0 - (atr_percentile / max(self.atr_percentile_threshold, 0.01)))
                confidence = round(min(0.5 + 0.4 * compression_factor, 0.9), 2)

            sig = {
                "symbol": snapshot.symbol,
                "signal": combo_type,
                "strategy": combo_type,
                "price": price,
                "combo_legs": combo_legs,
                "target_r": self.target_r,
                "stop_r": self.stop_r,
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "candidate_id": candidate_id,
                "confidence": confidence,
                "timestamp": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
                "diagnostics": {
                    "atr_percentile": round(atr_percentile, 3),
                    "atr_percentile_threshold": self.atr_percentile_threshold,
                    "atr": round(atr, 2),
                    "wing_strikes": self.wing_strikes,
                    "vol_signal_type": "realized_vol_proxy",  # NOT implied volatility — see module docstring
                },
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[StraddleStrangleStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
