#!/usr/bin/env python3
"""
Iron Condor Strategy — Neutral, defined-risk credit spread.
============================================================
Hypothesis: range-bound conditions (low RVOL, low efficiency, normal to low ATR)
allow us to collect theta by selling both OTM calls and OTM puts, buying further
OTM wings as defined-risk protection. This is a non-directional income strategy
highly suited for sideways or low-volatility regimes.
"""

import logging
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class IronCondorStrategy(BaseStrategy):
    """Iron Condor Strategy — defined-risk neutral credit play."""

    metadata = StrategyMetadata(
        id="iron_condor",
        name="Iron Condor (Sideways/Range)",
        hypothesis_id="iron_condor_neutral",
        hypothesis_family="Volatility",
        hypothesis_text=(
            "Under range-bound or volatile but sideways conditions, we sell both "
            "an OTM put spread and an OTM call spread. Profits from time decay "
            "and volatility contraction as long as the spot price stays within "
            "the sold strikes."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["options", "iron-condor", "combo", "neutral", "theta"],
    )

    def __init__(
        self,
        rvol_ceiling: float = 1.3,
        max_efficiency: float = 0.55,
        spread_width_strikes: int = 2,
        target_r: float = 0.4,
        stop_r: float = -1.0,
    ):
        self.rvol_ceiling = rvol_ceiling
        self.max_efficiency = max_efficiency
        self.spread_width_strikes = spread_width_strikes
        self.target_r = target_r
        self.stop_r = stop_r

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 5:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            move_efficiency = snapshot.features.get_float("move_efficiency")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            current_time = snapshot.timestamp

            combo_type = "IRON_CONDOR"

            rejection_reasons: List[str] = []
            if rvol > self.rvol_ceiling:
                rejection_reasons.append("HIGH_RVOL")
            if move_efficiency > self.max_efficiency:
                rejection_reasons.append("HIGH_EFFICIENCY")
            if current_time.hour >= 14:  # cut off early, needs time to decay
                rejection_reasons.append("LATE_SESSION")
            if self.spread_width_strikes <= 0:
                rejection_reasons.append("ZERO_WIDTH")

            # Legs for Iron Condor:
            # Sell -1 Put, Buy -1-width Put
            # Sell +1 Call, Buy +1+width Call
            combo_legs = [
                {"option_type": "PE", "side": "SELL", "strikes_away": -1},
                {"option_type": "PE", "side": "BUY", "strikes_away": -1 - self.spread_width_strikes},
                {"option_type": "CE", "side": "SELL", "strikes_away": 1},
                {"option_type": "CE", "side": "BUY", "strikes_away": 1 + self.spread_width_strikes},
            ]

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_"
                f"{combo_type}_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            confidence = 0.5
            if accepted:
                range_factor = min(max(1.0 - (rvol / self.rvol_ceiling), 0.0), 1.0)
                eff_factor = min(max(1.0 - (move_efficiency / self.max_efficiency), 0.0), 1.0)
                confidence = round(0.5 + 0.25 * range_factor + 0.25 * eff_factor, 2)

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
                    "rvol": round(rvol, 2),
                    "atr": round(atr, 2),
                    "move_efficiency": round(move_efficiency, 3),
                    "spread_width_strikes": self.spread_width_strikes,
                },
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[IronCondorStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
