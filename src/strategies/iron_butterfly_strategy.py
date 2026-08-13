#!/usr/bin/env python3
"""
Iron Butterfly Strategy — Neutral, defined-risk credit spread, ATM-centered.
==============================================================================
Hypothesis: same range-bound read as IronCondorStrategy, but sells the ATM
straddle directly (both short legs at strikes_away=0) instead of one strike
OTM each side. This collects far more credit (ATM options are always the
most expensive) at the cost of a much tighter profitable band and a bigger
per-lot max loss — a genuinely different risk/reward shape from the existing
IronCondorStrategy (wide, lower-credit, higher-probability-of-staying-in-range)
and the existing debit ButterflyStrategy (long-vol collapse bet, not a credit
theta-harvest). Fills a real gap: this system had no ATM-centered credit
structure before this.
"""

import logging
from datetime import timedelta
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class IronButterflyStrategy(BaseStrategy):
    """Iron Butterfly Strategy — ATM credit theta-harvest, defined risk."""

    metadata = StrategyMetadata(
        id="iron_butterfly",
        name="Iron Butterfly (ATM Credit)",
        hypothesis_id="iron_butterfly_neutral",
        hypothesis_family="Volatility",
        hypothesis_text=(
            "Under range-bound or compressed conditions, sell the ATM straddle "
            "(both a CE and a PE at the same strike) and buy protective wings "
            "further out for defined risk. Collects much more premium than an "
            "Iron Condor for the same wing width, but only profits within a "
            "tighter band around the short strike."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["options", "iron-butterfly", "combo", "neutral", "theta"],
    )

    def __init__(
        self,
        rvol_ceiling: float = 1.2,
        max_efficiency: float = 0.50,
        wing_width_strikes: int = 4,
        target_r: float = 0.35,
        stop_r: float = -1.0,
        loss_cooldown_minutes: float = 30.0,
    ):
        self.rvol_ceiling = rvol_ceiling
        self.max_efficiency = max_efficiency
        self.wing_width_strikes = wing_width_strikes
        self.target_r = target_r
        self.stop_r = stop_r
        self.loss_cooldown_minutes = loss_cooldown_minutes
        # Same lesson as ButterflyStrategy's cooldown — a tight ATM-centered
        # structure re-firing on the same strikes minutes after a stop-out
        # would just be re-entering the same losing read of the range.
        self._loss_cooldown_until: Dict[str, Any] = {}

    def notify_exit(self, symbol: str, pnl_r: float, timestamp) -> None:
        if pnl_r < 0:
            self._loss_cooldown_until[symbol] = timestamp + timedelta(minutes=self.loss_cooldown_minutes)

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

            combo_type = "IRON_BUTTERFLY"

            rejection_reasons: List[str] = []
            if rvol > self.rvol_ceiling:
                rejection_reasons.append("HIGH_RVOL")
            if move_efficiency > self.max_efficiency:
                rejection_reasons.append("HIGH_EFFICIENCY")
            if current_time.hour >= 14:  # cut off early, needs time to decay
                rejection_reasons.append("LATE_SESSION")
            if self.wing_width_strikes <= 0:
                rejection_reasons.append("ZERO_WIDTH")
            cooldown_until = self._loss_cooldown_until.get(snapshot.symbol)
            if cooldown_until is not None and current_time < cooldown_until:
                rejection_reasons.append("COOLDOWN_AFTER_LOSS")

            # Legs for Iron Butterfly: sell the ATM straddle, buy the wings.
            combo_legs = [
                {"option_type": "PE", "side": "SELL", "strikes_away": 0},
                {"option_type": "CE", "side": "SELL", "strikes_away": 0},
                {"option_type": "PE", "side": "BUY", "strikes_away": -self.wing_width_strikes},
                {"option_type": "CE", "side": "BUY", "strikes_away": self.wing_width_strikes},
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
                    "wing_width_strikes": self.wing_width_strikes,
                },
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[IronButterflyStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
