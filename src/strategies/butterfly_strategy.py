#!/usr/bin/env python3
"""
Butterfly Strategy — Neutral, defined-risk debit spread.
==========================================================
Hypothesis: range-bound conditions (low RVOL, low efficiency, low ATR percentile)
make it highly probable that the spot price will remain near the center strike.
A butterfly spread buys 1 ITM leg, sells 2 ATM legs, and buys 1 OTM leg.
Max profit is realized if the underlying asset expires exactly at the sold center strike.
"""

import logging
from datetime import timedelta
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class ButterflyStrategy(BaseStrategy):
    """Butterfly Strategy — defined-risk neutral debit play."""

    metadata = StrategyMetadata(
        id="butterfly",
        name="Butterfly Spread (Sideways/Range)",
        hypothesis_id="butterfly_neutral",
        hypothesis_family="Volatility",
        hypothesis_text=(
            "Under range-bound or low-volatility conditions, a long butterfly spread "
            "(buy ITM, sell 2x ATM, buy OTM) offers a high reward-to-risk ratio "
            "with defined risk if price remains close to the sold strike."
        ),
        version="v1.0",
        archetype="Theta-Harvest",
        exit_profile="PREMIUM_TARGET_R",
        maturity="RESEARCH",
        tags=["options", "butterfly", "combo", "neutral", "decay"],
    )

    def __init__(
        self,
        rvol_ceiling: float = 1.3,
        max_efficiency: float = 0.55,
        wing_width_strikes: int = 2,
        target_r: float = 1.5,
        stop_r: float = -0.5,
        loss_cooldown_minutes: float = 30.0,
    ):
        self.rvol_ceiling = rvol_ceiling
        self.max_efficiency = max_efficiency
        self.wing_width_strikes = wing_width_strikes
        self.target_r = target_r
        self.stop_r = stop_r
        self.loss_cooldown_minutes = loss_cooldown_minutes
        # Same strikes kept re-firing every 5-10min inside chop right after a
        # stop-out (13 fires in one session on Aug 13, 25% win rate) — suppress
        # re-entry on a symbol for a while after a loss. Wins re-enter immediately;
        # only losses trigger the cooldown.
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

            combo_type = "BUTTERFLY_SPREAD"

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

            # Butterfly Spread Legs:
            # Buy 1 ITM Call (-wing_width_strikes)
            # Sell 2 ATM Calls (0)
            # Buy 1 OTM Call (+wing_width_strikes)
            combo_legs = [
                {"option_type": "CE", "side": "BUY", "strikes_away": -self.wing_width_strikes},
                {"option_type": "CE", "side": "SELL", "strikes_away": 0},
                {"option_type": "CE", "side": "SELL", "strikes_away": 0},
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
            logger.error(f"[ButterflyStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
