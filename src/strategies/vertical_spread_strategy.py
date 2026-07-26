#!/usr/bin/env python3
"""
Vertical Spread Strategy — Bull Call Spread / Bear Put Spread
================================================================
Hypothesis: same directional thesis as any other trend strategy in this
framework (EMA20/50 cross aligned with daily bias, confirmed by RVOL and an
efficient move) — but financed as a debit spread (buy ATM, sell a further
OTM strike in the same direction) instead of a naked single-leg option.
This is the one combo shape that's still fully directional and reuses the
existing structural thesis, just with cheaper premium / defined max profit.

Unlike every other strategy in this framework, this one emits a `combo_legs`
signal instead of a single stop_loss/take_profit — see indian_trader.py's
_handle_combo_signal()/_enter_combo_position() for the separate execution
and PnL path (combined-premium R-multiples, not an index-price R-multiple).
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class VerticalSpreadStrategy(BaseStrategy):
    """Bull Call Spread / Bear Put Spread — directional thesis, debit-spread execution."""

    metadata = StrategyMetadata(
        id="vertical_spread",
        name="Vertical Spread (Bull Call / Bear Put)",
        hypothesis_id="vertical_spread_trend",
        hypothesis_family="Directional Options Combo",
        hypothesis_text=(
            "An EMA20/50 cross aligned with daily bias, backed by RVOL and an "
            "efficient move, is traded as a debit vertical spread (buy ATM, sell "
            "a further OTM strike) rather than a naked single-leg option — same "
            "directional thesis, cheaper premium, defined max profit."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["options", "vertical-spread", "combo", "directional"],
    )

    def __init__(
        self,
        rvol_threshold: float = 1.0,
        min_efficiency: float = 0.55,
        spread_width_strikes: int = 2,
        target_r: float = 1.0,
        stop_r: float = -0.6,
    ):
        self.rvol_threshold = rvol_threshold
        self.min_efficiency = min_efficiency
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

            ema_bullish = snapshot.features.get_bool("ema_bullish")
            move_efficiency = snapshot.features.get_float("move_efficiency")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            current_time = snapshot.timestamp

            combo_type = None
            if ema_bullish and snapshot.daily_bias != "BEARISH":
                combo_type = "BULL_CALL_SPREAD"
            elif not ema_bullish and snapshot.daily_bias != "BULLISH":
                combo_type = "BEAR_PUT_SPREAD"

            if combo_type is None:
                return self._empty_result(experiment_name, diagnostics={"reason": "no_ema_bias_alignment"})

            rejection_reasons: List[str] = []
            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")
            if move_efficiency < self.min_efficiency:
                rejection_reasons.append("LOW_EFFICIENCY")
            if current_time.hour >= 15:
                rejection_reasons.append("LATE_SESSION")

            if combo_type == "BULL_CALL_SPREAD":
                combo_legs = [
                    {"option_type": "CE", "side": "BUY", "strikes_away": 0},
                    {"option_type": "CE", "side": "SELL", "strikes_away": self.spread_width_strikes},
                ]
            else:
                combo_legs = [
                    {"option_type": "PE", "side": "BUY", "strikes_away": 0},
                    {"option_type": "PE", "side": "SELL", "strikes_away": -self.spread_width_strikes},
                ]

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_"
                f"{combo_type}_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            confidence = 0.5
            if accepted:
                eff_factor = min(move_efficiency / 1.0, 1.0)
                rvol_factor = min(rvol / 1.5, 1.0)
                confidence = round(0.5 + 0.25 * eff_factor + 0.25 * rvol_factor, 2)

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
                    "ema_bullish": ema_bullish,
                    "daily_bias": snapshot.daily_bias,
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
            logger.error(f"[VerticalSpreadStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
