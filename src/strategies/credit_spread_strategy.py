#!/usr/bin/env python3
"""
Credit Spread Strategy — Bull Put Spread / Bear Call Spread (RANGE regime).
=============================================================================
Hypothesis: same PCR-extreme contrarian read `PCRExtremeReversalStrategy` acts
on (heavy put OI below spot = sellers don't expect a fall past that level =
bullish; symmetric for heavy call OI) is financed as a defined-risk credit
spread — sell the near OTM strike, buy a further OTM strike as protection —
instead of a directional long option. This is the theta-positive counterpart
to `vertical_spread_strategy.py`'s debit spreads: it collects premium when
the market is range-bound rather than paying for a directional move, so it's
only routed real capital in the RANGE regime category (see regime_router.py).

Like vertical_spread_strategy.py, this emits a `combo_legs` signal instead of
a single stop_loss/take_profit — see indian_trader.py's
_handle_combo_signal()/_enter_combo_position() for the separate execution and
PnL path (combined-premium R-multiples, not an index-price R-multiple).
"""

import logging
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.options_intelligence_engine import OptionsIntelligence

logger = logging.getLogger(__name__)


class CreditSpreadStrategy(BaseStrategy):
    """Bull Put Spread / Bear Call Spread — PCR-extreme thesis, credit-spread execution."""

    metadata = StrategyMetadata(
        id="credit_spread",
        name="Credit Spread (Bull Put / Bear Call)",
        hypothesis_id="credit_spread_pcr_fade",
        hypothesis_family="Directional Options Combo",
        hypothesis_text=(
            "An extreme Put-Call Ratio, confirmed by low RVOL and inefficient "
            "movement (range conditions, not a trend), is traded as a credit "
            "vertical spread (sell the near OTM strike, buy a further OTM "
            "strike as protection) to collect premium on the side price is "
            "unlikely to reach, rather than paying for a directional bet."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["options", "credit-spread", "combo", "mean-reversion", "theta"],
    )

    def __init__(
        self,
        rvol_ceiling: float = 1.3,
        max_efficiency: float = 0.55,
        spread_width_strikes: int = 2,
        target_r: float = 0.5,
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
            options: Optional[OptionsIntelligence] = getattr(snapshot.market, "options", None)
            if options is None:
                return self._empty_result(experiment_name, errors=["OPTIONS_DATA_MISSING"])
            if options.is_stale:
                return self._empty_result(experiment_name, errors=["OPTIONS_DATA_STALE"])
            if options.pcr_bias not in ("BULLISH", "BEARISH"):
                # NEUTRAL/UNKNOWN isn't a data problem — there's just no PCR
                # extreme to fade right now.
                return self._empty_result(experiment_name)

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

            # PCR_BULLISH -> heavy put OI below spot -> sell the downside
            # (bull put spread). PCR_BEARISH -> heavy call OI above spot ->
            # sell the upside (bear call spread).
            if options.pcr_bias == "BULLISH":
                combo_type = "BULL_PUT_SPREAD"
            else:
                combo_type = "BEAR_CALL_SPREAD"

            rejection_reasons: List[str] = []
            if rvol > self.rvol_ceiling:
                rejection_reasons.append("HIGH_RVOL")
            if move_efficiency > self.max_efficiency:
                rejection_reasons.append("HIGH_EFFICIENCY")
            if current_time.hour >= 15:
                rejection_reasons.append("LATE_SESSION")
            if self.spread_width_strikes <= 0:
                rejection_reasons.append("ZERO_WIDTH")

            if combo_type == "BULL_PUT_SPREAD":
                combo_legs = [
                    {"option_type": "PE", "side": "SELL", "strikes_away": -1},
                    {"option_type": "PE", "side": "BUY", "strikes_away": -1 - self.spread_width_strikes},
                ]
            else:
                combo_legs = [
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
                    "pcr": options.pcr,
                    "pcr_bias": options.pcr_bias,
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
            logger.error(f"[CreditSpreadStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
