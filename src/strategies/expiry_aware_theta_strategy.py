#!/usr/bin/env python3
"""
Expiry-Aware Theta Strategy — Iron Condor whose own risk parameters adapt to
time-to-expiry, instead of a fixed wing width used the same way every day.
=============================================================================
Hypothesis: this system's existing credit structures (IronCondor, CreditSpread,
IronButterfly) use the same wing width and RVOL ceiling regardless of how much
time is actually left on the option — but gamma risk and theta reward both
scale with time-to-expiry, and in opposite directions. Far from expiry, gamma
risk is low and there's runway to adjust, so tighter wings and a looser RVOL
ceiling are fine. Close to expiry, gamma risk is highest (a same-day move
hurts far more) even though theta reward is also highest — so this strategy
gets MORE selective (wider wings for protection, tighter RVOL ceiling) as
expiry approaches, not less.

Important: this does NOT gate which days/windows the strategy is allowed to
trade — it evaluates every candle like every other strategy and can fire any
day. Time-to-expiry only continuously reshapes its own wing width / RVOL
ceiling / target, the same way every other strategy here already reshapes its
own SL/TP from ATR. "Mindful, not gated."
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


def _lerp(near_value: float, far_value: float, tte_days: float, window_days: float = 4.0) -> float:
    """Linear blend between "near expiry" and "far from expiry" behavior.
    progress=0 at tte_days=0 (expiry now), progress=1 at tte_days>=window_days."""
    progress = min(max(tte_days / window_days, 0.0), 1.0)
    return near_value + (far_value - near_value) * progress


class ExpiryAwareThetaStrategy(BaseStrategy):
    """Iron Condor whose wing width / RVOL ceiling / target scale continuously with time-to-expiry."""

    metadata = StrategyMetadata(
        id="expiry_aware_theta",
        name="Expiry-Aware Theta (Adaptive Iron Condor)",
        hypothesis_id="theta_tte_adaptive",
        hypothesis_family="Volatility",
        hypothesis_text=(
            "An Iron Condor whose wing width and acceptance thresholds scale "
            "continuously with time-to-expiry — wider and more selective near "
            "expiry (gamma risk is highest there), tighter and more permissive "
            "far from expiry (more runway, less gamma risk) — rather than using "
            "one fixed configuration regardless of where we are in the cycle."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["options", "iron-condor", "combo", "theta", "expiry-aware"],
    )

    def __init__(
        self,
        rvol_ceiling_far: float = 1.4,
        rvol_ceiling_near: float = 0.8,
        max_efficiency: float = 0.55,
        wing_width_far: int = 2,
        wing_width_near: int = 5,
        target_r_far: float = 0.5,
        target_r_near: float = 0.25,
        stop_r: float = -1.0,
        tte_window_days: float = 4.0,
    ):
        self.rvol_ceiling_far = rvol_ceiling_far
        self.rvol_ceiling_near = rvol_ceiling_near
        self.max_efficiency = max_efficiency
        self.wing_width_far = wing_width_far
        self.wing_width_near = wing_width_near
        self.target_r_far = target_r_far
        self.target_r_near = target_r_near
        self.stop_r = stop_r
        self.tte_window_days = tte_window_days
        # Lazy-init to avoid circular imports / a DB connection at construction
        # time — same pattern as PreMarketCollector._get_db().
        self._db = None
        self._expiry_resolver = None

    def _get_expiry_resolver(self):
        if self._expiry_resolver is None:
            from src.models.postgres_database import PostgresDatabase
            from src.core.options_execution_engine import ExpiryResolver
            self._db = PostgresDatabase()
            self._expiry_resolver = ExpiryResolver(self._db)
        return self._expiry_resolver

    def _time_to_expiry_days(self, symbol: str, now: datetime) -> float:
        """Days remaining to this symbol's active expiry, via the same
        DB-backed ExpiryResolver the real execution path uses (not the
        options_mapper.get_expiry_datetime import OptionsScalpingStrategy
        relies on — that function doesn't actually exist in options_mapper.py,
        so that strategy's TTE always silently falls back to 7 days; worth
        fixing separately). Falls back to a mid-cycle assumption (3 days) if
        expiry data isn't resolvable — a missing expiry lookup shouldn't
        block the whole strategy."""
        try:
            resolver = self._get_expiry_resolver()
            expiry_code = resolver.get_active_expiry(symbol)
            expiry_date = resolver.parse_expiry_to_date(expiry_code)
            if expiry_date is None:
                return 3.0
            return max((expiry_date - now.date()).days, 0.0)
        except Exception as e:
            logger.debug(f"[ExpiryAwareThetaStrategy] TTE resolution failed for {symbol}: {e}")
            return 3.0

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

            tte_days = self._time_to_expiry_days(snapshot.symbol, current_time)
            rvol_ceiling = _lerp(self.rvol_ceiling_near, self.rvol_ceiling_far, tte_days, self.tte_window_days)
            wing_width_strikes = round(_lerp(self.wing_width_near, self.wing_width_far, tte_days, self.tte_window_days))
            target_r = _lerp(self.target_r_near, self.target_r_far, tte_days, self.tte_window_days)
            # Far from expiry there's genuinely no time value left to harvest
            # this late in the session — the position would just carry
            # overnight decay risk with no edge captured today. Close to
            # expiry, the final hours ARE the edge (accelerated theta into
            # the close), so don't cut those off.
            late_session_cutoff_hour = 14 if tte_days > 1.0 else 15

            combo_type = "IRON_CONDOR"

            rejection_reasons: List[str] = []
            if rvol > rvol_ceiling:
                rejection_reasons.append("HIGH_RVOL")
            if move_efficiency > self.max_efficiency:
                rejection_reasons.append("HIGH_EFFICIENCY")
            if current_time.hour >= late_session_cutoff_hour:
                rejection_reasons.append("LATE_SESSION")
            if wing_width_strikes <= 0:
                rejection_reasons.append("ZERO_WIDTH")

            combo_legs = [
                {"option_type": "PE", "side": "SELL", "strikes_away": -1},
                {"option_type": "PE", "side": "BUY", "strikes_away": -1 - wing_width_strikes},
                {"option_type": "CE", "side": "SELL", "strikes_away": 1},
                {"option_type": "CE", "side": "BUY", "strikes_away": 1 + wing_width_strikes},
            ]

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_"
                f"{combo_type}_TTE_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            confidence = 0.5
            if accepted:
                range_factor = min(max(1.0 - (rvol / rvol_ceiling), 0.0), 1.0) if rvol_ceiling > 0 else 0.0
                eff_factor = min(max(1.0 - (move_efficiency / self.max_efficiency), 0.0), 1.0)
                confidence = round(0.5 + 0.25 * range_factor + 0.25 * eff_factor, 2)

            sig = {
                "symbol": snapshot.symbol,
                "signal": combo_type,
                "strategy": combo_type,
                "price": price,
                "combo_legs": combo_legs,
                "target_r": round(target_r, 3),
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
                    "wing_width_strikes": wing_width_strikes,
                    "tte_days": round(tte_days, 2),
                    "rvol_ceiling_used": round(rvol_ceiling, 2),
                },
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[ExpiryAwareThetaStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
