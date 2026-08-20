#!/usr/bin/env python3
"""
VWAP Reclaim Strategy
======================
Hypothesis: When price crosses back over intraday VWAP (having been on the
other side), that's a trend-continuation confirmation, not a mean-reversion
setup — the opposite thesis from vwap_reversion.py, which fades an
overstretched move back TOWARD VWAP. This strategy trades WITH the reclaim
direction, away from VWAP.

Distinguishing the two by name: "reversion" fades extension, "reclaim" trades
the break of the VWAP line itself as a continuation signal.
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class VwapReclaimStrategy(BaseStrategy):
    """VWAP Reclaim — trend continuation on a VWAP cross, not a fade."""

    metadata = StrategyMetadata(
        id="vwap_reclaim",
        name="VWAP Reclaim",
        hypothesis_id="vwap_reclaim_continuation",
        hypothesis_family="Trend Continuation",
        hypothesis_text=(
            "Price crossing back over intraday VWAP, aligned with daily bias and "
            "backed by RVOL, signals continuation in the reclaim direction — the "
            "opposite thesis from fading an overstretched VWAP deviation."
        ),
        version="v1.0",
        archetype="Trend-Continuation",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["vwap", "reclaim", "continuation", "trend"],
    )

    def __init__(
        self,
        rvol_threshold: float = 1.0,
        min_efficiency: float = 0.55,
        reclaim_buffer_atr_mult: float = 0.10,
    ):
        self.rvol_threshold = rvol_threshold
        self.min_efficiency = min_efficiency
        # Require the close to clear VWAP by a small ATR-scaled margin, not
        # just tick over it — avoids whipsaw right at the line.
        self.reclaim_buffer_atr_mult = reclaim_buffer_atr_mult

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 10:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            distance_to_vwap = snapshot.features.get_float("distance_to_vwap")
            move_efficiency = snapshot.features.get_float("move_efficiency")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            current_time = snapshot.timestamp

            last_candle = m5_df.iloc[-1]
            prev_candle = m5_df.iloc[-2]
            close = float(last_candle["close"])
            high = float(last_candle["high"])
            low = float(last_candle["low"])
            prev_close = float(prev_candle["close"])

            # distance_to_vwap = (price - vwap) / vwap  =>  vwap = price / (1 + distance_to_vwap)
            vwap = price / (1.0 + distance_to_vwap) if (1.0 + distance_to_vwap) > 0 else price
            buffer = atr * self.reclaim_buffer_atr_mult

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            # Bullish reclaim: previous close below VWAP, current close clears it upward
            if prev_close < vwap and close > (vwap + buffer):
                setup_type = "VWAP_RECLAIM"
                side = "BUY CALL"
                sl = min(low - (atr * 0.15), price - (atr * 0.5))
            # Bearish reclaim: previous close above VWAP, current close clears it downward
            elif prev_close > vwap and close < (vwap - buffer):
                setup_type = "VWAP_RECLAIM"
                side = "BUY PUT"
                sl = max(high + (atr * 0.15), price + (atr * 0.5))

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            rejection_reasons: List[str] = []

            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")

            # Continuation thesis — the reclaim must align with (or not oppose) daily bias.
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

            if move_efficiency < self.min_efficiency:
                rejection_reasons.append("LOW_EFFICIENCY")

            risk_dist = abs(price - sl) if sl else atr
            min_sl_dist = atr * 0.5
            if side == "BUY CALL" and (price - sl) < min_sl_dist:
                sl = price - min_sl_dist
                risk_dist = min_sl_dist
            elif side == "BUY PUT" and (sl - price) < min_sl_dist:
                sl = price + min_sl_dist
                risk_dist = min_sl_dist

            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")
                risk_dist = atr

            # Target: next opposing zone, floored at 2R (zone can only raise
            # the target, never lower it below the floor — same fix already
            # applied in orb.py/prev_day_extremes.py).
            tp_floor = (price + 2.0 * risk_dist) if side == "BUY CALL" else (price - 2.0 * risk_dist)
            tp_from_zone = None
            for z in (snapshot.h1_zones or []):
                if side == "BUY CALL" and z.level > price:
                    tp_from_zone = z.level
                    break
                if side == "BUY PUT" and z.level < price:
                    tp_from_zone = z.level
                    break

            if tp_from_zone is not None:
                take_profit = max(tp_floor, tp_from_zone) if side == "BUY CALL" else min(tp_floor, tp_from_zone)
            else:
                take_profit = tp_floor

            max_tp_dist = atr * 5.0
            if abs(take_profit - price) > max_tp_dist:
                take_profit = (price + max_tp_dist) if side == "BUY CALL" else (price - max_tp_dist)

            rr = round(abs(take_profit - price) / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < 1.5:
                rejection_reasons.append(f"LOW_RR:{rr}")

            confidence = 0.5
            if len(rejection_reasons) == 0:
                eff_factor = min(move_efficiency / 1.0, 1.0)
                rvol_factor = min(rvol / 1.5, 1.0)
                confidence = round(0.5 + 0.25 * eff_factor + 0.25 * rvol_factor, 2)

            diagnostics = {
                "vwap_price": round(vwap, 2),
                "distance_to_vwap": round(distance_to_vwap, 5),
                "reclaim_buffer": round(buffer, 2),
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "move_efficiency": round(move_efficiency, 3),
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_"
                f"VWAPRECLAIM_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            sig = {
                "symbol": snapshot.symbol,
                "signal": side,
                "strategy": setup_type,
                "price": price,
                "stop_loss": round(sl, 2),
                "take_profit": round(take_profit, 2),
                "tp1": round(price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5), 2),
                "rr_ratio": rr,
                "timestamp": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "candidate_id": candidate_id,
                "confidence": confidence,
                "diagnostics": diagnostics,
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[VwapReclaimStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
