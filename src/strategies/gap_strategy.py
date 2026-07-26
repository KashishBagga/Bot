#!/usr/bin/env python3
"""
Gap Strategy — Gap-and-Go (continuation) and Gap-Fill (reversion), as one
regime-gated setup rather than two independent contradictory ones.
============================================================================
A gap can resolve two opposite ways and a naive playbook that lists both as
separate "always-on" strategies has no way to pick between them. This
strategy instead watches how price behaves in the opening window AFTER the
gap, every candle, and classifies which regime is currently playing out:

    GAP_AND_GO — price is extending further away from today's open, in the
                 gap direction (continuation). Confirmed by RVOL + efficiency.
    GAP_FILL   — price is giving back the gap, heading back toward yesterday's
                 close, but hasn't reached it yet (so there's still a target
                 ahead, not a completed move being chased).

Only evaluated within `decision_window_minutes` of the open — a gap thesis
past that point is stale; ORB/structural strategies take over from there.
"""

import logging
from datetime import time
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class GapStrategy(BaseStrategy):
    """Gap-and-Go / Gap-Fill, resolved dynamically rather than as two fixed strategies."""

    metadata = StrategyMetadata(
        id="gap",
        name="Gap Continuation / Fill",
        hypothesis_id="gap_go_or_fill",
        hypothesis_family="Gap",
        hypothesis_text=(
            "An opening gap resolves as continuation (extension away from the "
            "open, backed by RVOL/efficiency) or as a fill (retracement back "
            "toward yesterday's close) — which one is decided by price action "
            "after the open, not assumed in advance."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["gap", "continuation", "reversion", "opening"],
    )

    def __init__(
        self,
        gap_threshold_pct: float = 0.15,
        rvol_threshold: float = 1.1,
        min_efficiency: float = 0.55,
        decision_window_minutes: int = 45,
    ):
        self.gap_threshold_pct = gap_threshold_pct
        self.rvol_threshold = rvol_threshold
        self.min_efficiency = min_efficiency
        self.decision_window_minutes = decision_window_minutes

    def _prev_close(self, d1_df, current_date) -> float:
        last_daily_date = d1_df.index[-1].date()
        row = d1_df.iloc[-2] if last_daily_date == current_date else d1_df.iloc[-1]
        return float(row["close"])

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            d1_df = snapshot.d1
            if m5_df is None or len(m5_df) < 5 or d1_df is None or len(d1_df) < 2:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            current_time = snapshot.timestamp
            today_date = current_time.date()

            # Decision window: only near the open — a stale gap thesis late in
            # the day is someone else's job (ORB/structural).
            session_open_dt = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            minutes_since_open = (current_time - session_open_dt).total_seconds() / 60.0
            if minutes_since_open < 5 or minutes_since_open > self.decision_window_minutes:
                return self._empty_result(experiment_name)

            today_mask = m5_df.index.date == today_date
            today_m5 = m5_df[today_mask]
            if len(today_m5) < 2:
                return self._empty_result(experiment_name)

            today_open = float(today_m5.iloc[0]["open"])
            prev_close = self._prev_close(d1_df, today_date)
            if prev_close <= 0:
                return self._empty_result(experiment_name)

            gap_pct = (today_open - prev_close) / prev_close * 100.0
            if abs(gap_pct) < self.gap_threshold_pct:
                return self._empty_result(experiment_name, diagnostics={"gap_pct": round(gap_pct, 3)})

            close = float(today_m5.iloc[-1]["close"])
            move_efficiency = snapshot.features.get_float("move_efficiency")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            extension_buffer = atr * 0.10

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            if gap_pct > 0:  # gap up
                if close > today_open + extension_buffer:
                    setup_type = "GAP_AND_GO"
                    side = "BUY CALL"
                    sl = min(today_open - (atr * 0.15), price - (atr * 0.5))
                elif close < today_open - extension_buffer and close > prev_close:
                    setup_type = "GAP_FILL"
                    side = "BUY PUT"
                    sl = max(today_open + (atr * 0.15), price + (atr * 0.5))
                    take_profit = prev_close
            else:  # gap down
                if close < today_open - extension_buffer:
                    setup_type = "GAP_AND_GO"
                    side = "BUY PUT"
                    sl = max(today_open + (atr * 0.15), price + (atr * 0.5))
                elif close > today_open + extension_buffer and close < prev_close:
                    setup_type = "GAP_FILL"
                    side = "BUY CALL"
                    sl = min(today_open - (atr * 0.15), price - (atr * 0.5))
                    take_profit = prev_close

            if setup_type == "NONE":
                return self._empty_result(experiment_name, diagnostics={"gap_pct": round(gap_pct, 3)})

            rejection_reasons: List[str] = []

            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")
            if setup_type == "GAP_AND_GO" and move_efficiency < self.min_efficiency:
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

            if setup_type == "GAP_AND_GO":
                # Continuation: next opposing zone, floored at 2R.
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
            # else: GAP_FILL already set take_profit = prev_close above

            max_tp_dist = atr * 5.0
            if abs(take_profit - price) > max_tp_dist:
                take_profit = (price + max_tp_dist) if side == "BUY CALL" else (price - max_tp_dist)

            rr = round(abs(take_profit - price) / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < 1.5:
                rejection_reasons.append(f"LOW_RR:{rr}")

            confidence = 0.5
            if len(rejection_reasons) == 0:
                rvol_factor = min(rvol / 2.0, 1.0)
                eff_factor = min(move_efficiency / 1.0, 1.0)
                confidence = round(0.5 + 0.25 * rvol_factor + 0.25 * eff_factor, 2)

            diagnostics = {
                "gap_pct": round(gap_pct, 3),
                "today_open": round(today_open, 2),
                "prev_close": round(prev_close, 2),
                "minutes_since_open": round(minutes_since_open, 1),
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "move_efficiency": round(move_efficiency, 3),
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_"
                f"{setup_type}_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
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
            logger.error(f"[GapStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
