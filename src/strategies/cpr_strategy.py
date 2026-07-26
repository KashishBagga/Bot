#!/usr/bin/env python3
"""
Central Pivot Range (CPR) Breakout Strategy
============================================
Hypothesis: The prior day's Central Pivot Range (Pivot, Top Central, Bottom
Central) marks the "fair value" zone institutions transacted around
yesterday. A confirmed break above TC or below BC, backed by volume, signals
a directional move away from that value area — popular among Indian
intraday/index traders specifically (CPR isn't in this codebase's existing
strategy set, unlike most of the generic ORB/VWAP/pullback playbook).

CPR formulas (standard floor-trader pivot geometry):
    Pivot (P)     = (PrevHigh + PrevLow + PrevClose) / 3
    Bottom Central (BC) = (PrevHigh + PrevLow) / 2
    Top Central (TC)    = 2*P - BC
(TC/BC are swapped if TC < BC, which can happen on some pivot geometries.)

"Virgin CPR" (price hasn't touched the CPR band yet today) is tracked as a
diagnostic, not a hard filter — consistent with how zone freshness is tracked
elsewhere without gating every strategy on it.
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class CprStrategy(BaseStrategy):
    """Central Pivot Range breakout strategy."""

    metadata = StrategyMetadata(
        id="cpr",
        name="Central Pivot Range Breakout",
        hypothesis_id="cpr_breakout",
        hypothesis_family="Pivot Breakout",
        hypothesis_text=(
            "A confirmed break of the prior day's Central Pivot Range (TC/BC), "
            "backed by RVOL and aligned with daily bias, signals a directional "
            "move away from yesterday's value area."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["cpr", "pivot", "breakout", "india"],
    )

    def __init__(
        self,
        rvol_threshold: float = 1.1,
        min_efficiency: float = 0.55,
    ):
        self.rvol_threshold = rvol_threshold
        self.min_efficiency = min_efficiency

    def _prev_day_ohlc(self, d1_df, current_date):
        """Same 'is the last d1 row today's still-forming bar' check used in
        IndicatorPipeline's dist_prev_high/low computation — replicated here
        rather than added as a shared feature, since only this strategy needs
        the raw prev close (not just distance ratios)."""
        last_daily_date = d1_df.index[-1].date()
        if last_daily_date == current_date:
            row = d1_df.iloc[-2]
        else:
            row = d1_df.iloc[-1]
        return float(row["high"]), float(row["low"]), float(row["close"])

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            d1_df = snapshot.d1
            if m5_df is None or len(m5_df) < 10 or d1_df is None or len(d1_df) < 2:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            move_efficiency = snapshot.features.get_float("move_efficiency")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            current_time = snapshot.timestamp
            today_date = current_time.date()

            prev_high, prev_low, prev_close = self._prev_day_ohlc(d1_df, today_date)

            pivot = (prev_high + prev_low + prev_close) / 3.0
            bc = (prev_high + prev_low) / 2.0
            tc = (2.0 * pivot) - bc
            if tc < bc:
                tc, bc = bc, tc
            cpr_width = abs(tc - bc)

            today_mask = m5_df.index.date == today_date
            today_m5 = m5_df[today_mask]
            if len(today_m5) < 2:
                return self._empty_result(experiment_name)

            last_candle = today_m5.iloc[-1]
            close = float(last_candle["close"])
            high = float(last_candle["high"])
            low = float(last_candle["low"])
            prev_candle = today_m5.iloc[-2]
            prev_candle_close = float(prev_candle["close"])

            # Virgin CPR: has any earlier candle today already traded inside/through the band?
            earlier = today_m5.iloc[:-1]
            virgin_cpr = True
            if len(earlier) > 0:
                touched = ((earlier["low"] <= tc) & (earlier["high"] >= bc)).any()
                virgin_cpr = not bool(touched)

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            if prev_candle_close <= tc and close > tc:
                setup_type = "CPR_BREAKOUT"
                side = "BUY CALL"
                sl = min(tc - (atr * 0.3), price - (atr * 0.5))
            elif prev_candle_close >= bc and close < bc:
                setup_type = "CPR_BREAKOUT"
                side = "BUY PUT"
                sl = max(bc + (atr * 0.3), price + (atr * 0.5))

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            rejection_reasons: List[str] = []

            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")
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
                narrow_cpr_factor = 1.0 if cpr_width < atr else max(0.0, 1.0 - (cpr_width - atr) / (2 * atr))
                virgin_factor = 1.0 if virgin_cpr else 0.6
                confidence = round(min(0.5 + 0.25 * narrow_cpr_factor + 0.2 * virgin_factor, 0.95), 2)

            diagnostics = {
                "pivot": round(pivot, 2),
                "tc": round(tc, 2),
                "bc": round(bc, 2),
                "cpr_width": round(cpr_width, 2),
                "cpr_width_vs_atr": round(cpr_width / atr, 3) if atr > 0 else None,
                "virgin_cpr": virgin_cpr,
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "move_efficiency": round(move_efficiency, 3),
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_"
                f"CPR_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
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
            logger.error(f"[CprStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
