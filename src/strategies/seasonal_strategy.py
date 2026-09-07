#!/usr/bin/env python3
"""
Seasonal (Calendar Bias) Strategy
===================================
Hypothesis: "Sell in May and go away" — Indian indices, like most equity
markets, have historically shown weaker average returns May-October than
November-April. Rather than trade the calendar alone (no intraday trigger),
this strategy uses the calendar month purely as a DIRECTIONAL FILTER on top
of a plain EMA20 pullback/rally trigger: only take the trade the seasonal
bias agrees with. Nov-Apr → only CALL pullback-bounces. May-Oct → only PUT
rally-fades. This is deliberately the loosest possible trigger (a simple
EMA20 touch + reversal candle) since the calendar is doing the direction
call, not the trigger.

Distinct from every other strategy in this system in one dimension: none of
them read wall-clock month. Self-contained — computes everything from
snapshot.m5 + snapshot.timestamp.month, no shared mutable state.
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)

# Nov-Apr: seasonally favorable months (bullish bias). May-Oct: seasonally
# weak months (bearish bias) — the "Sell in May" half of the year.
BULLISH_SEASON_MONTHS = {11, 12, 1, 2, 3, 4}


def _bullish_reversal(candle, min_body_fraction: float) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    rng = h - l
    if rng < 1e-9:
        return False
    return c > o and (abs(c - o) / rng) >= min_body_fraction


def _bearish_reversal(candle, min_body_fraction: float) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    rng = h - l
    if rng < 1e-9:
        return False
    return c < o and (abs(c - o) / rng) >= min_body_fraction


class SeasonalStrategy(BaseStrategy):
    """Calendar-month directional bias + EMA20 touch/reversal trigger."""

    metadata = StrategyMetadata(
        id="seasonal",
        name="Seasonal Calendar Bias",
        hypothesis_id="sell_in_may",
        hypothesis_family="Seasonal",
        hypothesis_text=(
            "Nov-Apr is seasonally bullish, May-Oct seasonally bearish for "
            "Indian indices — only take EMA20 touch/reversal setups whose "
            "direction agrees with the current calendar half."
        ),
        version="v1.0",
        archetype="Mean-Reversion",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["seasonal", "calendar", "ema"],
    )

    def __init__(
        self,
        ema_touch_tolerance_pct: float = 0.0015,
        min_body_fraction: float = 0.40,
        atr_sl_buffer_mult: float = 0.4,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
    ):
        self.ema_touch_tolerance_pct = ema_touch_tolerance_pct
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr

    def thesis_key(self, signal: dict) -> tuple:
        return (signal.get("symbol", ""), "SEASONAL", signal.get("signal", ""))

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 20:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            ema20 = snapshot.features.get_float("ema20")
            if atr <= 0 or ema20 <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr_or_ema20"])

            current_time = snapshot.timestamp
            month = current_time.month
            bullish_season = month in BULLISH_SEASON_MONTHS

            last_candle = m5_df.iloc[-1]
            touching_ema = abs(price - ema20) / ema20 <= self.ema_touch_tolerance_pct

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            if touching_ema and bullish_season and _bullish_reversal(last_candle, self.min_body_fraction):
                setup_type = "SEASONAL_BULLISH_BOUNCE"
                side = "BUY CALL"
                low = float(last_candle["low"])
                sl = min(low - (atr * self.atr_sl_buffer_mult), price - (atr * 0.5))
                take_profit = price + (atr * self.tp_atr_cap)
            elif touching_ema and not bullish_season and _bearish_reversal(last_candle, self.min_body_fraction):
                setup_type = "SEASONAL_BEARISH_FADE"
                side = "BUY PUT"
                high = float(last_candle["high"])
                sl = max(high + (atr * self.atr_sl_buffer_mult), price + (atr * 0.5))
                take_profit = price - (atr * self.tp_atr_cap)

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            rejection_reasons: List[str] = []

            # Seasonal bias is a soft directional prior, not a hard structural
            # read — still reject if it fights the (much more reliable)
            # intraday daily bias outright rather than blindly overriding it.
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

            risk_dist = abs(price - sl)
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price - min_sl_dist if side == "BUY CALL" else price + min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")

            reward = abs(take_profit - price)
            rr = round(reward / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < self.min_rr:
                rejection_reasons.append("LOW_RR")

            confidence = 0.5 if len(rejection_reasons) else 0.55
            diagnostics = {
                "month": month,
                "bullish_season": bullish_season,
                "ema20": round(ema20, 2),
                "atr": round(atr, 2),
                "rr_ratio": rr,
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_SEASONAL_"
                f"{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            sig = {
                "symbol": snapshot.symbol,
                "signal": side,
                "strategy": setup_type,
                "price": price,
                "stop_loss": sl,
                "take_profit": take_profit,
                "tp1": price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5),
                "rr_ratio": rr,
                "timestamp": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "features": snapshot.features.to_dict(),
                "candidate_id": candidate_id,
                "confidence": confidence,
                "diagnostics": diagnostics,
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[SeasonalStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
