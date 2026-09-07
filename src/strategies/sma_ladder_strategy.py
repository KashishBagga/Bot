#!/usr/bin/env python3
"""
SMA Ladder Break/Bounce Strategy
===================================
Hypothesis: the classic daily SMA ladder (20/50/100/150/200) still marks the
levels institutional/retail flow reacts to on an index, independent of any
5m/1h structure. Two setups off the NEAREST ladder rung to current price:
  - BOUNCE: price touches the rung from the side its own daily trend favors
    (price above the rung and daily trend up, or below it and down) and
    prints a reversal candle -> continuation in the trend direction.
  - BREAK: price decisively closes through the rung AGAINST the prevailing
    multi-rung alignment, with real body/volume -> trend-change signal.

Distinct from ema_pullback.py / htf_pullback_reversal.py (which read m5/1h
EMA20/50, a much faster mean) and cpr_strategy.py (prior-day pivot, not a
moving average at all) — this is the slow, purely daily-timeframe SMA ladder
gap in the framework. Self-contained: SMAs computed locally from
snapshot.d1, nothing added to the shared FeatureStore.
"""

import logging
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)

SMA_PERIODS = (20, 50, 100, 150, 200)


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


class SmaLadderStrategy(BaseStrategy):
    """Daily SMA(20/50/100/150/200) ladder — bounce in trend direction, or break against alignment."""

    metadata = StrategyMetadata(
        id="sma_ladder",
        name="SMA Ladder Break/Bounce",
        hypothesis_id="sma_ladder_break_bounce",
        hypothesis_family="Trend Continuation",
        hypothesis_text=(
            "Price touching the nearest daily SMA(20/50/100/150/200) rung "
            "either bounces in the direction that rung's own trend favors, "
            "or breaks through against the broader SMA alignment — both are "
            "real reactions to a slow, purely daily-timeframe level."
        ),
        version="v1.0",
        archetype="Trend-Continuation",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["sma", "daily", "ladder", "break", "bounce"],
    )

    def __init__(
        self,
        touch_tolerance_pct: float = 0.002,
        min_body_fraction: float = 0.40,
        atr_sl_buffer_mult: float = 0.3,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
    ):
        self.touch_tolerance_pct = touch_tolerance_pct
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr

    def thesis_key(self, signal: dict) -> tuple:
        return (
            signal.get("symbol", ""), "SMA_LADDER",
            signal.get("signal", ""), signal.get("diagnostics", {}).get("sma_period"),
        )

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            d1_df = snapshot.d1
            m5_df = snapshot.m5
            if d1_df is None or len(d1_df) < max(SMA_PERIODS) + 1 or m5_df is None or len(m5_df) < 5:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            daily_close = d1_df["close"]
            smas: Dict[int, float] = {
                p: float(daily_close.rolling(p).mean().iloc[-1]) for p in SMA_PERIODS
            }
            # Overall daily trend: how many rungs price sits above (bullish tilt).
            above_count = sum(1 for v in smas.values() if price > v)
            daily_trend_up = above_count >= 3  # majority of rungs below price

            # Nearest rung to current price.
            nearest_period, nearest_value = min(smas.items(), key=lambda kv: abs(price - kv[1]))
            touching = abs(price - nearest_value) / nearest_value <= self.touch_tolerance_pct
            if not touching:
                return self._empty_result(experiment_name)

            last_candle = m5_df.iloc[-1]

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            price_above_rung = price >= nearest_value
            if price_above_rung and daily_trend_up and _bullish_reversal(last_candle, self.min_body_fraction):
                setup_type = "SMA_LADDER_BOUNCE"
                side = "BUY CALL"
                sl = min(nearest_value - (atr * self.atr_sl_buffer_mult), price - (atr * 0.5))
                take_profit = price + (atr * self.tp_atr_cap)
            elif not price_above_rung and not daily_trend_up and _bearish_reversal(last_candle, self.min_body_fraction):
                setup_type = "SMA_LADDER_BOUNCE"
                side = "BUY PUT"
                sl = max(nearest_value + (atr * self.atr_sl_buffer_mult), price + (atr * 0.5))
                take_profit = price - (atr * self.tp_atr_cap)
            elif price_above_rung and not daily_trend_up and _bullish_reversal(last_candle, self.min_body_fraction):
                setup_type = "SMA_LADDER_BREAK"
                side = "BUY CALL"
                sl = min(nearest_value - (atr * self.atr_sl_buffer_mult), price - (atr * 0.5))
                take_profit = price + (atr * self.tp_atr_cap)
            elif not price_above_rung and daily_trend_up and _bearish_reversal(last_candle, self.min_body_fraction):
                setup_type = "SMA_LADDER_BREAK"
                side = "BUY PUT"
                sl = max(nearest_value + (atr * self.atr_sl_buffer_mult), price + (atr * 0.5))
                take_profit = price - (atr * self.tp_atr_cap)

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            current_time = snapshot.timestamp
            rejection_reasons: List[str] = []

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

            confidence = 0.5 if rejection_reasons else round(min(0.5 + 0.05 * above_count, 0.75), 2)

            diagnostics = {
                "sma_period": nearest_period,
                "sma_value": round(nearest_value, 2),
                "above_count": above_count,
                "daily_trend_up": daily_trend_up,
                "atr": round(atr, 2),
                "rr_ratio": rr,
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_SMA{nearest_period}_"
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
            logger.error(f"[SmaLadderStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
