#!/usr/bin/env python3
"""
RSI-2 Mean Reversion Strategy
==============================
Hypothesis: a genuine short-term momentum exhaustion (Larry Connors-style
2-period RSI at a real extreme, confirmed by a reversal candle) tends to
snap back toward the short-term mean. RSI-14 (already computed elsewhere in
this system) rarely swings past 20/80 on 5m index bars — it's built for
trend/momentum reads, not this. RSI-2 genuinely reaches <10 / >90 on ordinary
bars, which is what this edge needs.

This fills a real gap: every existing RANGE/COMPRESSION strategy in this
system reads VWAP-distance or OI/PCR extremes, not a plain price oscillator —
so this gives the RANGE bucket a signal basis uncorrelated with the rest of
it (if the regime classifier misreads RANGE, the existing bucket mostly fails
together since they're all reading similar structural cues; this one isn't).
"""

import logging
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


def _bullish_reversal_body(candle, min_body_fraction: float) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c > o and (body / candle_range) >= min_body_fraction


def _bearish_reversal_body(candle, min_body_fraction: float) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c < o and (body / candle_range) >= min_body_fraction


class Rsi2MeanReversionStrategy(BaseStrategy):
    """RSI-2 extreme fade — short-term mean reversion, confirmed by a reversal candle."""

    metadata = StrategyMetadata(
        id="rsi2_mean_reversion",
        name="RSI-2 Mean Reversion",
        hypothesis_id="rsi2_extreme_fade",
        hypothesis_family="Mean Reversion",
        hypothesis_text=(
            "A 2-period RSI reaching a genuine extreme (<10 or >90), confirmed "
            "by a reversal candle with a real body (not a doji), tends to snap "
            "back toward the short-term mean (EMA20) rather than continue."
        ),
        version="v1.0",
        archetype="Mean-Reversion",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["mean_reversion", "rsi", "oscillator"],
    )

    def __init__(
        self,
        rsi_oversold: float = 10.0,
        rsi_overbought: float = 90.0,
        min_body_fraction: float = 0.40,
        atr_sl_buffer_mult: float = 0.15,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
        rvol_ceiling: float = 1.5,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        self.rvol_ceiling = rvol_ceiling

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
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            rsi2 = snapshot.features.get_float("rsi2")
            ema20 = snapshot.features.get_float("ema20")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            current_time = snapshot.timestamp
            last_candle = m5_df.iloc[-1]

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            if rsi2 <= self.rsi_oversold and _bullish_reversal_body(last_candle, self.min_body_fraction):
                setup_type = "RSI2_OVERSOLD_BOUNCE"
                side = "BUY CALL"
                low = float(last_candle["low"])
                sl = min(low - (atr * self.atr_sl_buffer_mult), price - (atr * 0.5))
                take_profit = ema20
            elif rsi2 >= self.rsi_overbought and _bearish_reversal_body(last_candle, self.min_body_fraction):
                setup_type = "RSI2_OVERBOUGHT_FADE"
                side = "BUY PUT"
                high = float(last_candle["high"])
                sl = max(high + (atr * self.atr_sl_buffer_mult), price + (atr * 0.5))
                take_profit = ema20

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            rejection_reasons: List[str] = []

            # A genuine reversion setup happens in quiet/choppy conditions, not a
            # high-participation breakout day — very high RVOL at an RSI extreme
            # is more likely real continuation than exhaustion.
            if rvol > self.rvol_ceiling:
                rejection_reasons.append("HIGH_RVOL")

            # Don't fade an extreme against a strong daily bias — same rule
            # VwapReversionStrategy already uses for the same reason.
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

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

            # If EMA20 is already behind price in the trade direction (no room
            # to revert to), there's no real target — reject rather than
            # project a fake one that would never resolve as intended.
            reward = (take_profit - price) if side == "BUY CALL" else (price - take_profit)
            if reward <= 0:
                rejection_reasons.append("NO_REVERSION_ROOM")
                reward = 0.0

            max_tp_dist = atr * self.tp_atr_cap
            if reward > max_tp_dist:
                take_profit = (price + max_tp_dist) if side == "BUY CALL" else (price - max_tp_dist)
                reward = max_tp_dist

            rr = round(reward / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < self.min_rr:
                rejection_reasons.append("LOW_RR")

            confidence = 0.5
            if len(rejection_reasons) == 0:
                extreme_depth = (self.rsi_oversold - rsi2) if side == "BUY CALL" else (rsi2 - self.rsi_overbought)
                confidence = round(min(0.6 + 0.04 * max(extreme_depth, 0.0), 0.95), 2)

            diagnostics = {
                "rsi2": round(rsi2, 2),
                "ema20": round(ema20, 2),
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "rr_ratio": rr,
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_RSI2_"
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
            logger.error(f"[Rsi2MeanReversionStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
