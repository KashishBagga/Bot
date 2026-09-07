#!/usr/bin/env python3
"""
Wedge Compression Breakout Strategy
======================================
Hypothesis: a genuine wedge — successive swing highs falling AND successive
swing lows rising over the same window (converging trendlines, range
shrinking bar over bar) — represents a squeeze with directional bias baked
in (a rising-lows wedge into resistance is more likely to break up; a
falling-highs wedge into support more likely to break down). This is a
distinct geometric pattern from atr_squeeze.py's ATR-percentile compression
(a volatility-percentile read with no shape requirement) and
consolidation_breakout_strategy.py's touch-count box (a horizontal range,
not a converging one) — a wedge can compress while ATR percentile stays
completely normal.

Self-contained: swing highs/lows and convergence are computed locally from
snapshot.m5's own high/low columns using a simple fractal window, not via
StructureEngine (avoids coupling to its full swing-lifecycle machinery for
what's just a two-window high/low comparison here).
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


def _wedge_convergence(m5_df, window: int) -> Optional[Tuple[str, float, float]]:
    """Compares the most recent `window` bars against the `window` bars before
    that. Returns (wedge_direction, recent_high, recent_low) if highs are
    falling AND lows are rising (bar-range genuinely converging), else None.
    wedge_direction is 'RISING_LOWS' (bullish bias) or 'FALLING_HIGHS' — here
    both conditions must hold simultaneously for it to count as a wedge, so
    direction reflects which boundary is tighter (more likely to break).
    """
    if len(m5_df) < window * 2:
        return None
    recent = m5_df.iloc[-window:]
    prior = m5_df.iloc[-window * 2:-window]

    recent_high, recent_low = float(recent["high"].max()), float(recent["low"].min())
    prior_high, prior_low = float(prior["high"].max()), float(prior["low"].min())

    highs_falling = recent_high < prior_high
    lows_rising = recent_low > prior_low
    recent_range = recent_high - recent_low
    prior_range = prior_high - prior_low
    if not (highs_falling and lows_rising and recent_range < prior_range):
        return None

    # Which boundary compressed more decides the bias: highs falling faster
    # than lows are rising suggests sellers are more in control (higher odds
    # of a downside break through the rising-lows support), and vice versa.
    high_compression = prior_high - recent_high
    low_compression = recent_low - prior_low
    direction = "FALLING_HIGHS" if high_compression > low_compression else "RISING_LOWS"
    return direction, recent_high, recent_low


class WedgeBreakoutStrategy(BaseStrategy):
    """Converging trendline (wedge) compression + directional breakout."""

    metadata = StrategyMetadata(
        id="wedge_breakout",
        name="Wedge Compression Breakout",
        hypothesis_id="wedge_compression_breakout",
        hypothesis_family="Volatility Expansion",
        hypothesis_text=(
            "Bar-range genuinely converging over two windows (falling highs "
            "AND rising lows, not just low ATR percentile) breaks out in the "
            "direction its tighter boundary suggested, on real RVOL."
        ),
        version="v1.0",
        archetype="Breakout",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["wedge", "compression", "breakout", "pattern"],
    )

    def __init__(
        self,
        window: int = 10,
        rvol_threshold: float = 1.3,
        min_body_fraction: float = 0.45,
        atr_sl_buffer_mult: float = 0.2,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
    ):
        self.window = window
        self.rvol_threshold = rvol_threshold
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr

    def thesis_key(self, signal: dict) -> tuple:
        return (signal.get("symbol", ""), "WEDGE_BREAKOUT", signal.get("signal", ""))

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < self.window * 2 + 5:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            # Wedge boundary must be computed from bars BEFORE the current one
            # — including the live candle in the convergence window means a
            # genuine breakout bar (which necessarily blows past the prior
            # high/low) always poisons its own "highs falling" check, so no
            # breakout could ever be detected on the candle that actually
            # breaks out.
            wedge = _wedge_convergence(m5_df.iloc[:-1], self.window)
            if wedge is None:
                return self._empty_result(experiment_name)
            direction, boundary_high, boundary_low = wedge

            last_candle = m5_df.iloc[-1]
            close = float(last_candle["close"])
            body = abs(close - float(last_candle["open"]))
            candle_range = float(last_candle["high"]) - float(last_candle["low"])
            body_fraction = (body / candle_range) if candle_range > 1e-9 else 0.0

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            broke_up = close > boundary_high and body_fraction >= self.min_body_fraction
            broke_down = close < boundary_low and body_fraction >= self.min_body_fraction

            if broke_up:
                setup_type = "WEDGE_BREAKOUT_UP"
                side = "BUY CALL"
                sl = min(boundary_high - (atr * self.atr_sl_buffer_mult), price - (atr * 0.5))
                take_profit = price + (atr * self.tp_atr_cap)
            elif broke_down:
                setup_type = "WEDGE_BREAKOUT_DOWN"
                side = "BUY PUT"
                sl = max(boundary_low + (atr * self.atr_sl_buffer_mult), price + (atr * 0.5))
                take_profit = price - (atr * self.tp_atr_cap)

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            current_time = snapshot.timestamp
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            rejection_reasons: List[str] = []

            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")

            # The wedge's own compression bias should agree with the breakout
            # direction — a break the "wrong" way relative to which boundary
            # was tighter is a weaker read (fighting its own setup's tell).
            if side == "BUY CALL" and direction == "FALLING_HIGHS":
                rejection_reasons.append("WEDGE_BIAS_MISMATCH")
            elif side == "BUY PUT" and direction == "RISING_LOWS":
                rejection_reasons.append("WEDGE_BIAS_MISMATCH")

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

            confidence = 0.5 if rejection_reasons else round(min(0.5 + 0.15 * (rvol - 1.0), 0.85), 2)

            diagnostics = {
                "wedge_direction": direction,
                "boundary_high": round(boundary_high, 2),
                "boundary_low": round(boundary_low, 2),
                "rvol": round(rvol, 2),
                "body_fraction": round(body_fraction, 2),
                "atr": round(atr, 2),
                "rr_ratio": rr,
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_WEDGE_"
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
            logger.error(f"[WedgeBreakoutStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
