#!/usr/bin/env python3
"""
ChannelStrategy — trades parallel-trendline channels from Channel geometry.
============================================================================
Hypothesis:
    Price respects the boundaries of an established channel (two parallel
    opposite-role trendlines) until it doesn't — bounces off either boundary
    are high-probability mean-reversion entries; a clean RVOL-confirmed close
    beyond a boundary is a measured-move breakout continuation entry.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.market_geometry import GeometryContext, NarrativeBias, Channel

logger = logging.getLogger(__name__)


def _is_bullish_reversal_body(candle, min_body_fraction: float = 0.40) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c > o and (body / candle_range) >= min_body_fraction


def _is_bearish_reversal_body(candle, min_body_fraction: float = 0.40) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c < o and (body / candle_range) >= min_body_fraction


def _candidate_id(symbol: str, setup_type: str, price: float, ts: datetime) -> str:
    safe = symbol.replace(":", "_").replace("-", "_")
    return f"cand_{safe}_{setup_type}_{price:.2f}_{ts.strftime('%Y%m%d_%H%M%S')}"


class ChannelStrategy(BaseStrategy):
    """Channel bounce + breakout strategy, v1.0."""

    metadata = StrategyMetadata(
        id="channel",
        name="Channel Bounce/Breakout Strategy",
        hypothesis_id="parallel_trendline_channel",
        hypothesis_family="Geometry",
        hypothesis_text=(
            "Price respects the boundaries of an established parallel-trendline "
            "channel until a confirmed, RVOL-backed breakout invalidates it."
        ),
        version="v1.0",
        maturity="PAPER",
        tags=["geometry", "channel", "trendline", "breakout", "mean-reversion"],
    )

    def __init__(
        self,
        min_parallel_score: float = 0.3,
        zone_tolerance_pct: float = 0.0015,
        min_body_fraction: float = 0.40,
        atr_sl_buffer_mult: float = 0.15,
        breakout_rvol_threshold: float = 1.3,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
    ):
        self.min_parallel_score = min_parallel_score
        self.zone_tolerance_pct = zone_tolerance_pct
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.breakout_rvol_threshold = breakout_rvol_threshold
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        logger.info(
            f"📐 ChannelStrategy initialized [parallel>={min_parallel_score}, "
            f"breakout_rvol>={breakout_rvol_threshold}, RR>={min_rr}]"
        )

    def thesis_key(self, signal: dict) -> tuple:
        # One active CF per (symbol, channel_id, direction) — a specific channel's
        # bounce/breakout thesis, not just "any CHANNEL_BOUNCE on this symbol".
        return (
            signal.get("symbol", ""),
            signal.get("diagnostics", {}).get("channel_id", ""),
            signal.get("signal", ""),
        )

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            geo: Optional[GeometryContext] = getattr(snapshot.market, "geometry", None)
            if geo is None or not geo.channels:
                return self._empty_result(experiment_name)

            atr = snapshot.features.get_float("atr") or 0.0
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 2:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA:m5"])

            price = snapshot.current_price
            ts = snapshot.timestamp
            last_candle = m5_df.iloc[-1]
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            narrative = getattr(geo, "narrative", None)
            bias = narrative.bias if narrative else NarrativeBias.NEUTRAL
            bias_confidence = narrative.bias_confidence if narrative else 0.5

            for ch in geo.channels:
                if ch.parallel_score < self.min_parallel_score:
                    continue

                sig = self._evaluate_bounce(ch, price, atr, last_candle, ts, snapshot, experiment_name, bias, bias_confidence)
                if sig:
                    signals.append(sig)

                sig = self._evaluate_breakout(ch, price, atr, rvol, ts, snapshot, experiment_name, bias, bias_confidence)
                if sig:
                    signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[ChannelStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={"min_parallel_score": self.min_parallel_score, "min_rr": self.min_rr},
            errors=errors,
            warnings=warnings,
        )

    # ── Bounce off either boundary ──────────────────────────────────────────

    def _evaluate_bounce(
        self, ch: Channel, price: float, atr: float, last_candle, ts, snapshot,
        experiment_name: str, bias: NarrativeBias, bias_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        tolerance = price * self.zone_tolerance_pct

        # Near lower boundary (support) — fade upward
        if abs(price - ch.lower.price_at_now) <= tolerance and _is_bullish_reversal_body(last_candle, self.min_body_fraction):
            sl = ch.lower.price_at_now - (atr * self.atr_sl_buffer_mult)
            risk_dist = price - sl
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price - min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = min(ch.upper.price_at_now, price + atr * self.tp_atr_cap)
            return self._build_signal(
                ch, "CHANNEL_BOUNCE", "BUY CALL", price, sl, tp, risk_dist,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"boundary": "lower", "channel_width": ch.width},
            )

        # Near upper boundary (resistance) — fade downward
        if abs(price - ch.upper.price_at_now) <= tolerance and _is_bearish_reversal_body(last_candle, self.min_body_fraction):
            sl = ch.upper.price_at_now + (atr * self.atr_sl_buffer_mult)
            risk_dist = sl - price
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price + min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = max(ch.lower.price_at_now, price - atr * self.tp_atr_cap)
            return self._build_signal(
                ch, "CHANNEL_BOUNCE", "BUY PUT", price, sl, tp, risk_dist,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"boundary": "upper", "channel_width": ch.width},
            )

        return None

    # ── RVOL-confirmed breakout beyond either boundary ──────────────────────

    def _evaluate_breakout(
        self, ch: Channel, price: float, atr: float, rvol: float, ts, snapshot,
        experiment_name: str, bias: NarrativeBias, bias_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        if rvol < self.breakout_rvol_threshold:
            return None

        # Closed above the upper boundary — continuation long, former resistance
        # now acts as support (retest-style stop just below it).
        if price > ch.upper.price_at_now:
            sl = ch.upper.price_at_now - (atr * self.atr_sl_buffer_mult)
            risk_dist = price - sl
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price - min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = min(price + ch.width, price + atr * self.tp_atr_cap * 2)  # measured-move target
            return self._build_signal(
                ch, "CHANNEL_BREAKOUT", "BUY CALL", price, sl, tp, risk_dist,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"boundary": "upper", "rvol": round(rvol, 2), "channel_width": ch.width},
            )

        # Closed below the lower boundary — continuation short.
        if price < ch.lower.price_at_now:
            sl = ch.lower.price_at_now + (atr * self.atr_sl_buffer_mult)
            risk_dist = sl - price
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price + min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = max(price - ch.width, price - atr * self.tp_atr_cap * 2)
            return self._build_signal(
                ch, "CHANNEL_BREAKOUT", "BUY PUT", price, sl, tp, risk_dist,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"boundary": "lower", "rvol": round(rvol, 2), "channel_width": ch.width},
            )

        return None

    # ── Shared signal construction ──────────────────────────────────────────

    def _build_signal(
        self, ch: Channel, setup_type: str, side: str, price: float, sl: float, tp: float,
        risk_dist: float, snapshot: MarketSnapshot, ts, experiment_name: str,
        bias: NarrativeBias, bias_confidence: float, extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        tp_dist = abs(tp - price)
        rr = round(tp_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        rejection_reasons: List[str] = []
        if side == "BUY CALL" and bias == NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_BEARISH")
        elif side == "BUY PUT" and bias == NarrativeBias.CONTINUATION and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_BULLISH")
        if rr < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr}")

        accepted = len(rejection_reasons) == 0
        tp1 = price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5)

        cid = _candidate_id(snapshot.symbol, setup_type, price, ts)
        sig = {
            "symbol": snapshot.symbol,
            "candidate_id": cid,
            "signal": side,
            "price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "tp1": tp1,
            "rr_ratio": rr,
            "strategy": setup_type,
            "confidence": round(ch.parallel_score, 3),
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "timestamp": ts,
            "diagnostics": {
                "channel_id": ch.id,
                "channel_timeframe": ch.timeframe,
                "channel_direction": ch.direction.value,
                "channel_status": ch.status.value,
                "parallel_score": ch.parallel_score,
                **extra,
            },
        }
        return self._tag_signal(sig, experiment_name)
