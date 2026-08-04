#!/usr/bin/env python3
"""
OIWallReactionStrategy — trades reactions to real option-chain OI walls.
==========================================================================
Hypothesis:
    Strikes carrying outlier open interest act as de-facto strike-based S/R —
    price tends to fade at a heavily-written strike while momentum is decaying
    (market makers/writers defending the strike), but once RVOL-confirmed price
    actually breaches a wall, the unwind/short-covering that follows tends to
    continue rather than immediately reverse.

Requires REAL OI (src.core.options_intelligence_engine, fed by
src.warehouse.option_warehouse) — never trades on stale or missing options
data; that's an explicit error, not a silently-skipped rejection, since there
is no real candidate to reject without real OI.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.market_geometry import NarrativeBias
from src.core.options_intelligence_engine import OptionsIntelligence, OiWall

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


class OIWallReactionStrategy(BaseStrategy):
    """OI-wall fade + breakout strategy, v1.0."""

    metadata = StrategyMetadata(
        id="oi_wall_reaction",
        name="OI Wall Reaction Strategy",
        hypothesis_id="oi_wall_fade_and_break",
        hypothesis_family="OptionsIntelligence",
        hypothesis_text=(
            "Price fades at heavily-written OI strikes while momentum decays, "
            "and continues once an RVOL-confirmed break invalidates the wall."
        ),
        version="v1.0",
        maturity="PAPER",
        tags=["options", "open-interest", "oi-wall", "fade", "breakout"],
    )

    def __init__(
        self,
        zone_tolerance_pct: float = 0.0015,
        min_body_fraction: float = 0.40,
        atr_sl_buffer_mult: float = 0.15,
        breakout_rvol_threshold: float = 1.3,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
    ):
        self.zone_tolerance_pct = zone_tolerance_pct
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.breakout_rvol_threshold = breakout_rvol_threshold
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        logger.info(
            f"🧱 OIWallReactionStrategy initialized [breakout_rvol>={breakout_rvol_threshold}, RR>={min_rr}]"
        )

    def thesis_key(self, signal: dict) -> tuple:
        # One active CF per (symbol, wall strike, direction) — a specific wall's
        # thesis, not just "any OI_WALL setup on this symbol".
        return (
            signal.get("symbol", ""),
            signal.get("diagnostics", {}).get("wall_strike", ""),
            signal.get("signal", ""),
        )

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
            if options.call_oi_wall is None and options.put_oi_wall is None:
                return self._empty_result(experiment_name, errors=["NO_OI_WALLS_DETECTED"])

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

            geo = getattr(snapshot.market, "geometry", None)
            narrative = getattr(geo, "narrative", None) if geo else None
            bias = narrative.bias if narrative else NarrativeBias.NEUTRAL
            bias_confidence = narrative.bias_confidence if narrative else 0.5

            if options.call_oi_wall is not None:
                sig = self._evaluate_call_wall(
                    options.call_oi_wall, options.put_oi_wall, price, atr, rvol, last_candle,
                    ts, snapshot, experiment_name, bias, bias_confidence,
                )
                if sig:
                    signals.append(sig)

            if options.put_oi_wall is not None:
                sig = self._evaluate_put_wall(
                    options.put_oi_wall, options.call_oi_wall, price, atr, rvol, last_candle,
                    ts, snapshot, experiment_name, bias, bias_confidence,
                )
                if sig:
                    signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[OIWallReactionStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={"min_rr": self.min_rr, "breakout_rvol_threshold": self.breakout_rvol_threshold},
            errors=errors,
            warnings=warnings,
        )

    # ── Call wall (resistance): fade below it, continue above it ───────────

    def _evaluate_call_wall(
        self, wall: OiWall, opposite_wall: Optional[OiWall], price: float, atr: float, rvol: float,
        last_candle, ts, snapshot, experiment_name: str, bias: NarrativeBias, bias_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        tolerance = price * self.zone_tolerance_pct

        # Breakout above the wall, RVOL-confirmed — continuation long.
        if price > wall.strike and rvol >= self.breakout_rvol_threshold:
            sl = wall.strike - (atr * self.atr_sl_buffer_mult)
            risk_dist = price - sl
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price - min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = price + atr * self.tp_atr_cap
            return self._build_signal(
                "OI_WALL_BREAK", "BUY CALL", price, sl, tp, risk_dist, wall,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"wall_type": "call", "rvol": round(rvol, 2)},
            )

        # Approaching from below with decaying momentum — fade downward.
        if abs(price - wall.strike) <= tolerance and _is_bearish_reversal_body(last_candle, self.min_body_fraction):
            sl = wall.strike + (atr * self.atr_sl_buffer_mult)
            risk_dist = sl - price
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price + min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = opposite_wall.strike if opposite_wall else price - atr * self.tp_atr_cap
            tp = max(tp, price - atr * self.tp_atr_cap)  # never cap-exceed even if put wall is far
            return self._build_signal(
                "OI_WALL_FADE", "BUY PUT", price, sl, tp, risk_dist, wall,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"wall_type": "call"},
            )

        return None

    # ── Put wall (support): fade above it, continue below it ───────────────

    def _evaluate_put_wall(
        self, wall: OiWall, opposite_wall: Optional[OiWall], price: float, atr: float, rvol: float,
        last_candle, ts, snapshot, experiment_name: str, bias: NarrativeBias, bias_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        tolerance = price * self.zone_tolerance_pct

        # Breakout below the wall, RVOL-confirmed — continuation short.
        if price < wall.strike and rvol >= self.breakout_rvol_threshold:
            sl = wall.strike + (atr * self.atr_sl_buffer_mult)
            risk_dist = sl - price
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price + min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = price - atr * self.tp_atr_cap
            return self._build_signal(
                "OI_WALL_BREAK", "BUY PUT", price, sl, tp, risk_dist, wall,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"wall_type": "put", "rvol": round(rvol, 2)},
            )

        # Approaching from above with decaying momentum — fade upward.
        if abs(price - wall.strike) <= tolerance and _is_bullish_reversal_body(last_candle, self.min_body_fraction):
            sl = wall.strike - (atr * self.atr_sl_buffer_mult)
            risk_dist = price - sl
            min_sl_dist = atr * 0.5
            if risk_dist < min_sl_dist:
                sl = price - min_sl_dist
                risk_dist = min_sl_dist
            if risk_dist <= 0:
                return None
            tp = opposite_wall.strike if opposite_wall else price + atr * self.tp_atr_cap
            tp = min(tp, price + atr * self.tp_atr_cap)
            return self._build_signal(
                "OI_WALL_FADE", "BUY CALL", price, sl, tp, risk_dist, wall,
                snapshot, ts, experiment_name, bias, bias_confidence,
                extra={"wall_type": "put"},
            )

        return None

    # ── Shared signal construction ──────────────────────────────────────────

    def _build_signal(
        self, setup_type: str, side: str, price: float, sl: float, tp: float, risk_dist: float,
        wall: OiWall, snapshot: MarketSnapshot, ts, experiment_name: str,
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

        # OI normalized to a 0-1 confidence proxy — 1 lakh OI ~= 1.0, capped,
        # same heuristic already used when persisting OI walls into sr_zones.
        confidence = round(min(wall.oi / 100_000.0, 1.0), 3)

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
            "confidence": confidence,
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "timestamp": ts,
            "diagnostics": {
                "wall_strike": wall.strike,
                "wall_oi": wall.oi,
                **extra,
            },
        }
        return self._tag_signal(sig, experiment_name)
