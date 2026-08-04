#!/usr/bin/env python3
"""
PCRExtremeReversalStrategy — contrarian reversal on Put-Call Ratio extremes.
==============================================================================
Hypothesis:
    An extreme PCR reflects one-sided option-seller positioning (heavy put OI
    below spot = sellers comfortable being short puts there = they don't expect
    a fall past that level = bullish; symmetric for heavy call OI = bearish).
    PCR alone is noisy — this only fires when the extreme reading COINCIDES
    with a real price-action confirmation (a reversal candle at a genuine
    confluence zone), same gating geometry_strategy.py uses for its own
    confluence-bounce setups.

Requires REAL PCR (src.core.options_intelligence_engine, fed by
src.warehouse.option_warehouse) — never trades on stale or missing options
data.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.market_geometry import GeometryContext, NarrativeBias, ConfluenceZone
from src.core.options_intelligence_engine import OptionsIntelligence

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


def _distance_to_zone(price: float, zone: ConfluenceZone) -> float:
    if zone.band_low <= price <= zone.band_high:
        return 0.0
    return min(abs(price - zone.band_low), abs(price - zone.band_high))


def _candidate_id(symbol: str, setup_type: str, price: float, ts: datetime) -> str:
    safe = symbol.replace(":", "_").replace("-", "_")
    return f"cand_{safe}_{setup_type}_{price:.2f}_{ts.strftime('%Y%m%d_%H%M%S')}"


class PCRExtremeReversalStrategy(BaseStrategy):
    """PCR-extreme contrarian reversal, gated by zone confluence + candle confirmation. v1.0."""

    metadata = StrategyMetadata(
        id="pcr_extreme_reversal",
        name="PCR Extreme Reversal Strategy",
        hypothesis_id="pcr_extreme_contrarian",
        hypothesis_family="OptionsIntelligence",
        hypothesis_text=(
            "An extreme Put-Call Ratio combined with a confirmed reversal candle "
            "at a genuine confluence zone is a contrarian reversal signal."
        ),
        version="v1.0",
        maturity="PAPER",
        tags=["options", "pcr", "put-call-ratio", "reversal", "contrarian"],
    )

    def __init__(
        self,
        min_confluence_score: float = 40.0,
        zone_tolerance_pct: float = 0.0015,
        min_body_fraction: float = 0.40,
        atr_sl_buffer_mult: float = 0.15,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
    ):
        self.min_confluence_score = min_confluence_score
        self.zone_tolerance_pct = zone_tolerance_pct
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        logger.info(
            f"🔄 PCRExtremeReversalStrategy initialized [zone_score>={min_confluence_score}, RR>={min_rr}]"
        )

    def thesis_key(self, signal: dict) -> tuple:
        return (
            signal.get("symbol", ""),
            "PCR_EXTREME_REVERSAL",
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
            if options.pcr_bias not in ("BULLISH", "BEARISH"):
                # NEUTRAL/UNKNOWN isn't a data problem — there's just no extreme
                # reading to act on right now.
                return self._empty_result(experiment_name)

            geo: Optional[GeometryContext] = getattr(snapshot.market, "geometry", None)
            if geo is None:
                return self._empty_result(experiment_name, errors=["GEOMETRY_MISSING"])

            atr = snapshot.features.get_float("atr") or 0.0
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 2:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA:m5"])

            price = snapshot.current_price
            ts = snapshot.timestamp
            last_candle = m5_df.iloc[-1]

            narrative = getattr(geo, "narrative", None)
            bias = narrative.bias if narrative else NarrativeBias.NEUTRAL
            bias_confidence = narrative.bias_confidence if narrative else 0.5

            if options.pcr_bias == "BULLISH":
                sig = self._evaluate_bullish(geo, options, price, atr, last_candle, ts, snapshot, experiment_name, bias, bias_confidence)
            else:
                sig = self._evaluate_bearish(geo, options, price, atr, last_candle, ts, snapshot, experiment_name, bias, bias_confidence)
            if sig:
                signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[PCRExtremeReversalStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={"min_confluence_score": self.min_confluence_score, "min_rr": self.min_rr},
            errors=errors,
            warnings=warnings,
        )

    def _evaluate_bullish(
        self, geo: GeometryContext, options: OptionsIntelligence, price: float, atr: float,
        last_candle, ts, snapshot, experiment_name: str, bias: NarrativeBias, bias_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        zone = geo.support_confluence
        if zone is None or zone.total_score < self.min_confluence_score:
            return None
        if _distance_to_zone(price, zone) > price * self.zone_tolerance_pct:
            return None
        if not _is_bullish_reversal_body(last_candle, self.min_body_fraction):
            return None

        sl = zone.band_low - (atr * self.atr_sl_buffer_mult)
        risk_dist = price - sl
        min_sl_dist = atr * 0.5
        if risk_dist < min_sl_dist:
            sl = price - min_sl_dist
            risk_dist = min_sl_dist
        if risk_dist <= 0:
            return None
        tp = price + atr * self.tp_atr_cap

        return self._build_signal(
            "PCR_EXTREME_REVERSAL", "BUY CALL", price, sl, tp, risk_dist, zone, options,
            snapshot, ts, experiment_name, bias, bias_confidence,
        )

    def _evaluate_bearish(
        self, geo: GeometryContext, options: OptionsIntelligence, price: float, atr: float,
        last_candle, ts, snapshot, experiment_name: str, bias: NarrativeBias, bias_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        zone = geo.resistance_confluence
        if zone is None or zone.total_score < self.min_confluence_score:
            return None
        if _distance_to_zone(price, zone) > price * self.zone_tolerance_pct:
            return None
        if not _is_bearish_reversal_body(last_candle, self.min_body_fraction):
            return None

        sl = zone.band_high + (atr * self.atr_sl_buffer_mult)
        risk_dist = sl - price
        min_sl_dist = atr * 0.5
        if risk_dist < min_sl_dist:
            sl = price + min_sl_dist
            risk_dist = min_sl_dist
        if risk_dist <= 0:
            return None
        tp = price - atr * self.tp_atr_cap

        return self._build_signal(
            "PCR_EXTREME_REVERSAL", "BUY PUT", price, sl, tp, risk_dist, zone, options,
            snapshot, ts, experiment_name, bias, bias_confidence,
        )

    def _build_signal(
        self, setup_type: str, side: str, price: float, sl: float, tp: float, risk_dist: float,
        zone: ConfluenceZone, options: OptionsIntelligence, snapshot: MarketSnapshot, ts,
        experiment_name: str, bias: NarrativeBias, bias_confidence: float,
    ) -> Dict[str, Any]:
        tp_dist = abs(tp - price)
        rr = round(tp_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        rejection_reasons: List[str] = []
        # Note: this strategy is inherently contrarian to daily_bias-driven
        # narrative continuation — only oppose it when narrative REVERSAL
        # bias points the opposite way, mirroring geometry_strategy's gate.
        if side == "BUY CALL" and bias == NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_BEARISH")
        elif side == "BUY PUT" and bias == NarrativeBias.CONTINUATION and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_BULLISH")
        if rr < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr}")

        accepted = len(rejection_reasons) == 0
        tp1 = price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5)
        confidence = round((zone.total_score / 100.0), 3)

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
                "pcr": options.pcr,
                "pcr_bias": options.pcr_bias,
                "zone_score": round(zone.total_score, 1),
                "zone_explanation": zone.explanation,
            },
        }
        return self._tag_signal(sig, experiment_name)
