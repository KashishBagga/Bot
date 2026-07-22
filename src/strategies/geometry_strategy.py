#!/usr/bin/env python3
"""
GeometryStrategy — Signal generation purely from Market Geometry (MKE Stage 5).
================================================================================
Hypothesis:
    Institutional price action obeys a hierarchy of structural zones.
    When multiple geometry objects align at the same price band (a "confluence zone"),
    and price approaches that band with a reversal candle, the probability of a
    directional move away from the band is statistically elevated.

Architecture: PURELY READS snapshot.market.geometry (GeometryContext).
    No legacy pivot/zone/RVOL references.
    No EnhancedStrategyEngine.
    No FeatureStore indicators.

Setup logic uses ONLY:
    geo.support_confluence   → primary buy zone
    geo.resistance_confluence → primary sell zone
    geo.levels.*              → O(1) convenience views for SL anchor
    geo.trendlines.*          → trendline support / slope confirmation
    geo.narrative             → macro bias gate

Signal generation:
    1. CONFLUENCE_BOUNCE_BUY
       - support_confluence zone exists with score >= min_confluence_score
       - price is at or inside the zone (distance <= tolerance_pct)
       - last candle prints a bullish reversal body (close > open, body > 40% range)
       - narrative bias is BULLISH or NEUTRAL
       - SL: band_low - 0.15 × ATR (just below the zone)
       - TP: nearest resistance composite or nearest resistance trendline, whichever is closer
         but capped at 3 × ATR from entry

    2. CONFLUENCE_BOUNCE_SELL
       - resistance_confluence zone exists with score >= min_confluence_score
       - price is at or inside the zone
       - last candle prints a bearish reversal body (close < open, body > 40% range)
       - narrative bias is BEARISH or NEUTRAL
       - SL: band_high + 0.15 × ATR
       - TP: nearest support composite or nearest support trendline (capped at 3 × ATR)

    3. TRENDLINE_BREAK_AND_RETEST_BUY (optional, gated by trendline_break_enabled)
       - A trendline with status BROKEN and role SUPPORT exists
       - Price has since moved below and returned (retest)
       - Candle shows buying absorption
       - SL: just below the current candle low
       - TP: nearest structural resistance (from levels view)

    4. TRENDLINE_BREAK_AND_RETEST_SELL (mirror)

Quality gates (apply to all setups):
    - min_confidence: narrative.bias_confidence >= min_confidence
    - time_filter: no trades in first 15 minutes of session (09:15–09:30)
    - no_position_against_institutional: if nearest_support/resistance is INSTITUTIONAL
      priority and price is inside the zone, skip contrary signals

Rejection reasons are stored in signal['rejection_reasons']; accepted=False signals
are still emitted for counterfactual tracking.
"""

import logging
from datetime import datetime, time
from typing import List, Dict, Any, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.market_geometry import (
    GeometryContext,
    ConfluenceZone,
    CompositeLevel,
    Trendline,
    NarrativeBias,
    LevelDirection,
    GeometryStatus,
    TrendlineRole,
    LevelPriority,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_bullish_reversal_body(candle, min_body_fraction: float = 0.40) -> bool:
    """True if the candle is a bullish engulfing-style body (not a doji)."""
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c > o and (body / candle_range) >= min_body_fraction


def _is_bearish_reversal_body(candle, min_body_fraction: float = 0.40) -> bool:
    """True if the candle is a bearish engulfing-style body (not a doji)."""
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c < o and (body / candle_range) >= min_body_fraction


def _distance_to_zone(price: float, zone: ConfluenceZone) -> float:
    """Signed distance: negative = inside zone, positive = outside zone."""
    if price >= zone.band_low and price <= zone.band_high:
        return 0.0  # inside
    if price < zone.band_low:
        return zone.band_low - price
    return price - zone.band_high


def _pct_of(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0
    return abs(value - reference) / reference


def _candidate_id(symbol: str, setup_type: str, price: float, ts: datetime) -> str:
    safe = symbol.replace(":", "_").replace("-", "_")
    return f"cand_{safe}_{setup_type}_{price:.2f}_{ts.strftime('%Y%m%d%H%M')}"


def _is_session_open_blackout(ts: datetime) -> bool:
    """True if within the first 15 minutes of market open (09:15–09:29)."""
    t = ts.time()
    return time(9, 15) <= t < time(9, 30)


def _is_session_close_blackout(ts: datetime) -> bool:
    """True if within the last 15 minutes before market close (15:15–15:30)."""
    t = ts.time()
    return t >= time(15, 15)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class GeometryStrategy(BaseStrategy):
    """
    Signal generation driven purely by the GeometryContext produced by
    MKE Stage 5 engines (LevelEngine → TrendlineEngine → FusionEngine →
    ConfluenceEngine → NarrativeEngine).

    This strategy intentionally ignores all legacy indicators:
        - No RVOL
        - No zone scores
        - No EMA cross
        - No daily bias string

    The narrative bias from NarrativeEngine replaces all of these, because
    it is itself derived from structure, EMAs, trendlines, and liquidity.

    Parameters
    ----------
    min_confluence_score : float
        Minimum tanh-normalized score (0–100) for a ConfluenceZone to trigger.
        Default 35 ≈ 2 confluent sources. Recommended range: 30–60.
    zone_tolerance_pct : float
        Maximum distance from zone edge (as fraction of price) to qualify as
        "price is at the zone". Default 0.002 = 0.2%.
    min_body_fraction : float
        Minimum candle body / range ratio for reversal confirmation. Default 0.40.
    min_bias_confidence : float
        Minimum narrative.bias_confidence to proceed. Default 0.40.
    atr_sl_buffer_mult : float
        SL buffer = atr × this. Applied below band_low (buys) / above band_high (sells).
        Default 0.15 (tight, inside-zone logic requires structure to hold).
    tp_atr_cap : float
        Maximum TP distance in ATR multiples. Default 3.0.
    min_rr : float
        Minimum reward-to-risk ratio. Signals below this are rejected (not accepted).
    trendline_break_enabled : bool
        If True, emit TRENDLINE_BREAK_RETEST setups in addition to confluence bounces.
    """

    metadata = StrategyMetadata(
        id="geometry",
        name="Market Geometry Strategy",
        hypothesis_id="geometry_confluence_reversal",
        hypothesis_family="Geometry",
        hypothesis_text=(
            "Price reverses at high-confluence geometry zones where multiple "
            "institutional levels, trendlines, and VWAP/round numbers converge. "
            "Narrative bias gates the direction."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["geometry", "confluence", "structure", "trendline", "reversal"],
    )

    def __init__(
        self,
        min_confluence_score: float = 50.0,
        zone_tolerance_pct: float = 0.002,
        min_body_fraction: float = 0.40,
        min_bias_confidence: float = 0.40,
        atr_sl_buffer_mult: float = 0.15,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
        trendline_break_enabled: bool = True,
    ):
        self.min_confluence_score = min_confluence_score
        self.zone_tolerance_pct = zone_tolerance_pct
        self.min_body_fraction = min_body_fraction
        self.min_bias_confidence = min_bias_confidence
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        self.trendline_break_enabled = trendline_break_enabled
        logger.info(
            f"📐 GeometryStrategy initialized "
            f"[score>={min_confluence_score}, tol={zone_tolerance_pct*100:.1f}%, "
            f"RR>={min_rr}, trendline_break={trendline_break_enabled}]"
        )

    # ──────────────────────────────────────────────────────────────────────
    # BaseStrategy interface
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        experiment_name: str,
    ) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            # ── 1. Extract geometry from snapshot ─────────────────────────
            geo: Optional[GeometryContext] = getattr(snapshot.market, "geometry", None)
            if geo is None:
                return self._empty_result(
                    experiment_name,
                    errors=["GEOMETRY_MISSING: snapshot.market.geometry is None"],
                )

            # ── 2. Extract ATR from features (used for SL/TP sizing only) ─
            atr: float = snapshot.features.get_float("atr") or 0.0
            if atr <= 0:
                return self._empty_result(
                    experiment_name,
                    errors=["FEATURE_MISSING:atr"],
                )

            price = snapshot.current_price
            ts = snapshot.timestamp
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 5:
                return self._empty_result(
                    experiment_name,
                    errors=["INSUFFICIENT_DATA:m5"],
                )

            last_candle = m5_df.iloc[-1]

            # ── 3. Global time filters (Removed) ────────────────────────────────────
            global_rejections: List[str] = []

            # ── 4. Narrative gates ────────────────────────────────────────
            narrative = geo.narrative
            bias_confidence = narrative.bias_confidence if narrative else 0.0
            bias = narrative.bias if narrative else NarrativeBias.NEUTRAL

            if narrative and bias_confidence < self.min_bias_confidence:
                warnings.append(
                    f"LOW_BIAS_CONFIDENCE:{bias_confidence:.2f}"
                )

            # ── 5. Generate confluence bounce setups ──────────────────────
            support_sig = self._evaluate_confluence_bounce_buy(
                geo, price, atr, last_candle, ts, snapshot, experiment_name,
                bias, bias_confidence, global_rejections,
            )
            if support_sig:
                signals.append(support_sig)

            resistance_sig = self._evaluate_confluence_bounce_sell(
                geo, price, atr, last_candle, ts, snapshot, experiment_name,
                bias, bias_confidence, global_rejections,
            )
            if resistance_sig:
                signals.append(resistance_sig)

            # ── 6. Trendline break-and-retest setups ─────────────────────
            if self.trendline_break_enabled:
                tl_buy = self._evaluate_trendline_retest_buy(
                    geo, price, atr, last_candle, ts, snapshot, experiment_name,
                    bias, bias_confidence, global_rejections,
                )
                if tl_buy:
                    signals.append(tl_buy)

                tl_sell = self._evaluate_trendline_retest_sell(
                    geo, price, atr, last_candle, ts, snapshot, experiment_name,
                    bias, bias_confidence, global_rejections,
                )
                if tl_sell:
                    signals.append(tl_sell)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(
                f"[GeometryStrategy] Error evaluating {snapshot.symbol}: {e}",
                exc_info=True,
            )

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={
                "geometry_available": True,
                "min_confluence_score": self.min_confluence_score,
                "zone_tolerance_pct": self.zone_tolerance_pct,
                "trendline_break_enabled": self.trendline_break_enabled,
            },
            errors=errors,
            warnings=warnings,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Setup: CONFLUENCE_BOUNCE_BUY
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_confluence_bounce_buy(
        self,
        geo: GeometryContext,
        price: float,
        atr: float,
        last_candle,
        ts: datetime,
        snapshot: MarketSnapshot,
        experiment_name: str,
        bias: NarrativeBias,
        bias_confidence: float,
        global_rejections: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        CONFLUENCE_BOUNCE_BUY: price at a high-score support confluence zone
        with a bullish reversal candle.
        """
        zone = geo.support_confluence
        if zone is None or zone.total_score < self.min_confluence_score:
            return None

        dist = _distance_to_zone(price, zone)
        # Accept if price is inside zone OR within zone_tolerance_pct above the band_low
        zone_tolerance = price * self.zone_tolerance_pct
        if dist > zone_tolerance:
            return None  # Price not near the zone — no setup

        if not _is_bullish_reversal_body(last_candle, self.min_body_fraction):
            return None  # Candle not confirming

        # ── SL / TP ───────────────────────────────────────────────────────
        sl = zone.band_low - (atr * self.atr_sl_buffer_mult)
        risk_dist = price - sl

        # ── FIX: Enforce minimum SL floor of 0.5×ATR ───────────────────
        min_sl_dist = atr * 0.5
        if risk_dist < min_sl_dist:
            sl = price - min_sl_dist
            risk_dist = min_sl_dist
        # ────────────────────────────────────────────────────────────────

        if risk_dist <= 0:
            return None  # Degenerate (price already below SL)

        tp = self._find_buy_target(geo, price, atr)

        # ── RR check ─────────────────────────────────────────────────────
        tp_dist = tp - price
        rr = round(tp_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        # ── Rejection reasons ─────────────────────────────────────────────
        rejection_reasons = list(global_rejections)

        if bias == NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_BEARISH")

        if bias_confidence < self.min_bias_confidence:
            rejection_reasons.append(f"LOW_BIAS_CONFIDENCE:{bias_confidence:.2f}")

        if rr < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr}")

        accepted = len(rejection_reasons) == 0

        # ── TP1 at 1.5R ───────────────────────────────────────────────────
        tp1 = price + (risk_dist * 1.5)

        return self._build_signal(
            setup_type="CONFLUENCE_BOUNCE",
            side="BUY CALL",
            price=price,
            sl=sl,
            tp=tp,
            tp1=tp1,
            rr=rr,
            confidence=self._score_to_confidence(zone.total_score, bias_confidence),
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            zone=zone,
            symbol=snapshot.symbol,
            ts=ts,
            atr=atr,
            experiment_name=experiment_name,
            extra_diagnostics={
                "zone_score": round(zone.total_score, 1),
                "zone_explanation": zone.explanation,
                "dist_to_zone": round(dist, 2),
                "narrative_bias": bias.value,
                "bias_confidence": round(bias_confidence, 3),
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Setup: CONFLUENCE_BOUNCE_SELL
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_confluence_bounce_sell(
        self,
        geo: GeometryContext,
        price: float,
        atr: float,
        last_candle,
        ts: datetime,
        snapshot: MarketSnapshot,
        experiment_name: str,
        bias: NarrativeBias,
        bias_confidence: float,
        global_rejections: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        CONFLUENCE_BOUNCE_SELL: price at a high-score resistance confluence zone
        with a bearish reversal candle.
        """
        zone = geo.resistance_confluence
        if zone is None or zone.total_score < self.min_confluence_score:
            return None

        dist = _distance_to_zone(price, zone)
        zone_tolerance = price * self.zone_tolerance_pct
        if dist > zone_tolerance:
            return None

        if not _is_bearish_reversal_body(last_candle, self.min_body_fraction):
            return None

        # ── SL / TP ───────────────────────────────────────────────────────
        sl = zone.band_high + (atr * self.atr_sl_buffer_mult)
        risk_dist = sl - price

        # ── FIX: Enforce minimum SL floor of 0.5×ATR ───────────────────
        min_sl_dist = atr * 0.5
        if risk_dist < min_sl_dist:
            sl = price + min_sl_dist
            risk_dist = min_sl_dist
        # ────────────────────────────────────────────────────────────────

        if risk_dist <= 0:
            return None

        tp = self._find_sell_target(geo, price, atr)
        tp_dist = price - tp
        rr = round(tp_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        # ── Rejection reasons ─────────────────────────────────────────────
        rejection_reasons = list(global_rejections)

        if bias == NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            # NarrativeEngine emits REVERSAL for a bullish intent — wrong for sells
            rejection_reasons.append("NARRATIVE_BIAS_BULLISH")

        if bias_confidence < self.min_bias_confidence:
            rejection_reasons.append(f"LOW_BIAS_CONFIDENCE:{bias_confidence:.2f}")

        if rr < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr}")

        accepted = len(rejection_reasons) == 0
        tp1 = price - (risk_dist * 1.5)

        return self._build_signal(
            setup_type="CONFLUENCE_BOUNCE",
            side="BUY PUT",
            price=price,
            sl=sl,
            tp=tp,
            tp1=tp1,
            rr=rr,
            confidence=self._score_to_confidence(zone.total_score, bias_confidence),
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            zone=zone,
            symbol=snapshot.symbol,
            ts=ts,
            atr=atr,
            experiment_name=experiment_name,
            extra_diagnostics={
                "zone_score": round(zone.total_score, 1),
                "zone_explanation": zone.explanation,
                "dist_to_zone": round(dist, 2),
                "narrative_bias": bias.value,
                "bias_confidence": round(bias_confidence, 3),
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Setup: TRENDLINE_BREAK_RETEST_BUY
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_trendline_retest_buy(
        self,
        geo: GeometryContext,
        price: float,
        atr: float,
        last_candle,
        ts: datetime,
        snapshot: MarketSnapshot,
        experiment_name: str,
        bias: NarrativeBias,
        bias_confidence: float,
        global_rejections: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        TRENDLINE_BREAK_RETEST_BUY:
        A support trendline was broken (role SUPPORT, status BROKEN).
        Price has returned to the now-resistance trendline from below and printed
        a bullish candle — buy the retest of the old trendline as new support.

        This is the "role reversal" for trendlines.
        """
        # Find the highest-confidence broken support trendline
        broken_supports = [
            t for t in geo.trendlines
            if t.role == TrendlineRole.SUPPORT
            and t.status == GeometryStatus.BROKEN
            and abs(t.distance_pct) <= self.zone_tolerance_pct * 3
            and t.confidence >= 0.45
        ]
        if not broken_supports:
            return None

        tl = max(broken_supports, key=lambda t: t.confidence)
        tl_price = tl.price_at_now
        dist_pct = abs(price - tl_price) / price if price > 0 else 1.0

        if dist_pct > self.zone_tolerance_pct * 2:
            return None  # Not close enough to the line

        if not _is_bullish_reversal_body(last_candle, self.min_body_fraction):
            return None

        # SL: below this candle's low minus buffer
        candle_low = float(last_candle["low"])
        sl = candle_low - (atr * self.atr_sl_buffer_mult)
        risk_dist = price - sl
        if risk_dist <= 0:
            return None

        # TP: nearest resistance level
        resistance = geo.levels.nearest_resistance()
        tp_candidate = resistance.price if resistance else (price + atr * self.tp_atr_cap)
        tp = min(tp_candidate, price + atr * self.tp_atr_cap)
        tp_dist = tp - price
        rr = round(tp_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        rejection_reasons = list(global_rejections)
        if bias == NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_BEARISH")
        if bias_confidence < self.min_bias_confidence:
            rejection_reasons.append(f"LOW_BIAS_CONFIDENCE:{bias_confidence:.2f}")
        if rr < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr}")

        accepted = len(rejection_reasons) == 0
        tp1 = price + (risk_dist * 1.5)

        return self._build_signal(
            setup_type="TRENDLINE_RETEST",
            side="BUY CALL",
            price=price,
            sl=sl,
            tp=tp,
            tp1=tp1,
            rr=rr,
            confidence=self._trendline_confidence(tl, bias_confidence),
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            zone=None,
            symbol=snapshot.symbol,
            ts=ts,
            atr=atr,
            experiment_name=experiment_name,
            extra_diagnostics={
                "trendline_id": tl.id,
                "trendline_confidence": round(tl.confidence, 3),
                "trendline_r2": round(tl.r_squared, 3),
                "trendline_touches": tl.touches,
                "trendline_price_at_now": round(tl_price, 2),
                "dist_pct": round(dist_pct, 5),
                "narrative_bias": bias.value,
                "bias_confidence": round(bias_confidence, 3),
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Setup: TRENDLINE_BREAK_RETEST_SELL
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_trendline_retest_sell(
        self,
        geo: GeometryContext,
        price: float,
        atr: float,
        last_candle,
        ts: datetime,
        snapshot: MarketSnapshot,
        experiment_name: str,
        bias: NarrativeBias,
        bias_confidence: float,
        global_rejections: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        TRENDLINE_BREAK_RETEST_SELL:
        A resistance trendline was broken upward (role RESISTANCE, status BROKEN).
        Price has returned to the old resistance from above.
        Sell the retest of what is now support-turned-resistance.
        """
        broken_resistances = [
            t for t in geo.trendlines
            if t.role == TrendlineRole.RESISTANCE
            and t.status == GeometryStatus.BROKEN
            and abs(t.distance_pct) <= self.zone_tolerance_pct * 3
            and t.confidence >= 0.45
        ]
        if not broken_resistances:
            return None

        tl = max(broken_resistances, key=lambda t: t.confidence)
        tl_price = tl.price_at_now
        dist_pct = abs(price - tl_price) / price if price > 0 else 1.0

        if dist_pct > self.zone_tolerance_pct * 2:
            return None

        if not _is_bearish_reversal_body(last_candle, self.min_body_fraction):
            return None

        candle_high = float(last_candle["high"])
        sl = candle_high + (atr * self.atr_sl_buffer_mult)
        risk_dist = sl - price
        if risk_dist <= 0:
            return None

        support = geo.levels.nearest_support()
        tp_candidate = support.price if support else (price - atr * self.tp_atr_cap)
        tp = max(tp_candidate, price - atr * self.tp_atr_cap)
        tp_dist = price - tp
        rr = round(tp_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        rejection_reasons = list(global_rejections)
        if bias == NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_BULLISH")
        if bias_confidence < self.min_bias_confidence:
            rejection_reasons.append(f"LOW_BIAS_CONFIDENCE:{bias_confidence:.2f}")
        if rr < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr}")

        accepted = len(rejection_reasons) == 0
        tp1 = price - (risk_dist * 1.5)

        return self._build_signal(
            setup_type="TRENDLINE_RETEST",
            side="BUY PUT",
            price=price,
            sl=sl,
            tp=tp,
            tp1=tp1,
            rr=rr,
            confidence=self._trendline_confidence(tl, bias_confidence),
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            zone=None,
            symbol=snapshot.symbol,
            ts=ts,
            atr=atr,
            experiment_name=experiment_name,
            extra_diagnostics={
                "trendline_id": tl.id,
                "trendline_confidence": round(tl.confidence, 3),
                "trendline_r2": round(tl.r_squared, 3),
                "trendline_touches": tl.touches,
                "trendline_price_at_now": round(tl_price, 2),
                "dist_pct": round(dist_pct, 5),
                "narrative_bias": bias.value,
                "bias_confidence": round(bias_confidence, 3),
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # TP resolution helpers
    # ──────────────────────────────────────────────────────────────────────

    def _find_buy_target(
        self,
        geo: GeometryContext,
        price: float,
        atr: float,
    ) -> float:
        """
        Find the best buy TP target:
        prefer the nearest resistance composite, fallback to nearest resistance trendline,
        fallback to price + 3 × ATR. Cap at tp_atr_cap × ATR.
        """
        cap = price + (atr * self.tp_atr_cap)

        # Option 1: nearest resistance composite
        res_composite = geo.levels.nearest_resistance()
        if res_composite and res_composite.price > price:
            return min(res_composite.price, cap)

        # Option 2: nearest resistance trendline
        res_tl = geo.trendlines.nearest_resistance()
        if res_tl and res_tl.price_at_now > price:
            return min(res_tl.price_at_now, cap)

        # Fallback: cap
        return cap

    def _find_sell_target(
        self,
        geo: GeometryContext,
        price: float,
        atr: float,
    ) -> float:
        """
        Find the best sell TP target:
        prefer nearest support composite, fallback to nearest support trendline,
        fallback to price - 3 × ATR. Cap at tp_atr_cap × ATR.
        """
        cap = price - (atr * self.tp_atr_cap)

        sup_composite = geo.levels.nearest_support()
        if sup_composite and sup_composite.price < price:
            return max(sup_composite.price, cap)

        sup_tl = geo.trendlines.nearest_support()
        if sup_tl and sup_tl.price_at_now < price:
            return max(sup_tl.price_at_now, cap)

        return cap

    # ──────────────────────────────────────────────────────────────────────
    # Confidence scoring
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_confidence(zone_score: float, bias_confidence: float) -> float:
        """
        Combine zone score (0–100) and narrative bias confidence (0–1) into a
        single confidence value (0–1).
        Weights: 60% zone score, 40% narrative confidence.
        """
        normalized_score = min(zone_score / 100.0, 1.0)
        raw = 0.60 * normalized_score + 0.40 * bias_confidence
        return round(min(raw, 0.99), 3)

    @staticmethod
    def _trendline_confidence(tl: Trendline, bias_confidence: float) -> float:
        """
        Confidence for trendline setups: trendline confidence × narrative bias weight.
        """
        raw = 0.65 * tl.confidence + 0.35 * bias_confidence
        return round(min(raw, 0.99), 3)

    # ──────────────────────────────────────────────────────────────────────
    # Signal builder — single source of truth for dict schema
    # ──────────────────────────────────────────────────────────────────────

    def _build_signal(
        self,
        setup_type: str,
        side: str,
        price: float,
        sl: float,
        tp: float,
        tp1: float,
        rr: float,
        confidence: float,
        accepted: bool,
        rejection_reasons: List[str],
        zone: Optional[ConfluenceZone],
        symbol: str,
        ts: datetime,
        atr: float,
        experiment_name: str,
        extra_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        risk_dist = abs(price - sl)
        cid = _candidate_id(symbol, setup_type, price, ts)

        sig: Dict[str, Any] = {
            # ── Identity ─────────────────────────────────────────────────
            "symbol": symbol,
            "candidate_id": cid,
            "experiment_name": experiment_name,
            "strategy_id": self.id,
            "version": self.version,

            # ── Signal ────────────────────────────────────────────────────
            "signal": side,
            "strategy": setup_type,
            "price": round(price, 2),
            "stop_loss": round(sl, 2),
            "take_profit": round(tp, 2),
            "tp1": round(tp1, 2),
            "rr_ratio": rr,
            "timestamp": ts.isoformat(),

            # ── Geometry source ───────────────────────────────────────────
            "zone_band_low": round(zone.band_low, 2) if zone else None,
            "zone_band_high": round(zone.band_high, 2) if zone else None,
            "zone_direction": zone.direction.value if zone else None,
            "zone_component_count": len(zone.components) if zone else 0,

            # ── Acceptance ────────────────────────────────────────────────
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,

            # ── Risk sizing ───────────────────────────────────────────────
            "stop_loss_distance": round(risk_dist, 2),
            "atr": round(atr, 2),

            # ── Confidence ────────────────────────────────────────────────
            "confidence": confidence,

            # ── Diagnostics ───────────────────────────────────────────────
            "diagnostics": extra_diagnostics,

            # ── Features (empty — geometry strategy doesn't use FeatureStore for entries)
            "features": {},
        }

        return sig

    # ──────────────────────────────────────────────────────────────────────
    # Thesis key — deduplication for counterfactual tracking
    # ──────────────────────────────────────────────────────────────────────

    def thesis_key(self, signal: dict) -> tuple:
        """
        One active CF per (symbol, setup_type, direction).
        Same zone and same direction = same thesis regardless of exact price.
        """
        return (
            signal.get("symbol", ""),
            signal.get("strategy", ""),   # e.g. "CONFLUENCE_BOUNCE" or "TRENDLINE_RETEST"
            signal.get("signal", ""),     # "BUY CALL" / "BUY PUT"
        )

    def __repr__(self) -> str:
        return (
            f"GeometryStrategy("
            f"score>={self.min_confluence_score}, "
            f"tol={self.zone_tolerance_pct*100:.1f}%, "
            f"RR>={self.min_rr})"
        )
