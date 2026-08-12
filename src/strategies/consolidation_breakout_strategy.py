#!/usr/bin/env python3
"""
Consolidation Breakout Strategy (v1.0)
======================================
Hypothesis: Low-volatility consolidation zones on the H1 timeframe followed
by M5 close above/below the zone boundary, confirmed by RVOL ≥ threshold and
BreakoutEngine score ≥ 60, produces positive expectancy.

Two experiments:
  Consolidation_Breakout_v1.0       — RVOL ≥ 1.5 (standard participation)
  Consolidation_Breakout_Tight_v1.0 — RVOL ≥ 2.0 (high-conviction only)

ATR: uses the project's canonical True-Range formula from IndicatorPipeline._compute_atr.
Touch counting: boundary-defining candles excluded; 2-bar minimum cluster separation.
RSI: "RSI Momentum Confirmation" (RSI > 50 for bullish, < 50 for bearish).
      This is NOT RSI divergence. Real divergence detection deferred to v2.
SL/TP: explicit R-based. TP = entry + max(2R, 2×zone_range). Always ≥ 2R.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.breakout_engine import BreakoutEngine, detect_consolidation_zone, ConsolidationZone
from src.core.indicator_pipeline import IndicatorPipeline

logger = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI — reuses same logic as IndicatorPipeline."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs    = gain / loss.replace(0, float('nan'))
    return 100.0 - (100.0 / (1.0 + rs))


class ConsolidationBreakoutStrategy(BaseStrategy):
    """
    1H consolidation squeeze + M5 breakout + RSI Momentum Confirmation.
    PAPER maturity only. Promotion criteria defined in implementation_plan.md.
    """

    metadata = StrategyMetadata(
        id="consolidation_breakout",
        name="Consolidation Breakout Strategy",
        hypothesis_id="h1_squeeze_m5_breakout",
        hypothesis_family="Breakout",
        hypothesis_text=(
            "Low-ATR H1 consolidation zones followed by M5 close breakout with "
            "RVOL confirmation produce positive expectancy over ≥200 PAPER trades."
        ),
        version="v1.0",
        maturity="PAPER",
        tags=["breakout", "consolidation", "squeeze", "h1", "m5", "rvol"],
        expected_holding=(15, 60),
    )

    def __init__(
        self,
        rvol_threshold: float = 1.5,
        breakout_score_min: int = 60,
        atr_pct_threshold: float = 30.0,    # ATR must be in bottom 30th percentile
        max_zone_atr_mult: float = 1.5,
        min_touches: int = 3,
        tp_zone_mult: float = 2.0,          # TP = entry + max(2R, tp_zone_mult×zone_range)
        rsi_period: int = 14,
        sl_atr_buffer: float = 0.10,        # SL buffer below zone (×ATR)
        min_r: float = 2.0,
    ):
        self.rvol_threshold      = rvol_threshold
        self.breakout_score_min  = breakout_score_min
        self.atr_pct_threshold   = atr_pct_threshold
        self.max_zone_atr_mult   = max_zone_atr_mult
        self.min_touches         = min_touches
        self.tp_zone_mult        = tp_zone_mult
        self.rsi_period          = rsi_period
        self.sl_atr_buffer       = sl_atr_buffer
        self.min_r               = min_r
        self._breakout_engine    = BreakoutEngine()

    # ── H1 resampling ─────────────────────────────────────────────────────────

    def _get_h1(self, snapshot: MarketSnapshot) -> Optional[pd.DataFrame]:
        """Use snapshot.h1 if available, otherwise resample from m5."""
        if snapshot.h1 is not None and len(snapshot.h1) >= 30:
            return snapshot.h1
        if snapshot.m5 is not None and len(snapshot.m5) >= 60:
            return (
                snapshot.m5
                .resample("1h")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna()
            )
        return None

    # ── Breakout detection ────────────────────────────────────────────────────

    def _is_breakout_candle(
        self, m5: pd.DataFrame, zone: ConsolidationZone
    ) -> Optional[str]:
        """
        Returns 'BULL' | 'BEAR' | None.
        Breakout requires M5 close outside zone AND prior M5 close inside zone.
        """
        if len(m5) < 2:
            return None
        curr_close = float(m5.iloc[-1]["close"])
        prev_close = float(m5.iloc[-2]["close"])
        if curr_close > zone.top and prev_close <= zone.top:
            return "BULL"
        if curr_close < zone.bottom and prev_close >= zone.bottom:
            return "BEAR"
        return None

    # ── SL / TP calculation ───────────────────────────────────────────────────

    def _sl_tp(
        self, direction: str, entry: float, zone: ConsolidationZone
    ) -> tuple:
        """
        Explicit R-based SL/TP.
        SL: just outside the opposite zone boundary.
        TP: entry + max(min_r × R, tp_zone_mult × zone_range).
        Guaranteed: (tp - entry) >= min_r × R.
        """
        buf = self.sl_atr_buffer * zone.atr
        if direction == "BULL":
            sl = zone.bottom - buf
            R  = entry - sl
            tp = entry + max(self.min_r * R, self.tp_zone_mult * zone.range)
        else:
            sl = zone.top + buf
            R  = sl - entry
            tp = entry - max(self.min_r * R, self.tp_zone_mult * zone.range)
        return round(sl, 2), round(R, 2), round(tp, 2)

    # ── Main evaluate() ────────────────────────────────────────────────────────

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str]   = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5 = snapshot.m5
            if m5 is None or len(m5) < 30:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_M5"])

            h1 = self._get_h1(snapshot)
            if h1 is None or len(h1) < 30:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_H1"])

            # ── Detect consolidation zone ────────────────────────────────────
            zone = detect_consolidation_zone(
                h1,
                atr_pct_threshold=self.atr_pct_threshold,
                max_zone_atr_mult=self.max_zone_atr_mult,
                min_touches=self.min_touches,
            )
            if zone is None:
                return self._empty_result(
                    experiment_name,
                    diagnostics={"reason": "NO_CONSOLIDATION_ZONE"},
                )

            # ── Breakout candle check (M5 close outside zone) ────────────────
            direction = self._is_breakout_candle(m5, zone)
            if direction is None:
                return self._empty_result(
                    experiment_name,
                    diagnostics={"zone": f"{zone.bottom:.0f}–{zone.top:.0f}",
                                 "reason": "NO_BREAKOUT_CANDLE"},
                )

            entry = snapshot.current_price
            rvol  = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            # ── BreakoutEngine score ─────────────────────────────────────────
            levels = {"resistance": zone.top, "support": zone.bottom}
            bo_result = self._breakout_engine.analyze(m5, levels)
            bo_score  = bo_result.get("confidence", 0)

            # ── RSI Momentum Confirmation ─────────────────────────────────────
            # NOT RSI divergence. RSI >50 = bullish momentum, <50 = bearish.
            # Divergence detection deferred to v2.
            h1_rsi_series = _compute_rsi(h1["close"], period=self.rsi_period)
            h1_rsi = float(h1_rsi_series.iloc[-1]) if not h1_rsi_series.dropna().empty else 50.0

            # ── SL / TP ──────────────────────────────────────────────────────
            sl, R, tp = self._sl_tp(direction, entry, zone)

            # ── Collect rejection reasons ────────────────────────────────────
            rejection_reasons: List[str] = []

            if rvol < self.rvol_threshold:
                rejection_reasons.append(f"LOW_RVOL:{rvol:.2f}<{self.rvol_threshold}")
            if bo_score < self.breakout_score_min:
                rejection_reasons.append(f"LOW_BREAKOUT_SCORE:{bo_score}<{self.breakout_score_min}")
            if direction == "BULL" and h1_rsi < 50:
                rejection_reasons.append(f"RSI_MOMENTUM_WEAK:{h1_rsi:.1f}<50")
            if direction == "BEAR" and h1_rsi > 50:
                rejection_reasons.append(f"RSI_MOMENTUM_WEAK:{h1_rsi:.1f}>50")
            if R <= 0:
                rejection_reasons.append("ZERO_RISK")

            now = snapshot.timestamp
            if isinstance(now, datetime) and now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)

            side = "BUY CALL" if direction == "BULL" else "BUY PUT"
            accepted = len(rejection_reasons) == 0

            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}"
                f"_CONSOL_BRK_{direction}"
                f"_{now.strftime('%Y%m%d_%H%M%S') if hasattr(now, 'strftime') else ''}"
            )

            # RR ratio
            rr = round(abs(tp - entry) / R, 2) if R > 0 else 0.0

            sig: Dict[str, Any] = {
                "symbol":           snapshot.symbol,
                "signal":           side,
                "strategy":         "CONSOLIDATION_BREAKOUT",
                "direction":        direction,
                "price":            entry,
                "stop_loss":        sl,
                "take_profit":      tp,
                "rr_ratio":         rr,
                "R":                R,
                # ── Zone metadata ────────────────────────────────────────────
                "zone_top":         zone.top,
                "zone_bottom":      zone.bottom,
                "zone_range":       round(zone.range, 2),
                "zone_atr":         round(zone.atr, 2),
                "zone_atr_pct":     round(zone.atr_percentile, 1),
                "top_touches":      zone.top_touches,
                "bot_touches":      zone.bot_touches,
                # ── Confirmation ─────────────────────────────────────────────
                "rvol":             round(rvol, 3),
                "breakout_score":   bo_score,
                "h1_rsi":           round(h1_rsi, 1),
                # ── Experiment provenance ─────────────────────────────────────
                "strategy_version": f"Consolidation_Breakout_v1.0_{self.rvol_threshold}rvol",
                # ── Result ───────────────────────────────────────────────────
                "timestamp":        now.isoformat() if hasattr(now, "isoformat") else str(now),
                "accepted":         accepted,
                "rejection_reasons": rejection_reasons,
                "candidate_id":     candidate_id,
                "diagnostics": {
                    "rvol_threshold":    self.rvol_threshold,
                    "bo_score":          bo_score,
                    "bo_is_trap":        bo_result.get("is_trap", False),
                },
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[ConsolidationBreakoutStrategy] {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.metadata.id,
            version=self.metadata.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
