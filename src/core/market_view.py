#!/usr/bin/env python3
"""
MarketViewEngine — the shared confluence layer.
===============================================
Computed once per snapshot (in IndicatorPipeline). Aggregates chart patterns,
structural bias, regime, and participation (RVOL) into a single ``MarketView``
that answers: *what is the market doing right now, and how strong is the case?*

Two jobs:
  1. Give the dashboard/operator a plain-language "current market view".
  2. Provide composability: every strategy can call
        view.confluence_boost("BUY CALL")
     to nudge its own confidence up (agreement) or down (conflict), so a pattern
     that stands alone can also strengthen another strategy's signal — without
     any strategy needing to know about any other strategy.

Scoring (all 0..1 unless noted):
  bull_score / bear_score  — weighted agreement for each direction
  vol_score                — volatility-expansion pressure (squeeze, low ATR pct)
  directional_score        — bull_score - bear_score, in [-1, +1]
  regime label             — TREND_UP / TREND_DOWN / RANGE / VOLATILE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.core.chart_patterns import (
    PatternSignal, detect_all, BULLISH, BEARISH, VOLATILE, NEUTRAL,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketView:
    symbol: str
    directional_score: float          # [-1, +1]  (+ bullish, - bearish)
    bull_score: float                 # [0, 1]
    bear_score: float                 # [0, 1]
    vol_score: float                  # [0, 1] volatility-expansion pressure
    regime_label: str                 # TREND_UP | TREND_DOWN | RANGE | VOLATILE
    bias: str                         # daily structural bias
    rvol: float
    patterns: List[PatternSignal] = field(default_factory=list)
    summary: str = ""

    @property
    def dominant_direction(self) -> str:
        if self.regime_label == "VOLATILE" and self.vol_score >= max(self.bull_score, self.bear_score):
            return VOLATILE
        if self.bull_score > self.bear_score and self.bull_score > 0.15:
            return BULLISH
        if self.bear_score > self.bull_score and self.bear_score > 0.15:
            return BEARISH
        return NEUTRAL

    def confluence_boost(self, side: str) -> float:
        """Confidence multiplier for a directional signal, in [0.7, 1.3].

        Agreement with the aggregated view boosts; conflict dampens. Designed to
        be applied as ``confidence *= view.confluence_boost(side)``.
        """
        want = BULLISH if "CALL" in (side or "") else BEARISH
        agree = self.bull_score if want == BULLISH else self.bear_score
        oppose = self.bear_score if want == BULLISH else self.bull_score
        # base 1.0, +/- up to 0.3 based on net agreement
        return float(max(0.7, min(1.3, 1.0 + 0.3 * (agree - oppose))))

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "directional_score": round(self.directional_score, 3),
            "bull_score": round(self.bull_score, 3),
            "bear_score": round(self.bear_score, 3),
            "vol_score": round(self.vol_score, 3),
            "regime_label": self.regime_label,
            "dominant_direction": self.dominant_direction,
            "bias": self.bias,
            "rvol": round(self.rvol, 2),
            "patterns": [p.as_dict() for p in self.patterns],
            "summary": self.summary,
        }


class MarketViewEngine:
    """Builds a MarketView from the raw frames + already-computed context."""

    def build(
        self,
        symbol: str,
        m5: pd.DataFrame,
        h1: Optional[pd.DataFrame],
        atr: float,
        atr_percentile: float,
        daily_bias: str,
        market_regime: str,
        rvol: float,
        rsi: Optional[pd.Series] = None,
    ) -> MarketView:
        try:
            # Patterns on the entry timeframe (m5) plus larger H1 reversal patterns.
            patterns: List[PatternSignal] = detect_all(m5, atr, rsi)
            if h1 is not None and len(h1) >= 30:
                from src.core.chart_patterns import detect_head_shoulders, detect_double_top_bottom
                for det in (detect_head_shoulders, detect_double_top_bottom):
                    p = det(h1, atr)
                    if p is not None:
                        # H1 patterns carry more weight — mark and slightly boost
                        patterns.append(PatternSignal(
                            f"H1_{p.name}", p.direction, min(1.0, p.confidence * 1.1),
                            "H1 " + p.rationale, p.key_level, p.target, p.stop_hint))

            bull = sum(p.confidence for p in patterns if p.direction == BULLISH)
            bear = sum(p.confidence for p in patterns if p.direction == BEARISH)
            vol = sum(p.confidence for p in patterns if p.direction == VOLATILE)

            # Structural bias contributes to the directional case.
            if daily_bias == BULLISH:
                bull += 0.4
            elif daily_bias == BEARISH:
                bear += 0.4

            # Regime string (from RegimeEngine) contributes.
            reg = (market_regime or "").upper()
            if "UP" in reg or "BULL" in reg:
                bull += 0.3
            elif "DOWN" in reg or "BEAR" in reg:
                bear += 0.3
            if "RANGE" in reg or "SIDEWAYS" in reg:
                vol += 0.1

            # Participation amplifies whatever directional case exists.
            part = max(0.0, min(1.0, (rvol - 1.0)))  # 0 at rvol=1, 1 at rvol>=2
            bull *= (1.0 + 0.25 * part)
            bear *= (1.0 + 0.25 * part)

            # Low ATR percentile = compressed volatility = expansion pressure.
            if atr_percentile <= 0.25:
                vol += (0.25 - atr_percentile) * 2.0  # up to +0.5

            # Normalise to 0..1 with a soft cap.
            def norm(x: float) -> float:
                return float(max(0.0, min(1.0, x / 1.5)))

            bull_s, bear_s, vol_s = norm(bull), norm(bear), norm(vol)
            directional = round(bull_s - bear_s, 3)

            # Regime label.
            if vol_s >= 0.6 and vol_s > max(bull_s, bear_s):
                label = "VOLATILE"
            elif bull_s > bear_s + 0.2:
                label = "TREND_UP"
            elif bear_s > bull_s + 0.2:
                label = "TREND_DOWN"
            else:
                label = "RANGE"

            names = ", ".join(p.name for p in patterns) or "none"
            summary = (
                f"{label} | dir={directional:+.2f} (bull {bull_s:.2f} / bear {bear_s:.2f}) "
                f"| vol {vol_s:.2f} | bias {daily_bias} | RVOL {rvol:.2f} | patterns: {names}"
            )

            return MarketView(
                symbol=symbol, directional_score=directional,
                bull_score=bull_s, bear_score=bear_s, vol_score=vol_s,
                regime_label=label, bias=daily_bias, rvol=rvol,
                patterns=patterns, summary=summary,
            )
        except Exception as e:
            logger.error(f"[MarketViewEngine] build failed for {symbol}: {e}", exc_info=True)
            return MarketView(symbol, 0.0, 0.0, 0.0, 0.0, "RANGE", daily_bias, rvol, [], "unavailable")
