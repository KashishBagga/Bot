#!/usr/bin/env python3
"""
Level Engine (MKE Stage 5 — Geometry Layer)
============================================
Detects raw horizontal price levels from multiple sources.
One HorizontalLevel per source — no merging (FusionEngine's responsibility).

Sources:
  PDH / PDL      → Previous day's high / low (from d1 DataFrame)
  PWH / PWL      → Previous week's high / low (from d1 DataFrame)
  OPEN_OF_DAY    → First bar of current session
  SUPPORT        → Confirmed swing lows (strength >= 0.45)
  RESISTANCE     → Confirmed swing highs (strength >= 0.45)
  ROUND_NUMBER   → 50pt grid (±3% of price), 100pt grid (±8% of price)
  VWAP / EMA20 / EMA50 → From FeatureStore (already computed)

Touch counting: abs(candle extreme - level) / level < 0.0005 (0.05%)
"""

import hashlib
import logging
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from src.core.market_geometry import (
    HorizontalLevel, LevelType, LevelDirection, LevelPriority,
    GeometryStatus, FormationReason
)
from src.core.market_knowledge import SwingPoint, StructureState, ResearchEvent

logger = logging.getLogger(__name__)

TOUCH_TOLERANCE = 0.0005          # 0.05% — price within this band = a touch
SWING_STRENGTH_THRESHOLD = 0.45   # Minimum swing strength to create a level
ROUND_NUMBER_STEP_NEAR = 50       # 50-point increments within ±3% of price
ROUND_NUMBER_STEP_FAR  = 100      # 100-point increments within ±8% of price


class LevelEngine:
    """Detects raw horizontal levels from all sources. No merging."""

    required_history = 2

    def detect_levels(
        self,
        m5:  pd.DataFrame,
        h1:  Optional[pd.DataFrame],
        d1:  Optional[pd.DataFrame],
        structure: StructureState,
        atr: float,
        vwap: float,
        ema20: float,
        ema50: float,
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> Tuple[List[HorizontalLevel], List[ResearchEvent]]:
        """
        Detect all raw horizontal levels for one tick.

        Returns:
            (levels, events) — flat list of all detected levels +
                               ResearchEvents for lifecycle changes.
        """
        levels: List[HorizontalLevel] = []
        events: List[ResearchEvent]   = []

        # ── 1. Institutional levels (PDH / PDL / PWH / PWL) ─────────────────
        inst_levels, inst_events = self._detect_institutional(d1, current_price, symbol, now)
        levels.extend(inst_levels)
        events.extend(inst_events)

        # ── 2. Open of day ────────────────────────────────────────────────────
        ood = self._detect_open_of_day(m5, current_price, symbol, now)
        if ood:
            levels.append(ood)

        # ── 3. Structural levels (swing highs / swing lows) ──────────────────
        struct_levels, struct_events = self._detect_structural(
            structure, m5, current_price, symbol, now
        )
        levels.extend(struct_levels)
        events.extend(struct_events)

        # ── 4. Round numbers ─────────────────────────────────────────────────
        round_levels = self._detect_round_numbers(current_price, symbol, now)
        levels.extend(round_levels)

        # ── 5. Technical indicators (VWAP, EMAs) ────────────────────────────
        tech_levels = self._detect_technical(
            vwap, ema20, ema50, m5, current_price, symbol, now
        )
        levels.extend(tech_levels)

        return levels, events

    # ── Institutional ────────────────────────────────────────────────────────

    def _detect_institutional(
        self,
        d1: Optional[pd.DataFrame],
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> Tuple[List[HorizontalLevel], List[ResearchEvent]]:
        levels: List[HorizontalLevel] = []
        events: List[ResearchEvent]   = []

        if d1 is None or len(d1) < 2:
            return levels, events

        try:
            # PDH / PDL: previous session row
            current_date = now.date() if hasattr(now, 'date') else date.today()
            d1_dates = [ts.date() if hasattr(ts, 'date') else ts for ts in d1.index]

            # Find the most recent completed session (not today)
            prev_rows = [(i, d) for i, d in enumerate(d1_dates) if d < current_date]
            if prev_rows:
                prev_idx = prev_rows[-1][0]
                prev_row = d1.iloc[prev_idx]
                pdh_price = float(prev_row["high"])
                pdl_price = float(prev_row["low"])

                for price, lt, direction in [
                    (pdh_price, LevelType.PDH, LevelDirection.RESISTANCE),
                    (pdl_price, LevelType.PDL, LevelDirection.SUPPORT),
                ]:
                    levels.append(self._build_level(
                        price=price, lt=lt, direction=direction,
                        priority=LevelPriority.INSTITUTIONAL,
                        current_price=current_price, symbol=symbol,
                        touches=1, strength=0.85, freshness=0.9,
                        provenance={"source": lt.value, "session_date": str(d1_dates[prev_idx])}
                    ))

            # PWH / PWL: previous 7 calendar days of d1 bars
            week_rows = [(i, d) for i, d in enumerate(d1_dates) if d < current_date]
            week_rows = week_rows[-7:] if len(week_rows) >= 7 else week_rows
            if week_rows:
                indices = [r[0] for r in week_rows]
                pwh_price = float(d1.iloc[indices]["high"].max())
                pwl_price = float(d1.iloc[indices]["low"].min())

                for price, lt, direction in [
                    (pwh_price, LevelType.PWH, LevelDirection.RESISTANCE),
                    (pwl_price, LevelType.PWL, LevelDirection.SUPPORT),
                ]:
                    levels.append(self._build_level(
                        price=price, lt=lt, direction=direction,
                        priority=LevelPriority.INSTITUTIONAL,
                        current_price=current_price, symbol=symbol,
                        touches=1, strength=0.80, freshness=0.6,
                        provenance={"source": lt.value, "lookback_days": len(week_rows)}
                    ))

        except Exception as e:
            logger.warning(f"[LevelEngine] Institutional level error: {e}")

        return levels, events

    # ── Open of Day ──────────────────────────────────────────────────────────

    def _detect_open_of_day(
        self,
        m5: pd.DataFrame,
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> Optional[HorizontalLevel]:
        try:
            current_date = now.date() if hasattr(now, 'date') else date.today()
            today_mask = [ts.date() == current_date for ts in m5.index]
            today_bars = m5[today_mask]
            if len(today_bars) > 0:
                open_price = float(today_bars.iloc[0]["open"])
                return self._build_level(
                    price=open_price, lt=LevelType.OPEN_OF_DAY,
                    direction=LevelDirection.SUPPORT if open_price <= current_price else LevelDirection.RESISTANCE,
                    priority=LevelPriority.STRUCTURAL,
                    current_price=current_price, symbol=symbol,
                    touches=1, strength=0.60, freshness=0.8,
                    provenance={"source": "OPEN_OF_DAY"}
                )
        except Exception as e:
            logger.warning(f"[LevelEngine] Open of day error: {e}")
        return None

    # ── Structural (swing-derived) ────────────────────────────────────────────

    def _detect_structural(
        self,
        structure: StructureState,
        m5: pd.DataFrame,
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> Tuple[List[HorizontalLevel], List[ResearchEvent]]:
        levels: List[HorizontalLevel] = []
        events: List[ResearchEvent]   = []

        closes  = m5["close"].values
        highs   = m5["high"].values
        lows    = m5["low"].values

        for swing in structure.swings:
            if swing.strength < SWING_STRENGTH_THRESHOLD:
                continue

            is_high = swing.type == "HIGH"
            direction = LevelDirection.RESISTANCE if is_high else LevelDirection.SUPPORT
            lt = LevelType.RESISTANCE if is_high else LevelType.SUPPORT

            # Count how many bars have touched this swing level
            touches = self._count_touches(
                swing.price, highs if is_high else lows, TOUCH_TOLERANCE
            )

            # Detect role reversal: was resistance, now price is consistently above it
            role_reversal = self._check_role_reversal(
                swing.price, closes, is_high
            )

            # Confidence: starts from swing strength, decays with distance
            confidence = round(swing.confidence * 0.9, 3)

            # Freshness: recent swings = fresh
            freshness = round(max(0.1, swing.confidence), 3)

            levels.append(self._build_level(
                price=swing.price, lt=lt, direction=direction,
                priority=LevelPriority.STRUCTURAL,
                current_price=current_price, symbol=symbol,
                touches=max(1, touches), strength=swing.strength,
                freshness=freshness, role_reversal=role_reversal,
                confidence=confidence,
                provenance={
                    "source": "SWING", "swing_id": swing.id,
                    "swing_strength": swing.strength
                }
            ))

        return levels, events

    # ── Round Numbers ─────────────────────────────────────────────────────────

    def _detect_round_numbers(
        self,
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> List[HorizontalLevel]:
        levels: List[HorizontalLevel] = []

        # 50-point grid within ±3%
        near_low  = current_price * 0.97
        near_high = current_price * 1.03
        far_low   = current_price * 0.92
        far_high  = current_price * 1.08

        seen_prices = set()

        def add_grid(step: float, lo: float, hi: float):
            start = int(lo / step) * step
            p = start
            while p <= hi + step:
                rp = round(p, 2)
                if lo <= rp <= hi and rp not in seen_prices:
                    seen_prices.add(rp)
                    direction = (LevelDirection.RESISTANCE if rp > current_price
                                 else LevelDirection.SUPPORT)
                    # Stronger confidence for rounder numbers (100 > 50)
                    conf = 0.75 if step >= 100 else 0.60
                    levels.append(self._build_level(
                        price=rp, lt=LevelType.ROUND_NUMBER, direction=direction,
                        priority=LevelPriority.TECHNICAL,
                        current_price=current_price, symbol=symbol,
                        touches=0, strength=conf, freshness=1.0,
                        confidence=conf,
                        provenance={"source": "ROUND_NUMBER", "step": step}
                    ))
                p += step

        add_grid(ROUND_NUMBER_STEP_NEAR, near_low, near_high)
        add_grid(ROUND_NUMBER_STEP_FAR, far_low, far_high)

        return levels

    # ── Technical Indicators ──────────────────────────────────────────────────

    def _detect_technical(
        self,
        vwap_price: float,
        ema20: float,
        ema50: float,
        m5: pd.DataFrame,
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> List[HorizontalLevel]:
        levels: List[HorizontalLevel] = []

        for price, lt, conf in [
            (vwap_price, LevelType.VWAP,  0.70),
            (ema20,      LevelType.EMA20,  0.60),
            (ema50,      LevelType.EMA50,  0.65),
        ]:
            if price <= 0:
                continue
            direction = (LevelDirection.RESISTANCE if price > current_price
                         else LevelDirection.SUPPORT)
            levels.append(self._build_level(
                price=price, lt=lt, direction=direction,
                priority=LevelPriority.TECHNICAL,
                current_price=current_price, symbol=symbol,
                touches=0, strength=conf, freshness=1.0,
                confidence=conf,
                provenance={"source": lt.value}
            ))

        return levels

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_level(
        self,
        price: float,
        lt: LevelType,
        direction: LevelDirection,
        priority: LevelPriority,
        current_price: float,
        symbol: str,
        touches: int,
        strength: float,
        freshness: float,
        role_reversal: bool = False,
        confidence: float = 0.7,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> HorizontalLevel:
        distance_pct = abs(price - current_price) / current_price if current_price > 0 else 0.0
        level_id = f"lv_{symbol.replace(':', '_').replace('-', '_')}_{lt.value}_{int(price * 100)}"
        return HorizontalLevel(
            id=level_id,
            price=round(price, 2),
            type=lt,
            direction=direction,
            priority=priority,
            status=GeometryStatus.ACTIVE,
            touches=touches,
            strength=round(strength, 3),
            freshness=round(freshness, 3),
            role_reversal=role_reversal,
            confidence=round(confidence, 3),
            distance_pct=round(distance_pct, 6),
            provenance=provenance or {},
        )

    def _count_touches(self, level_price: float, extremes: np.ndarray, tol: float) -> int:
        """Count how many bars came within tolerance of the level price."""
        return int(np.sum(np.abs(extremes - level_price) / level_price < tol))

    def _check_role_reversal(
        self, level_price: float, closes: np.ndarray, was_resistance: bool
    ) -> bool:
        """
        Detect role reversal:
        - A level that was resistance (swing high) and price has since closed consistently above it.
        - A level that was support (swing low) and price has since closed consistently below it.
        """
        recent_closes = closes[-20:] if len(closes) >= 20 else closes
        if was_resistance:
            above_count = int(np.sum(recent_closes > level_price))
            return above_count >= len(recent_closes) * 0.7
        else:
            below_count = int(np.sum(recent_closes < level_price))
            return below_count >= len(recent_closes) * 0.7
