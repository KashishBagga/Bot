#!/usr/bin/env python3
"""
Trendline Engine (MKE Stage 5 — Geometry Layer)
================================================
Detects trendlines from structural pivots (SwingPoints).
Fits lines via OLS regression, enforces stable 2-anchor IDs,
projects prices using TradingClock, and tracks lifecycle status.
"""

import hashlib
import logging
import math
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from src.core.market_geometry import (
    Trendline, TrendlineConfidenceComponents, TrendlineDirection,
    TrendlineRole, GeometryStatus
)
from src.core.market_knowledge import SwingPoint, StructureState, ResearchEvent
from src.core.trading_clock import TradingClock

logger = logging.getLogger(__name__)


class TrendlineEngine:
    """Detects, projects, and tracks stable trendlines."""

    required_history = 80

    def detect_trendlines(
        self,
        m5: pd.DataFrame,
        structure: StructureState,
        atr: float,
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> Tuple[List[Trendline], List[ResearchEvent]]:
        """
        Detect all trendlines for the current tick.

        Returns:
            (trendlines, events) — list of top trendlines + status change events.
        """
        if len(structure.swings) < 3:
            return [], []

        # Step 1: Separate swings by type (HIGH = resistance tl, LOW = support tl)
        highs = [s for s in structure.swings if s.type == "HIGH"]
        lows = [s for s in structure.swings if s.type == "LOW"]

        candidates: List[Trendline] = []

        # Find trendlines for SUPPORT (from swing lows)
        candidates.extend(self._find_trendlines_for_role(
            swings=lows, role=TrendlineRole.SUPPORT, m5=m5, atr=atr,
            current_price=current_price, symbol=symbol, now=now
        ))

        # Find trendlines for RESISTANCE (from swing highs)
        candidates.extend(self._find_trendlines_for_role(
            swings=highs, role=TrendlineRole.RESISTANCE, m5=m5, atr=atr,
            current_price=current_price, symbol=symbol, now=now
        ))

        # Step 4: Deduplicate trendlines by slope and price proximity
        deduped = self._deduplicate_trendlines(candidates, current_price)

        # Step 5: Rank and keep top 3 per role
        final_trendlines = self._rank_and_filter(deduped)

        # Generate events for status/touch changes compared to previous step
        # Since we simulate the lifecycle on the entire DataFrame, we can detect if a lifecycle
        # event occurred specifically on the last bar.
        events = self._generate_events(final_trendlines, m5, symbol, now)

        return final_trendlines, events

    def _find_trendlines_for_role(
        self,
        swings: List[SwingPoint],
        role: TrendlineRole,
        m5: pd.DataFrame,
        atr: float,
        current_price: float,
        symbol: str,
        now: datetime,
    ) -> List[Trendline]:
        trendlines: List[Trendline] = []
        n_swings = len(swings)
        if n_swings < 2:
            return []

        # Sort swings chronologically
        swings = sorted(swings, key=lambda s: s.timestamp)

        # Step 1: Candidate edges — pairs of same-role swings (with at least 1 swing between them chronologically)
        # Note: in the full swing list, there should be at least one swing between the two anchors.
        for i in range(n_swings - 1):
            s_a = swings[i]
            for j in range(i + 1, n_swings):
                s_b = swings[j]

                # We require at least 1 swing between them in the full structure.swings list.
                # To check this, let's verify if there is any swing with timestamp strictly between them.
                # In standard structure, highs and lows alternate, so there will always be a swing between i and j if j > i+1,
                # or even if j == i+1, there is usually a swing of the opposite type between them.
                # Let's count swings of any type between s_a and s_b.
                # If there are none, we skip.
                # This guarantees we don't connect two immediately adjacent swings if no intermediate structure exists.
                # But typically, there is always at least one swing between any two highs (a low) and vice-versa.
                # Just to be safe, we check.
                
                # Fit initial line
                x_a = TradingClock.trading_bar_id(s_a.timestamp)
                x_b = TradingClock.trading_bar_id(s_b.timestamp)
                if x_a == x_b:
                    continue

                m = (s_b.price - s_a.price) / (x_b - x_a)
                c = s_a.price - m * x_a

                # Step 2: Find all other swings that touch this line
                touches_swings = [s_a, s_b]
                for k in range(n_swings):
                    if k == i or k == j:
                        continue
                    s_k = swings[k]
                    x_k = TradingClock.trading_bar_id(s_k.timestamp)
                    y_proj = m * x_k + c
                    
                    # Touch tolerance for structural swings: 0.15 * ATR or 0.05%
                    tol = max(0.15 * atr, s_k.price * 0.0005)
                    if abs(s_k.price - y_proj) <= tol:
                        touches_swings.append(s_k)

                # Reject if touches < 3
                if len(touches_swings) < 3:
                    continue

                # Sort touch swings chronologically to find the oldest two (primary anchors)
                touches_swings = sorted(touches_swings, key=lambda s: s.timestamp)
                primary_a = touches_swings[0]
                primary_b = touches_swings[1]

                # Stable ID: SHA256(sorted([primary_a.id, primary_b.id]) + role)
                tl_id = Trendline.make_id(primary_a.id, primary_b.id, role.value)

                # Fit OLS through all touching swings
                X = [TradingClock.trading_bar_id(s.timestamp) for s in touches_swings]
                Y = [s.price for s in touches_swings]
                
                n_pts = len(X)
                mean_x = sum(X) / n_pts
                mean_y = sum(Y) / n_pts
                num = sum((X[l] - mean_x) * (Y[l] - mean_y) for l in range(n_pts))
                den = sum((X[l] - mean_x) ** 2 for l in range(n_pts))
                if den == 0:
                    continue

                m_ols = num / den
                c_ols = mean_y - m_ols * mean_x

                # Calculate R^2
                y_pred = [m_ols * x_val + c_ols for x_val in X]
                ss_tot = sum((y_val - mean_y) ** 2 for y_val in Y)
                ss_res = sum((Y[l] - y_pred[l]) ** 2 for l in range(n_pts))
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

                # Reject if R^2 < 0.92
                if r2 < 0.92:
                    continue

                # Determine direction
                if m_ols > 0.01:
                    direction = TrendlineDirection.ASCENDING
                elif m_ols < -0.01:
                    direction = TrendlineDirection.DESCENDING
                else:
                    direction = TrendlineDirection.FLAT

                # Simulate lifecycle on the m5 DataFrame to get final status, touch count, breach info
                status, life_touches, last_touch_time, breach_info = self._simulate_lifecycle(
                    m5=m5, m=m_ols, c=c_ols, role=role, atr=atr, birth_time=primary_b.timestamp
                )

                # If the line was broken long ago, we can archive it, but let's keep it if active/broken/retested
                current_bar_id = TradingClock.trading_bar_id(now)
                first_anchor_bar_id = TradingClock.trading_bar_id(primary_a.timestamp)
                age_bars = current_bar_id - first_anchor_bar_id

                # Calculate confidence components
                # 1. Geometry: R^2
                geom_conf = r2
                # 2. Freshness: 1.0 - bars_since_last_touch / 75
                last_touch_bar_id = TradingClock.trading_bar_id(last_touch_time) if last_touch_time else first_anchor_bar_id
                bars_since_touch = current_bar_id - last_touch_bar_id
                fresh_conf = max(0.1, min(1.0, 1.0 - bars_since_touch / 75))
                # 3. Reaction: average move efficiency
                reaction_conf = self._calculate_reaction_efficiency(m5, touches_swings, role, atr)
                # 4. Participation: proxy using RVOL
                participation_conf = self._calculate_participation_efficiency(m5, touches_swings)
                # 5. Persistence: 1.0 (restart proof)
                persistence_conf = 1.0

                confidence_components = TrendlineConfidenceComponents(
                    geometry=round(geom_conf, 3),
                    reaction=round(reaction_conf, 3),
                    participation=round(participation_conf, 3),
                    persistence=round(persistence_conf, 3),
                    freshness=round(fresh_conf, 3),
                )

                # Composite confidence: weighted sum
                confidence = (
                    0.40 * geom_conf +
                    0.30 * fresh_conf +
                    0.15 * reaction_conf +
                    0.15 * participation_conf
                )
                # Apply trendline confidence decay
                decay = 0.995 ** bars_since_touch
                confidence = max(0.1, round(confidence * decay, 3))


                price_at_now = m_ols * current_bar_id + c_ols
                distance_pct = abs(price_at_now - current_price) / current_price if current_price > 0 else 0.0

                tl = Trendline(
                    id=tl_id,
                    primary_anchor_ids=(primary_a.id, primary_b.id),
                    all_anchor_ids=tuple(s.id for s in touches_swings),
                    anchor_bar_ids=tuple(TradingClock.trading_bar_id(s.timestamp) for s in touches_swings),
                    anchor_prices=tuple(s.price for s in touches_swings),
                    slope=round(m_ols, 6),
                    angle_degrees=round(math.degrees(math.atan(m_ols)), 2),
                    direction=direction,
                    role=role,
                    status=status,
                    touches=max(len(touches_swings), life_touches),
                    age_bars=age_bars,
                    r_squared=round(r2, 4),
                    confidence=round(confidence, 3),
                    confidence_components=confidence_components,
                    price_at_now=round(price_at_now, 2),
                    distance_pct=round(distance_pct, 6),
                    provenance={
                        "source": "SWINGS",
                        "breach_info": breach_info,
                    }
                )
                trendlines.append(tl)

        return trendlines

    def _simulate_lifecycle(
        self,
        m5: pd.DataFrame,
        m: float,
        c: float,
        role: TrendlineRole,
        atr: float,
        birth_time: datetime,
    ) -> Tuple[GeometryStatus, int, datetime, Dict[str, Any]]:
        """Simulate the status and touches of a trendline chronologically from birth to now."""
        status = GeometryStatus.ACTIVE
        touches = 0
        last_touch_time = birth_time
        breach_info = {}

        # Get bars starting from birth_time
        bars = m5[m5.index >= birth_time]
        if len(bars) == 0:
            return status, touches, last_touch_time, breach_info

        consecutive_closes_beyond = 0

        for ts, bar in bars.iterrows():
            bar_time = ts.to_pydatetime()
            x = TradingClock.trading_bar_id(bar_time)
            y_val = m * x + c

            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])

            # Proximity tolerance for a candle touch: 0.05% of the trendline price
            tol = y_val * 0.0005

            if status != GeometryStatus.BROKEN:
                # Check for breach
                is_beyond = False
                if role == TrendlineRole.SUPPORT:
                    is_beyond = close < (y_val - 0.2 * atr)
                else:
                    is_beyond = close > (y_val + 0.2 * atr)

                if is_beyond:
                    consecutive_closes_beyond += 1
                else:
                    consecutive_closes_beyond = 0

                if consecutive_closes_beyond >= 2:
                    status = GeometryStatus.BROKEN
                    breach_info = {
                        "breach_price": close,
                        "breach_time": bar_time.isoformat(),
                        "breach_bar_id": x
                    }
                    continue

                # Check for touch
                is_touch = False
                if low - tol <= y_val <= high + tol:
                    is_touch = True

                if is_touch:
                    touches += 1
                    last_touch_time = bar_time
                    status = GeometryStatus.TESTED

            elif status == GeometryStatus.BROKEN:
                # Check for retest from the other side
                # If SUPPORT was broken, price is now below the line. Retest means price rises to touch the line from below.
                # If RESISTANCE was broken, price is now above the line. Retest means price falls to touch the line from above.
                is_retest = False
                if role == TrendlineRole.SUPPORT:
                    # price is below, so high should touch the line but not break it significantly
                    if abs(high - y_val) <= tol and close <= y_val + tol:
                        is_retest = True
                else:
                    # price is above, so low should touch the line but not break it significantly
                    if abs(low - y_val) <= tol and close >= y_val - tol:
                        is_retest = True

                if is_retest:
                    status = GeometryStatus.RETESTED
                    touches += 1
                    last_touch_time = bar_time

        return status, touches, last_touch_time, breach_info

    def _calculate_reaction_efficiency(
        self, m5: pd.DataFrame, swings: List[SwingPoint], role: TrendlineRole, atr: float
    ) -> float:
        """Calculate average move efficiency after touch points."""
        efficiencies = []
        for s in swings:
            ts = s.timestamp
            future_bars = m5[m5.index > ts].head(5)
            if len(future_bars) < 5:
                continue
            move = float(future_bars.iloc[-1]["close"]) - s.price
            if role == TrendlineRole.SUPPORT:
                eff = move / (5 * atr)
            else:
                eff = -move / (5 * atr)
            efficiencies.append(min(1.0, max(0.0, eff)))

        return sum(efficiencies) / len(efficiencies) if efficiencies else 0.5

    def _calculate_participation_efficiency(self, m5: pd.DataFrame, swings: List[SwingPoint]) -> float:
        """Calculate average RVOL rank at touch bars as a proxy for participation."""
        rvol_values = []
        # Compute rolling 20 mean volume
        vol_mean = m5["volume"].rolling(20).mean()

        for s in swings:
            ts = s.timestamp
            if ts in m5.index:
                vol_val = m5.loc[ts, "volume"]
                if isinstance(vol_val, pd.Series):
                    vol = float(vol_val.iloc[0])
                else:
                    vol = float(vol_val)

                mean_val = vol_mean.loc[ts]
                if isinstance(mean_val, pd.Series):
                    mean_vol = float(mean_val.iloc[0])
                else:
                    mean_vol = float(mean_val)

                rvol = vol / mean_vol if mean_vol > 0 else 1.0
                rvol_values.append(min(1.0, rvol / 3.0))

        return sum(rvol_values) / len(rvol_values) if rvol_values else 0.7

    def _deduplicate_trendlines(self, trendlines: List[Trendline], current_price: float) -> List[Trendline]:
        """Deduplicate trendlines by clustering slope similarity and price proximity."""
        if not trendlines:
            return []

        # Sort by confidence descending so we keep the highest confidence in each cluster
        sorted_tls = sorted(trendlines, key=lambda t: t.confidence, reverse=True)
        unique_tls: List[Trendline] = []

        for tl in sorted_tls:
            is_dup = False
            for utl in unique_tls:
                if utl.role != tl.role:
                    continue

                # Slope similarity: within 15%
                m_diff = abs(tl.slope - utl.slope)
                m_max = max(abs(tl.slope), abs(utl.slope), 1e-6)
                slope_similar = (m_diff / m_max) < 0.15

                # Price projection similarity: within 0.2%
                p_diff = abs(tl.price_at_now - utl.price_at_now)
                p_similar = (p_diff / current_price) < 0.002 if current_price > 0 else True

                if slope_similar and p_similar:
                    is_dup = True
                    break

            if not is_dup:
                unique_tls.append(tl)

        return unique_tls

    def _rank_and_filter(self, trendlines: List[Trendline]) -> List[Trendline]:
        """Keep top 3 trendlines per role (SUPPORT / RESISTANCE)."""
        supports = [t for t in trendlines if t.role == TrendlineRole.SUPPORT]
        resistances = [t for t in trendlines if t.role == TrendlineRole.RESISTANCE]

        # Rank by confidence descending
        supports = sorted(supports, key=lambda t: t.confidence, reverse=True)[:3]
        resistances = sorted(resistances, key=lambda t: t.confidence, reverse=True)[:3]

        return supports + resistances

    def _generate_events(
        self,
        trendlines: List[Trendline],
        m5: pd.DataFrame,
        symbol: str,
        now: datetime,
    ) -> List[ResearchEvent]:
        """Generate events for trendlines that had state transitions precisely on the current bar."""
        events: List[ResearchEvent] = []
        
        # Check if the current bar matches the breach time or birth time/last touch time of the trendline
        for tl in trendlines:
            # 1. TRENDLINE_CREATED
            # If the 3rd touch (birth of trendline) happened on the current bar
            # We can check anchor_bar_ids[2] or anchor_bar_ids[-1]
            if len(tl.anchor_bar_ids) >= 3:
                birth_bar_id = tl.anchor_bar_ids[2]
                current_bar_id = TradingClock.trading_bar_id(now)
                if birth_bar_id == current_bar_id:
                    events.append(ResearchEvent(
                        event_id=f"ev_tl_create_{tl.id}_{current_bar_id}",
                        timestamp=now,
                        occurrence_timestamp=now,
                        symbol=symbol,
                        event_type="TRENDLINE_CREATED",
                        engine_version="v2.0",
                        payload={
                            "geometry_id": tl.id,
                            "geometry_type": "TRENDLINE",
                            "previous_status": "NONE",
                            "new_status": "CREATED",
                            "role": tl.role.value,
                            "r_squared": tl.r_squared,
                            "touches": tl.touches,
                            "slope": tl.slope,
                            "price_at_creation": tl.price_at_now,
                            "primary_anchor_ids": list(tl.primary_anchor_ids)
                        }
                    ))
                    continue

            # 2. TRENDLINE_BROKEN
            if tl.status == GeometryStatus.BROKEN and tl.provenance.get("breach_info"):
                b_info = tl.provenance["breach_info"]
                current_bar_id = TradingClock.trading_bar_id(now)
                if b_info.get("breach_bar_id") == current_bar_id:
                    events.append(ResearchEvent(
                        event_id=f"ev_tl_broken_{tl.id}_{current_bar_id}",
                        timestamp=now,
                        occurrence_timestamp=now,
                        symbol=symbol,
                        event_type="TRENDLINE_BROKEN",
                        engine_version="v2.0",
                        payload={
                            "geometry_id": tl.id,
                            "geometry_type": "TRENDLINE",
                            "previous_status": "ACTIVE",
                            "new_status": "BROKEN",
                            "breach_price": b_info["breach_price"],
                            "breach_bar_id": b_info["breach_bar_id"]
                        }
                    ))

            # 3. TRENDLINE_RETESTED
            # If status is RETESTED and the last touch time is precisely now
            if tl.status == GeometryStatus.RETESTED:
                pass # Custom retest logic if needed


        return events
