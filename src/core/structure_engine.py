#!/usr/bin/env python3
"""
Structure Engine v2 (MKE Stage 1 Context)
=========================================
Tracks swing status lifecycles, completes legs, maps developing legs and candidate swings.
Emits verified ResearchEvents (BOS, CHOCH) for database persistence.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

from src.core.market_knowledge import (
    SwingPoint, SwingStatus, SwingRelationship, CompletedLeg,
    DevelopingLeg, SessionAnchor, AnchorType, ResearchEvent, StructureState
)
from src.core.trading_clock import TradingClock

logger = logging.getLogger(__name__)


class StructureEngine:
    """Orchestrates market structure tracking and event emissions."""
    required_history = 150
    engine_version = "v2.0"

    def __init__(self, pivot_window: int = 3):
        self.w = pivot_window

    def analyze(
        self,
        df: pd.DataFrame,
        raw_swings: List[SwingPoint],
        clusters: List[Any],
        symbol: str,
        timeframe: str = "m5"
    ) -> Tuple[StructureState, List[ResearchEvent]]:
        """
        Runs the structure analysis pass.
        Computes swing relationships, updates statuses, maps completed/developing legs,
        and generates high-value ResearchEvents.
        """
        if df is None or len(df) < 20:
            return StructureState(), []

        # Resolve interval minutes based on timeframe
        interval_mins = 5
        tf_lower = timeframe.lower() if timeframe else "m5"
        if tf_lower == "h1":
            interval_mins = 60
        elif tf_lower in ["d1", "daily"]:
            interval_mins = 375

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index
        current_time = timestamps[-1].to_pydatetime()
        current_price = float(closes[-1])
        
        # Calculate current ATR for volatility/reversal checks
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = float(highs[-1] - lows[-1])
        atr = max(atr, 1.0)

        # ── 1. Copy and Sort Swings ──────────────────────────────
        # Swings are sorted by timestamp
        swings = sorted(list(raw_swings), key=lambda s: s.timestamp)
        relationships: Dict[str, SwingRelationship] = {}

        # ── 2. Swing Relationships (HH/HL/LH/LL) ─────────────────
        high_history: List[SwingPoint] = []
        low_history: List[SwingPoint] = []

        for s in swings:
            label = "NEUTRAL"
            if s.type == "HIGH":
                if len(high_history) > 0:
                    prev = high_history[-1]
                    label = "HH" if s.price > prev.price else "LH"
                high_history.append(s)
            elif s.type == "LOW":
                if len(low_history) > 0:
                    prev = low_history[-1]
                    label = "HL" if s.price > prev.price else "LL"
                low_history.append(s)
            relationships[s.id] = SwingRelationship(swing_id=s.id, label=label)

        # ── 3. Session Anchor & Completed Legs ───────────────────
        # Identify start day anchor
        start_candle_time = timestamps[0].to_pydatetime()
        start_anchor = SessionAnchor(
            id=f"anchor_{symbol.replace(':', '_')}_{timeframe}_{start_candle_time.strftime('%Y%m%d_%H%M%S')}",
            timestamp=start_candle_time,
            price=float(df['open'].iloc[0]),
            type=AnchorType.SESSION
        )

        completed_legs: List[CompletedLeg] = []
        anchors_and_swings: List[Any] = [start_anchor] + swings

        for k in range(len(anchors_and_swings) - 1):
            start = anchors_and_swings[k]
            end = anchors_and_swings[k + 1]
            leg_type = "UP_LEG" if end.price > start.price else "DOWN_LEG"
            
            bars_held = TradingClock.bars_between(start.timestamp, end.timestamp, interval_minutes=interval_mins)
            leg_id = f"leg_{symbol.replace(':', '_')}_{timeframe}_{start.timestamp.strftime('%Y%m%d_%H%M')}_{end.timestamp.strftime('%Y%m%d_%H%M')}"
            
            completed_legs.append(CompletedLeg(
                id=leg_id,
                start_anchor=start,
                end_pivot=end,
                type=leg_type,
                price_range=round(abs(end.price - start.price), 2),
                bars_held=bars_held
            ))

        # ── 4. Developing Leg ────────────────────────────────────
        developing_leg = None
        if len(anchors_and_swings) > 0:
            last_anchor = anchors_and_swings[-1]
            leg_type = "UP_LEG" if current_price > last_anchor.price else "DOWN_LEG"
            dur_bars = TradingClock.bars_between(last_anchor.timestamp, current_time, interval_minutes=interval_mins)
            
            # Estimate live velocity: points per bar
            velocity = round((current_price - last_anchor.price) / max(dur_bars, 1), 3)
            
            developing_leg = DevelopingLeg(
                start_anchor=last_anchor,
                current_price=current_price,
                current_high=float(highs[-1]),
                current_low=float(lows[-1]),
                current_duration_bars=dur_bars,
                current_extension_atr=round(abs(current_price - last_anchor.price) / atr, 2),
                live_velocity=velocity
            )

        # ── 5. Candidate Swing (with reversal filter) ───────────
        candidate_swing = None
        if len(swings) > 0:
            last_confirmed = swings[-1]
            # Price range since last swing
            sub_df = df[df.index > pd.Timestamp(last_confirmed.timestamp)]
            if len(sub_df) > 0:
                sub_highs = sub_df['high'].values
                sub_lows = sub_df['low'].values
                sub_closes = sub_df['close'].values
                sub_times = sub_df.index
                
                if last_confirmed.type == "HIGH":
                    # We are looking for a swing LOW candidate.
                    # Find absolute low since last swing
                    min_idx = int(np.argmin(sub_lows))
                    candidate_price = float(sub_lows[min_idx])
                    # Reversal check: price must have reversed up off this low by >= 0.5 ATR
                    reversal_pct = current_price - candidate_price
                    if reversal_pct >= (atr * 0.5):
                        candidate_ts = sub_times[min_idx].to_pydatetime()
                        cand_id = f"cand_sw_{symbol.replace(':', '_')}_{timeframe}_low_{candidate_ts.strftime('%Y%m%d_%H%M%S')}"
                        candidate_swing = SwingPoint(
                            id=cand_id,
                            timestamp=candidate_ts,
                            price=candidate_price,
                            type="LOW",
                            status=SwingStatus.ACTIVE,
                            confidence=0.8, # unconfirmed
                            strength=0.5,
                            strength_components={},
                            provenance={"engine": "StructureEngine", "version": self.engine_version}
                        )
                else:
                    # We are looking for a swing HIGH candidate
                    max_idx = int(np.argmax(sub_highs))
                    candidate_price = float(sub_highs[max_idx])
                    reversal_pct = candidate_price - current_price
                    if reversal_pct >= (atr * 0.5):
                        candidate_ts = sub_times[max_idx].to_pydatetime()
                        cand_id = f"cand_sw_{symbol.replace(':', '_')}_{timeframe}_high_{candidate_ts.strftime('%Y%m%d_%H%M%S')}"
                        candidate_swing = SwingPoint(
                            id=cand_id,
                            timestamp=candidate_ts,
                            price=candidate_price,
                            type="HIGH",
                            status=SwingStatus.ACTIVE,
                            confidence=0.8,
                            strength=0.5,
                            strength_components={},
                            provenance={"engine": "StructureEngine", "version": self.engine_version}
                        )

        # ── 6. Swing Status Lifecycle & Event Emission ───────────
        active_swings: List[SwingPoint] = []
        events: List[ResearchEvent] = []
        
        last_high: Optional[SwingPoint] = None
        last_low: Optional[SwingPoint] = None

        # Build dynamic list of updated swings based on current close breaches
        updated_swings: List[SwingPoint] = []
        
        for s in swings:
            status = s.status
            confidence = s.confidence
            
            # Confidence decay: reduce confidence by 0.5% per bar since pivot confirmation
            bars_since = TradingClock.bars_between(s.timestamp, current_time, interval_minutes=interval_mins)
            confidence = max(0.1, round(s.confidence * (0.995 ** bars_since), 3))
            
            # Check for LEVEL_CLOSED_BEYOND interaction -> Breach
            is_breached = False
            if s.type == "HIGH" and current_price > s.price:
                is_breached = True
            elif s.type == "LOW" and current_price < s.price:
                is_breached = True
                
            if is_breached and status == SwingStatus.ACTIVE:
                status = SwingStatus.BREACHED
                confidence = 0.5 # Reset/stabilize confidence
                
                # Emit high-value ResearchEvent
                event_type = "BOS_CONFIRMED"
                # If relationship shows reversal, classify as CHOCH
                rel = relationships.get(s.id)
                if rel and rel.label in ["LH", "HL"]:
                    event_type = "CHOCH_CONFIRMED"

                event_id = f"evt_{symbol.replace(':', '_')}_{timeframe}_{event_type.lower()}_{current_time.strftime('%Y%m%d_%H%M%S')}"
                events.append(ResearchEvent(
                    event_id=event_id,
                    timestamp=current_time,
                    occurrence_timestamp=current_time,
                    symbol=symbol,
                    event_type=event_type,
                    engine_version=self.engine_version,
                    payload={
                        "breached_swing_id": s.id,
                        "breach_price": current_price,
                        "swing_price": s.price,
                        "swing_type": s.type,
                        "rvol": float(df['volume'].iloc[-1])
                    }
                ))
            
            # Retest check (role reversal support/resistance touch)
            if status == SwingStatus.BREACHED:
                # If price returns to test this breached level within 0.3 ATR
                if abs(current_price - s.price) <= (atr * 0.3):
                    status = SwingStatus.RETESTED
                    confidence = 1.0 # Refreshed on test

            updated_swing = SwingPoint(
                id=s.id,
                timestamp=s.timestamp,
                price=s.price,
                type=s.type,
                status=status,
                confidence=confidence,
                strength=s.strength,
                strength_components=s.strength_components,
                provenance=s.provenance
            )
            updated_swings.append(updated_swing)
            
            if status == SwingStatus.ACTIVE:
                active_swings.append(updated_swing)
                if s.type == "HIGH":
                    last_high = updated_swing
                else:
                    last_low = updated_swing

        # Compression check
        is_compressed = False
        if len(closes) >= 20:
            std_short = float(np.std(closes[-5:]))
            std_long = float(np.std(closes[-20:]))
            is_compressed = std_short < std_long * 0.6

        state = StructureState(
            swings=updated_swings,
            relationships=relationships,
            legs=completed_legs,
            clusters=clusters,
            developing_leg=developing_leg,
            candidate_swing=candidate_swing,
            last_swing_high=last_high,
            last_swing_low=last_low,
            is_compressed=is_compressed
        )

        return state, events
