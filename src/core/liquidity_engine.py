#!/usr/bin/env python3
"""
Liquidity Engine (MKE Stage 7 — Liquidity Layer)
================================================
Identifies stateful imbalances (Fair Value Gaps), constructs multi-component
liquidity pools, checks for stop hunt sweeps with hierarchical volume metrics,
and projects a trace-ready directional LiquidityMap.
"""

import logging
from collections import defaultdict
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from src.core.market_liquidity import (
    Imbalance, ImbalanceType, ImbalanceConfidenceComponents,
    LiquidityPool, LiquiditySweep, SweepType, SweepState, SweptLevelType,
    SweepConfidenceComponents, LiquidityMap, LiquidityPressureType,
    LiquidityTransitionEvent, LiquidityContext, LiquidityIDBuilder
)
from src.core.market_facts import MarketFacts
from src.core.market_knowledge import SwingPoint
from src.core.market_geometry import CompositeLevel, Trendline
from src.core.market_patterns import PatternsContext, PatternDirection
from src.core.trading_clock import TradingClock

logger = logging.getLogger(__name__)


class StopDensityScorer:
    """Pluggable scorer for estimating institutional stop density at a pool."""

    def score(
        self,
        type_score: float,
        touch_score: float,
        volume_score: float,
        confluence_score: float,
        pattern_score: float
    ) -> float:
        """Calculate weighted stop density from decomposed features."""
        # Clean heuristic combination that can be replaced by a machine learning model
        w_type = 0.25
        w_touch = 0.25
        w_vol = 0.20
        w_conf = 0.15
        w_pat = 0.15
        
        score_val = (
            w_type * type_score +
            w_touch * touch_score +
            w_vol * volume_score +
            w_conf * confluence_score +
            w_pat * pattern_score
        )
        return round(min(1.0, max(0.0, score_val)), 3)


class LiquidityEngine:
    """Stateful Order-Flow Liquidity Engine."""

    required_history = 80
    MAX_AGE_BARS = 300

    def __init__(self, min_gap_atr: float = 0.05, max_resolution_bars: int = 50, tolerance_pct: float = 0.0005, **kwargs):
        self.min_gap_atr = min_gap_atr
        self.max_resolution_bars = max_resolution_bars
        self.tolerance_pct = tolerance_pct
        self.stop_scorer = StopDensityScorer()
        
        # Stateful Caches: symbol -> list of entities
        self._imbalances = defaultdict(list)
        self._sweeps = defaultdict(list)

    def analyze(
        self,
        facts: MarketFacts,
        composites: List[CompositeLevel],
        trendlines: List[Trendline],
        patterns: PatternsContext,
        m5: pd.DataFrame,
        d1: Optional[pd.DataFrame] = None
    ) -> LiquidityContext:
        """
        Runs Stage 7 Liquidity Engine.
        """
        symbol = facts.symbol
        ts = facts.timestamp
        current_bar = facts.current_bar
        atr = facts.atr if facts.atr > 0 else 10.0
        
        # 1. Initialize caches if empty
        if symbol not in self._imbalances:
            self._imbalances[symbol] = []
        if symbol not in self._sweeps:
            self._sweeps[symbol] = []

        # 2. Build active LiquidityPools
        pools = self._build_liquidity_pools(facts, composites, patterns, m5, d1)

        # 3. Detect new Imbalances (FVGs)
        new_imbalances, creation_events = self._detect_new_imbalances(facts, m5, current_bar, ts, atr)
        self._imbalances[symbol].extend(new_imbalances)

        # 4. Update and Mitigate active Imbalances statefully
        mitigation_events = self._update_imbalances(symbol, facts.current_price, current_bar, ts)

        # 5. Detect Stop Sweeps
        new_sweeps, sweep_events = self._detect_sweeps(
            facts, pools, trendlines, patterns, composites, m5, current_bar, ts, atr
        )
        self._sweeps[symbol].extend(new_sweeps)

        # 6. Update pending Sweeps and resolve outcomes statefully
        outcome_events = self._update_sweeps(symbol, m5, current_bar, ts, atr)

        # 7. Filter active/mitigated lists
        active_imbs = tuple(i for i in self._imbalances[symbol] if not i.is_fully_filled)
        mitigated_imbs = tuple(i for i in self._imbalances[symbol] if i.is_mitigated)

        # 8. Compile the Liquidity Map
        liq_map = self._compile_liquidity_map(facts.current_price, active_imbs, pools, self._sweeps[symbol], ts)

        # Collect transition events
        transition_events = tuple(creation_events + mitigation_events + sweep_events + outcome_events)

        return LiquidityContext(
            symbol=symbol,
            timestamp=ts,
            current_bar=current_bar,
            active_imbalances=active_imbs,
            mitigated_imbalances=mitigated_imbs,
            pools=tuple(pools),
            sweeps=tuple(self._sweeps[symbol]),
            liq_map=liq_map,
            transition_events=transition_events
        )

    # ── Liquidity Pool Builder ───────────────────────────────────────────────

    def _build_liquidity_pools(
        self,
        facts: MarketFacts,
        composites: List[CompositeLevel],
        patterns: PatternsContext,
        m5: pd.DataFrame,
        d1: Optional[pd.DataFrame]
    ) -> List[LiquidityPool]:
        """Group structural swings and daily extremes into LiquidityPool nodes."""
        pools: List[LiquidityPool] = []
        symbol = facts.symbol
        ts = facts.timestamp

        # 1. Fetch PDH / PDL extremes
        pdh, pdl = None, None
        if d1 is not None and len(d1) >= 2:
            pdh = float(d1['high'].iloc[-2])
            pdl = float(d1['low'].iloc[-2])
        else:
            dates = m5.index.date
            unique_dates = sorted(list(set(dates)))
            if len(unique_dates) >= 2:
                prev_day_data = m5[m5.index.date == unique_dates[-2]]
                pdh = float(prev_day_data['high'].max())
                pdl = float(prev_day_data['low'].min())

        # Wrap PDH
        if pdh is not None:
            pool_id = LiquidityIDBuilder.make_pool_id(symbol, "pdh", pdh, ts)
            pools.append(self._create_pool_object(pool_id, "PDH", pdh, 0.0, 1, ts, composites, patterns, ()))

        # Wrap PDL
        if pdl is not None:
            pool_id = LiquidityIDBuilder.make_pool_id(symbol, "pdl", pdl, ts)
            pools.append(self._create_pool_object(pool_id, "PDL", pdl, 0.0, 1, ts, composites, patterns, ()))

        # 2. Group clustered swing points (EQH/EQL)
        clustered_ids = set()
        for cluster in facts.clusters:
            pool_id = cluster.id
            touches = len(cluster.member_swing_ids)
            
            # Find price width from members
            member_swings = [s for s in facts.swings if s.id in cluster.member_swing_ids]
            if len(member_swings) >= 2:
                prices = [s.price for s in member_swings]
                width = max(prices) - min(prices)
            else:
                width = 0.0

            pool = self._create_pool_object(
                pool_id, cluster.type, cluster.price, width, touches, cluster.last_touched,
                composites, patterns, tuple(cluster.member_swing_ids)
            )
            pools.append(pool)
            for sid in cluster.member_swing_ids:
                clustered_ids.add(sid)

        # 3. Wrap unclustered swings as single swing pools (highly swept by stops)
        for s in facts.swings:
            if s.id in clustered_ids:
                continue
            pool_id = LiquidityIDBuilder.make_pool_id(symbol, f"swing_{s.type.lower()}", s.price, s.timestamp)
            pool = self._create_pool_object(
                pool_id, f"SWING_{s.type}", s.price, 0.0, 1, s.timestamp,
                composites, patterns, (s.id,)
            )
            pools.append(pool)

        return pools

    def _create_pool_object(
        self,
        pool_id: str,
        pool_type: str,
        price: float,
        width: float,
        touches: int,
        last_interaction: datetime,
        composites: List[CompositeLevel],
        patterns: PatternsContext,
        member_swing_ids: Tuple[str, ...]
    ) -> LiquidityPool:
        """Helper to score and construct a LiquidityPool dataclass."""
        # 1. Type Score
        if pool_type in ("PDH", "PDL"):
            type_score = 1.0
        elif pool_type in ("EQH", "EQL"):
            type_score = 0.8
        else:
            type_score = 0.4

        # 2. Touch Score
        touch_score = min(1.0, 0.3 + 0.15 * touches)

        # 3. Volume Score (dummy/placeholder or derived from composites)
        volume_score = 0.7

        # 4. Confluence Score (check if composites sit near pool center)
        has_composite = any(abs(c.price - price) / price < 0.001 for c in composites)
        confluence_score = 1.0 if has_composite else 0.3

        # 5. Pattern Score (check if pattern anchors align with member swings)
        has_pattern_anchor = False
        for p in patterns.patterns:
            for anc in p.anchors:
                if anc.source_swing_id in member_swing_ids:
                    has_pattern_anchor = True
                    break
        pattern_score = 1.0 if has_pattern_anchor else 0.2

        # Compute stop density score using pluggable scorer
        density = self.stop_scorer.score(type_score, touch_score, volume_score, confluence_score, pattern_score)

        # Confidence is heavily correlated with stops density
        confidence = round(0.7 * density + 0.3 * touch_score, 3)

        return LiquidityPool(
            id=pool_id,
            type=pool_type,
            center_price=round(price, 2),
            width=round(width, 2),
            touches=touches,
            last_interaction=last_interaction,
            type_score=type_score,
            touch_score=touch_score,
            volume_score=volume_score,
            confluence_score=confluence_score,
            pattern_score=pattern_score,
            estimated_stop_density=density,
            confidence=confidence,
            member_swing_ids=member_swing_ids
        )

    # ── Imbalance (FVG) Detection ────────────────────────────────────────────

    def _detect_new_imbalances(
        self,
        facts: MarketFacts,
        m5: pd.DataFrame,
        current_bar: int,
        now: datetime,
        atr: float
    ) -> Tuple[List[Imbalance], List[LiquidityTransitionEvent]]:
        """Scan last candles to detect newly formed Fair Value Gaps (FVGs)."""
        if len(m5) < 3:
            return [], []

        symbol = facts.symbol
        new_imbalances: List[Imbalance] = []
        events: List[LiquidityTransitionEvent] = []

        # Current completed index is current_bar (usually len(m5)-1)
        i = len(m5) - 1
        
        # Fetch OHLC values
        o1, c1 = float(m5["open"].iloc[i-1]), float(m5["close"].iloc[i-1])
        h2, l2 = float(m5["high"].iloc[i-2]), float(m5["low"].iloc[i-2])
        h0, l0 = float(m5["high"].iloc[i]), float(m5["low"].iloc[i])
        v1 = float(m5["volume"].iloc[i-1])

        # Rolling volume mean
        vol_mean = m5["volume"].rolling(20).mean().iloc[i-1]
        vol_mean = vol_mean if vol_mean > 0 else 1.0

        # Hierarchical volume calculation
        if facts.rvol_tod is not None:
            volume_score = min(1.0, facts.rvol_tod / 3.0)
        else:
            volume_score = min(1.0, (v1 / vol_mean) / 3.0)

        # Bullish Imbalance (BISI / FVG)
        # Candle i-1 is expansion up, low[i] > high[i-2]
        if c1 > o1 and l0 > h2:
            gap_size = l0 - h2
            if gap_size >= self.min_gap_atr * atr:
                imb_id = LiquidityIDBuilder.make_imbalance_id(symbol, current_bar - 1, ImbalanceType.FVG)
                
                # Check duplication
                if not any(x.id == imb_id for x in self._imbalances[symbol]):
                    # Score components
                    gap_size_score = min(1.0, gap_size / atr)
                    
                    body_size = c1 - o1
                    total_range = float(m5["high"].iloc[i-1]) - float(m5["low"].iloc[i-1]) + 1e-6
                    displacement_score = min(1.0, body_size / total_range)
                    
                    structure_score = 1.0 if facts.last_swing_high is not None and facts.current_price > facts.last_swing_high.price else 0.5
                    freshness_score = 1.0

                    conf_comp = ImbalanceConfidenceComponents(
                        gap_size_score=round(gap_size_score, 3),
                        displacement_score=round(displacement_score, 3),
                        volume_score=round(volume_score, 3),
                        structure_score=round(structure_score, 3),
                        freshness_score=round(freshness_score, 3)
                    )
                    confidence = round(0.3 * gap_size_score + 0.3 * displacement_score + 0.2 * volume_score + 0.2 * structure_score, 3)

                    imb = Imbalance(
                        id=imb_id,
                        type=ImbalanceType.FVG,
                        direction=PatternDirection.LONG,
                        top=round(l0, 2),
                        bottom=round(h2, 2),
                        creation_bar=current_bar - 1,
                        creation_time=m5.index[i-1],
                        last_seen_bar=current_bar,
                        is_mitigated=False,
                        mitigated_bar=None,
                        mitigated_time=None,
                        mitigated_price=None,
                        is_fully_filled=False,
                        fill_percentage=0.0,
                        remaining_gap=round(gap_size, 2),
                        deepest_fill=round(l0, 2),
                        confidence=confidence,
                        confidence_components=conf_comp,
                        raw_components={
                            "gap_size": gap_size,
                            "displacement": displacement_score,
                            "volume": volume_score
                        },
                        explanation=f"Bullish Fair Value Gap formed at bar {current_bar - 1}."
                    )
                    new_imbalances.append(imb)
                    
                    events.append(LiquidityTransitionEvent(
                        event_id=f"ev_imb_create_{imb_id}_{current_bar}",
                        event_type="IMBALANCE_CREATED",
                        symbol=symbol,
                        bar=current_bar,
                        timestamp=now,
                        payload={
                            "imbalance_id": imb.id,
                            "type": imb.type.value,
                            "direction": imb.direction.value,
                            "top": imb.top,
                            "bottom": imb.bottom,
                            "confidence": imb.confidence
                        }
                    ))

        # Bearish Imbalance (SIBI / FVG)
        # Candle i-1 is expansion down, high[i] < low[i-2]
        elif c1 < o1 and h0 < h2:
            l2 = float(m5["low"].iloc[i-2]) # Correct reference for bearish gap
            if h0 < l2:
                gap_size = l2 - h0
                if gap_size >= self.min_gap_atr * atr:
                    imb_id = LiquidityIDBuilder.make_imbalance_id(symbol, current_bar - 1, ImbalanceType.FVG)
                    
                    if not any(x.id == imb_id for x in self._imbalances[symbol]):
                        gap_size_score = min(1.0, gap_size / atr)
                        
                        body_size = o1 - c1
                        total_range = float(m5["high"].iloc[i-1]) - float(m5["low"].iloc[i-1]) + 1e-6
                        displacement_score = min(1.0, body_size / total_range)
                        
                        structure_score = 1.0 if facts.last_swing_low is not None and facts.current_price < facts.last_swing_low.price else 0.5
                        freshness_score = 1.0

                        conf_comp = ImbalanceConfidenceComponents(
                            gap_size_score=round(gap_size_score, 3),
                            displacement_score=round(displacement_score, 3),
                            volume_score=round(volume_score, 3),
                            structure_score=round(structure_score, 3),
                            freshness_score=round(freshness_score, 3)
                        )
                        confidence = round(0.3 * gap_size_score + 0.3 * displacement_score + 0.2 * volume_score + 0.2 * structure_score, 3)

                        imb = Imbalance(
                            id=imb_id,
                            type=ImbalanceType.FVG,
                            direction=PatternDirection.SHORT,
                            top=round(l2, 2),
                            bottom=round(h0, 2),
                            creation_bar=current_bar - 1,
                            creation_time=m5.index[i-1],
                            last_seen_bar=current_bar,
                            is_mitigated=False,
                            mitigated_bar=None,
                            mitigated_time=None,
                            mitigated_price=None,
                            is_fully_filled=False,
                            fill_percentage=0.0,
                            remaining_gap=round(gap_size, 2),
                            deepest_fill=round(bottom, 2) if 'bottom' in locals() else round(h0, 2),
                            confidence=confidence,
                            confidence_components=conf_comp,
                            raw_components={
                                "gap_size": gap_size,
                                "displacement": displacement_score,
                                "volume": volume_score
                            },
                            explanation=f"Bearish Fair Value Gap formed at bar {current_bar - 1}."
                        )
                        new_imbalances.append(imb)
                        
                        events.append(LiquidityTransitionEvent(
                            event_id=f"ev_imb_create_{imb_id}_{current_bar}",
                            event_type="IMBALANCE_CREATED",
                            symbol=symbol,
                            bar=current_bar,
                            timestamp=now,
                            payload={
                                "imbalance_id": imb.id,
                                "type": imb.type.value,
                                "direction": imb.direction.value,
                                "top": imb.top,
                                "bottom": imb.bottom,
                                "confidence": imb.confidence
                            }
                        ))

        return new_imbalances, events

    # ── Imbalance Updates (Mitigation & Pruning) ────────────────────────────

    def _update_imbalances(
        self,
        symbol: str,
        current_price: float,
        current_bar: int,
        now: datetime
    ) -> List[LiquidityTransitionEvent]:
        """Update active imbalances statefully against current price action."""
        updated: List[Imbalance] = []
        events: List[LiquidityTransitionEvent] = []

        for imb in self._imbalances[symbol]:
            if imb.is_fully_filled:
                updated.append(imb)
                continue

            # Check if age exceeded
            if current_bar - imb.creation_bar > self.MAX_AGE_BARS:
                # Expire it by marking fully filled (filled status acts as archived/invalid)
                updated.append(imb)
                continue

            # Update mitigation status
            is_mit = imb.is_mitigated
            mit_bar = imb.mitigated_bar
            mit_time = imb.mitigated_time
            mit_price = imb.mitigated_price
            is_filled = imb.is_fully_filled
            deepest = imb.deepest_fill
            
            total_gap = imb.top - imb.bottom

            if imb.direction == PatternDirection.LONG:
                # Bullish imbalance is mitigated if price dips below FVG top
                if current_price < imb.top:
                    if not is_mit:
                        is_mit = True
                        mit_bar = current_bar
                        mit_time = now
                        mit_price = current_price
                        events.append(LiquidityTransitionEvent(
                            event_id=f"ev_imb_mit_{imb.id}_{current_bar}",
                            event_type="IMBALANCE_MITIGATED",
                            symbol=symbol,
                            bar=current_bar,
                            timestamp=now,
                            payload={"imbalance_id": imb.id, "price": current_price}
                        ))
                    deepest = min(deepest, current_price)
                if current_price <= imb.bottom:
                    is_filled = True
                    events.append(LiquidityTransitionEvent(
                        event_id=f"ev_imb_fill_{imb.id}_{current_bar}",
                        event_type="IMBALANCE_FILLED",
                        symbol=symbol,
                        bar=current_bar,
                        timestamp=now,
                        payload={"imbalance_id": imb.id}
                    ))

            else: # Bearish FVG
                # Bearish imbalance is mitigated if price spikes above FVG bottom
                if current_price > imb.bottom:
                    if not is_mit:
                        is_mit = True
                        mit_bar = current_bar
                        mit_time = now
                        mit_price = current_price
                        events.append(LiquidityTransitionEvent(
                            event_id=f"ev_imb_mit_{imb.id}_{current_bar}",
                            event_type="IMBALANCE_MITIGATED",
                            symbol=symbol,
                            bar=current_bar,
                            timestamp=now,
                            payload={"imbalance_id": imb.id, "price": current_price}
                        ))
                    deepest = max(deepest, current_price)
                if current_price >= imb.top:
                    is_filled = True
                    events.append(LiquidityTransitionEvent(
                        event_id=f"ev_imb_fill_{imb.id}_{current_bar}",
                        event_type="IMBALANCE_FILLED",
                        symbol=symbol,
                        bar=current_bar,
                        timestamp=now,
                        payload={"imbalance_id": imb.id}
                    ))

            fill_pct = 1.0 if is_filled else round(max(0.0, min(1.0, (imb.top - deepest) / total_gap if imb.direction == PatternDirection.LONG else (deepest - imb.bottom) / total_gap)), 3)
            rem_gap = 0.0 if is_filled else round(total_gap * (1.0 - fill_pct), 2)

            # Re-create updated Imbalance object
            new_imb = Imbalance(
                id=imb.id,
                type=imb.type,
                direction=imb.direction,
                top=imb.top,
                bottom=imb.bottom,
                creation_bar=imb.creation_bar,
                creation_time=imb.creation_time,
                last_seen_bar=current_bar,
                is_mitigated=is_mit,
                mitigated_bar=mit_bar,
                mitigated_time=mit_time,
                mitigated_price=mit_price,
                is_fully_filled=is_filled,
                fill_percentage=fill_pct,
                remaining_gap=rem_gap,
                deepest_fill=round(deepest, 2),
                confidence=imb.confidence,
                confidence_components=imb.confidence_components,
                raw_components=imb.raw_components,
                explanation=imb.explanation
            )
            updated.append(new_imb)

        self._imbalances[symbol] = updated
        return events

    # ── Sweep Detection ──────────────────────────────────────────────────────

    def _detect_sweeps(
        self,
        facts: MarketFacts,
        pools: List[LiquidityPool],
        trendlines: List[Trendline],
        patterns: PatternsContext,
        composites: List[CompositeLevel],
        m5: pd.DataFrame,
        current_bar: int,
        now: datetime,
        atr: float
    ) -> Tuple[List[LiquiditySweep], List[LiquidityTransitionEvent]]:
        """Detect stop-loss hunt sweeps against active liquidity pools."""
        symbol = facts.symbol
        new_sweeps: List[LiquiditySweep] = []
        events: List[LiquidityTransitionEvent] = []

        i = len(m5) - 1
        curr_h, curr_l, curr_c = float(m5["high"].iloc[i]), float(m5["low"].iloc[i]), float(m5["close"].iloc[i])
        curr_o = float(m5["open"].iloc[i])
        curr_v = float(m5["volume"].iloc[i])

        # Hierarchical volume calculation
        vol_mean = m5["volume"].rolling(20).mean().iloc[i]
        vol_mean = vol_mean if vol_mean > 0 else 1.0
        
        if facts.rvol_tod is not None:
            rvol = facts.rvol_tod
        else:
            rvol = curr_v / vol_mean

        # We require elevated volume for sweep validation
        if rvol < 1.1:
            return [], []

        for pool in pools:
            is_sweep = False
            sweep_type = None
            break_dist = 0.0
            rejection_wick = 0.0

            # Bearish Sweep (swept high liquidity, rejecting down)
            if curr_h > pool.center_price and curr_c < pool.center_price:
                break_dist = curr_h - pool.center_price
                rejection_wick = curr_h - max(curr_o, curr_c)
                # Check wick size threshold
                if rejection_wick >= 0.08 * atr:
                    is_sweep = True
                    sweep_type = SweepType.BEARISH

            # Bullish Sweep (swept low liquidity, rejecting up)
            elif curr_l < pool.center_price and curr_c > pool.center_price:
                break_dist = pool.center_price - curr_l
                rejection_wick = min(curr_o, curr_c) - curr_l
                if rejection_wick >= 0.08 * atr:
                    is_sweep = True
                    sweep_type = SweepType.BULLISH

            if is_sweep and sweep_type is not None:
                sweep_id = LiquidityIDBuilder.make_sweep_id(symbol, current_bar, pool.center_price, sweep_type)
                
                # Check duplication
                if any(s.id == sweep_id for s in self._sweeps[symbol]):
                    continue

                # Score confidence components
                wick_length_score = min(1.0, break_dist / atr)
                reclaim_score = min(1.0, rejection_wick / atr)
                volume_score = min(1.0, rvol / 3.0)
                liquidity_score = pool.estimated_stop_density
                structure_score = 0.8  # neutral base
                
                # Check pattern alignment
                has_pat = False
                aligned_pats = []
                for p in patterns.patterns:
                    for anc in p.anchors:
                        if anc.source_swing_id in pool.member_swing_ids:
                            has_pat = True
                            aligned_pats.append(p.id)
                pattern_score = 1.0 if has_pat else 0.2

                conf_comp = SweepConfidenceComponents(
                    wick_length_score=round(wick_length_score, 3),
                    reclaim_score=round(reclaim_score, 3),
                    volume_score=round(volume_score, 3),
                    liquidity_score=round(liquidity_score, 3),
                    structure_score=round(structure_score, 3),
                    pattern_score=round(pattern_score, 3)
                )
                confidence = round(0.2 * wick_length_score + 0.2 * reclaim_score + 0.2 * volume_score + 0.2 * liquidity_score + 0.2 * pattern_score, 3)

                # Context matches
                geom_matches = [c.id for c in composites if abs(c.price - pool.center_price) / pool.center_price < 0.001]

                sweep = LiquiditySweep(
                    id=sweep_id,
                    type=sweep_type,
                    state=SweepState.CREATED,
                    level_swept=pool.center_price,
                    level_type=SweptLevelType.POOL,
                    swept_object_id=pool.id,
                    swept_object_type="LIQUIDITY_POOL",
                    object_confidence=pool.confidence,
                    bar_index=current_bar,
                    timestamp=now,
                    volume_multiplier=round(rvol, 2),
                    wick_size_atr=round(break_dist / atr, 3),
                    rejection_wick_size=round(rejection_wick, 2),
                    confidence=confidence,
                    confidence_components=conf_comp,
                    raw_components={
                        "break_distance": break_dist,
                        "rejection_wick": rejection_wick,
                        "volume": rvol
                    },
                    pattern_ids=tuple(aligned_pats),
                    geometry_ids=tuple(geom_matches),
                    confluence_zone_id=None,
                    outcome=SweepState.PENDING,
                    bars_until_resolution=0,
                    max_excursion=curr_c,
                    max_adverse_excursion=curr_c
                )
                new_sweeps.append(sweep)

                events.append(LiquidityTransitionEvent(
                    event_id=f"ev_sweep_detect_{sweep_id}_{current_bar}",
                    event_type="SWEEP_DETECTED",
                    symbol=symbol,
                    bar=current_bar,
                    timestamp=now,
                    payload={
                        "sweep_id": sweep.id,
                        "type": sweep.type.value,
                        "level_swept": sweep.level_swept,
                        "level_type": sweep.level_type.value,
                        "confidence": sweep.confidence,
                        "object_type": sweep.swept_object_type,
                        "patterns": list(sweep.pattern_ids)
                    }
                ))

        return new_sweeps, events

    # ── Sweep Outcome Tracking ───────────────────────────────────────────────

    def _update_sweeps(
        self,
        symbol: str,
        m5: pd.DataFrame,
        current_bar: int,
        now: datetime,
        atr: float
    ) -> List[LiquidityTransitionEvent]:
        """Statefully track and update pending sweeps until resolution outcome."""
        updated: List[LiquiditySweep] = []
        events: List[LiquidityTransitionEvent] = []

        for s in self._sweeps[symbol]:
            if s.outcome != SweepState.PENDING:
                updated.append(s)
                continue

            # Increment bars since creation
            bars_held = (s.bars_until_resolution or 0) + 1
            curr_c = float(m5["close"].iloc[-1])
            curr_h = float(m5["high"].iloc[-1])
            curr_l = float(m5["low"].iloc[-1])

            # Update excursions
            max_exc = s.max_excursion
            max_adv = s.max_adverse_excursion
            outcome = s.outcome

            # Determine sweep bar high/low boundaries
            sweep_bar_idx = s.bar_index
            # Find the position of the sweep bar in the current df
            # Usually we can find it relative to current tail
            offset = current_bar - sweep_bar_idx
            if offset < len(m5):
                sweep_h = float(m5["high"].iloc[-1 - offset])
                sweep_l = float(m5["low"].iloc[-1 - offset])
            else:
                sweep_h = s.level_swept + atr
                sweep_l = s.level_swept - atr

            if s.type == SweepType.BULLISH:
                # expecting price to reverse UP
                max_exc = max(max_exc, curr_h)
                max_adv = min(max_adv, curr_l)
                
                # Fail condition (continuation down, breaching sweep bar low)
                if curr_c < sweep_l:
                    outcome = SweepState.CONTINUATION
                # Win condition (reversal confirmed, moving up by 1.5 ATR)
                elif curr_c >= s.level_swept + 1.5 * atr:
                    outcome = SweepState.REVERSAL
                # Timeout check
                elif bars_held >= self.max_resolution_bars:
                    outcome = SweepState.TIMEOUT
                    
            else: # BEARISH Sweep
                # expecting price to reverse DOWN
                max_exc = min(max_exc, curr_l)
                max_adv = max(max_adv, curr_h)
                
                # Fail condition (continuation up, breaching sweep bar high)
                if curr_c > sweep_h:
                    outcome = SweepState.CONTINUATION
                # Win condition (reversal confirmed, moving down by 1.5 ATR)
                elif curr_c <= s.level_swept - 1.5 * atr:
                    outcome = SweepState.REVERSAL
                elif bars_held >= self.max_resolution_bars:
                    outcome = SweepState.TIMEOUT

            # If resolved, trigger outcome event
            if outcome != SweepState.PENDING:
                events.append(LiquidityTransitionEvent(
                    event_id=f"ev_sweep_res_{s.id}_{current_bar}",
                    event_type="SWEEP_OUTCOME_RESOLVED",
                    symbol=symbol,
                    bar=current_bar,
                    timestamp=now,
                    payload={
                        "sweep_id": s.id,
                        "type": s.type.value,
                        "outcome": outcome.value,
                        "bars_until_resolution": bars_held,
                        "max_excursion_atr": round((max_exc - s.level_swept) / atr if s.type == SweepType.BULLISH else (s.level_swept - max_exc) / atr, 3),
                        "max_adverse_excursion_atr": round((s.level_swept - max_adv) / atr if s.type == SweepType.BULLISH else (max_adv - s.level_swept) / atr, 3)
                    }
                ))

            new_s = LiquiditySweep(
                id=s.id,
                type=s.type,
                state=SweepState.PENDING if outcome == SweepState.PENDING else SweepState.ARCHIVED,
                level_swept=s.level_swept,
                level_type=s.level_type,
                swept_object_id=s.swept_object_id,
                swept_object_type=s.swept_object_type,
                object_confidence=s.object_confidence,
                bar_index=s.bar_index,
                timestamp=s.timestamp,
                volume_multiplier=s.volume_multiplier,
                wick_size_atr=s.wick_size_atr,
                rejection_wick_size=s.rejection_wick_size,
                confidence=s.confidence,
                confidence_components=s.confidence_components,
                raw_components=s.raw_components,
                pattern_ids=s.pattern_ids,
                geometry_ids=s.geometry_ids,
                confluence_zone_id=s.confluence_zone_id,
                outcome=outcome,
                bars_until_resolution=bars_held,
                max_excursion=max_exc,
                max_adverse_excursion=max_adv
            )
            updated.append(new_s)

        self._sweeps[symbol] = updated
        return events

    # ── Liquidity Map Compilation ───────────────────────────────────────────

    def _compile_liquidity_map(
        self,
        current_price: float,
        imbalances: Tuple[Imbalance, ...],
        pools: List[LiquidityPool],
        sweeps: List[LiquiditySweep],
        now: datetime
    ) -> LiquidityMap:
        """Expose a LiquidityMap view of nearest imbalances, pools, and directional pressure."""
        nearest_bull = None
        nearest_bear = None
        min_bull_dist = float("inf")
        min_bear_dist = float("inf")

        for imb in imbalances:
            if imb.direction == PatternDirection.LONG: # Bullish FVG sits below price
                if imb.top < current_price:
                    dist = current_price - imb.top
                    if dist < min_bull_dist:
                        min_bull_dist = dist
                        nearest_bull = imb
            else: # Bearish FVG sits above price
                if imb.bottom > current_price:
                    dist = imb.bottom - current_price
                    if dist < min_bear_dist:
                        min_bear_dist = dist
                        nearest_bear = imb

        nearest_liq_above = None
        nearest_liq_below = None
        min_above_dist = float("inf")
        min_below_dist = float("inf")

        for pool in pools:
            if pool.center_price > current_price:
                dist = pool.center_price - current_price
                if dist < min_above_dist:
                    min_above_dist = dist
                    nearest_liq_above = pool
            elif pool.center_price < current_price:
                dist = current_price - pool.center_price
                if dist < min_below_dist:
                    min_below_dist = dist
                    nearest_liq_below = pool

        # Find if there is an active sweep occurring on the latest bar
        active_sweep = None
        pending_sweeps = [sw for sw in sweeps if sw.outcome == SweepState.PENDING]
        if pending_sweeps:
            active_sweep = pending_sweeps[-1]

        # Calculate directional order-flow pressure scores
        bullish_pressure = 0.5
        bearish_pressure = 0.5

        if active_sweep:
            if active_sweep.type == SweepType.BULLISH:
                bullish_pressure += 0.3 # Reversal impulse up
            else:
                bearish_pressure += 0.3

        # Near imbalances attract price as support/resistance zones
        if nearest_bull:
            # Closeness factor
            closeness = 1.0 / (1.0 + min_bull_dist / 50.0) # normal scaling
            bullish_pressure += 0.15 * closeness * nearest_bull.confidence
        if nearest_bear:
            closeness = 1.0 / (1.0 + min_bear_dist / 50.0)
            bearish_pressure += 0.15 * closeness * nearest_bear.confidence

        # Limit scores to [0.0, 1.0]
        bullish_pressure = round(min(1.0, max(0.0, bullish_pressure)), 3)
        bearish_pressure = round(min(1.0, max(0.0, bearish_pressure)), 3)

        # Categorize pressure state
        diff = bullish_pressure - bearish_pressure
        if abs(diff) < 0.08:
            pressure_state = LiquidityPressureType.BALANCED
        elif diff > 0.08:
            pressure_state = LiquidityPressureType.SEEKING_HIGHER
        else:
            pressure_state = LiquidityPressureType.SEEKING_LOWER

        # Check for vacuum: price sandwiched between two major untapped zones
        if min_above_dist < 100.0 and min_below_dist < 100.0:
            if nearest_liq_above and nearest_liq_below:
                if nearest_liq_above.confidence > 0.6 and nearest_liq_below.confidence > 0.6:
                    pressure_state = LiquidityPressureType.VACUUM

        return LiquidityMap(
            nearest_bullish_imbalance=nearest_bull,
            nearest_bearish_imbalance=nearest_bear,
            nearest_liquidity_above=nearest_liq_above,
            nearest_liquidity_below=nearest_liq_below,
            active_sweep=active_sweep,
            bullish_pressure=bullish_pressure,
            bearish_pressure=bearish_pressure,
            pressure_state=pressure_state
        )
