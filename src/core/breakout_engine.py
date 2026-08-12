#!/usr/bin/env python3
"""
Breakout Acceptance Engine (Tier 2, Item 6)
===========================================
Quantifies breakout validity to avoid fakeouts.
Rules:
1. Breakout Candle: Close > Resistance OR Close < Support
2. Volume Check: RVOL > 2.0 (Institutional participation)
3. Follow-through: Candle[n+1] High > Candle[n] High (for bullish)
4. Retest Logic: Price returns to break-level but rejects (Role Reversal)
"""

import pandas as pd
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class BreakoutEngine:
    def __init__(self, rvol_threshold: float = 1.8, consolidation_bars: int = 15):
        self.rvol_threshold = rvol_threshold
        self.consolidation_bars = consolidation_bars

    def analyze(self, df: pd.DataFrame, levels: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive breakout analysis.
        Checks for breakouts of:
        - Resistance/Support
        - PDH/PDL
        - Consolidation Boxes
        """
        if len(df) < self.consolidation_bars: 
            return {'status': 'NEUTRAL', 'confidence': 0, 'is_trap': False}

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. Level Detection
        resistance = levels.get('resistance', levels.get('pdh', 0))
        support = levels.get('support', levels.get('pdl', 999999))
        
        # 2. Basic Breakout Flags
        is_bull_break = curr['close'] > resistance and prev['close'] <= resistance
        is_bear_break = curr['close'] < support and prev['close'] >= support
        
        # 3. Consolidation Detection (High ROI)
        # Check if price was ranging in a tight band before the break
        recent_window = df.iloc[-self.consolidation_bars : -1]
        recent_range = (recent_window['high'].max() - recent_window['low'].min()) / recent_window['low'].min()
        is_tight_consolidation = recent_range < 0.01  # < 1% range over 15 bars
        
        # 4. Volume/Velocity Analysis
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        rvol = curr['volume'] / avg_vol if avg_vol > 0 else 0
        
        body_size = abs(curr['close'] - curr['open'])
        candle_size = curr['high'] - curr['low']
        velocity = body_size / (df['high'].rolling(20).max() - df['low'].rolling(20).min()).iloc[-1]

        # 5. Trap Detection (Failed Follow-Through)
        # Price spikes above level but closes back inside with high RVOL
        is_bull_trap = curr['high'] > resistance and curr['close'] < resistance and rvol > 1.5
        is_bear_trap = curr['low'] < support and curr['close'] > support and rvol > 1.5

        # 6. Scoring
        score = 0
        if is_bull_break or is_bear_break:
            score = 30 # Base score for a break
            if rvol > self.rvol_threshold: score += 20
            if is_tight_consolidation: score += 20
            if velocity > 0.5: score += 15 # Impulsive move
            if (curr['high'] - curr['close']) / candle_size < 0.15 if is_bull_break else (curr['close'] - curr['low']) / candle_size < 0.15:
                score += 15 # Closed near extreme
        
        # Traps are strong signals too, but for the opposite direction
        trap_score = 0
        if is_bull_trap or is_bear_trap:
            trap_score = 60 # Traps are high conviction
            if rvol > 2.5: trap_score += 20

        status = 'NONE'
        if is_bull_break: status = 'BULL_BREAKOUT'
        elif is_bear_break: status = 'BEAR_BREAKOUT'
        elif is_bull_trap: status = 'BULL_TRAP'
        elif is_bear_trap: status = 'BEAR_TRAP'

        return {
            'status': status,
            'confidence': max(score, trap_score),
            'rvol': round(rvol, 2),
            'is_trap': is_bull_trap or is_bear_trap,
            'consolidation_tightness': round(recent_range * 100, 2),
            'velocity': round(velocity, 2)
        }

# ── Consolidation Zone detection (v3.1 addition) ──────────────────────────────

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConsolidationZone:
    """A detected low-volatility consolidation range on a given timeframe."""
    top: float
    bottom: float
    range: float
    atr: float
    atr_percentile: float
    top_touches: int
    bot_touches: int
    bar_start: object   # pd.Timestamp
    bar_end: object


def _count_clustered_touches(
    df,
    col: str,
    level: float,
    atr: float,
    tolerance: float = 0.10,
    min_separation_bars: int = 2,
    exclude_idx=None,
) -> int:
    """
    Count distinct touches of `level` in column `col`.

    A touch: candle's col value within tolerance×ATR of level.
    Two candles in the same continuous cluster = ONE touch.
    A new cluster requires at least `min_separation_bars` consecutive
    non-touching bars between it and the previous touch.

    exclude_idx: index of the boundary-defining candle (skipped so the candle
    that established zone_high/zone_low doesn't auto-count as its own touch.
    """
    threshold = tolerance * atr
    touch_count = 0
    bars_since_last_touch = min_separation_bars + 1  # start: no active cluster

    for idx, row in df.iterrows():
        if exclude_idx is not None and idx == exclude_idx:
            continue  # skip boundary-defining candle
        near = abs(row[col] - level) <= threshold
        if near:
            if bars_since_last_touch >= min_separation_bars:
                touch_count += 1   # new distinct cluster
            bars_since_last_touch = 0
        else:
            bars_since_last_touch += 1

    return touch_count


def detect_consolidation_zone(
    h1_df,
    lookback: int = 12,
    atr_pct_threshold: float = 30.0,
    max_zone_atr_mult: float = 1.5,
    min_touches: int = 3,
    min_top_touches: int = 1,
    min_bot_touches: int = 1,
) -> Optional[ConsolidationZone]:
    """
    Detect a consolidation zone in the last `lookback` H1 bars.

    Uses the canonical True-Range ATR (same formula as IndicatorPipeline._compute_atr).
    ATR percentile is computed from the H1 ATR series — a squeeze requires ATR
    to be in the bottom `atr_pct_threshold`-th percentile.

    Boundary-defining candles are excluded from touch counts so the extrema
    themselves don't automatically satisfy the touch requirement.

    Returns a ConsolidationZone or None.
    Requires at least 30 bars of history for a meaningful ATR percentile.
    """
    import pandas as pd
    from scipy.stats import percentileofscore

    if h1_df is None or len(h1_df) < 30:
        return None

    # ── Canonical True-Range ATR ──────────────────────────────────────────────
    close_prev = h1_df["close"].shift(1)
    tr = pd.concat([
        h1_df["high"] - h1_df["low"],
        (h1_df["high"] - close_prev).abs(),
        (h1_df["low"]  - close_prev).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(window=14).mean()

    current_atr = float(atr_series.iloc[-1])
    if pd.isna(current_atr) or current_atr <= 0:
        return None

    atr_pct = percentileofscore(atr_series.dropna().values, current_atr)
    if atr_pct > atr_pct_threshold:
        return None   # not a volatility squeeze

    window = h1_df.iloc[-lookback:]
    zone_high = float(window["high"].max())
    zone_low  = float(window["low"].min())
    zone_range = zone_high - zone_low

    if zone_range > max_zone_atr_mult * current_atr:
        return None   # zone too wide

    # ── Touch counting (boundary candles excluded) ────────────────────────────
    zone_high_idx = window["high"].idxmax()
    zone_low_idx  = window["low"].idxmin()

    top_touches = _count_clustered_touches(
        window, "high", zone_high, current_atr,
        exclude_idx=zone_high_idx,
    )
    bot_touches = _count_clustered_touches(
        window, "low", zone_low, current_atr,
        exclude_idx=zone_low_idx,
    )

    if top_touches < min_top_touches or bot_touches < min_bot_touches:
        return None
    if (top_touches + bot_touches) < min_touches:
        return None

    return ConsolidationZone(
        top=zone_high,
        bottom=zone_low,
        range=zone_range,
        atr=current_atr,
        atr_percentile=atr_pct,
        top_touches=top_touches,
        bot_touches=bot_touches,
        bar_start=window.index[0],
        bar_end=window.index[-1],
    )
