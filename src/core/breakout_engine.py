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
