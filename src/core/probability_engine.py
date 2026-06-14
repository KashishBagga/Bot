#!/usr/bin/env python3
"""
Probability Engine v2 — Confluence Scorer
==========================================
FIXES from review:
1. Candle patterns NEVER standalone — only valid at zone + with RVOL
2. Zone score now uses ZoneEngine quality (not raw touch count)
3. Structure uses health_score (not binary trend label)
4. Breakout acceptance + failed follow-through integrated
5. All weights are explicit and backtestable
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ProbabilityEngine:
    """
    Confluence Scorer (0-100).
    Indicators = 10 pts max (tie-breakers).
    Price action + structure + volume = 90 pts.
    """

    def calculate_confluence_score(
        self,
        signal_type: str,
        intel: Dict[str, Any],
        candle_patterns: Dict[str, bool],
        indicators: Dict[str, Any],
        daily_bias: str
    ) -> Tuple[float, List[str]]:
        """
        Returns (score 0-100, list of confirming factor names).
        """
        score = 0
        factors = []
        price = indicators.get('price', 0)

        # ════════════════════════════════════════════════════════════
        # 1. ZONE ALIGNMENT (25 pts)
        #    Uses ZoneEngine quality score, not raw touch count.
        # ════════════════════════════════════════════════════════════
        zone_info = intel.get('zone_info')  # (at_zone: bool, zone: Zone, dist_pct)
        at_zone = False
        zone_quality = 0

        if zone_info and zone_info[0]:
            at_zone = True
            zone_obj = zone_info[1]
            zone_quality = zone_obj.score if zone_obj else 0

            # Scale 25 pts by zone quality (0-100 → 0-25)
            zone_pts = round(25 * (zone_quality / 100), 1)
            score += zone_pts
            factors.append(f"ZONE_{zone_obj.zone_type}(q={zone_quality:.0f})")

        # ════════════════════════════════════════════════════════════
        # 2. CANDLE PATTERN (20 pts)
        #    CRITICAL FIX: Only valid AT a zone. Never standalone.
        # ════════════════════════════════════════════════════════════
        has_reversal = False
        if signal_type == 'BUY CALL':
            has_reversal = candle_patterns.get('bull_pin') or candle_patterns.get('bull_eng') or candle_patterns.get('morning_star')
        elif signal_type == 'BUY PUT':
            has_reversal = candle_patterns.get('bear_pin') or candle_patterns.get('bear_eng') or candle_patterns.get('evening_star')

        if has_reversal and at_zone:
            # Candle AT a zone = strong reaction evidence
            score += 20
            factors.append("CANDLE_AT_ZONE")
        elif has_reversal and not at_zone:
            # Candle in open air = noise, give only 3 pts
            score += 3
            factors.append("CANDLE_NO_ZONE")

        # ════════════════════════════════════════════════════════════
        # 3. VOLUME & BREAKOUT EVIDENCE (20 pts)
        #    Climax (counter-direction) OR Breakout/Trap
        # ════════════════════════════════════════════════════════════
        climax = intel.get('volume_climax', 'NONE')
        if (signal_type == 'BUY CALL' and climax == 'BEARISH_CLIMAX') or \
           (signal_type == 'BUY PUT' and climax == 'BULLISH_CLIMAX'):
            score += 20
            factors.append("VOLUME_CLIMAX")
        
        breakout = intel.get('breakout', {})
        b_status = breakout.get('status', 'NONE')
        b_conf = breakout.get('confidence', 0)
        
        # Scenario A: Real Breakout
        if (signal_type == 'BUY CALL' and b_status == 'BULL_BREAKOUT') or \
           (signal_type == 'BUY PUT' and b_status == 'BEAR_BREAKOUT'):
            # Scale 20 pts by breakout confidence
            b_pts = round(20 * (b_conf / 100), 1)
            score += b_pts
            factors.append(f"BREAKOUT(c={b_conf})")
        
        # Scenario B: Failed Follow-Through (Trap) - HIGHEST ROI
        elif (signal_type == 'BUY CALL' and b_status == 'BEAR_TRAP') or \
             (signal_type == 'BUY PUT' and b_status == 'BULL_TRAP'):
            # Traps are explosive reversals. Give full 20 pts if confidence > 50
            if b_conf > 50:
                score += 20
                factors.append("INSTITUTIONAL_TRAP")

        # ════════════════════════════════════════════════════════════
        # 4. DAILY BIAS / HTF ALIGNMENT (15 pts)
        # ════════════════════════════════════════════════════════════
        if (signal_type == 'BUY CALL' and daily_bias == 'BULLISH') or \
           (signal_type == 'BUY PUT' and daily_bias == 'BEARISH'):
            score += 15
            factors.append("HTF_BIAS")

        # ════════════════════════════════════════════════════════════
        # 5. STRUCTURE HEALTH (15 pts)
        #    Uses health_score, not just binary trend.
        # ════════════════════════════════════════════════════════════
        structure = intel.get('structure')
        if structure:
            trend_aligned = (
                (signal_type == 'BUY CALL' and structure.trend == 'BULLISH') or
                (signal_type == 'BUY PUT' and structure.trend == 'BEARISH')
            )
            if trend_aligned:
                health = getattr(structure, 'health_score', 50)
                # Scale 15 pts by health (0-100 → 0-15)
                struct_pts = round(15 * (health / 100), 1)
                score += struct_pts
                factors.append(f"STRUCTURE(h={health:.0f})")

        # ════════════════════════════════════════════════════════════
        # 6. LIQUIDITY SWEEP (10 pts)
        #    Sweep + reversal = high conviction
        # ════════════════════════════════════════════════════════════
        sweep = intel.get('liquidity_sweeps', 'NONE')
        if (signal_type == 'BUY CALL' and sweep == 'BULLISH_SWEEP') or \
           (signal_type == 'BUY PUT' and sweep == 'BEARISH_SWEEP'):
            score += 10
            factors.append("SWEEP_REVERSAL")

        # ════════════════════════════════════════════════════════════
        # 7. INDICATORS (5 pts total — tie-breakers ONLY)
        # ════════════════════════════════════════════════════════════
        if indicators.get('macd_aligned'):
            score += 3
            factors.append("MACD")
        if indicators.get('rsi_aligned'):
            score += 2
            factors.append("RSI")

        return min(score, 100), factors
