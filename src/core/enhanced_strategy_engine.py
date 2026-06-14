#!/usr/bin/env python3
"""
Structural Decision Engine (Phase 3.2 - Final Polish)
=====================================================
Implements:
- Adaptive Stops: max(Structural, 0.5*ATR)
- Relaxed Context: Daily Aligned + (1H Aligned OR Neutral)
- FFT Trap Detection: Trapped Participation setups
- Wickiness Filter: Rejects unstable breakouts
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.structure_engine import StructureEngine
from src.core.liquidity_engine import LiquidityEngine
from src.core.volume_engine import VolumeEngine
from src.core.zone_engine import ZoneEngine
from src.core.quant_utils import QuantUtils
from src.core.fft_engine import FFTEngine

logger = logging.getLogger(__name__)

class EnhancedStrategyEngine:
    def __init__(self, symbols: List[str], 
                 min_zone_score: float = 60.0,
                 rvol_threshold: float = 2.0):
        self.symbols = symbols
        self.min_zone_score = min_zone_score
        self.rvol_threshold = rvol_threshold
        self.tz = ZoneInfo("Asia/Kolkata")
        
        self.structure_engine = StructureEngine(pivot_window=3)
        self.liquidity_engine = LiquidityEngine(tolerance_pct=0.0005)
        self.volume_engine = VolumeEngine(historical_days=20)
        self.zone_engine = ZoneEngine(cluster_pct=0.002)
        self.zone_engine.MIN_ZONE_SCORE = min_zone_score
        self.fft_engine = FFTEngine(failure_threshold=0.8)

        logger.info(f"🏛️ Structural Decision Engine initialized | Phase 3.2 (Final Polish) Active")

    def generate_signals_for_all_symbols(
        self,
        historical_data: Dict[str, Dict[str, pd.DataFrame]],
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        all_signals = []

        for symbol in self.symbols:
            if symbol not in historical_data or symbol not in current_prices: continue
            
            data = historical_data[symbol]
            d1_df, h1_df, m5_df = data.get('1d'), data.get('1h'), data.get('5m')
            if h1_df is None or m5_df is None: continue

            # ── 1. Relaxed Fractal Structural Bias ─────────────────
            daily_bias = QuantUtils.get_structural_bias(d1_df) if d1_df is not None else "NEUTRAL"
            h1_struct = self.structure_engine.analyze(h1_df)
            
            # Relaxed Logic: 1H can be Aligned OR Neutral, but NOT Opposed
            is_opposed = (daily_bias == "BULLISH" and h1_struct.trend == "BEARISH") or \
                        (daily_bias == "BEARISH" and h1_struct.trend == "BULLISH")
            
            is_bias_aligned = (daily_bias != "NEUTRAL") and (not is_opposed)
            bias = daily_bias if is_bias_aligned else "NEUTRAL"

            # ── 2. Zone Quality & Execution ────────────────────────
            h1_zones = self.zone_engine.detect_zones(h1_df)
            
            signals = self._evaluate_structural_setups(
                symbol, m5_df, h1_zones, bias, current_prices[symbol]
            )
            
            all_signals.extend(signals)

        return all_signals

    def _evaluate_structural_setups(self, symbol, m5_df, zones, htf_bias, price) -> List[Dict]:
        signals = []
        
        # Time Filter
        current_time = pd.to_datetime(m5_df.index[-1])
        if current_time.hour == 9 and current_time.minute < 45: return []
        
        # Participation Filter (ToD RVOL)
        vol_report = self.volume_engine.analyze(m5_df, symbol)
        if vol_report.rvol_tod < self.rvol_threshold: return []
        
        # Check Location
        at_zone, target_zone, _ = self.zone_engine.is_price_at_zone(price, zones)
        
        # Adaptive Stop Buffer (0.5 * ATR)
        atr = m5_df['high'].tail(10).mean() - m5_df['low'].tail(10).mean()
        min_stop_dist = atr * 0.5

        # --- Setup A: Sweep Reversal ---
        if at_zone and QuantUtils.is_strong_rejection(m5_df):
            if target_zone.zone_type == 'DEMAND' and htf_bias != "BEARISH":
                # Structural SL + 0.5 ATR Buffer
                sl = min(m5_df['low'].iloc[-1] - 1.0, price - min_stop_dist)
                sig = self._create_structural_signal(symbol, "BUY CALL", price, sl, zones, "SWEEP")
                if sig: signals.append(sig)

            elif target_zone.zone_type == 'SUPPLY' and htf_bias != "BULLISH":
                sl = max(m5_df['high'].iloc[-1] + 1.0, price + min_stop_dist)
                sig = self._create_structural_signal(symbol, "BUY PUT", price, sl, zones, "SWEEP")
                if sig: signals.append(sig)

        # --- Setup B: Breakout Acceptance ---
        m5_struct = self.structure_engine.analyze(m5_df)
        move_efficiency = QuantUtils.calculate_move_efficiency(m5_df, 10)
        wickiness = QuantUtils.calculate_wickiness(m5_df, 5)
        
        # Logic: Clean BOS + High RVOL + Low Wickiness (< 0.5)
        if m5_struct.bos_count > 0 and move_efficiency > 0.6 and wickiness < 0.5:
            # Use swing high/low as the breakout level
            bos_level = m5_struct.last_swing_high if m5_struct.trend == "BULLISH" else m5_struct.last_swing_low
            
            if m5_struct.trend == "BULLISH" and htf_bias == "BULLISH":
                sl = min(bos_level - (atr * 0.3), price - min_stop_dist)
                sig = self._create_structural_signal(symbol, "BUY CALL", price, sl, zones, "BREAKOUT")
                if sig: signals.append(sig)
            elif m5_struct.trend == "BEARISH" and htf_bias == "BEARISH":
                sl = max(bos_level + (atr * 0.3), price + min_stop_dist)
                sig = self._create_structural_signal(symbol, "BUY PUT", price, sl, zones, "BREAKOUT")
                if sig: signals.append(sig)

        # --- Setup C: Failed Follow-Through (FFT Trap) ---
        if m5_struct.bos_count > 0:
            bos_level = m5_struct.last_swing_high if m5_struct.trend == "BULLISH" else m5_struct.last_swing_low
            trap_type = self.fft_engine.detect_trap(m5_df, bos_level, m5_struct.trend)
            if trap_type:
                # Target is the OTHER side of the structure
                sl = m5_df['high'].iloc[-1] + 1.0 if trap_type == "BUY PUT" else m5_df['low'].iloc[-1] - 1.0
                sig = self._create_structural_signal(symbol, trap_type, price, sl, zones, "TRAP")
                if sig: signals.append(sig)

        return signals

    def _create_structural_signal(self, symbol, side, entry, sl, zones, strategy) -> Optional[Dict]:
        next_zone_level = None
        for z in zones:
            if side == "BUY CALL" and z.level > entry:
                next_zone_level = z.level; break
            if side == "BUY PUT" and z.level < entry:
                next_zone_level = z.level; break
        
        if not next_zone_level: return None 

        risk = abs(entry - sl)
        reward = abs(next_zone_level - entry)
        if risk == 0: return None
        
        rr = reward / risk
        if rr < 1.5: return None 

        return {
            'symbol': symbol, 'signal': side, 'strategy': strategy,
            'price': entry, 'stop_loss': sl, 'take_profit': next_zone_level,
            'tp1': entry + (risk * 1.5) if side == "BUY CALL" else entry - (risk * 1.5),
            'rr_ratio': round(rr, 2), 'timestamp': datetime.now(self.tz).isoformat()
        }
