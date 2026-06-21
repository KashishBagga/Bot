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
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.structure_engine import StructureEngine
from src.core.liquidity_engine import LiquidityEngine
from src.core.volume_engine import VolumeEngine
from src.core.zone_engine import ZoneEngine
from src.core.quant_utils import QuantUtils
from src.core.fft_engine import FFTEngine
from src.core.regime_engine import RegimeEngine
from src.models.postgres_database import PostgresDatabase

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
        self.regime_engine = RegimeEngine()
        self.db = PostgresDatabase()
        
        # Versioning (Priority 3)
        self.strategy_version = "v3.2"
        self.feature_version = "v1.1"

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
            market_regime = self.regime_engine.detect_regime(m5_df)
            
            signals = self._evaluate_structural_setups(
                symbol, m5_df, h1_zones, daily_bias, h1_struct, market_regime, current_prices[symbol]
            )
            all_signals.extend(signals)

        return all_signals

    def _evaluate_structural_setups(self, symbol, m5_df, zones, daily_bias, h1_struct, regime, price) -> List[Dict]:
        signals = []
        htf_bias = daily_bias # Using daily as main HTF bias
        
        is_opposed = (daily_bias == "BULLISH" and h1_struct.trend == "BEARISH") or \
                    (daily_bias == "BEARISH" and h1_struct.trend == "BULLISH")
        
        # Time Filter
        current_time = pd.to_datetime(m5_df.index[-1])
        vol_report = self.volume_engine.analyze(m5_df, symbol)
        
        # Clean symbol name for deterministic candidate ID
        symbol_clean = symbol.replace(":", "_").replace("-", "_")
        
        # Primary Rejections Accumulator
        primary_rejections = []
        if current_time.hour == 9 and current_time.minute < 45:
            primary_rejections.append("TIME_FILTER")
        if vol_report.rvol_tod < self.rvol_threshold:
            primary_rejections.append("LOW_RVOL")
        if is_opposed:
            primary_rejections.append("BIAS_OPPOSED")

        # ── Intelligence Metadata ──────────────────────────────
        atr = m5_df['high'].tail(14).mean() - m5_df['low'].tail(14).mean()
        at_zone, target_zone, dist = self.zone_engine.is_price_at_zone(price, zones)
        
        dist_supply = next((abs(z.level - price) for z in zones if z.zone_type == 'SUPPLY'), 9999.0)
        dist_demand = next((abs(z.level - price) for z in zones if z.zone_type == 'DEMAND'), 9999.0)
        
        nearest_supply = next((z for z in zones if z.zone_type == 'SUPPLY'), None)
        nearest_demand = next((z for z in zones if z.zone_type == 'DEMAND'), None)
        
        base_features = {
            'date': current_time.strftime("%Y-%m-%d"),
            'strategy_name': 'Structural_v3',
            'strategy_version': self.strategy_version,
            'feature_version': self.feature_version,
            'market_regime': regime,
            'day_type': self.regime_engine.get_day_type(m5_df),
            'session_type': self.regime_engine.get_session_type(current_time),
            'daily_bias': daily_bias,
            'hourly_bias': h1_struct.trend,
            'trend_strength': h1_struct.quality_score,
            'rvol': vol_report.rvol_tod,
            'atr': round(atr, 2),
            'distance_from_supply': round(dist_supply, 2),
            'distance_from_demand': round(dist_demand, 2),
            'nearest_supply_strength': nearest_supply.score if nearest_supply else 0.0,
            'nearest_demand_strength': nearest_demand.score if nearest_demand else 0.0,
            'distance_to_liquidity_pool': dist_supply if htf_bias == "BULLISH" else dist_demand,
            'liquidity_score': vol_report.rvol_tod * 10,
            'indicator_snapshot': {
                'rvol_tod': vol_report.rvol_tod,
                'is_compressed': h1_struct.is_compressed,
                'bos_count': h1_struct.bos_count
            }
        }

        # Setup Version Splits
        signal_logic_v = self.strategy_version
        position_logic_v = "v3.1"
        risk_logic_v = "v1.1"

        # Adaptive Stop Buffer (0.5 * ATR)
        min_stop_dist = atr * 0.5
        setup_checked = None
        setup_rejection_reasons = []
        candidate_id = None
        rr_ratio = 0.0
        zone_score = target_zone.score if (at_zone and target_zone) else 0.0
        
        side = None
        entry = price
        sl = None
        take_profit = None

        # --- Setup A: Sweep Reversal ---
        if at_zone and QuantUtils.is_strong_rejection(m5_df):
            setup_checked = "SWEEP"
            trigger_level = target_zone.level
            candidate_id = f"cand_{symbol_clean}_SWEEP_{trigger_level:.2f}_{current_time.strftime('%Y%m%d')}"
            
            if target_zone.zone_type == 'DEMAND':
                side = "BUY CALL"
                sl = min(m5_df['low'].iloc[-1] - 1.0, price - min_stop_dist)
                if htf_bias == "BEARISH":
                    setup_rejection_reasons.append("BIAS_MISMATCH")
            elif target_zone.zone_type == 'SUPPLY':
                side = "BUY PUT"
                sl = max(m5_df['high'].iloc[-1] + 1.0, price + min_stop_dist)
                if htf_bias == "BULLISH":
                    setup_rejection_reasons.append("BIAS_MISMATCH")
            else:
                setup_rejection_reasons.append("BIAS_MISMATCH")

        # --- Setup B: Breakout Acceptance ---
        m5_struct = self.structure_engine.analyze(m5_df)
        move_efficiency = QuantUtils.calculate_move_efficiency(m5_df, 10)
        wickiness = QuantUtils.calculate_wickiness(m5_df, 5)
        
        base_features['move_efficiency'] = move_efficiency
        base_features['wickiness'] = wickiness
        
        if not setup_checked and m5_struct.bos_count > 0:
            setup_checked = "BREAKOUT"
            bos_level = m5_struct.last_swing_high if m5_struct.trend == "BULLISH" else m5_struct.last_swing_low
            trigger_level = bos_level
            candidate_id = f"cand_{symbol_clean}_BREAKOUT_{trigger_level:.2f}_{current_time.strftime('%Y%m%d')}"
            
            side = "BUY CALL" if m5_struct.trend == "BULLISH" else "BUY PUT"
            if side == "BUY CALL":
                sl = min(bos_level - (atr * 0.3), price - min_stop_dist)
                if htf_bias != "BULLISH":
                    setup_rejection_reasons.append("BIAS_MISMATCH")
            else:
                sl = max(bos_level + (atr * 0.3), price + min_stop_dist)
                if htf_bias != "BEARISH":
                    setup_rejection_reasons.append("BIAS_MISMATCH")
                    
            if move_efficiency <= 0.6:
                setup_rejection_reasons.append("LOW_EFFICIENCY")
            if wickiness >= 0.5:
                setup_rejection_reasons.append("HIGH_WICKINESS")

        # --- Setup C: Failed Follow-Through (FFT Trap) ---
        if not setup_checked and m5_struct.bos_count > 0:
            bos_level = m5_struct.last_swing_high if m5_struct.trend == "BULLISH" else m5_struct.last_swing_low
            trap_type = self.fft_engine.detect_trap(m5_df, bos_level, m5_struct.trend)
            if trap_type:
                setup_checked = "TRAP"
                trigger_level = bos_level
                candidate_id = f"cand_{symbol_clean}_TRAP_{trigger_level:.2f}_{current_time.strftime('%Y%m%d')}"
                side = trap_type
                sl = m5_df['high'].iloc[-1] + 1.0 if side == "BUY PUT" else m5_df['low'].iloc[-1] - 1.0
                if side == "BUY CALL" and htf_bias == "BEARISH":
                    setup_rejection_reasons.append("BIAS_MISMATCH")
                elif side == "BUY PUT" and htf_bias == "BULLISH":
                    setup_rejection_reasons.append("BIAS_MISMATCH")

        if not setup_checked:
            setup_checked = "NONE"
            candidate_id = f"cand_{symbol_clean}_NONE_{current_time.strftime('%Y%m%d_%H%M%S')}"
            setup_rejection_reasons.append("NO_SETUP")

        if setup_checked == "NONE":
            final_reasons = primary_rejections + setup_rejection_reasons
            score_breakdown = {
                'rvol': vol_report.rvol_tod,
                'move_efficiency': move_efficiency,
                'wickiness': wickiness,
                'atr': round(atr, 2),
                'daily_bias': daily_bias,
                'hourly_bias': h1_struct.trend,
                'zone_score': zone_score,
                'rr_ratio': 0.0
            }
            audit_record = {
                'candidate_id': candidate_id,
                'timestamp': current_time,
                'symbol': symbol,
                'accepted': False,
                'setup_type': setup_checked,
                'rejection_reasons': final_reasons,
                'score_breakdown': score_breakdown,
                'daily_bias': daily_bias,
                'hourly_bias': h1_struct.trend,
                'market_regime': regime,
                'signal_logic_version': signal_logic_v,
                'position_logic_version': position_logic_v,
                'risk_logic_version': risk_logic_v,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'rr_ratio': 0.0
            }
            self.db.save_signal_audit(audit_record)
            return []

        # Find target zone (take profit)
        for z in zones:
            if side == "BUY CALL" and z.level > entry:
                take_profit = z.level
                break
            if side == "BUY PUT" and z.level < entry:
                take_profit = z.level
                break

        if not take_profit:
            setup_rejection_reasons.append("NO_TARGET_ZONE")
            # Use a 2R projection as the fallback target instead of entry price.
            # TP = entry means reward = 0 → the position manager fires TP_EXPANSION
            # on the very first tick, creating a phantom R-multiple ladder.
            risk_dist = abs(entry - sl) if sl else atr
            take_profit = (entry + 2.0 * risk_dist) if side == "BUY CALL" else (entry - 2.0 * risk_dist)

        # Cap TP at 5× ATR from entry.
        # Stale demand/supply zones from weeks ago can produce TP targets 900–3000 pts
        # away, making RR > 100x. Those trades will ALWAYS exit via initial SL and
        # produce meaningless counterfactual data.
        max_tp_dist = atr * 5.0
        current_tp_dist = abs(take_profit - entry)
        if current_tp_dist > max_tp_dist:
            take_profit = (entry + max_tp_dist) if side == "BUY CALL" else (entry - max_tp_dist)
            if "NO_TARGET_ZONE" not in setup_rejection_reasons:
                setup_rejection_reasons.append("TP_CAPPED")

        # Check risk and RR
        risk = abs(entry - sl) if sl else 0.0
        reward = abs(take_profit - entry)
        if risk == 0.0:
            setup_rejection_reasons.append("ZERO_RISK")
            rr_ratio = 0.0
        else:
            rr_ratio = round(reward / risk, 2)
            if rr_ratio < 1.5:
                setup_rejection_reasons.append("LOW_RR")


        # Combine rejections
        final_reasons = primary_rejections + setup_rejection_reasons
        accepted = (len(final_reasons) == 0)

        # Create structured Core Score Breakdown
        score_breakdown = {
            'rvol': vol_report.rvol_tod,
            'move_efficiency': move_efficiency,
            'wickiness': wickiness,
            'atr': round(atr, 2),
            'daily_bias': daily_bias,
            'hourly_bias': h1_struct.trend,
            'zone_score': zone_score,
            'rr_ratio': rr_ratio
        }

        # Save to signal_audit table
        audit_record = {
            'candidate_id': candidate_id,
            'timestamp': current_time,
            'symbol': symbol,
            'accepted': accepted,
            'setup_type': setup_checked,
            'rejection_reasons': final_reasons,
            'score_breakdown': score_breakdown,
            'daily_bias': daily_bias,
            'hourly_bias': h1_struct.trend,
            'market_regime': regime,
            'signal_logic_version': signal_logic_v,
            'position_logic_version': position_logic_v,
            'risk_logic_version': risk_logic_v,
            'entry_price': entry,
            'stop_loss': sl,
            'take_profit': take_profit,
            'rr_ratio': rr_ratio
        }
        self.db.save_signal_audit(audit_record)

        # Create the signal candidate structure
        features = base_features.copy()
        features['fft_score'] = 1.0 if setup_checked == "TRAP" else 0.0
        features['setup_type'] = setup_checked
        features['candidate_id'] = candidate_id

        sig = {
            'symbol': symbol,
            'signal': side,
            'strategy': setup_checked,
            'price': entry,
            'stop_loss': sl,
            'take_profit': take_profit,
            'tp1': entry + (risk * 1.5) if side == "BUY CALL" else entry - (risk * 1.5) if sl else entry,
            'rr_ratio': rr_ratio,
            'timestamp': current_time.isoformat(),
            'accepted': accepted,
            'rejection_reasons': final_reasons,
            'features': features,
            'candidate_id': candidate_id
        }

        if accepted:
            signal_record = {
                'signal_id': f"{symbol}_{current_time.timestamp()}",
                'candidate_id': candidate_id,
                'timestamp': current_time,
                'strategy': setup_checked,
                'symbol': symbol,
                'regime': regime,
                'strength': h1_struct.quality_score,
                'accepted': True,
                'rejected_reason': None,
                'executed': False,
                'context': features,
                'setup_type': setup_checked,
                'score_breakdown': score_breakdown,
                'daily_bias': daily_bias,
                'hourly_bias': h1_struct.trend,
                'market_regime': regime,
                'signal_logic_version': signal_logic_v,
                'position_logic_version': position_logic_v,
                'risk_logic_version': risk_logic_v
            }
            self.db.save_signal(signal_record)

        return [sig]

    def _create_structural_signal(self, symbol, side, entry, sl, zones, strategy, features=None) -> Tuple[Optional[Dict], Optional[str]]:
        next_zone_level = None
        for z in zones:
            if side == "BUY CALL" and z.level > entry:
                next_zone_level = z.level; break
            if side == "BUY PUT" and z.level < entry:
                next_zone_level = z.level; break
        
        if not next_zone_level: return None, "NO_HTF_TARGET_ZONE" 

        risk = abs(entry - sl)
        reward = abs(next_zone_level - entry)
        if risk == 0: return None, "ZERO_RISK"
        
        rr = reward / risk
        if rr < 1.5: return None, f"LOW_RR_{rr:.2f}" 

        return {
            'symbol': symbol, 'signal': side, 'strategy': strategy,
            'price': entry, 'stop_loss': sl, 'take_profit': next_zone_level,
            'tp1': entry + (risk * 1.5) if side == "BUY CALL" else entry - (risk * 1.5),
            'rr_ratio': round(rr, 2), 'timestamp': datetime.now(self.tz).isoformat(),
            'features': features
        }, None
