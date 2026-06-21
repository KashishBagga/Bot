#!/usr/bin/env python3
"""
ATR Squeeze Breakout Strategy
==============================
Hypothesis: Periods of volatility compression (low ATR percentile) represent 
a coiled spring. Breakouts from these periods with high volume participation 
yield high-momentum, explosive extensions.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.structure_engine import StructureEngine
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)


class AtrSqueezeStrategy(BaseStrategy):
    """
    ATR Squeeze Breakout Strategy.
    Monitors volatility compression via rolling ATR percentiles.
    """

    metadata = StrategyMetadata(
        id="atr_squeeze",
        name="ATR Squeeze Breakout",
        hypothesis_id="volatility_compression",
        hypothesis_family="Volatility Expansion",
        hypothesis_text="Trades momentum breakouts out of low volatility compression (low ATR percentile) periods.",
        version="v1.0",
        maturity="PAPER",
        tags=["atr", "squeeze", "compression", "breakout"]
    )

    def __init__(
        self,
        rvol_threshold: float = 1.0,
        atr_percentile_threshold: float = 0.20,
    ):
        self.rvol_threshold = rvol_threshold
        self.atr_percentile_threshold = atr_percentile_threshold
        self.structure_engine = StructureEngine(pivot_window=3)

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        experiment_name: str,
    ) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            if len(m5_df) < 50:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            atr_percentile = snapshot.features.get_float("atr_percentile")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            # Volatility Squeeze condition: ATR is in the lower percentile
            is_squeezed = atr_percentile <= self.atr_percentile_threshold

            # Detect local 5m breakouts
            m5_struct = self.structure_engine.analyze(m5_df)
            has_breakout = m5_struct.bos_count > 0

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            if is_squeezed and has_breakout:
                setup_type = "SQUEEZE_BREAKOUT"
                bos_level = m5_struct.last_swing_high if m5_struct.trend == "BULLISH" else m5_struct.last_swing_low
                
                if m5_struct.trend == "BULLISH":
                    side = "BUY CALL"
                    sl = min(bos_level - (atr * 0.3), price - (atr * 0.5))
                    take_profit = price + (atr * 3.0)
                elif m5_struct.trend == "BEARISH":
                    side = "BUY PUT"
                    sl = max(bos_level + (atr * 0.3), price + (atr * 0.5))
                    take_profit = price - (atr * 3.0)

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            # Rejections
            rejection_reasons = []

            # 1. Open hours blackout
            current_time = snapshot.timestamp
            if current_time.hour == 9 and current_time.minute < 45:
                rejection_reasons.append("TIME_FILTER")

            # 2. RVOL filter (squeeze breakouts need high volume participation)
            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")

            # 3. Daily bias alignment
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

            # Invalidation buffer
            risk_dist = abs(price - sl) if sl else atr
            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")

            # Find take profit (Next 1H opposite zone)
            for z in snapshot.h1_zones:
                if side == "BUY CALL" and z.level > price:
                    take_profit = z.level
                    break
                if side == "BUY PUT" and z.level < price:
                    take_profit = z.level
                    break

            # Cap TP at 5x ATR
            max_tp_dist = atr * 5.0
            if abs(take_profit - price) > max_tp_dist:
                take_profit = (price + max_tp_dist) if side == "BUY CALL" else (price - max_tp_dist)
                rejection_reasons.append("TP_CAPPED")

            rr = round(abs(take_profit - price) / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < 1.5:
                rejection_reasons.append("LOW_RR")

            # Calculate confidence
            confidence = 0.5
            if len(rejection_reasons) == 0:
                # Squeezes that release with extremely high volume get higher confidence
                confidence = round(min(0.6 + 0.3 * (rvol - 1.0), 0.95), 2)

            # Diagnostics payload for Feature Attribution Engine
            diagnostics = {
                "atr_percentile": round(atr_percentile, 3),
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "bos_trend": m5_struct.trend,
                "bos_count": m5_struct.bos_count,
                "move_efficiency": round(snapshot.features.get_float("move_efficiency"), 3)
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_SQBRK_{price:.2f}_{current_time.strftime('%Y%m%d')}"

            sig = {
                'symbol': snapshot.symbol,
                'signal': side,
                'strategy': setup_type,
                'price': price,
                'stop_loss': sl,
                'take_profit': take_profit,
                'tp1': price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5),
                'rr_ratio': rr,
                'timestamp': current_time.isoformat(),
                'accepted': accepted,
                'rejection_reasons': rejection_reasons,
                'features': snapshot.features.to_dict(),
                'candidate_id': candidate_id,
                'confidence': confidence,
                'diagnostics': diagnostics
            }

            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[AtrSqueezeStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings
        )
