#!/usr/bin/env python3
"""
EMA Pullback Strategy
=====================
Hypothesis: Trend continuation after pullback to key moving averages.
Exploits established trends by buying/selling the pullback to the 20 EMA.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)


class EmaPullbackStrategy(BaseStrategy):
    """
    EMA Pullback Strategy.
    """

    metadata = StrategyMetadata(
        id="ema_pullback",
        name="EMA Pullback",
        hypothesis_id="trend_continuation_pullback",
        hypothesis_family="Trend Following",
        hypothesis_text="Trades resumption of established trends after a pullback to the 20 EMA.",
        version="v1.0",
        maturity="PAPER",
        tags=["trend", "ema", "pullback"]
    )

    def __init__(
        self,
        rvol_threshold: float = 1.0,
        min_efficiency: float = 0.6,
    ):
        self.rvol_threshold = rvol_threshold
        self.min_efficiency = min_efficiency

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
            ema20 = snapshot.features.get_float("ema20")
            ema50 = snapshot.features.get_float("ema50")
            ema_bullish = snapshot.features.get_bool("ema_bullish")
            move_efficiency = snapshot.features.get_float("move_efficiency")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            # Last candle
            last_candle = m5_df.iloc[-1]
            high = float(last_candle["high"])
            low = float(last_candle["low"])
            close = float(last_candle["close"])

            # Setup detection
            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            # Pullback checks
            # Bullish: ema_bullish is active, low dipped below or touched ema20, close held above ema20
            if ema_bullish:
                if low <= ema20 and close >= ema20 * 0.999:
                    setup_type = "PULLBACK"
                    side = "BUY CALL"
                    # SL set below ema50 with small buffer
                    sl = min(ema50 - (atr * 0.2), price - (atr * 0.5))
            else:
                # Bearish: ema_bullish is false, high peaked above or touched ema20, close held below ema20
                if high >= ema20 and close <= ema20 * 1.001:
                    setup_type = "PULLBACK"
                    side = "BUY PUT"
                    # SL set above ema50 with small buffer
                    sl = max(ema50 + (atr * 0.2), price + (atr * 0.5))

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            # Rejections
            rejection_reasons = []
            
            # 1. Open hours blackout
            current_time = snapshot.timestamp
            if current_time.hour == 9 and current_time.minute < 45:
                rejection_reasons.append("TIME_FILTER")

            # 2. RVOL filter
            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")

            # 3. Macro trend bias filter (cannot trade opposed to daily bias)
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

            # 4. Trend quality/efficiency filter
            if move_efficiency < self.min_efficiency:
                rejection_reasons.append("LOW_EFFICIENCY")

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

            if not take_profit:
                take_profit = (price + 2.0 * risk_dist) if side == "BUY CALL" else (price - 2.0 * risk_dist)

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
                # Scale confidence based on move efficiency and RVOL
                eff_factor = min(move_efficiency / 1.0, 1.0)
                rvol_factor = min(rvol / 2.0, 1.0)
                confidence = round(0.5 + 0.3 * eff_factor + 0.2 * rvol_factor, 2)

            # Diagnostics payload for Feature Attribution Engine
            diagnostics = {
                "ema_distance": round((price - ema20) / ema20, 5) if ema20 > 0 else 0.0,
                "ema20": round(ema20, 2),
                "ema50": round(ema50, 2),
                "move_efficiency": round(move_efficiency, 3),
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "atr_percentile": round(snapshot.features.get_float("atr_percentile"), 3),
                "distance_to_vwap": round(snapshot.features.get_float("distance_to_vwap"), 4)
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_EMAPULL_{price:.2f}_{current_time.strftime('%Y%m%d')}"

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
            logger.error(f"[EmaPullbackStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings
        )
