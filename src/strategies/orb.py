#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) Strategy
=====================================
Hypothesis: The high/low range created during the opening auction period 
determines the key institutional levels for the day. A breakout of this range 
indicates the dominant market direction.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime, time

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)


class OrbStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy.
    Monitors 15-minute or 30-minute opening range.
    """

    metadata = StrategyMetadata(
        id="orb",
        name="Opening Range Breakout",
        hypothesis_id="opening_range_breakout",
        hypothesis_family="Breakouts",
        hypothesis_text="Trades momentum breakouts of the session's initial high-low range.",
        version="v1.0",
        maturity="PAPER",
        tags=["orb", "opening", "breakout"]
    )

    def __init__(
        self,
        rvol_threshold: float = 1.2,
        opening_range_minutes: int = 15,
        min_efficiency: float = 0.6,
    ):
        self.rvol_threshold = rvol_threshold
        self.opening_range_minutes = opening_range_minutes
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
            if len(m5_df) < 20:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            move_efficiency = snapshot.features.get_float("move_efficiency")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            current_time = snapshot.timestamp
            today_date = current_time.date()

            # Filter candles belonging to today
            today_mask = m5_df.index.date == today_date
            today_m5 = m5_df[today_mask]

            if len(today_m5) == 0:
                return self._empty_result(experiment_name)

            # Determine the cutoff time for the opening range
            # Session start is assumed to be 09:15 IST
            if self.opening_range_minutes == 15:
                cutoff_time = time(9, 30)
            elif self.opening_range_minutes == 30:
                cutoff_time = time(9, 45)
            else:
                cutoff_time = time(9, 30)  # Default fallback 15m

            # Find opening candles
            opening_candles = today_m5[today_m5.index.time <= cutoff_time]
            if len(opening_candles) == 0:
                return self._empty_result(experiment_name)

            # If current time is still within the opening range window, do not trade
            if current_time.time() <= cutoff_time:
                return self._empty_result(experiment_name)

            # Calculate opening range High/Low
            range_high = float(opening_candles["high"].max())
            range_low = float(opening_candles["low"].min())

            # Last candle
            last_candle = today_m5.iloc[-1]
            close = float(last_candle["close"])
            high = float(last_candle["high"])
            low = float(last_candle["low"])

            # Previous candle
            prev_candle = today_m5.iloc[-2]
            prev_close = float(prev_candle["close"])

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            # Bullish Breakout: close crosses above range_high
            if prev_close <= range_high and close > range_high:
                setup_type = f"ORB_{self.opening_range_minutes}M"
                side = "BUY CALL"
                sl = min(range_high - (atr * 0.3), price - (atr * 0.5))
                take_profit = price + (atr * 3.0)

            # Bearish Breakout: close crosses below range_low
            elif prev_close >= range_low and close < range_low:
                setup_type = f"ORB_{self.opening_range_minutes}M"
                side = "BUY PUT"
                sl = max(range_low + (atr * 0.3), price + (atr * 0.5))
                take_profit = price - (atr * 3.0)

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            # Rejections
            rejection_reasons = []

            # 1. RVOL filter
            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")

            # 2. Daily bias filter (cannot trade opposed to daily bias)
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

            # 3. Efficiency filter
            if move_efficiency < self.min_efficiency:
                rejection_reasons.append("LOW_EFFICIENCY")

            # 4. Late session entry guard
            if current_time.hour >= 15:
                rejection_reasons.append("LATE_SESSION")

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
                # Breakouts with higher move efficiency and volume get higher conviction
                eff_factor = min(move_efficiency / 1.0, 1.0)
                rvol_factor = min(rvol / 2.0, 1.0)
                confidence = round(0.5 + 0.25 * eff_factor + 0.25 * rvol_factor, 2)

            # Diagnostics payload for Feature Attribution Engine
            diagnostics = {
                "orb_high": round(range_high, 2),
                "orb_low": round(range_low, 2),
                "orb_minutes": self.opening_range_minutes,
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "atr_percentile": round(snapshot.features.get_float("atr_percentile"), 3),
                "move_efficiency": round(move_efficiency, 3)
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_ORB_{self.opening_range_minutes}M_{price:.2f}_{current_time.strftime('%Y%m%d')}"

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
            logger.error(f"[OrbStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings
        )
