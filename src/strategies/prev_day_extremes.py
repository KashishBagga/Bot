#!/usr/bin/env python3
"""
Previous Day High / Low (PDH/PDL) Strategy
=========================================
Hypothesis: Major liquidity pools accumulate around yesterday's extremes. 
Price sweeps or breakouts of yesterday's high/low yield strong reversals or momentum continuation.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)


class PrevDayExtremesStrategy(BaseStrategy):
    """
    Previous Day High/Low Strategy.
    Supports both Reversals (fakeouts/sweeps) and Breakouts (volume-backed continuation).
    """

    metadata = StrategyMetadata(
        id="prev_day_extremes",
        name="Previous Day High/Low",
        hypothesis_id="liquidity_extremes",
        hypothesis_family="Liquidity Sweeps",
        hypothesis_text="Trades sweeps (fakeouts) or volume-backed breakouts of the previous day's high/low.",
        version="v1.0",
        maturity="PAPER",
        tags=["prev_day", "reversal", "breakout", "liquidity"]
    )

    def __init__(
        self,
        reversal_rvol_threshold: float = 1.0,
        breakout_rvol_threshold: float = 1.2,
        proximity_multiplier: float = 0.3,
    ):
        self.reversal_rvol_threshold = reversal_rvol_threshold
        self.breakout_rvol_threshold = breakout_rvol_threshold
        self.proximity_multiplier = proximity_multiplier

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
            d1_df = snapshot.d1
            if len(m5_df) < 50 or d1_df is None or len(d1_df) < 2:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            dist_prev_high = snapshot.features.get_float("dist_prev_high")
            dist_prev_low = snapshot.features.get_float("dist_prev_low")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            # Derive PDH/PDL levels
            # dist_prev_high = (price - prev_high) / prev_high  => prev_high = price / (1 + dist_prev_high)
            prev_high = price / (1.0 + dist_prev_high) if (1.0 + dist_prev_high) > 0 else price
            prev_low = price / (1.0 + dist_prev_low) if (1.0 + dist_prev_low) > 0 else price

            # Last candle
            last_candle = m5_df.iloc[-1]
            high = float(last_candle["high"])
            low = float(last_candle["low"])
            close = float(last_candle["close"])
            open_ = float(last_candle["open"])

            # Previous candle
            prev_candle = m5_df.iloc[-2]
            prev_close = float(prev_candle["close"])

            # Proximity
            proximity_buffer = self.proximity_multiplier * atr
            is_near_high = abs(price - prev_high) <= proximity_buffer or (low <= prev_high <= high)
            is_near_low = abs(price - prev_low) <= proximity_buffer or (low <= prev_low <= high)

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None
            rejection_reasons = []

            # --- 1. Reversal (Fakeout/Sweep) Logic ---
            if setup_type == "NONE" and QuantUtils.is_strong_rejection(m5_df):
                # High Reversal (Bearish Sweep)
                if is_near_high and high >= prev_high and close < prev_high:
                    setup_type = "REVERSAL"
                    side = "BUY PUT"
                    # BUG FIX: ATR-scaled buffer (was fixed +1.0pt — dangerous for BankNifty)
                    sl = max(high + (atr * 0.15), price + (atr * 0.5))
                    # TP set to opposite low or default
                    take_profit = prev_low
                    if rvol < self.reversal_rvol_threshold:
                        rejection_reasons.append("LOW_RVOL")
                    if snapshot.daily_bias == "BULLISH":
                        rejection_reasons.append("BIAS_MISMATCH")

                # Low Reversal (Bullish Sweep)
                elif is_near_low and low <= prev_low and close > prev_low:
                    setup_type = "REVERSAL"
                    side = "BUY CALL"
                    # BUG FIX: ATR-scaled buffer (was fixed -1.0pt — dangerous for BankNifty)
                    sl = min(low - (atr * 0.15), price - (atr * 0.5))
                    take_profit = prev_high
                    if rvol < self.reversal_rvol_threshold:
                        rejection_reasons.append("LOW_RVOL")
                    if snapshot.daily_bias == "BEARISH":
                        rejection_reasons.append("BIAS_MISMATCH")

            # --- 2. Breakout Logic ---
            if setup_type == "NONE":
                # High Breakout (Bullish Continuation)
                if prev_close <= prev_high and close > prev_high:
                    setup_type = "BREAKOUT"
                    side = "BUY CALL"
                    sl = min(prev_high - (atr * 0.3), price - (atr * 0.5))
                    take_profit = price + (atr * 3.0)  # Default target
                    if rvol < self.breakout_rvol_threshold:
                        rejection_reasons.append("LOW_RVOL")
                    if snapshot.daily_bias != "BULLISH":
                        rejection_reasons.append("BIAS_MISMATCH")

                # Low Breakout (Bearish Continuation)
                elif prev_close >= prev_low and close < prev_low:
                    setup_type = "BREAKOUT"
                    side = "BUY PUT"
                    sl = max(prev_low + (atr * 0.3), price + (atr * 0.5))
                    take_profit = price - (atr * 3.0)
                    if rvol < self.breakout_rvol_threshold:
                        rejection_reasons.append("LOW_RVOL")
                    if snapshot.daily_bias != "BEARISH":
                        rejection_reasons.append("BIAS_MISMATCH")

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            # --- 3. Filter checks ---
            # Open hours blackout (Removed)
            current_time = snapshot.timestamp

            # Invalidation buffer
            risk_dist = abs(price - sl) if sl else atr

            # SL floor: minimum 0.5×ATR distance from entry (same fix as all other strategies)
            min_sl_dist = atr * 0.5
            if side == "BUY PUT" and (sl - price) < min_sl_dist:
                sl = price + min_sl_dist
                risk_dist = min_sl_dist
            elif side == "BUY CALL" and (price - sl) < min_sl_dist:
                sl = price - min_sl_dist
                risk_dist = min_sl_dist

            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")

            # Target zones
            for z in snapshot.h1_zones:
                if side == "BUY CALL" and z.level > price:
                    take_profit = z.level
                    break
                if side == "BUY PUT" and z.level < price:
                    take_profit = z.level
                    break

            # Cap TP at 5x ATR from entry
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
                # Breakouts with RVOL > 1.8 have higher confidence
                if setup_type == "BREAKOUT":
                    confidence = round(min(0.6 + 0.2 * (rvol / 1.5), 0.95), 2)
                else:
                    # Reversals with deep rejection wicks
                    wick_ratio = snapshot.features.get_float("wickiness")
                    confidence = round(min(0.6 + 0.3 * wick_ratio, 0.9), 2)

            # Diagnostics payload for Feature Attribution Engine
            diagnostics = {
                "dist_prev_high": round(dist_prev_high, 5),
                "dist_prev_low": round(dist_prev_low, 5),
                "prev_high": round(prev_high, 2),
                "prev_low": round(prev_low, 2),
                "setup_type": setup_type,
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "atr_percentile": round(snapshot.features.get_float("atr_percentile"), 3)
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_PDHL_{setup_type}_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"

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
            logger.error(f"[PrevDayExtremesStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings
        )
