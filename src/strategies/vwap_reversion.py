#!/usr/bin/env python3
"""
VWAP Reversion Strategy
=======================
Hypothesis: Price mean-reverts when overstretched from intraday VWAP.
Exploits statistical stretch using distance to VWAP and price action confirmation.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)


class VwapReversionStrategy(BaseStrategy):
    """
    VWAP Reversion Strategy.
    """

    metadata = StrategyMetadata(
        id="vwap_reversion",
        name="VWAP Reversion",
        hypothesis_id="mean_reversion_vwap",
        hypothesis_family="Mean Reversion",
        hypothesis_text="Trades reversion to VWAP when price is overstretched and prints a rejection candle.",
        version="v1.0",
        maturity="PAPER",
        tags=["mean_reversion", "vwap", "stretch"]
    )

    def __init__(
        self,
        rvol_threshold: float = 1.0,
        vwap_stretch_multiplier: float = 1.5,
    ):
        self.rvol_threshold = rvol_threshold
        self.vwap_stretch_multiplier = vwap_stretch_multiplier

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
            distance_to_vwap = snapshot.features.get_float("distance_to_vwap")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            # Last candle
            last_candle = m5_df.iloc[-1]
            high = float(last_candle["high"])
            low = float(last_candle["low"])

            # Derived VWAP price
            # distance_to_vwap = (price - vwap) / vwap  => vwap = price / (1 + distance_to_vwap)
            vwap = price / (1.0 + distance_to_vwap) if (1.0 + distance_to_vwap) > 0 else price
            abs_dist = abs(price - vwap)

            # Setup detection
            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            # Stretch requirement: absolute distance >= stretch_multiplier * atr
            stretch_threshold = self.vwap_stretch_multiplier * atr
            
            if abs_dist >= stretch_threshold and QuantUtils.is_strong_rejection(m5_df):
                setup_type = "REVERSION"
                if distance_to_vwap > 0.0:
                    # Overstretched upward -> Mean revert downward (BUY PUT)
                    side = "BUY PUT"
                    # ATR-scaled buffer instead of fixed 1.0pt — important for BankNifty
                    sl = max(high + (atr * 0.15), price + (atr * 0.5))
                    take_profit = vwap
                elif distance_to_vwap < 0.0:
                    # Overstretched downward -> Mean revert upward (BUY CALL)
                    side = "BUY CALL"
                    sl = min(low - (atr * 0.15), price - (atr * 0.5))
                    take_profit = vwap

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            # Rejections
            rejection_reasons = []
            
            # 1. Open hours blackout (Removed)
            current_time = snapshot.timestamp

            # 2. RVOL filter
            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")

            # 3. Macro trend bias filter (do not reversion trade opposite to strong daily bias)
            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

            # Invalidation buffer
            risk_dist = abs(price - sl) if sl else atr

            # ── FIX: Enforce minimum SL floor of 0.5×ATR from entry ──────────
            min_sl_dist = atr * 0.5
            if side == "BUY PUT" and (sl - price) < min_sl_dist:
                sl = price + min_sl_dist
                risk_dist = min_sl_dist
            elif side == "BUY CALL" and (price - sl) < min_sl_dist:
                sl = price - min_sl_dist
                risk_dist = min_sl_dist
            # ─────────────────────────────────────────────────────────────────

            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")

            # ── FIX: Move efficiency gate — choppy moves don't revert cleanly
            move_efficiency = snapshot.features.get_float("move_efficiency")
            if move_efficiency < 0.5:
                rejection_reasons.append("LOW_EFFICIENCY")

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
                # Scale confidence based on stretch depth (more stretched = higher probability of reversion)
                stretch_ratio = abs_dist / max(stretch_threshold, 1.0)
                confidence = round(min(0.5 + 0.3 * (stretch_ratio - 1.0), 0.95), 2)

            # Diagnostics payload for Feature Attribution Engine
            diagnostics = {
                "distance_to_vwap": round(distance_to_vwap, 5),
                "abs_distance": round(abs_dist, 2),
                "vwap_price": round(vwap, 2),
                "stretch_threshold": round(stretch_threshold, 2),
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
                "atr_percentile": round(snapshot.features.get_float("atr_percentile"), 3),
                "move_efficiency": round(snapshot.features.get_float("move_efficiency"), 3)
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_VWAPREV_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"

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
            logger.error(f"[VwapReversionStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings
        )
