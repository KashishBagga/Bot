#!/usr/bin/env python3
"""
ATR-Based Risk Manager with Trailing Stops
==========================================
Provides ATR-based stop-loss / take-profit calculations and
per-trade trailing stop management.

Design:
  - SL  = entry ± ATR_MULT_SL  × ATR   (default 1.5×)
  - TP1 = entry ± ATR_MULT_TP1 × ATR   (default 2.5×)
  - TP2 = entry ± ATR_MULT_TP2 × ATR   (default 4.0×)
  - Trailing stop activates after price moves ATR_TRAIL_TRIGGER × ATR
    in profit, then trails by ATR_TRAIL_DIST × ATR.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Configurable multipliers ──────────────────────────────────────────────────
ATR_MULT_SL      = 1.5   # Stop-loss distance in ATR units
ATR_MULT_TP1     = 2.5   # Take-profit-1 distance (partial exit)
ATR_MULT_TP2     = 4.0   # Take-profit-2 distance (full exit)
ATR_TRAIL_TRIGGER = 1.5  # Move this many ATRs in profit before trailing activates
ATR_TRAIL_DIST   = 1.0   # Keep trailing stop this many ATRs behind high-water mark
ATR_PERIOD       = 14    # Rolling window for ATR
MIN_ATR_FALLBACK = 0.01  # Fallback: 1% of price when ATR is 0 / NaN


@dataclass
class ATRLevels:
    """Pre-computed ATR-based price levels for a trade."""
    entry_price:      float
    atr:              float
    direction:        str    # 'BUY CALL' or 'BUY PUT'

    # Calculated levels
    stop_loss:        float = 0.0
    take_profit_1:    float = 0.0
    take_profit_2:    float = 0.0

    # State for trailing stop
    trailing_active:  bool  = False
    high_water_mark:  float = 0.0   # best price seen (favourable direction)
    trailing_stop:    float = 0.0   # current trailing SL price

    def __post_init__(self):
        atr = self.atr if self.atr and self.atr > 0 else self.entry_price * MIN_ATR_FALLBACK
        if self.direction == 'BUY CALL':
            self.stop_loss     = self.entry_price - ATR_MULT_SL  * atr
            self.take_profit_1 = self.entry_price + ATR_MULT_TP1 * atr
            self.take_profit_2 = self.entry_price + ATR_MULT_TP2 * atr
            self.high_water_mark = self.entry_price
            self.trailing_stop   = self.stop_loss
        else:  # BUY PUT
            self.stop_loss     = self.entry_price + ATR_MULT_SL  * atr
            self.take_profit_1 = self.entry_price - ATR_MULT_TP1 * atr
            self.take_profit_2 = self.entry_price - ATR_MULT_TP2 * atr
            self.high_water_mark = self.entry_price
            self.trailing_stop   = self.stop_loss


class ATRRiskManager:
    """
    Provides ATR-based SL/TP levels and real-time trailing stop management.
    One instance lives inside the Trader; it keeps a registry of open trade
    ATR levels and updates them each monitoring cycle.
    """

    def __init__(self):
        # trade_id → ATRLevels
        self._levels: Dict[str, ATRLevels] = {}

    # ── ATR helper ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_atr(data: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        """
        Compute ATR from an OHLCV DataFrame.
        Returns a scalar float.
        """
        try:
            if data is None or len(data) < period + 1:
                return 0.0
            required = {'high', 'low', 'close'}
            if not required.issubset(data.columns):
                return 0.0

            hi  = data['high']
            lo  = data['low']
            pc  = data['close'].shift(1)

            tr = pd.concat([
                hi - lo,
                (hi - pc).abs(),
                (lo - pc).abs()
            ], axis=1).max(axis=1)

            atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
            return float(atr) if not pd.isna(atr) else 0.0
        except Exception as e:
            logger.error(f"ATR computation error: {e}")
            return 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def register_trade(
        self,
        trade_id:    str,
        direction:   str,
        entry_price: float,
        atr:         float,
    ) -> ATRLevels:
        """
        Create and store ATR levels for a new trade.
        Returns the ATRLevels object (caller may store it too).
        """
        levels = ATRLevels(
            entry_price=entry_price,
            atr=atr,
            direction=direction,
        )
        self._levels[trade_id] = levels
        logger.info(
            f"📐 ATR levels registered [{trade_id}] {direction} @ {entry_price:.2f} "
            f"| ATR={atr:.2f} | SL={levels.stop_loss:.2f} "
            f"| TP1={levels.take_profit_1:.2f} | TP2={levels.take_profit_2:.2f}"
        )
        return levels

    def remove_trade(self, trade_id: str):
        """Remove a closed trade from the registry."""
        self._levels.pop(trade_id, None)

    def get_levels(self, trade_id: str) -> Optional[ATRLevels]:
        return self._levels.get(trade_id)

    def update_trailing_stop(
        self, trade_id: str, current_price: float
    ) -> Tuple[Optional[float], bool]:
        """
        Update the trailing stop for a trade given the current price.

        Returns:
            (current_trailing_stop_price, should_exit)
        """
        levels = self._levels.get(trade_id)
        if levels is None:
            return None, False

        atr = levels.atr if levels.atr > 0 else levels.entry_price * MIN_ATR_FALLBACK

        if levels.direction == 'BUY CALL':
            # ── LONG trade ────────────────────────────────────────────────
            # Update high-water mark
            if current_price > levels.high_water_mark:
                levels.high_water_mark = current_price

            profit_in_atr = (levels.high_water_mark - levels.entry_price) / atr

            # Activate trailing once we're +TRIGGER ATRs in profit
            if profit_in_atr >= ATR_TRAIL_TRIGGER:
                levels.trailing_active = True

            if levels.trailing_active:
                # Trail: keep SL at (HWM - TRAIL_DIST × ATR)
                new_trail = levels.high_water_mark - ATR_TRAIL_DIST * atr
                if new_trail > levels.trailing_stop:
                    levels.trailing_stop = new_trail
                    logger.debug(
                        f"📈 Trailing stop updated [{trade_id}]: "
                        f"HWM={levels.high_water_mark:.2f} → TSL={levels.trailing_stop:.2f}"
                    )

            # Exit if current price falls below trailing or hard SL
            effective_sl = max(levels.trailing_stop, levels.stop_loss)
            should_exit  = current_price <= effective_sl

        else:
            # ── SHORT / BUY PUT trade ─────────────────────────────────────
            if current_price < levels.high_water_mark:
                levels.high_water_mark = current_price

            profit_in_atr = (levels.entry_price - levels.high_water_mark) / atr

            if profit_in_atr >= ATR_TRAIL_TRIGGER:
                levels.trailing_active = True

            if levels.trailing_active:
                new_trail = levels.high_water_mark + ATR_TRAIL_DIST * atr
                if new_trail < levels.trailing_stop:
                    levels.trailing_stop = new_trail

            effective_sl = min(levels.trailing_stop, levels.stop_loss)
            should_exit  = current_price >= effective_sl

        return levels.trailing_stop, should_exit

    def check_take_profit(self, trade_id: str, current_price: float) -> Optional[str]:
        """
        Check whether TP1 or TP2 have been hit.
        Returns 'TP1', 'TP2', or None.
        """
        levels = self._levels.get(trade_id)
        if levels is None:
            return None

        if levels.direction == 'BUY CALL':
            if current_price >= levels.take_profit_2:
                return 'TP2'
            if current_price >= levels.take_profit_1:
                return 'TP1'
        else:  # BUY PUT
            if current_price <= levels.take_profit_2:
                return 'TP2'
            if current_price <= levels.take_profit_1:
                return 'TP1'
        return None

    def check_exit(self, trade_id: str, current_price: float) -> Optional[str]:
        """
        Convenience method: update trailing then check all exits.
        Returns exit reason string or None.
        """
        _, sl_hit = self.update_trailing_stop(trade_id, current_price)
        if sl_hit:
            levels = self._levels.get(trade_id)
            reason = "TRAILING_STOP" if (levels and levels.trailing_active) else "STOP_LOSS"
            return reason

        tp = self.check_take_profit(trade_id, current_price)
        if tp:
            return f"TARGET_HIT_{tp}"

        return None

    def get_summary(self, trade_id: str) -> dict:
        """Return a readable summary of a trade's ATR levels."""
        levels = self._levels.get(trade_id)
        if not levels:
            return {}
        return {
            "trade_id":        trade_id,
            "direction":       levels.direction,
            "entry":           levels.entry_price,
            "atr":             levels.atr,
            "stop_loss":       levels.stop_loss,
            "take_profit_1":   levels.take_profit_1,
            "take_profit_2":   levels.take_profit_2,
            "trailing_active": levels.trailing_active,
            "trailing_stop":   levels.trailing_stop,
            "high_water_mark": levels.high_water_mark,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
atr_risk_manager = ATRRiskManager()
