#!/usr/bin/env python3
"""
ADX/DMI + CCI Trend-Strength Strategy
=======================================
Hypothesis: ADX>threshold confirms a genuine trend exists (filters out the
sideways chop every other strategy in this system already struggles with).
+DI/-DI crossover gives trend direction. CCI crossing +-100 in that same
direction, off a fresh crossover, is the actual entry trigger — it catches
the burst of momentum that kicks a newly-confirmed trend off, rather than
entering on a trend that's already extended.

This fills a real gap: nothing else in this system reads a DMI/CCI momentum
combination — the other trend/momentum entrants (EMA pullback, VWAP reclaim,
momentum burst) all key off price structure or moving averages, not a
smoothed-directional-movement oscillator pair. Self-contained: +DI/-DI/CCI
are computed locally from snapshot.m5, not added to the shared FeatureStore.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


def _compute_di(df: pd.DataFrame, period: int = 14):
    """+DI / -DI via Wilder smoothing — same TR/DM construction as
    RegimeEngine._compute_adx, extended to return the directional lines
    themselves (that helper only returns the scalar ADX + its slope)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    dm_plus_smooth = pd.Series(dm_plus, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    dm_minus_smooth = pd.Series(dm_minus, index=df.index).ewm(alpha=1 / period, adjust=False).mean()

    di_plus = 100 * dm_plus_smooth / tr_smooth.replace(0, np.nan)
    di_minus = 100 * dm_minus_smooth / tr_smooth.replace(0, np.nan)
    return di_plus, di_minus


def _compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = typical_price.rolling(period).mean()
    mean_dev = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical_price - sma) / (0.015 * mean_dev.replace(0, np.nan))


class AdxDmiCciStrategy(BaseStrategy):
    """ADX-confirmed trend + DMI crossover direction + CCI momentum trigger."""

    metadata = StrategyMetadata(
        id="adx_dmi_cci",
        name="ADX/DMI + CCI Trend Strength",
        hypothesis_id="adx_dmi_cci_trend_trigger",
        hypothesis_family="Trend Continuation",
        hypothesis_text=(
            "When ADX confirms a real trend (>threshold) and +DI/-DI just "
            "crossed in a direction, a CCI cross past +-100 in that same "
            "direction is a genuine momentum trigger, not noise."
        ),
        version="v1.0",
        archetype="Trend-Continuation",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["adx", "dmi", "cci", "trend_strength"],
    )

    def __init__(
        self,
        adx_threshold: float = 20.0,
        cci_period: int = 20,
        cci_entry_level: float = 100.0,
        di_period: int = 14,
        atr_sl_buffer_mult: float = 0.3,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
        rvol_floor: float = 0.7,
    ):
        self.adx_threshold = adx_threshold
        self.cci_period = cci_period
        self.cci_entry_level = cci_entry_level
        self.di_period = di_period
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        self.rvol_floor = rvol_floor

    def thesis_key(self, signal: dict) -> tuple:
        return (signal.get("symbol", ""), "ADX_DMI_CCI", signal.get("signal", ""))

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            min_bars = max(self.cci_period, self.di_period) + 5
            if m5_df is None or len(m5_df) < min_bars:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            adx = snapshot.features.get_float("adx")

            di_plus, di_minus = _compute_di(m5_df, period=self.di_period)
            cci = _compute_cci(m5_df, period=self.cci_period)

            if pd.isna(di_plus.iloc[-1]) or pd.isna(di_minus.iloc[-1]) or pd.isna(cci.iloc[-1]) or pd.isna(cci.iloc[-2]):
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:dmi_cci"])

            di_plus_now, di_minus_now = float(di_plus.iloc[-1]), float(di_minus.iloc[-1])
            cci_now, cci_prev = float(cci.iloc[-1]), float(cci.iloc[-2])
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            current_time = snapshot.timestamp

            setup_type = "NONE"
            side = None
            sl = None
            take_profit = None

            trend_up = di_plus_now > di_minus_now
            trend_down = di_minus_now > di_plus_now
            cci_cross_up = cci_prev <= self.cci_entry_level and cci_now > self.cci_entry_level
            cci_cross_down = cci_prev >= -self.cci_entry_level and cci_now < -self.cci_entry_level

            if trend_up and cci_cross_up:
                setup_type = "ADX_DMI_CCI_LONG"
                side = "BUY CALL"
                sl = price - max(atr * self.atr_sl_buffer_mult, atr * 0.5)
                take_profit = price + (atr * self.tp_atr_cap)
            elif trend_down and cci_cross_down:
                setup_type = "ADX_DMI_CCI_SHORT"
                side = "BUY PUT"
                sl = price + max(atr * self.atr_sl_buffer_mult, atr * 0.5)
                take_profit = price - (atr * self.tp_atr_cap)

            if setup_type == "NONE":
                return self._empty_result(experiment_name)

            rejection_reasons: List[str] = []

            if adx < self.adx_threshold:
                rejection_reasons.append("WEAK_TREND_ADX")

            if rvol < self.rvol_floor:
                rejection_reasons.append("LOW_RVOL")

            if side == "BUY CALL" and snapshot.daily_bias == "BEARISH":
                rejection_reasons.append("BIAS_MISMATCH")
            elif side == "BUY PUT" and snapshot.daily_bias == "BULLISH":
                rejection_reasons.append("BIAS_MISMATCH")

            risk_dist = abs(price - sl)
            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")

            reward = abs(take_profit - price)
            rr = round(reward / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < self.min_rr:
                rejection_reasons.append("LOW_RR")

            confidence = 0.5
            if len(rejection_reasons) == 0:
                confidence = round(min(0.5 + 0.01 * (adx - self.adx_threshold), 0.9), 2)

            diagnostics = {
                "adx": round(adx, 2),
                "di_plus": round(di_plus_now, 2),
                "di_minus": round(di_minus_now, 2),
                "cci": round(cci_now, 2),
                "rvol": round(rvol, 2),
                "rr_ratio": rr,
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_ADXDMICCI_"
                f"{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            sig = {
                "symbol": snapshot.symbol,
                "signal": side,
                "strategy": setup_type,
                "price": price,
                "stop_loss": sl,
                "take_profit": take_profit,
                "tp1": price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5),
                "rr_ratio": rr,
                "timestamp": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "features": snapshot.features.to_dict(),
                "candidate_id": candidate_id,
                "confidence": confidence,
                "diagnostics": diagnostics,
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[AdxDmiCciStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
