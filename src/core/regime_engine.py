#!/usr/bin/env python3
"""
Market Regime Detection Engine (Enhanced)
==========================================
Classifies market conditions using a multi-signal approach:

  - ADX (14-period):          Trend strength — strong (>25), weak (15-25), range (<15)
  - ATR percentile:           Volatility state — expansion (>80th), compression (<20th)
  - Gap detection:            Is today a gap-open day (|open - prev_close| > 0.4%)?
  - EMA cross (m5 frame):     Direction — TREND_UP vs TREND_DOWN
  - Session:                  OPEN (09:15-09:45 IST) | MID | CLOSE (14:45+)

Returns a `RegimeLabel` dataclass. The `.label` property produces the legacy
string format ("STRONG_TREND_UP_HIGH_VOL") already expected by MarketSnapshot
and trade_performance.market_regime -- fully backward compatible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RegimeLabel
# ---------------------------------------------------------------------------

@dataclass
class RegimeLabel:
    """
    Structured regime classification for one symbol at one point in time.

    Attributes
    ----------
    primary   : Coarse directional + strength label.
                One of: STRONG_TREND_UP | WEAK_TREND_UP | STRONG_TREND_DOWN |
                        WEAK_TREND_DOWN | RANGE | COMPRESSION | GAP_UP | GAP_DOWN
    adx       : Raw 14-period ADX value (float).
    adx_slope : Change in ADX over the last 3 bars (positive = strengthening).
    atr_pct   : ATR percentile rank vs. trailing 250 bars (0.0-1.0).
    is_gap_day: True if the session opened with a gap > 0.4% vs. prior close.
    gap_pct   : Raw gap size as a fraction of prior close (signed: + = gap up).
    session   : "OPEN" | "MID" | "CLOSE"
    vol_state : "HIGH_VOL" | "NORMAL" | "LOW_VOL"
    """

    primary: str
    adx: float
    adx_slope: float
    atr_pct: float
    is_gap_day: bool
    gap_pct: float
    session: str
    vol_state: str
    # MTF context — additive, does not affect `primary`/`vol_state` classification.
    # h1_trend_aligned: True if h1 structure agrees with the m5 EMA20/50 direction,
    # False if it conflicts, None if h1 trend is NEUTRAL/unavailable.
    h1_trend_aligned: Optional[bool] = None
    d1_trend: str = "FLAT"   # "UP" | "DOWN" | "FLAT"

    @property
    def label(self) -> str:
        """
        Backward-compatible string for MarketSnapshot.market_regime.
        Produces the same format as the old RegimeEngine (e.g. "RANGE_HIGH_VOL").
        """
        return f"{self.primary}_{self.vol_state}"

    def to_dict(self) -> dict:
        """Serialize for storage in diagnostics / feature store."""
        return {
            "primary":    self.primary,
            "adx":        round(self.adx, 2),
            "adx_slope":  round(self.adx_slope, 4),
            "atr_pct":    round(self.atr_pct, 4),
            "is_gap_day": self.is_gap_day,
            "gap_pct":    round(self.gap_pct, 4),
            "session":    self.session,
            "vol_state":  self.vol_state,
            "label":      self.label,
            "h1_trend_aligned": self.h1_trend_aligned,
            "d1_trend":         self.d1_trend,
        }


# ---------------------------------------------------------------------------
# RegimeEngine
# ---------------------------------------------------------------------------

class RegimeEngine:
    """
    Classifies the current market regime for one symbol per candle.

    Usage (same interface as the old RegimeEngine, enhanced):
        engine = RegimeEngine()
        label  = engine.detect_regime(m5, d1=d1, now=timestamp)
        # label is a RegimeLabel; label.label is the legacy string
    """

    # Thresholds
    ADX_STRONG    = 25.0   # ADX > this -> strong trend
    ADX_WEAK      = 15.0   # ADX 15-25  -> weak trend; < 15 -> range
    GAP_THRESHOLD = 0.004  # |gap| > 0.4% of prior close -> gap day
    ATR_HIGH_VOL  = 0.80   # ATR percentile above this -> HIGH_VOL
    ATR_LOW_VOL   = 0.20   # ATR percentile below this -> LOW_VOL (compression)
    ATR_HIST_BARS = 250    # bars used for ATR percentile ranking

    def __init__(self, vol_window: int = 20, trend_window: int = 50):
        # kept for backward compat
        self.vol_window   = vol_window
        self.trend_window = trend_window

    # Public interface

    def detect_regime(
        self,
        m5: pd.DataFrame,
        d1: Optional[pd.DataFrame] = None,
        h1_structure: Optional[object] = None,
        now: Optional[datetime] = None,
    ) -> RegimeLabel:
        """
        Classify the current market regime.

        Parameters
        ----------
        m5  : 5-minute OHLCV DataFrame (primary timeframe).
        d1  : Daily OHLCV DataFrame (used for gap detection + d1_trend).
        h1_structure : object exposing `.trend` ("BULLISH"/"BEARISH"/"NEUTRAL"),
                       e.g. MarketSnapshot.h1_structure — reused as-is, not
                       recomputed here, for h1_trend_aligned.
        now : Current timestamp (for session classification).

        Returns
        -------
        RegimeLabel with full breakdown, plus a .label string for backward compat.
        """
        if m5 is None or len(m5) < 30:
            return RegimeLabel(
                primary="UNKNOWN", adx=0.0, adx_slope=0.0, atr_pct=0.5,
                is_gap_day=False, gap_pct=0.0, session="MID", vol_state="NORMAL"
            )

        # 1. ADX (14-period)
        adx, adx_slope = self._compute_adx(m5, period=14)

        # 2. ATR percentile
        atr_pct = self._compute_atr_percentile(m5, hist_bars=self.ATR_HIST_BARS)

        # 3. EMA direction (m5 frame)
        ema20  = m5["close"].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50  = m5["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        is_up  = ema20 > ema50

        # 4. Gap detection
        is_gap_day, gap_pct = self._compute_gap(m5, d1)

        # 5. Session
        if now is None:
            try:
                now = m5.index[-1].to_pydatetime()
            except Exception:
                now = datetime.utcnow()
        session = self._classify_session(now)

        # 6. Volatility state
        if atr_pct >= self.ATR_HIGH_VOL:
            vol_state = "HIGH_VOL"
        elif atr_pct <= self.ATR_LOW_VOL:
            vol_state = "LOW_VOL"
        else:
            vol_state = "NORMAL"

        # 7. Primary regime
        if is_gap_day:
            primary = "GAP_UP" if gap_pct > 0 else "GAP_DOWN"
        elif atr_pct <= self.ATR_LOW_VOL:
            primary = "COMPRESSION"
        elif adx >= self.ADX_STRONG:
            primary = "STRONG_TREND_UP" if is_up else "STRONG_TREND_DOWN"
        elif adx >= self.ADX_WEAK:
            primary = "WEAK_TREND_UP" if is_up else "WEAK_TREND_DOWN"
        else:
            primary = "RANGE"

        # 8. MTF context (additive — does not affect `primary`/`vol_state` above)
        h1_trend_aligned = self._compute_h1_alignment(h1_structure, is_up)
        d1_trend = self._compute_d1_trend(d1)

        return RegimeLabel(
            primary=primary,
            adx=float(adx),
            adx_slope=float(adx_slope),
            atr_pct=float(atr_pct),
            is_gap_day=is_gap_day,
            gap_pct=float(gap_pct),
            session=session,
            vol_state=vol_state,
            h1_trend_aligned=h1_trend_aligned,
            d1_trend=d1_trend,
        )

    # Legacy compatibility

    def get_day_type(self, df: pd.DataFrame) -> str:
        """Legacy method -- kept for backward compatibility."""
        if df is None or len(df) < 10:
            return "NORMAL"
        high        = df["high"].max()
        low         = df["low"].min()
        open_price  = df["open"].iloc[0]
        close_price = df["close"].iloc[-1]
        move_pct    = abs(close_price - open_price) / open_price
        range_pct   = (high - low) / low if low > 0 else 0.0
        if move_pct > 0.015 and range_pct > 0.02:
            return "TREND_DAY"
        elif range_pct < 0.008:
            return "RANGE_DAY"
        return "NORMAL_DAY"

    def get_session_type(self, timestamp: datetime) -> str:
        """Legacy method -- kept for backward compatibility."""
        return self._classify_session(timestamp)

    # Private helpers

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14):
        """Compute ADX and its 3-bar slope."""
        try:
            high  = df["high"]
            low   = df["low"]
            close = df["close"]

            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)

            up_move   = high.diff()
            down_move = -low.diff()

            dm_plus  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
            dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            dm_plus_s  = pd.Series(dm_plus,  index=df.index)
            dm_minus_s = pd.Series(dm_minus, index=df.index)

            # Wilder smoothing
            tr_smooth       = tr.ewm(alpha=1/period, adjust=False).mean()
            dm_plus_smooth  = dm_plus_s.ewm(alpha=1/period, adjust=False).mean()
            dm_minus_smooth = dm_minus_s.ewm(alpha=1/period, adjust=False).mean()

            di_plus  = 100 * dm_plus_smooth  / tr_smooth.replace(0, np.nan)
            di_minus = 100 * dm_minus_smooth / tr_smooth.replace(0, np.nan)

            dx_denom   = (di_plus + di_minus).replace(0, np.nan)
            dx         = 100 * (di_plus - di_minus).abs() / dx_denom
            adx_series = dx.ewm(alpha=1/period, adjust=False).mean()

            adx_val   = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0
            adx_prev3 = float(adx_series.iloc[-4]) if len(adx_series) >= 4 and not pd.isna(adx_series.iloc[-4]) else adx_val
            slope     = adx_val - adx_prev3

            return adx_val, slope

        except Exception as e:
            logger.warning(f"[RegimeEngine] ADX computation failed: {e}")
            return 0.0, 0.0

    @staticmethod
    def _compute_atr_percentile(df: pd.DataFrame, hist_bars: int = 250) -> float:
        """Returns the current ATR percentile rank vs. trailing hist_bars."""
        try:
            prev_close = df["close"].shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"]  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_rolling = tr.rolling(window=14).mean()
            current_atr = atr_rolling.iloc[-1]
            lookback    = atr_rolling.tail(hist_bars).dropna()
            if len(lookback) < 2 or pd.isna(current_atr):
                return 0.5
            return float((lookback < current_atr).sum() / len(lookback))
        except Exception as e:
            logger.warning(f"[RegimeEngine] ATR percentile failed: {e}")
            return 0.5

    @staticmethod
    def _compute_gap(m5: pd.DataFrame, d1):
        """Detect an opening gap. Returns (is_gap_day, gap_pct)."""
        if d1 is None or len(d1) < 2:
            return False, 0.0
        try:
            today_date = m5.index[-1].date()
            today_m5   = m5[m5.index.date == today_date]
            if len(today_m5) == 0:
                return False, 0.0
            today_open = float(today_m5["open"].iloc[0])

            last_d1_date = d1.index[-1].date()
            if last_d1_date == today_date:
                prev_close = float(d1.iloc[-2]["close"]) if len(d1) >= 2 else None
            else:
                prev_close = float(d1.iloc[-1]["close"])

            if prev_close is None or prev_close <= 0:
                return False, 0.0

            gap_pct = (today_open - prev_close) / prev_close
            is_gap  = abs(gap_pct) > RegimeEngine.GAP_THRESHOLD
            return is_gap, gap_pct

        except Exception as e:
            logger.warning(f"[RegimeEngine] Gap detection failed: {e}")
            return False, 0.0

    @staticmethod
    def _compute_h1_alignment(h1_structure: Optional[object], m5_is_up: bool) -> Optional[bool]:
        """True if h1 structure trend agrees with the m5 EMA20/50 direction,
        False if it conflicts, None if h1 trend is NEUTRAL/unavailable."""
        trend = getattr(h1_structure, "trend", None) if h1_structure is not None else None
        if trend not in ("BULLISH", "BEARISH"):
            return None
        h1_is_up = (trend == "BULLISH")
        return h1_is_up == m5_is_up

    @staticmethod
    def _compute_d1_trend(d1: Optional[pd.DataFrame]) -> str:
        """SMA(10) vs SMA(20) of daily close over the already-fetched d1 window."""
        if d1 is None or len(d1) < 20:
            return "FLAT"
        try:
            sma10 = d1["close"].tail(10).mean()
            sma20 = d1["close"].tail(20).mean()
            if sma10 > sma20 * 1.001:
                return "UP"
            elif sma10 < sma20 * 0.999:
                return "DOWN"
            return "FLAT"
        except Exception as e:
            logger.warning(f"[RegimeEngine] d1_trend computation failed: {e}")
            return "FLAT"

    @staticmethod
    def _classify_session(now: datetime) -> str:
        """Classify IST market session: OPEN | MID | CLOSE."""
        t = now.time() if hasattr(now, "time") else now
        if time(9, 15) <= t < time(9, 45):
            return "OPEN"
        elif t >= time(14, 45):
            return "CLOSE"
        return "MID"
