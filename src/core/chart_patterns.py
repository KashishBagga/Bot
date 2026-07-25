#!/usr/bin/env python3
"""
Chart Pattern Detection Library
===============================
Pure, dependency-light detectors for the classic price-action patterns, plus a
volatility-regime detector. Every detector:

  * takes an OHLCV DataFrame (+ an ATR scale) and returns Optional[PatternSignal]
  * NEVER raises — returns None on insufficient/degenerate data
  * is stateless and side-effect free (safe to call from any strategy or the
    shared MarketViewEngine)

A ``PatternSignal`` is the common currency of the confluence model: it carries a
direction, a 0..1 confidence, and the trade-relevant levels (trigger, target,
stop hint). Strategies can either trade a pattern directly OR read the aggregated
MarketView (see market_view.py) to boost/dampen their own confidence.

Directions:
  BULLISH  — favours BUY CALL
  BEARISH  — favours BUY PUT
  VOLATILE — expansion imminent / non-directional (favours straddle/strangle)
  NEUTRAL  — no actionable signal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BULLISH = "BULLISH"
BEARISH = "BEARISH"
VOLATILE = "VOLATILE"
NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class PatternSignal:
    name: str            # e.g. "DOUBLE_BOTTOM", "HEAD_SHOULDERS", "TRIANGLE_ASC"
    direction: str       # BULLISH | BEARISH | VOLATILE | NEUTRAL
    confidence: float    # 0.0 .. 1.0
    rationale: str       # human-readable trigger description (for docs/dashboard)
    key_level: float = 0.0   # the trigger level (neckline / breakout edge)
    target: float = 0.0      # measured-move projection (0 if n/a)
    stop_hint: float = 0.0    # structural invalidation level (0 if n/a)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "key_level": round(self.key_level, 2),
            "target": round(self.target, 2),
            "stop_hint": round(self.stop_hint, 2),
        }


# ────────────────────────────────────────────────────────────────────────────
# Swing / fractal helpers
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Swing:
    pos: int      # integer index into the DataFrame
    price: float
    kind: str     # "H" (swing high) | "L" (swing low)


def find_swings(df: pd.DataFrame, window: int = 3) -> List[_Swing]:
    """Fractal swing points: a swing high has ``window`` lower highs on each side
    (mirror for lows). Confirmed swings only — the last ``window`` bars can never
    be a swing (no look-ahead)."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    out: List[_Swing] = []
    if n < 2 * window + 1:
        return out
    for i in range(window, n - window):
        hi_seg = highs[i - window:i + window + 1]
        lo_seg = lows[i - window:i + window + 1]
        if highs[i] == hi_seg.max() and (hi_seg == highs[i]).sum() == 1:
            out.append(_Swing(i, float(highs[i]), "H"))
        elif lows[i] == lo_seg.min() and (lo_seg == lows[i]).sum() == 1:
            out.append(_Swing(i, float(lows[i]), "L"))
    return out


def _last_close(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1])


def _safe_atr(df: pd.DataFrame, atr: float) -> float:
    if atr and atr > 0:
        return atr
    rng = float((df["high"] - df["low"]).tail(14).mean())
    return rng if rng > 0 else max(1.0, _last_close(df) * 0.001)


# ────────────────────────────────────────────────────────────────────────────
# Reversal patterns
# ────────────────────────────────────────────────────────────────────────────

def detect_double_top_bottom(df: pd.DataFrame, atr: float, tol_mult: float = 0.6) -> Optional[PatternSignal]:
    """Double Top (bearish) / Double Bottom (bullish).

    Two swing extremes within ``tol_mult*ATR`` of each other, separated by a
    counter-swing, with price now breaking the intervening level (neckline).
    """
    atr = _safe_atr(df, atr)
    swings = find_swings(df, window=3)
    if len(swings) < 3:
        return None
    close = _last_close(df)
    tol = tol_mult * atr

    highs = [s for s in swings if s.kind == "H"]
    lows = [s for s in swings if s.kind == "L"]

    # Double Top: last two swing highs ~equal, a swing low between them = neckline
    if len(highs) >= 2 and len(lows) >= 1:
        h2, h1 = highs[-1], highs[-2]
        if abs(h1.price - h2.price) <= tol:
            mids = [l for l in lows if h1.pos < l.pos < h2.pos]
            if mids:
                neckline = min(m.price for m in mids)
                if close < neckline:  # confirmed break of the neckline
                    peak = max(h1.price, h2.price)
                    sym = 1.0 - min(1.0, abs(h1.price - h2.price) / tol)
                    depth = (peak - neckline) / atr
                    conf = float(np.clip(0.45 + 0.25 * sym + 0.1 * min(depth, 3.0), 0.0, 0.9))
                    return PatternSignal(
                        "DOUBLE_TOP", BEARISH, conf,
                        f"Two equal highs @~{peak:.0f} rejected; neckline {neckline:.0f} broken down",
                        key_level=neckline, target=neckline - (peak - neckline), stop_hint=peak,
                    )

    # Double Bottom: last two swing lows ~equal, a swing high between them = neckline
    if len(lows) >= 2 and len(highs) >= 1:
        l2, l1 = lows[-1], lows[-2]
        if abs(l1.price - l2.price) <= tol:
            mids = [h for h in highs if l1.pos < h.pos < l2.pos]
            if mids:
                neckline = max(m.price for m in mids)
                if close > neckline:
                    trough = min(l1.price, l2.price)
                    sym = 1.0 - min(1.0, abs(l1.price - l2.price) / tol)
                    depth = (neckline - trough) / atr
                    conf = float(np.clip(0.45 + 0.25 * sym + 0.1 * min(depth, 3.0), 0.0, 0.9))
                    return PatternSignal(
                        "DOUBLE_BOTTOM", BULLISH, conf,
                        f"Two equal lows @~{trough:.0f} held; neckline {neckline:.0f} broken up",
                        key_level=neckline, target=neckline + (neckline - trough), stop_hint=trough,
                    )
    return None


def detect_head_shoulders(df: pd.DataFrame, atr: float) -> Optional[PatternSignal]:
    """Head & Shoulders (bearish) / Inverse H&S (bullish).

    Three consecutive swing highs (lows) where the middle is the most extreme and
    the two shoulders are roughly symmetric; confirmed on a close through the
    neckline drawn across the intervening opposite swings.
    """
    atr = _safe_atr(df, atr)
    swings = find_swings(df, window=3)
    close = _last_close(df)

    highs = [s for s in swings if s.kind == "H"]
    lows = [s for s in swings if s.kind == "L"]

    # Standard H&S (bearish): LS < Head > RS, shoulders similar height
    if len(highs) >= 3 and len(lows) >= 2:
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        if head.price > ls.price and head.price > rs.price:
            shoulder_diff = abs(ls.price - rs.price)
            if shoulder_diff <= 0.8 * atr:
                necks = [l.price for l in lows if ls.pos < l.pos < rs.pos]
                if necks:
                    neckline = float(np.mean(necks))
                    if close < neckline:
                        sym = 1.0 - min(1.0, shoulder_diff / (0.8 * atr))
                        prom = (head.price - max(ls.price, rs.price)) / atr
                        conf = float(np.clip(0.5 + 0.2 * sym + 0.1 * min(prom, 3.0), 0.0, 0.92))
                        return PatternSignal(
                            "HEAD_SHOULDERS", BEARISH, conf,
                            f"H&S: head {head.price:.0f} > shoulders {ls.price:.0f}/{rs.price:.0f}; neckline {neckline:.0f} broken",
                            key_level=neckline, target=neckline - (head.price - neckline), stop_hint=head.price,
                        )

    # Inverse H&S (bullish)
    if len(lows) >= 3 and len(highs) >= 2:
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        if head.price < ls.price and head.price < rs.price:
            shoulder_diff = abs(ls.price - rs.price)
            if shoulder_diff <= 0.8 * atr:
                necks = [h.price for h in highs if ls.pos < h.pos < rs.pos]
                if necks:
                    neckline = float(np.mean(necks))
                    if close > neckline:
                        sym = 1.0 - min(1.0, shoulder_diff / (0.8 * atr))
                        prom = (min(ls.price, rs.price) - head.price) / atr
                        conf = float(np.clip(0.5 + 0.2 * sym + 0.1 * min(prom, 3.0), 0.0, 0.92))
                        return PatternSignal(
                            "INVERSE_HEAD_SHOULDERS", BULLISH, conf,
                            f"iH&S: head {head.price:.0f} < shoulders {ls.price:.0f}/{rs.price:.0f}; neckline {neckline:.0f} broken",
                            key_level=neckline, target=neckline + (neckline - head.price), stop_hint=head.price,
                        )
    return None


# ────────────────────────────────────────────────────────────────────────────
# Continuation patterns
# ────────────────────────────────────────────────────────────────────────────

def detect_triangle(df: pd.DataFrame, atr: float, lookback: int = 40) -> Optional[PatternSignal]:
    """Ascending / descending / symmetric triangle, confirmed on breakout.

    Fits trendlines to recent swing highs and lows; a converging range that price
    then closes out of is the trigger. Ascending (flat top rising lows) → bullish,
    descending → bearish, symmetric → breaks in the direction of the close.
    """
    atr = _safe_atr(df, atr)
    seg = df.tail(lookback)
    swings = find_swings(seg.reset_index(drop=True), window=2)
    highs = [s for s in swings if s.kind == "H"]
    lows = [s for s in swings if s.kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    close = _last_close(df)
    hi_x = np.array([s.pos for s in highs]); hi_y = np.array([s.price for s in highs])
    lo_x = np.array([s.pos for s in lows]); lo_y = np.array([s.price for s in lows])
    hi_slope = float(np.polyfit(hi_x, hi_y, 1)[0])
    lo_slope = float(np.polyfit(lo_x, lo_y, 1)[0])

    top = float(hi_y.max())
    bottom = float(lo_y.min())
    height = top - bottom
    if height <= 0:
        return None

    # Convergence: opposite-signed (or near-flat) slopes narrowing the range
    converging = (hi_slope <= 0.02 * atr) and (lo_slope >= -0.02 * atr) and not (hi_slope > 0 and lo_slope < 0)
    flat = 0.05 * atr

    name = direction = None
    if abs(hi_slope) < flat and lo_slope > flat:      # flat top, rising lows
        name, direction, level = "TRIANGLE_ASC", BULLISH, top
    elif abs(lo_slope) < flat and hi_slope < -flat:    # flat bottom, falling highs
        name, direction, level = "TRIANGLE_DESC", BEARISH, bottom
    elif hi_slope < -flat and lo_slope > flat:         # symmetric convergence
        name = "TRIANGLE_SYM"
        direction = BULLISH if close > top else (BEARISH if close < bottom else NEUTRAL)
        level = top if direction == BULLISH else bottom
    else:
        return None

    if direction == BULLISH and close > level:
        conf = float(np.clip(0.4 + 0.15 * min((close - level) / atr, 3.0), 0.0, 0.85))
        return PatternSignal(name, BULLISH, conf,
                             f"{name}: upside break of {level:.0f}", key_level=level,
                             target=level + height, stop_hint=bottom)
    if direction == BEARISH and close < level:
        conf = float(np.clip(0.4 + 0.15 * min((level - close) / atr, 3.0), 0.0, 0.85))
        return PatternSignal(name, BEARISH, conf,
                             f"{name}: downside break of {level:.0f}", key_level=level,
                             target=level - height, stop_hint=top)
    return None


def detect_flag(df: pd.DataFrame, atr: float, impulse_bars: int = 6, flag_bars: int = 6) -> Optional[PatternSignal]:
    """Bull/Bear flag: a strong impulse leg followed by a shallow counter-trend
    consolidation, resolving in the impulse direction."""
    atr = _safe_atr(df, atr)
    need = impulse_bars + flag_bars + 1
    if len(df) < need:
        return None
    closes = df["close"].values
    impulse_start = closes[-(flag_bars + impulse_bars) - 1]
    impulse_end = closes[-flag_bars - 1]
    impulse = impulse_end - impulse_start
    if abs(impulse) < 1.5 * atr:
        return None  # impulse not strong enough

    flag = df.tail(flag_bars)
    flag_range = float(flag["high"].max() - flag["low"].min())
    flag_drift = float(flag["close"].iloc[-1] - flag["close"].iloc[0])
    close = _last_close(df)

    # Consolidation must be tight relative to the impulse and drift mildly against it
    if flag_range > abs(impulse) * 0.75:
        return None
    if impulse > 0 and flag_drift <= 0 and close > float(flag["high"].iloc[:-1].max()):
        conf = float(np.clip(0.45 + 0.15 * min(abs(impulse) / atr, 4.0) / 4.0, 0.0, 0.82))
        return PatternSignal("BULL_FLAG", BULLISH, conf,
                             f"Bull flag: +{impulse/atr:.1f} ATR impulse, tight pullback, breakout",
                             key_level=float(flag["high"].max()),
                             target=close + abs(impulse), stop_hint=float(flag["low"].min()))
    if impulse < 0 and flag_drift >= 0 and close < float(flag["low"].iloc[:-1].min()):
        conf = float(np.clip(0.45 + 0.15 * min(abs(impulse) / atr, 4.0) / 4.0, 0.0, 0.82))
        return PatternSignal("BEAR_FLAG", BEARISH, conf,
                             f"Bear flag: -{abs(impulse)/atr:.1f} ATR impulse, tight pullback, breakdown",
                             key_level=float(flag["low"].min()),
                             target=close - abs(impulse), stop_hint=float(flag["high"].max()))
    return None


# ────────────────────────────────────────────────────────────────────────────
# Mean-reversion / momentum
# ────────────────────────────────────────────────────────────────────────────

def detect_rsi_divergence(df: pd.DataFrame, atr: float, rsi: Optional[pd.Series] = None,
                          lookback: int = 30) -> Optional[PatternSignal]:
    """Regular RSI divergence.

    Bullish: price makes a lower low but RSI makes a higher low (selling
    exhaustion). Bearish: price higher high, RSI lower high.
    """
    atr = _safe_atr(df, atr)
    if rsi is None:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
    seg = df.tail(lookback).reset_index(drop=True)
    rsi_seg = rsi.tail(lookback).reset_index(drop=True)
    swings = find_swings(seg, window=2)
    lows = [s for s in swings if s.kind == "L"]
    highs = [s for s in swings if s.kind == "H"]

    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        r1, r2 = float(rsi_seg.iloc[l1.pos]), float(rsi_seg.iloc[l2.pos])
        if l2.price < l1.price and r2 > r1 and r2 < 45:
            conf = float(np.clip(0.45 + 0.2 * min((r2 - r1) / 15.0, 1.0), 0.0, 0.85))
            return PatternSignal("RSI_BULL_DIVERGENCE", BULLISH, conf,
                                 f"Price LL but RSI HL ({r1:.0f}→{r2:.0f}) — downside exhaustion",
                                 key_level=l2.price, stop_hint=l2.price - atr)
    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        r1, r2 = float(rsi_seg.iloc[h1.pos]), float(rsi_seg.iloc[h2.pos])
        if h2.price > h1.price and r2 < r1 and r2 > 55:
            conf = float(np.clip(0.45 + 0.2 * min((r1 - r2) / 15.0, 1.0), 0.0, 0.85))
            return PatternSignal("RSI_BEAR_DIVERGENCE", BEARISH, conf,
                                 f"Price HH but RSI LH ({r1:.0f}→{r2:.0f}) — upside exhaustion",
                                 key_level=h2.price, stop_hint=h2.price + atr)
    return None


# ────────────────────────────────────────────────────────────────────────────
# Volatility regime
# ────────────────────────────────────────────────────────────────────────────

def detect_squeeze(df: pd.DataFrame, atr: float, length: int = 20) -> Optional[PatternSignal]:
    """Bollinger-in-Keltner squeeze → volatility compression that precedes an
    expansion. Non-directional (VOLATILE): the setup that favours a straddle /
    strangle. If price is already breaking out of the squeeze, a directional hint
    is included in the rationale.
    """
    if len(df) < length + 2:
        return None
    atr = _safe_atr(df, atr)
    close = df["close"]
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    bb_upper = ma + 2 * sd
    bb_lower = ma - 2 * sd
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift(1)).abs(),
        (df["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_k = tr.rolling(length).mean()
    kc_upper = ma + 1.5 * atr_k
    kc_lower = ma - 1.5 * atr_k

    if pd.isna(bb_upper.iloc[-1]) or pd.isna(kc_upper.iloc[-1]):
        return None
    squeeze_on = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
    if not squeeze_on:
        return None

    # Tightness → confidence: how compressed BB width is vs its own recent history
    bb_width = (bb_upper - bb_lower)
    width_now = float(bb_width.iloc[-1])
    width_ref = float(bb_width.tail(length * 3).median())
    tight = 1.0 - min(1.0, width_now / width_ref) if width_ref > 0 else 0.5
    conf = float(np.clip(0.4 + 0.4 * tight, 0.0, 0.85))
    return PatternSignal("VOL_SQUEEZE", VOLATILE, conf,
                         "Bollinger inside Keltner — volatility compressed; expansion imminent",
                         key_level=float(ma.iloc[-1]))


# ────────────────────────────────────────────────────────────────────────────
# Aggregate runner
# ────────────────────────────────────────────────────────────────────────────

def detect_all(df: pd.DataFrame, atr: float, rsi: Optional[pd.Series] = None) -> List[PatternSignal]:
    """Run every detector and return all non-None PatternSignals. Never raises."""
    detectors = [
        lambda: detect_head_shoulders(df, atr),
        lambda: detect_double_top_bottom(df, atr),
        lambda: detect_triangle(df, atr),
        lambda: detect_flag(df, atr),
        lambda: detect_rsi_divergence(df, atr, rsi),
        lambda: detect_squeeze(df, atr),
    ]
    out: List[PatternSignal] = []
    for d in detectors:
        try:
            sig = d()
            if sig is not None:
                out.append(sig)
        except Exception as e:  # a detector bug must never break the pipeline
            logger.warning(f"[chart_patterns] detector error: {e}")
    return out
