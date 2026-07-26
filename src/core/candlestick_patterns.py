#!/usr/bin/env python3
"""
Candlestick Pattern Detector
============================
Single/multi-candle Japanese candlestick patterns — distinct from the
multi-swing chart patterns in pattern_engine.py (Double Top, H&S, etc.).

Other strategies (geometry_strategy.py, order_flow_strategy.py, ...) already
use a crude "big real body, not a doji" proxy (_is_bullish_reversal_body /
_is_bearish_reversal_body) as a stand-in for a real reversal-candle check.
This module names the actual patterns instead — doji, engulfing, hammer,
shooting star, morning/evening star, marubozu, piercing line, dark cloud
cover, three white soldiers / three black crows — so a strategy can react to
a *specific* pattern rather than just "body is big".

Reversal-context patterns (hammer vs. hanging man, inverted hammer vs.
shooting star) are shape-identical — what makes them bullish or bearish is
whether they appear after a downtrend or an uptrend. `prior_trend()` is a
lightweight local-lookback proxy for that context, consistent with this
codebase's style of simple structural heuristics rather than a full trend
classifier (which already exists upstream as daily_bias / market_regime for
strategies that want it).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd


class CandleDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class CandlestickSignal:
    """One detected candlestick pattern at the current candle."""
    name: str
    direction: CandleDirection
    strength: float  # 0.0-1.0, relative confidence in the pattern's textbook shape


# ── Single-candle shape helpers ────────────────────────────────────────────

def _parts(candle) -> tuple:
    o, h, l, c = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
    rng = h - l
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return o, h, l, c, rng, body, upper_wick, lower_wick


def prior_trend(df: pd.DataFrame, idx: int, lookback: int = 5) -> CandleDirection:
    """Local trend context for the candle(s) leading into `idx` — a lightweight
    proxy for 'was this a downtrend (so a hammer here is a bottom) or an
    uptrend (so the same shape is a hanging man)'."""
    start = max(0, idx - lookback)
    if start >= idx:
        return CandleDirection.NEUTRAL
    lead_close = float(df["close"].iloc[start])
    curr_close = float(df["close"].iloc[idx - 1]) if idx > 0 else lead_close
    if curr_close > lead_close * 1.001:
        return CandleDirection.BULLISH
    if curr_close < lead_close * 0.999:
        return CandleDirection.BEARISH
    return CandleDirection.NEUTRAL


# ── Single-candle patterns ─────────────────────────────────────────────────

def _is_doji(candle, doji_body_frac: float = 0.10) -> Optional[CandlestickSignal]:
    o, h, l, c, rng, body, uw, lw = _parts(candle)
    if rng < 1e-9:
        return None
    if (body / rng) <= doji_body_frac:
        return CandlestickSignal("DOJI", CandleDirection.NEUTRAL, round(1.0 - (body / rng), 2))
    return None


def _is_hammer_shape(candle, min_lower_wick_ratio: float = 2.0, max_upper_wick_frac: float = 0.15) -> Optional[float]:
    """Shape check shared by Hammer and Hanging Man: small body near the top
    of the range, long lower wick, negligible upper wick. Returns strength or None."""
    o, h, l, c, rng, body, uw, lw = _parts(candle)
    if rng < 1e-9 or body < 1e-9:
        return None
    if lw >= min_lower_wick_ratio * body and uw <= max_upper_wick_frac * rng:
        return round(min(1.0, lw / (rng + 1e-9) + 0.3), 2)
    return None


def _is_inverted_hammer_shape(candle, min_upper_wick_ratio: float = 2.0, max_lower_wick_frac: float = 0.15) -> Optional[float]:
    """Shape check shared by Inverted Hammer and Shooting Star: small body near
    the bottom of the range, long upper wick, negligible lower wick."""
    o, h, l, c, rng, body, uw, lw = _parts(candle)
    if rng < 1e-9 or body < 1e-9:
        return None
    if uw >= min_upper_wick_ratio * body and lw <= max_lower_wick_frac * rng:
        return round(min(1.0, uw / (rng + 1e-9) + 0.3), 2)
    return None


def _is_marubozu(candle, max_wick_frac: float = 0.05) -> Optional[CandlestickSignal]:
    o, h, l, c, rng, body, uw, lw = _parts(candle)
    if rng < 1e-9:
        return None
    if uw <= max_wick_frac * rng and lw <= max_wick_frac * rng and (body / rng) >= 0.90:
        direction = CandleDirection.BULLISH if c > o else CandleDirection.BEARISH
        name = "BULLISH_MARUBOZU" if direction == CandleDirection.BULLISH else "BEARISH_MARUBOZU"
        return CandlestickSignal(name, direction, round(body / rng, 2))
    return None


def _is_spinning_top(candle, max_body_frac: float = 0.30, min_wick_frac: float = 0.25) -> Optional[CandlestickSignal]:
    o, h, l, c, rng, body, uw, lw = _parts(candle)
    if rng < 1e-9:
        return None
    if (body / rng) <= max_body_frac and (uw / rng) >= min_wick_frac and (lw / rng) >= min_wick_frac:
        return CandlestickSignal("SPINNING_TOP", CandleDirection.NEUTRAL, round(1.0 - (body / rng), 2))
    return None


# ── Multi-candle patterns ──────────────────────────────────────────────────

def _is_bullish_engulfing(prev, curr) -> Optional[CandlestickSignal]:
    po, ph, pl, pc, prng, pbody, _, _ = _parts(prev)
    o, h, l, c, rng, body, _, _ = _parts(curr)
    if pc >= po:  # prior candle must be bearish
        return None
    if c > o and c >= po and o <= pc and body > pbody:
        strength = round(min(1.0, body / (pbody + 1e-9) - 1.0), 2)
        return CandlestickSignal("BULLISH_ENGULFING", CandleDirection.BULLISH, max(0.3, min(1.0, strength)))
    return None


def _is_bearish_engulfing(prev, curr) -> Optional[CandlestickSignal]:
    po, ph, pl, pc, prng, pbody, _, _ = _parts(prev)
    o, h, l, c, rng, body, _, _ = _parts(curr)
    if pc <= po:  # prior candle must be bullish
        return None
    if c < o and o >= pc and c <= po and body > pbody:
        strength = round(min(1.0, body / (pbody + 1e-9) - 1.0), 2)
        return CandlestickSignal("BEARISH_ENGULFING", CandleDirection.BEARISH, max(0.3, min(1.0, strength)))
    return None


def _is_piercing_line(prev, curr) -> Optional[CandlestickSignal]:
    po, ph, pl, pc, prng, pbody, _, _ = _parts(prev)
    o, h, l, c, rng, body, _, _ = _parts(curr)
    if pc >= po or prng < 1e-9:
        return None  # prior candle must be a real bearish body
    midpoint = pc + (po - pc) / 2.0
    if o < pc and c > midpoint and c < po:
        return CandlestickSignal("PIERCING_LINE", CandleDirection.BULLISH, round((c - midpoint) / (po - midpoint + 1e-9), 2))
    return None


def _is_dark_cloud_cover(prev, curr) -> Optional[CandlestickSignal]:
    po, ph, pl, pc, prng, pbody, _, _ = _parts(prev)
    o, h, l, c, rng, body, _, _ = _parts(curr)
    if pc <= po or prng < 1e-9:
        return None  # prior candle must be a real bullish body
    midpoint = po + (pc - po) / 2.0
    if o > pc and c < midpoint and c > po:
        return CandlestickSignal("DARK_CLOUD_COVER", CandleDirection.BEARISH, round((midpoint - c) / (midpoint - po + 1e-9), 2))
    return None


def _is_morning_star(c1, c2, c3) -> Optional[CandlestickSignal]:
    o1, h1, l1, cl1, rng1, body1, _, _ = _parts(c1)
    o2, h2, l2, cl2, rng2, body2, _, _ = _parts(c2)
    o3, h3, l3, cl3, rng3, body3, _, _ = _parts(c3)
    if cl1 >= o1 or rng1 < 1e-9:
        return None  # candle 1: real bearish body
    if body2 > 0.5 * body1:
        return None  # candle 2: small body (indecision) relative to candle 1
    if cl3 <= o3:
        return None  # candle 3: bullish body
    if cl3 <= (o1 + cl1) / 2.0:
        return None  # closes above the midpoint of candle 1's body
    return CandlestickSignal("MORNING_STAR", CandleDirection.BULLISH, round(min(1.0, body3 / (body1 + 1e-9)), 2))


def _is_evening_star(c1, c2, c3) -> Optional[CandlestickSignal]:
    o1, h1, l1, cl1, rng1, body1, _, _ = _parts(c1)
    o2, h2, l2, cl2, rng2, body2, _, _ = _parts(c2)
    o3, h3, l3, cl3, rng3, body3, _, _ = _parts(c3)
    if cl1 <= o1 or rng1 < 1e-9:
        return None  # candle 1: real bullish body
    if body2 > 0.5 * body1:
        return None
    if cl3 >= o3:
        return None  # candle 3: bearish body
    if cl3 >= (o1 + cl1) / 2.0:
        return None
    return CandlestickSignal("EVENING_STAR", CandleDirection.BEARISH, round(min(1.0, body3 / (body1 + 1e-9)), 2))


def _is_three_white_soldiers(c1, c2, c3) -> Optional[CandlestickSignal]:
    candles = [c1, c2, c3]
    parts = [_parts(c) for c in candles]
    if not all(c > o for o, h, l, c, rng, body, uw, lw in parts):
        return None
    if not (parts[1][3] > parts[0][3] and parts[2][3] > parts[1][3]):
        return None  # each closes higher than the last
    if not all((body / rng) >= 0.5 for o, h, l, c, rng, body, uw, lw in parts if rng > 1e-9):
        return None  # each has a real body, not just a wick-heavy doji-like candle
    return CandlestickSignal("THREE_WHITE_SOLDIERS", CandleDirection.BULLISH, 0.8)


def _is_three_black_crows(c1, c2, c3) -> Optional[CandlestickSignal]:
    candles = [c1, c2, c3]
    parts = [_parts(c) for c in candles]
    if not all(c < o for o, h, l, c, rng, body, uw, lw in parts):
        return None
    if not (parts[1][3] < parts[0][3] and parts[2][3] < parts[1][3]):
        return None
    if not all((body / rng) >= 0.5 for o, h, l, c, rng, body, uw, lw in parts if rng > 1e-9):
        return None
    return CandlestickSignal("THREE_BLACK_CROWS", CandleDirection.BEARISH, 0.8)


# ── Public API ──────────────────────────────────────────────────────────────

def detect(df: pd.DataFrame, idx: int = -1) -> List[CandlestickSignal]:
    """All candlestick patterns matching at `idx` (default: last candle).

    Returns every match — a candle can be both a Doji and, in context, part
    of a Morning/Evening Star. Trend-context patterns (hammer family) are
    resolved using `prior_trend()` so the same shape yields Hammer at the
    bottom of a downtrend vs. Hanging Man at the top of an uptrend.
    """
    n = len(df)
    pos = idx if idx >= 0 else n + idx
    if pos < 0 or pos >= n:
        return []

    curr = df.iloc[pos]
    signals: List[CandlestickSignal] = []

    doji = _is_doji(curr)
    if doji:
        signals.append(doji)

    marubozu = _is_marubozu(curr)
    if marubozu:
        signals.append(marubozu)

    spinning = _is_spinning_top(curr)
    if spinning:
        signals.append(spinning)

    trend = prior_trend(df, pos)
    hammer_strength = _is_hammer_shape(curr)
    if hammer_strength is not None:
        if trend == CandleDirection.BEARISH:
            signals.append(CandlestickSignal("HAMMER", CandleDirection.BULLISH, hammer_strength))
        elif trend == CandleDirection.BULLISH:
            signals.append(CandlestickSignal("HANGING_MAN", CandleDirection.BEARISH, hammer_strength))

    inv_hammer_strength = _is_inverted_hammer_shape(curr)
    if inv_hammer_strength is not None:
        if trend == CandleDirection.BEARISH:
            signals.append(CandlestickSignal("INVERTED_HAMMER", CandleDirection.BULLISH, inv_hammer_strength))
        elif trend == CandleDirection.BULLISH:
            signals.append(CandlestickSignal("SHOOTING_STAR", CandleDirection.BEARISH, inv_hammer_strength))

    if pos >= 1:
        prev = df.iloc[pos - 1]
        for fn in (_is_bullish_engulfing, _is_bearish_engulfing, _is_piercing_line, _is_dark_cloud_cover):
            sig = fn(prev, curr)
            if sig:
                signals.append(sig)

    if pos >= 2:
        c1, c2 = df.iloc[pos - 2], df.iloc[pos - 1]
        for fn in (_is_morning_star, _is_evening_star, _is_three_white_soldiers, _is_three_black_crows):
            sig = fn(c1, c2, curr)
            if sig:
                signals.append(sig)

    return signals


def strongest_signal(signals: List[CandlestickSignal], direction: Optional[CandleDirection] = None) -> Optional[CandlestickSignal]:
    """Highest-strength signal, optionally filtered to one direction."""
    candidates = [s for s in signals if direction is None or s.direction == direction]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.strength)
