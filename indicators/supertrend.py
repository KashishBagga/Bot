import pandas as pd
from collections import deque

class Supertrend:
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.atr_values = deque(maxlen=period)
        self.prev_atr = None
        self.prev_supertrend = None
        self.prev_direction = None

    def calculate_atr(self, high, low, close, prev_close):
        tr = max(
            high - low,
            abs(high - prev_close) if prev_close is not None else 0,
            abs(low - prev_close) if prev_close is not None else 0
        )

        if self.prev_atr is None and len(self.atr_values) < self.period:
            self.atr_values.append(tr)
            if len(self.atr_values) == self.period:
                self.prev_atr = sum(self.atr_values) / self.period
            return None

        if self.prev_atr is None:
            self.prev_atr = sum(self.atr_values) / self.period

        # Wilder's smoothing
        atr = (self.prev_atr * (self.period - 1) + tr) / self.period
        self.prev_atr = atr
        return atr

    def update(self, candle: dict, prev_candle: dict = None):
        high, low, close = candle['high'], candle['low'], candle['close']
        prev_close = prev_candle['close'] if prev_candle else None

        atr = self.calculate_atr(high, low, close, prev_close)
        if atr is None:
            return None, None

        hl2 = (high + low) / 2
        upperband = hl2 + (self.multiplier * atr)
        lowerband = hl2 - (self.multiplier * atr)

        if self.prev_supertrend is None:
            supertrend = upperband
            direction = 1  # default uptrend
        else:
            if self.prev_supertrend == upperband and close <= upperband:
                supertrend = upperband
                direction = -1
            elif self.prev_supertrend == upperband and close > upperband:
                supertrend = lowerband
                direction = 1
            elif self.prev_supertrend == lowerband and close >= lowerband:
                supertrend = lowerband
                direction = 1
            else:
                supertrend = upperband
                direction = -1

        self.prev_supertrend = supertrend
        self.prev_direction = direction
        return supertrend, direction
