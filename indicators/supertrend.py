from collections import deque

class Supertrend:
    def __init__(self, period=7, multiplier=3):
        self.period = period
        self.multiplier = multiplier
        self.atr_values = deque(maxlen=period)
        self.prev_candle = None
        self.prev_supertrend = None

    def _calculate_true_range(self, candle, prev_candle):
        tr1 = abs(candle['high'] - candle['low'])
        tr2 = abs(candle['high'] - (prev_candle['close'] if prev_candle else candle['close']))
        tr3 = abs(candle['low'] - (prev_candle['close'] if prev_candle else candle['close']))
        return max(tr1, tr2, tr3)

    def _calculate_atr(self, tr):
        self.atr_values.append(tr)
        return sum(self.atr_values) / len(self.atr_values) if self.atr_values else tr

    def update(self, candle):
        """
        Update Supertrend with a new candle.

        Parameters:
            candle (dict): Must contain 'high', 'low', 'close'
        
        Returns:
            dict: {
                'value': Supertrend value,
                'direction': 1 (uptrend) or -1 (downtrend),
                'upperband': Final upper band,
                'lowerband': Final lower band
            }
        """
        tr = self._calculate_true_range(candle, self.prev_candle)
        atr = self._calculate_atr(tr)

        hl2 = (candle['high'] + candle['low']) / 2
        basic_upper = hl2 + (self.multiplier * atr)
        basic_lower = hl2 - (self.multiplier * atr)

        if self.prev_supertrend and self.prev_candle:
            final_upper = (
                min(basic_upper, self.prev_supertrend['upperband'])
                if self.prev_candle['close'] <= self.prev_supertrend['upperband']
                else basic_upper
            )
            final_lower = (
                max(basic_lower, self.prev_supertrend['lowerband'])
                if self.prev_candle['close'] >= self.prev_supertrend['lowerband']
                else basic_lower
            )
        else:
            final_upper = basic_upper
            final_lower = basic_lower

        # Trend decision
        if candle['close'] > final_upper:
            direction = 1
            supertrend = final_lower
        elif candle['close'] < final_lower:
            direction = -1
            supertrend = final_upper
        else:
            direction = self.prev_supertrend['direction'] if self.prev_supertrend else 1
            supertrend = self.prev_supertrend['value'] if self.prev_supertrend else final_lower

        result = {
            'value': supertrend,
            'direction': direction,
            'upperband': final_upper,
            'lowerband': final_lower
        }

        self.prev_candle = candle
        self.prev_supertrend = result
        return result

def calculate_supertrend_live(candle, prev_candle=None, prev_supertrend=None, multiplier=3):
    """
    Live-compatible Supertrend calculation using previous state.

    Parameters:
        candle (dict): Current candle with keys 'high', 'low', 'close', 'atr'
        prev_candle (dict or None): Previous candle (used to track trend)
        prev_supertrend (dict or None): Previous supertrend state
        multiplier (float): ATR multiplier, default is 3

    Returns:
        dict: {
            'value': supertrend value (line),
            'direction': +1 for uptrend, -1 for downtrend,
            'upperband': final upper band,
            'lowerband': final lower band
        }
    """
    high = candle['high']
    low = candle['low']
    close = candle['close']
    atr = candle['atr']

    hl2 = (high + low) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)

    if prev_candle and prev_supertrend:
        prev_close = prev_candle['close']
        prev_upperband = prev_supertrend['upperband']
        prev_lowerband = prev_supertrend['lowerband']

        final_upperband = min(basic_upperband, prev_upperband) if prev_close <= prev_upperband else basic_upperband
        final_lowerband = max(basic_lowerband, prev_lowerband) if prev_close >= prev_lowerband else basic_lowerband

        # Trend direction logic
        if close > final_upperband:
            direction = 1
            supertrend = final_lowerband
        elif close < final_lowerband:
            direction = -1
            supertrend = final_upperband
        else:
            direction = prev_supertrend['direction']
            supertrend = prev_supertrend['value']
    else:
        # First candle: initialize direction based on position
        final_upperband = basic_upperband
        final_lowerband = basic_lowerband
        direction = 1 if close > hl2 else -1
        supertrend = final_lowerband if direction == 1 else final_upperband

    return {
        'value': supertrend,
        'direction': direction,
        'upperband': final_upperband,
        'lowerband': final_lowerband
    }
