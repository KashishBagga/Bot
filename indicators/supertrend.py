"""
Supertrend indicator implementation
"""

import pandas as pd
import numpy as np

def get_supertrend_instance(timeframe="5min", period=10, multiplier=3.0):
    """Get supertrend indicator instance."""
    return SupertrendIndicator(period, multiplier)

class SupertrendIndicator:
    """Supertrend indicator implementation."""
    
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
    
    def calculate(self, data):
        """Calculate supertrend indicator."""
        if len(data) < self.period:
            return None
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        # Calculate basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)
        
        # Convert to numpy for fast looping
        close_arr = close.values
        ub_arr = upper_band.values
        lb_arr = lower_band.values
        
        n = len(data)
        final_ub = np.zeros(n)
        final_lb = np.zeros(n)
        
        final_ub[0] = ub_arr[0]
        final_lb[0] = lb_arr[0]
        
        for i in range(1, n):
            if ub_arr[i] < final_ub[i-1] or close_arr[i-1] > final_ub[i-1]:
                final_ub[i] = ub_arr[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if lb_arr[i] > final_lb[i-1] or close_arr[i-1] < final_lb[i-1]:
                final_lb[i] = lb_arr[i]
            else:
                final_lb[i] = final_lb[i-1]
        
        st_arr = np.zeros(n)
        dir_arr = np.zeros(n, dtype=int)
        
        st_arr[0] = final_lb[0]
        dir_arr[0] = 1
        
        for i in range(1, n):
            if st_arr[i-1] == final_ub[i-1] and close_arr[i] <= final_ub[i]:
                st_arr[i] = final_ub[i]
                dir_arr[i] = -1
            elif st_arr[i-1] == final_ub[i-1] and close_arr[i] > final_ub[i]:
                st_arr[i] = final_lb[i]
                dir_arr[i] = 1
            elif st_arr[i-1] == final_lb[i-1] and close_arr[i] >= final_lb[i]:
                st_arr[i] = final_lb[i]
                dir_arr[i] = 1
            elif st_arr[i-1] == final_lb[i-1] and close_arr[i] < final_lb[i]:
                st_arr[i] = final_ub[i]
                dir_arr[i] = -1
            else:
                st_arr[i] = st_arr[i-1]
                dir_arr[i] = dir_arr[i-1]
        
        return {
            'supertrend': pd.Series(st_arr, index=data.index),
            'direction': pd.Series(dir_arr, index=data.index),
            'upper_band': pd.Series(final_ub, index=data.index),
            'lower_band': pd.Series(final_lb, index=data.index)
        }

    def update(self, data):
        """Update method for compatibility with strategies."""
        # Handle single candle (pandas Series) by converting to DataFrame
        if hasattr(data, 'name') and hasattr(data, 'index'):
            # It's a pandas Series (single candle)
            df = pd.DataFrame([data])
            result = self.calculate(df)
            if result is not None:
                # Return the last values from the calculation
                return (result['supertrend'].iloc[-1], result['direction'].iloc[-1])
            return None
        else:
            # It's already a DataFrame
            return self.calculate(data)
