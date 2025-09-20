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
        
        # Calculate final bands
        final_upper_band = pd.Series(index=data.index, dtype=float)
        final_lower_band = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            if i == 0:
                final_upper_band.iloc[i] = upper_band.iloc[i]
                final_lower_band.iloc[i] = lower_band.iloc[i]
            else:
                if upper_band.iloc[i] < final_upper_band.iloc[i-1] or close.iloc[i-1] > final_upper_band.iloc[i-1]:
                    final_upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
                
                if lower_band.iloc[i] > final_lower_band.iloc[i-1] or close.iloc[i-1] < final_lower_band.iloc[i-1]:
                    final_lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
        
        # Calculate supertrend
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)
        
        for i in range(len(data)):
            if i == 0:
                supertrend.iloc[i] = final_lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if supertrend.iloc[i-1] == final_upper_band.iloc[i-1] and close.iloc[i] <= final_upper_band.iloc[i]:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
                    direction.iloc[i] = -1
                elif supertrend.iloc[i-1] == final_upper_band.iloc[i-1] and close.iloc[i] > final_upper_band.iloc[i]:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                    direction.iloc[i] = 1
                elif supertrend.iloc[i-1] == final_lower_band.iloc[i-1] and close.iloc[i] >= final_lower_band.iloc[i]:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                    direction.iloc[i] = 1
                elif supertrend.iloc[i-1] == final_lower_band.iloc[i-1] and close.iloc[i] < final_lower_band.iloc[i]:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'upper_band': final_upper_band,
            'lower_band': final_lower_band
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
