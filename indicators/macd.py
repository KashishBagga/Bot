"""
MACD indicator implementation
"""

import pandas as pd
import numpy as np

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return {
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    }
