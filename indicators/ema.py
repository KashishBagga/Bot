"""
EMA indicator implementation
"""

import pandas as pd
import numpy as np

def calculate_ema(data, period):
    """Calculate EMA."""
    return data.ewm(span=period).mean()
