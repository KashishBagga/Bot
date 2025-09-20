#!/usr/bin/env python3
"""
Optimized Technical Indicators with Vectorized Operations
Eliminates redundant calculations and improves performance
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import numba as nb
from numba import jit

@jit(nopython=True)
def _calculate_rsi_numba(prices, period=14):
    """Numba-optimized RSI calculation."""
    n = len(prices)
    rsi = np.full(n, 50.0)  # Default to 50
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate remaining RSI values
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@jit(nopython=True)
def _calculate_ema_numba(prices, period):
    """Numba-optimized EMA calculation."""
    n = len(prices)
    ema = np.full(n, np.nan)
    
    if n == 0:
        return ema
    
    # Calculate smoothing factor
    alpha = 2.0 / (period + 1.0)
    
    # Initialize with first price
    ema[0] = prices[0]
    
    # Calculate EMA
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    
    return ema

@jit(nopython=True)
def _calculate_atr_numba(high, low, close, period=14):
    """Numba-optimized ATR calculation."""
    n = len(high)
    atr = np.full(n, np.nan)
    
    if n < 2:
        return atr
    
    # Calculate true range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr

class OptimizedIndicators:
    """Optimized technical indicators with vectorized operations."""
    
    @staticmethod
    def calculate_all_indicators_optimized(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators with optimized operations."""
        df = data.copy()
        
        if len(df) < 50:
            return df
        
        # Convert to numpy arrays for numba operations
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        # Calculate EMAs using numba
        df['ema_9'] = _calculate_ema_numba(close_prices, 9)
        df['ema_12'] = _calculate_ema_numba(close_prices, 12)
        df['ema_21'] = _calculate_ema_numba(close_prices, 21)
        df['ema_26'] = _calculate_ema_numba(close_prices, 26)
        df['ema_50'] = _calculate_ema_numba(close_prices, 50)
        
        # Calculate RSI using numba
        df['rsi'] = _calculate_rsi_numba(close_prices, 14)
        
        # Calculate ATR using numba
        df['atr'] = _calculate_atr_numba(high_prices, low_prices, close_prices, 14)
        
        # Calculate MACD using pandas (already optimized)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands using pandas
        df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
        bb_std = df['close'].rolling(20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Fill NaN values
        df['bb_upper'] = df['bb_upper'].fillna(df['close'])
        df['bb_lower'] = df['bb_lower'].fillna(df['close'])
        
        # Calculate SuperTrend
        hl2 = (df['high'] + df['low']) / 2
        df['supertrend'] = hl2 + (df['atr'] * 3)
        
        # Calculate Stochastic
        low_14 = df['low'].rolling(14, min_periods=1).min()
        high_14 = df['high'].rolling(14, min_periods=1).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14).replace(0, np.inf))
        df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
        
        # Fill NaN values
        df['stoch_k'] = df['stoch_k'].fillna(50)
        df['stoch_d'] = df['stoch_d'].fillna(50)
        
        # Calculate Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14).replace(0, np.inf))
        df['williams_r'] = df['williams_r'].fillna(-50)
        
        # Calculate CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20, min_periods=1).mean()
        mad = typical_price.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())) if len(x) > 0 else 0)
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.inf))
        df['cci'] = df['cci'].fillna(0)
        
        return df
    
    @staticmethod
    def validate_indicators_optimized(data: pd.DataFrame) -> bool:
        """Validate indicators with optimized checks."""
        required_indicators = [
            'ema_9', 'ema_12', 'ema_21', 'ema_26', 'ema_50',
            'rsi', 'atr', 'macd', 'macd_signal', 'macd_histogram',
            'bb_middle', 'bb_upper', 'bb_lower', 'supertrend'
        ]
        
        # Check if all required indicators exist
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        if missing_indicators:
            return False
        
        # Check for NaN values in recent data (last 10 rows)
        recent_data = data.tail(10)
        nan_counts = recent_data[required_indicators].isna().sum()
        
        # Allow some NaN values but not too many
        return nan_counts.sum() <= len(required_indicators) * 2
