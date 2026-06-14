#!/usr/bin/env python3
"""
Technical Indicators Calculator - FIXED VERSION
Calculates all required indicators from raw OHLCV data with proper NaN handling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators with proper NaN handling."""
    df = data.copy()
    
    # Ensure we have enough data
    if len(df) < 50:
        print(f"⚠️ Insufficient data: {len(df)} candles (need at least 50)")
        return df
    
    # Calculate EMAs with proper initialization
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate RSI with proper handling
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.inf)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Fill initial NaN values with neutral RSI
    df['rsi'] = df['rsi'].fillna(50)
    
    # Calculate ATR with proper handling
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14, min_periods=1).mean()
    
    # Fill initial NaN values with a reasonable ATR estimate
    if df['atr'].isna().any():
        avg_price = df['close'].mean()
        df['atr'] = df['atr'].fillna(avg_price * 0.02)  # 2% of average price
    
    # Calculate MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Calculate Bollinger Bands with proper handling
    df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Fill NaN values in Bollinger Bands
    df['bb_upper'] = df['bb_upper'].fillna(df['close'])
    df['bb_lower'] = df['bb_lower'].fillna(df['close'])
    
    # Calculate SuperTrend (proper implementation with direction tracking)
    hl2 = (df['high'] + df['low']) / 2
    multiplier = 3
    basic_upper = hl2 + (df['atr'] * multiplier)
    basic_lower = hl2 - (df['atr'] * multiplier)

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=df.index, data=0.0, dtype=float)
    st_direction = pd.Series(index=df.index, data=1, dtype=int)  # 1=up, -1=down

    supertrend.iloc[0] = basic_upper.iloc[0]

    for i in range(1, len(df)):
        # Track upper / lower bands
        if df['close'].iloc[i - 1] <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(basic_upper.iloc[i], final_upper.iloc[i - 1])
        else:
            final_upper.iloc[i] = basic_upper.iloc[i]

        if df['close'].iloc[i - 1] >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(basic_lower.iloc[i], final_lower.iloc[i - 1])
        else:
            final_lower.iloc[i] = basic_lower.iloc[i]

        # Direction logic
        if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            if df['close'].iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                st_direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
                st_direction.iloc[i] = 1
        elif supertrend.iloc[i - 1] == final_lower.iloc[i - 1]:
            if df['close'].iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                st_direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
                st_direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i - 1]
            st_direction.iloc[i] = st_direction.iloc[i - 1]

    df['supertrend'] = supertrend
    df['supertrend_direction'] = st_direction  # 1 = bullish (price above ST), -1 = bearish
    
    # Calculate Stochastic
    low_14 = df['low'].rolling(14, min_periods=1).min()
    high_14 = df['high'].rolling(14, min_periods=1).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14).replace(0, np.inf))
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
    
    # Fill NaN values in Stochastic
    df['stoch_k'] = df['stoch_k'].fillna(50)
    df['stoch_d'] = df['stoch_d'].fillna(50)
    
    # Calculate Williams %R
    df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14).replace(0, np.inf))
    df['williams_r'] = df['williams_r'].fillna(-50)
    
    # Calculate CCI (Commodity Channel Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20, min_periods=1).mean()
    mad = typical_price.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())) if len(x) > 0 else 0)
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.inf))
    df['cci'] = df['cci'].fillna(0)
    
    # Calculate ADX (Average Directional Index)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift())
    tr3 = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    dm_plus = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), 
                       np.maximum(df['high'] - df['high'].shift(), 0), 0)
    dm_minus = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), 
                        np.maximum(df['low'].shift() - df['low'], 0), 0)
    
    # Calculate Choppiness Index (CHOP)
    # 100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
    chop_period = 14
    tr_sum = true_range.rolling(chop_period).sum()
    hh = df['high'].rolling(chop_period).max()
    ll = df['low'].rolling(chop_period).min()
    df['chop'] = 100 * np.log10(tr_sum / (hh - ll).replace(0, np.inf)) / np.log10(chop_period)
    df['chop'] = df['chop'].fillna(50)  # Neutral chop

    # Trend Intensity (Custom)
    df['trend_intensity'] = (df['close'] - df['ema_50']).abs() / df['atr'].replace(0, np.inf)
    
    # Calculate smoothed values
    period = 14
    tr_smooth = tr.rolling(period, min_periods=1).mean()
    dm_plus_smooth = pd.Series(dm_plus, index=df.index).rolling(period, min_periods=1).mean()
    dm_minus_smooth = pd.Series(dm_minus, index=df.index).rolling(period, min_periods=1).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / tr_smooth.replace(0, np.inf))
    di_minus = 100 * (dm_minus_smooth / tr_smooth.replace(0, np.inf))
    
    # Calculate DX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.inf)
    
    # Calculate ADX
    df['adx'] = dx.rolling(period, min_periods=1).mean()
    df['adx'] = df['adx'].fillna(25)  # Neutral ADX value
    
    return df

def validate_indicators(data: pd.DataFrame) -> bool:
    """Validate indicators with better NaN handling."""
    required_indicators = [
        'ema_9', 'ema_12', 'ema_21', 'ema_26', 'ema_50',
        'rsi', 'atr', 'macd', 'macd_signal', 'macd_histogram',
        'bb_middle', 'bb_upper', 'bb_lower', 'supertrend', 'supertrend_direction', 'adx'
    ]
    
    missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
    
    if missing_indicators:
        print(f"⚠️ Missing indicators: {missing_indicators}")
        return False
    
    # Check for NaN values in the last 20 rows (recent data)
    recent_data = data.tail(20)
    nan_counts = recent_data[required_indicators].isna().sum()
    
    if nan_counts.sum() > 0:
        print(f"⚠️ NaN values in recent data: {nan_counts[nan_counts > 0].to_dict()}")
        # Allow some NaN values in early periods, but not in recent data
        if nan_counts.sum() > len(required_indicators) * 5:  # Allow up to 5 NaN per indicator
            return False
    
    return True
