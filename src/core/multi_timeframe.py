"""
Multi-timeframe data handling module.
Provides functionality to fetch and process data from multiple timeframes.
"""
import pandas as pd
from typing import Dict, Optional

def resample_to_higher_timeframe(df: pd.DataFrame, target_resolution: str) -> pd.DataFrame:
    """Resample data to a higher timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        target_resolution: Target resolution (e.g., '15min', '1H')
        
    Returns:
        pd.DataFrame: Resampled data
    """
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df = df.set_index('time')
        else:
            raise ValueError("DataFrame must have a datetime index or 'time' column")
    
    # Define aggregation rules
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Add any technical indicators that should be resampled
    if 'ema_20' in df.columns:
        agg_dict['ema_20'] = 'last'
    if 'atr' in df.columns:
        agg_dict['atr'] = 'last'
    if 'rsi' in df.columns:
        agg_dict['rsi'] = 'last'
    
    # Resample and aggregate
    resampled = df.resample(target_resolution).agg(agg_dict)
    
    # Forward fill any missing values
    resampled = resampled.fillna(method='ffill')
    
    return resampled

def prepare_multi_timeframe_data(df: pd.DataFrame, base_resolution: str = '3min') -> Dict[str, pd.DataFrame]:
    """Prepare data for multiple timeframes.
    
    Args:
        df: Base timeframe DataFrame
        base_resolution: Base timeframe resolution (default: '3min')
        
    Returns:
        dict: Dictionary with data for each timeframe
    """
    # Ensure base DataFrame has DatetimeIndex
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    timeframes = {
        base_resolution: df,  # Base timeframe (e.g., 3min)
        '15min': resample_to_higher_timeframe(df, '15min'),  # Higher timeframe for confirmation
        '30min': resample_to_higher_timeframe(df, '30min'),   # Additional higher timeframe
        '1H': resample_to_higher_timeframe(df, '1H')  # Additional higher timeframe if needed
    }
    # Ensure all resampled DataFrames have DatetimeIndex
    for tf, tf_df in timeframes.items():
        if 'time' in tf_df.columns:
            tf_df['time'] = pd.to_datetime(tf_df['time'])
            tf_df = tf_df.set_index('time')
            timeframes[tf] = tf_df
    return timeframes

def get_higher_timeframe_data(df: pd.DataFrame, current_index: int, timeframe_data: Dict[str, pd.DataFrame], 
                            target_resolution: str = '15min') -> Optional[pd.DataFrame]:
    """Get higher timeframe data up to the current point.
    
    Args:
        df: Base timeframe DataFrame
        current_index: Current index in base timeframe
        timeframe_data: Dictionary with data for each timeframe
        target_resolution: Target higher timeframe resolution
        
    Returns:
        pd.DataFrame: Higher timeframe data up to current point, or None if not available
    """
    if target_resolution not in timeframe_data:
        return None
    
    # Get current timestamp
    current_time = df.index[current_index]
    
    # Get higher timeframe data up to current time
    higher_tf_data = timeframe_data[target_resolution]
    higher_tf_data = higher_tf_data[higher_tf_data.index <= current_time]
    
    if higher_tf_data.empty:
        return None
        
    return higher_tf_data 