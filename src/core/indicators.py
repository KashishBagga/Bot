"""
Technical indicators module.
Provides a consistent interface for all technical indicators used in the trading strategies.
"""
import math
import pandas as pd
import ta
from typing import Union, Dict, Any

def safe_float(val):
    """Safely convert value to float with rounding and edge case handling."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return round(f, 2)
    except (ValueError, TypeError):
        return 0.0

class Indicators:
    """Technical indicators for trading strategies."""
    
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        return ta.momentum.RSIIndicator(data[column], window=period).rsi().apply(safe_float)
    
    @staticmethod
    def macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9, column: str = 'close') -> Dict[str, pd.Series]:
        """Calculate Moving Average Convergence Divergence (MACD)."""
        macd_indicator = ta.trend.MACD(
            data[column], 
            window_fast=fast_period, 
            window_slow=slow_period, 
            window_sign=signal_period
        )
        return {
            'macd': macd_indicator.macd().apply(safe_float),
            'signal': macd_indicator.macd_signal().apply(safe_float),
            'histogram': macd_indicator.macd_diff().apply(safe_float)
        }
    
    @staticmethod
    def ema(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        return ta.trend.EMAIndicator(data[column], window=period).ema_indicator().apply(safe_float)
    
    @staticmethod
    def sma(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        return ta.trend.SMAIndicator(data[column], window=period).sma_indicator().apply(safe_float)
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        return ta.volatility.AverageTrueRange(
            data['high'], data['low'], data['close'], window=period
        ).average_true_range().apply(safe_float)
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                         column: str = 'close') -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        indicator = ta.volatility.BollingerBands(
            data[column], window=period, window_dev=std_dev
        )
        return {
            'upper': indicator.bollinger_hband().apply(safe_float),
            'middle': indicator.bollinger_mavg().apply(safe_float),
            'lower': indicator.bollinger_lband().apply(safe_float)
        }
    
    @staticmethod
    def supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """Calculate SuperTrend indicator using vectorized operations."""
        if len(data) < period + 1:
            # Not enough data for calculation
            return {
                'supertrend': pd.Series(0.0, index=data.index).apply(safe_float),
                'direction': pd.Series(0, index=data.index),
                'upper': pd.Series(0.0, index=data.index).apply(safe_float),
                'lower': pd.Series(0.0, index=data.index).apply(safe_float)
            }
        
        atr = Indicators.atr(data, period)
        hl2 = (data['high'] + data['low']) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        supertrend = pd.Series(index=data.index, data=0.0)  # Initialize with 0.0 instead of False
        direction = pd.Series(1, index=data.index)  # 1 for uptrend, -1 for downtrend

        # Initialize first value
        supertrend.iloc[0] = basic_upper.iloc[0]
        
        for i in range(1, len(data)):
            # Bounds check to prevent index errors
            if i >= len(data) or i-1 < 0:
                continue
                
            final_upper.iloc[i] = min(basic_upper.iloc[i], final_upper.iloc[i-1]) if data['close'].iloc[i-1] <= final_upper.iloc[i-1] else basic_upper.iloc[i]
            final_lower.iloc[i] = max(basic_lower.iloc[i], final_lower.iloc[i-1]) if data['close'].iloc[i-1] >= final_lower.iloc[i-1] else basic_lower.iloc[i]

            if supertrend.iloc[i-1] == final_upper.iloc[i-1] and data['close'].iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and data['close'].iloc[i] > final_upper.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and data['close'].iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and data['close'].iloc[i] < final_lower.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                # Default case - maintain previous values
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]

        return {
            'supertrend': supertrend.apply(safe_float),
            'direction': direction,
            'upper': final_upper.apply(safe_float),
            'lower': final_lower.apply(safe_float)
        }
    
    @staticmethod
    def donchian_channel(data: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Donchian Channel."""
        upper = data['high'].rolling(window=period).max().apply(safe_float)
        lower = data['low'].rolling(window=period).min().apply(safe_float)
        middle = ((upper + lower) / 2).apply(safe_float)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

# Create a callable interface for easy access to indicators
indicators = Indicators() 