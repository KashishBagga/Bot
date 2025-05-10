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
        return round(f, 2) if not (math.isnan(f) or math.isinf(f)) else 0.0
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
        """Calculate SuperTrend indicator."""
        atr = Indicators.atr(data, period)
        
        hl2 = (data['high'] + data['low']) / 2
        
        # Calculate basic upper and lower bands
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        # Initialize final bands and supertrend series
        final_upper = pd.Series(0.0, index=data.index)
        final_lower = pd.Series(0.0, index=data.index)
        supertrend = pd.Series(0.0, index=data.index)
        direction = pd.Series(1, index=data.index)  # 1 for uptrend, -1 for downtrend
        
        # Calculate SuperTrend
        for i in range(1, len(data)):
            # Calculate final upper band
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or data['close'].iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
                
            # Calculate final lower band
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or data['close'].iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
                
            # Determine trend direction
            if final_upper.iloc[i-1] == supertrend.iloc[i-1] and data['close'].iloc[i] <= final_upper.iloc[i]:
                # Remains in downtrend
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            elif final_upper.iloc[i-1] == supertrend.iloc[i-1] and data['close'].iloc[i] > final_upper.iloc[i]:
                # Changes to uptrend
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            elif final_lower.iloc[i-1] == supertrend.iloc[i-1] and data['close'].iloc[i] >= final_lower.iloc[i]:
                # Remains in uptrend
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            elif final_lower.iloc[i-1] == supertrend.iloc[i-1] and data['close'].iloc[i] < final_lower.iloc[i]:
                # Changes to downtrend
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
        
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