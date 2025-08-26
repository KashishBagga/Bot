"""
Technical indicators module.
Provides a consistent interface for all technical indicators used in the trading strategies.
"""
import math
import pandas as pd
import ta
import numpy as np
from typing import Union, Dict, Any
import logging

# Small epsilon to prevent division by zero
EPSILON = 1e-10

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
        """Calculate Relative Strength Index (RSI) with NaN guards."""
        try:
            rsi_series = ta.momentum.RSIIndicator(data[column], window=period).rsi()
            # Fill NaN values with 50 (neutral RSI)
            return rsi_series.fillna(50).apply(safe_float)
        except Exception as e:
            logging.error(f"RSI calculation error: {e}")
            return pd.Series(50, index=data.index)
    
    @staticmethod
    def macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9, column: str = 'close') -> Dict[str, pd.Series]:
        """Calculate Moving Average Convergence Divergence (MACD)."""
        try:
            macd_indicator = ta.trend.MACD(
                data[column], 
                window_fast=fast_period, 
                window_slow=slow_period, 
                window_sign=signal_period
            )
            return {
                'macd': macd_indicator.macd().fillna(0).apply(safe_float),
                'signal': macd_indicator.macd_signal().fillna(0).apply(safe_float),
                'histogram': macd_indicator.macd_diff().fillna(0).apply(safe_float)
            }
        except Exception as e:
            logging.error(f"MACD calculation error: {e}")
            return {
                'macd': pd.Series(0, index=data.index),
                'signal': pd.Series(0, index=data.index),
                'histogram': pd.Series(0, index=data.index)
            }
    
    @staticmethod
    def ema(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        try:
            ema_series = ta.trend.EMAIndicator(data[column], window=period).ema_indicator()
            return ema_series.fillna(method='bfill').fillna(data[column].iloc[0]).apply(safe_float)
        except Exception as e:
            logging.error(f"EMA calculation error: {e}")
            return pd.Series(data[column].iloc[0], index=data.index)
    
    @staticmethod
    def sma(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        try:
            sma_series = ta.trend.SMAIndicator(data[column], window=period).sma_indicator()
            return sma_series.fillna(method='bfill').fillna(data[column].iloc[0]).apply(safe_float)
        except Exception as e:
            logging.error(f"SMA calculation error: {e}")
            return pd.Series(data[column].iloc[0], index=data.index)
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) with NaN guards."""
        try:
            atr_series = ta.volatility.AverageTrueRange(
                data['high'], data['low'], data['close'], window=period
            ).average_true_range()
            return atr_series.fillna(method='bfill').fillna(EPSILON).apply(safe_float)
        except Exception as e:
            logging.error(f"ATR calculation error: {e}")
            return pd.Series(EPSILON, index=data.index)
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                         column: str = 'close') -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            indicator = ta.volatility.BollingerBands(
                data[column], window=period, window_dev=std_dev
            )
            return {
                'upper': indicator.bollinger_hband().fillna(method='bfill').fillna(data[column].iloc[0]).apply(safe_float),
                'middle': indicator.bollinger_mavg().fillna(method='bfill').fillna(data[column].iloc[0]).apply(safe_float),
                'lower': indicator.bollinger_lband().fillna(method='bfill').fillna(data[column].iloc[0]).apply(safe_float)
            }
        except Exception as e:
            logging.error(f"Bollinger Bands calculation error: {e}")
            return {
                'upper': pd.Series(data[column].iloc[0], index=data.index),
                'middle': pd.Series(data[column].iloc[0], index=data.index),
                'lower': pd.Series(data[column].iloc[0], index=data.index)
            }
    
    @staticmethod
    def supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """Calculate SuperTrend indicator using vectorized operations with NaN guards."""
        if len(data) < period + 1:
            # Not enough data for calculation
            return {
                'supertrend': pd.Series(0.0, index=data.index).apply(safe_float),
                'direction': pd.Series(0, index=data.index),
                'upper': pd.Series(0.0, index=data.index).apply(safe_float),
                'lower': pd.Series(0.0, index=data.index).apply(safe_float)
            }
        
        try:
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
        except Exception as e:
            logging.error(f"SuperTrend calculation error: {e}")
            return {
                'supertrend': pd.Series(0.0, index=data.index).apply(safe_float),
                'direction': pd.Series(0, index=data.index),
                'upper': pd.Series(0.0, index=data.index).apply(safe_float),
                'lower': pd.Series(0.0, index=data.index).apply(safe_float)
            }
    
    @staticmethod
    def donchian_channel(data: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Donchian Channel."""
        try:
            upper = data['high'].rolling(window=period).max().fillna(method='bfill').fillna(data['high'].iloc[0]).apply(safe_float)
            lower = data['low'].rolling(window=period).min().fillna(method='bfill').fillna(data['low'].iloc[0]).apply(safe_float)
            middle = ((upper + lower) / 2).apply(safe_float)
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
        except Exception as e:
            logging.error(f"Donchian Channel calculation error: {e}")
            return {
                'upper': pd.Series(data['high'].iloc[0], index=data.index),
                'middle': pd.Series(data['close'].iloc[0], index=data.index),
                'lower': pd.Series(data['low'].iloc[0], index=data.index)
            }

# Create a callable interface for easy access to indicators
indicators = Indicators()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe with proper NaN guards."""
    df = df.copy()
    
    try:
        # Add basic indicators
        df['rsi'] = indicators.rsi(df)
        df['ema_9'] = indicators.ema(df, period=9)
        df['ema_21'] = indicators.ema(df, period=21)
        df['atr'] = indicators.atr(df)
        
        # Add MACD
        macd_data = indicators.macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Add volume and price indicators with NaN guards
        df['volume_ratio'] = df['volume'].pct_change().fillna(0)
        
        # Price position with division by zero protection
        range_ = (df['high'] - df['low']).replace(0, np.nan)
        df['price_position'] = ((df['close'] - df['low']) / range_).clip(0, 1).fillna(0)
        
        # Candle size with division by zero protection
        den_close = df['close'].replace(0, np.nan)
        df['candle_size'] = (df['high'] - df['low']) / den_close
        df['candle_size'] = df['candle_size'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Body ratio with division by zero protection
        body_range = (df['high'] - df['low']).replace(0, np.nan)
        df['body_ratio'] = abs(df['close'] - df['open']) / body_range
        df['body_ratio'] = df['body_ratio'].fillna(0)
        
        # Add price momentum
        df['price_momentum'] = df['close'].pct_change(3).fillna(0)
        
        return df
        
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        return df 