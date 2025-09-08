#!/usr/bin/env python3
"""
Market Condition Analysis
Identifies trending vs ranging markets and optimal trading conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class MarketConditionAnalyzer:
    """Analyzes market conditions to determine optimal trading environments"""
    
    def __init__(self):
        self.min_trend_duration = 3  # Minimum candles for trend confirmation
        self.adx_threshold = 25      # ADX threshold for trend strength
        self.volatility_range = (0.3, 2.0)  # Optimal ATR range
        self.volume_threshold = 0.8  # Minimum volume ratio
        
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        try:
            if len(df) < period + 1:
                return pd.Series([0] * len(df), index=df.index)
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            dm_plus = high - high.shift(1)
            dm_minus = low.shift(1) - low
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # Smooth the values
            tr_smooth = tr.rolling(period).mean()
            dm_plus_smooth = dm_plus.rolling(period).mean()
            dm_minus_smooth = dm_minus.rolling(period).mean()
            
            # Calculate DI+ and DI-
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(period).mean()
            
            return adx.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple indicators"""
        try:
            # Calculate EMA slope
            ema_21 = df['ema_21']
            ema_slope = (ema_21 - ema_21.shift(5)) / ema_21.shift(5) * 100
            
            # Calculate price momentum
            price_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
            
            # Calculate RSI trend
            rsi = df['rsi']
            rsi_trend = rsi - rsi.shift(5)
            
            # Combine indicators for trend strength
            trend_strength = (
                (ema_slope.abs() * 0.4) +
                (price_momentum.abs() * 0.3) +
                (rsi_trend.abs() * 0.3)
            )
            
            return trend_strength.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def identify_market_condition(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Identify current market condition"""
        try:
            if current_index < 20:  # Need enough data
                return {
                    'condition': 'INSUFFICIENT_DATA',
                    'trend_direction': 0,
                    'trend_strength': 0,
                    'volatility': 'LOW',
                    'volume_condition': 'LOW',
                    'tradeable': False,
                    'reason': 'insufficient data for analysis'
                }
            
            # Get recent data
            recent_data = df.iloc[max(0, current_index-20):current_index+1]
            
            # Calculate indicators
            adx = self.calculate_adx(recent_data).iloc[-1]
            trend_strength = self.calculate_trend_strength(recent_data).iloc[-1]
            
            # Current candle data
            current_candle = df.iloc[current_index]
            prev_candle = df.iloc[current_index-1] if current_index > 0 else current_candle
            
            # Trend direction
            ema_9 = current_candle['ema_9']
            ema_21 = current_candle['ema_21']
            trend_direction = 1 if ema_9 > ema_21 else -1
            
            # Volatility analysis
            atr = current_candle['atr']
            avg_atr = df['atr'].rolling(20).mean().iloc[current_index]
            volatility_ratio = atr / avg_atr if avg_atr > 0 else 1
            
            if volatility_ratio < 0.5:
                volatility_condition = 'LOW'
            elif volatility_ratio > 2.0:
                volatility_condition = 'HIGH'
            else:
                volatility_condition = 'OPTIMAL'
            
            # Volume analysis
            volume_ratio = current_candle['volume_ratio']
            if volume_ratio < self.volume_threshold:
                volume_condition = 'LOW'
            elif volume_ratio > 2.0:
                volume_condition = 'HIGH'
            else:
                volume_condition = 'OPTIMAL'
            
            # Market condition classification
            if adx > self.adx_threshold and trend_strength > 10:
                if trend_direction == 1:
                    condition = 'STRONG_UPTREND'
                else:
                    condition = 'STRONG_DOWNTREND'
            elif adx > 20 and trend_strength > 5:
                if trend_direction == 1:
                    condition = 'WEAK_UPTREND'
                else:
                    condition = 'WEAK_DOWNTREND'
            else:
                condition = 'RANGING'
            
            # Determine if market is tradeable
            tradeable = (
                adx > 20 and  # Some trend strength
                trend_strength > 5 and  # Minimum trend strength
                volatility_condition == 'OPTIMAL' and  # Good volatility
                volume_condition in ['OPTIMAL', 'HIGH']  # Sufficient volume
            )
            
            reason = f"ADX: {adx:.1f}, Trend Strength: {trend_strength:.1f}, Vol: {volatility_condition}, VolRatio: {volume_condition}"
            
            return {
                'condition': condition,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'adx': adx,
                'volatility': volatility_condition,
                'volume_condition': volume_condition,
                'tradeable': tradeable,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market condition: {e}")
            return {
                'condition': 'ERROR',
                'trend_direction': 0,
                'trend_strength': 0,
                'volatility': 'UNKNOWN',
                'volume_condition': 'UNKNOWN',
                'tradeable': False,
                'reason': f'analysis error: {str(e)}'
            }
    
    def get_session_filter(self, timestamp) -> Tuple[bool, str]:
        """Check if current time is optimal for trading"""
        try:
            if hasattr(timestamp, 'time'):
                current_time = timestamp.time()
            else:
                # If timestamp is string, parse it
                from datetime import datetime
                if isinstance(timestamp, str):
                    current_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').time()
                else:
                    return True, "timestamp format unknown"
            
            # Market hours: 9:15 AM to 3:30 PM
            market_start = datetime.strptime('09:15:00', '%H:%M:%S').time()
            market_end = datetime.strptime('15:30:00', '%H:%M:%S').time()
            
            # Avoid first 15 minutes and last 15 minutes
            avoid_start = datetime.strptime('09:30:00', '%H:%M:%S').time()
            avoid_end = datetime.strptime('15:15:00', '%H:%M:%S').time()
            
            # Avoid lunch hour (12:00-13:00)
            lunch_start = datetime.strptime('12:00:00', '%H:%M:%S').time()
            lunch_end = datetime.strptime('13:00:00', '%H:%M:%S').time()
            
            if current_time < market_start or current_time > market_end:
                return False, "outside market hours"
            elif current_time < avoid_start or current_time > avoid_end:
                return False, "avoiding market open/close periods"
            elif lunch_start <= current_time <= lunch_end:
                return False, "avoiding lunch hour"
            else:
                return True, "optimal trading time"
                
        except Exception as e:
            logger.error(f"Error in session filter: {e}")
            return True, "session filter error"

def analyze_market_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Add market condition analysis to the dataframe"""
    try:
        analyzer = MarketConditionAnalyzer()
        
        # Add basic market condition analysis without ADX for now
        conditions = []
        for i in range(len(df)):
            # Simplified market condition check
            if i < 20:
                condition_dict = {
                    'condition': 'INSUFFICIENT_DATA',
                    'trend_direction': 0,
                    'trend_strength': 0,
                    'adx': 0,
                    'tradeable': False,
                    'reason': 'insufficient data'
                }
            else:
                # Basic trend direction
                ema_9 = df['ema_9'].iloc[i]
                ema_21 = df['ema_21'].iloc[i]
                trend_direction = 1 if ema_9 > ema_21 else -1
                
                # Basic trend strength - MUCH LESS RESTRICTIVE
                price_momentum = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5] * 100 if i >= 5 else 0
                trend_strength = abs(price_momentum)
                
                # Simple market condition - LESS RESTRICTIVE
                if trend_strength > 1:  # Much lower threshold
                    if trend_direction == 1:
                        condition = 'WEAK_UPTREND'
                    else:
                        condition = 'WEAK_DOWNTREND'
                else:
                    condition = 'RANGING'
                
                # Proper volume validation
                volume_ok = True
                if volume is not None and len(volume) >= 20:
                    avg_volume = np.mean(volume[-20:])
                    current_volume = volume[-1]
                    volume_ok = current_volume >= avg_volume * 0.5  # At least 50% of average
                
                # Proper volatility validation
                volatility_ok = True
                if len(close_prices) >= 20:
                    returns = np.diff(close_prices[-20:]) / close_prices[-21:-1]
                    volatility = np.std(returns)
                    volatility_ok = 0.001 <= volatility <= 0.1  # Reasonable volatility range
                
                # Proper tradeable condition
                tradeable = volume_ok and volatility_ok and trend_strength > 1.0                
                condition_dict = {
                    'condition': condition,
                    'trend_direction': trend_direction,
                    'trend_strength': trend_strength,
                    'adx': 20,  # Default ADX value
                    'tradeable': tradeable,
                    'reason': f"trend strength: {trend_strength:.1f}, volume: {volume_ok}, volatility: {volatility_ok}"
                }
            
            conditions.append(condition_dict)
        
        # Convert to DataFrame columns
        df['market_condition'] = [c['condition'] for c in conditions]
        df['trend_direction'] = [c['trend_direction'] for c in conditions]
        df['trend_strength'] = [c['trend_strength'] for c in conditions]
        df['adx'] = [c['adx'] for c in conditions]
        df['market_tradeable'] = [c['tradeable'] for c in conditions]
        df['market_reason'] = [c['reason'] for c in conditions]
        
        return df
        
    except Exception as e:
        logger.error(f"Error in market condition analysis: {e}")
        # Return original dataframe if analysis fails
        return df 