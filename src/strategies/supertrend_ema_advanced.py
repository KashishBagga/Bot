"""
Advanced SuperTrend EMA Strategy - Fixed and Enhanced
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SuperTrendEmaAdvanced:
    def __init__(self):
        self.name = "supertrend_ema_advanced"
        self.min_confidence_threshold = 25  # Reduced from 60
        self.adx_threshold = 10  # Reduced from 20
        self.atr_filter = 0.02  # Reduced from 0.5
        self.rsi_bullish = 35  # Reduced from 45
        self.rsi_bearish = 65  # Increased from 55
        self.volume_filter = 0.5  # Reduced from 1.0
        self.momentum_bullish = -0.2  # More lenient
        self.momentum_bearish = 0.2  # More lenient
        
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate SuperTrend indicator with improved logic"""
        try:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(period).mean()
            
            # Calculate basic bands
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Calculate SuperTrend
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            for i in range(len(df)):
                if i == 0:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    # Current lower basic band
                    if lower_band.iloc[i] > supertrend.iloc[i-1] or df['close'].iloc[i-1] > supertrend.iloc[i-1]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                    else:
                        supertrend.iloc[i] = supertrend.iloc[i-1]
                    
                    # Current upper basic band
                    if upper_band.iloc[i] < supertrend.iloc[i-1] or df['close'].iloc[i-1] < supertrend.iloc[i-1]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                    else:
                        supertrend.iloc[i] = supertrend.iloc[i-1]
                    
                    # Direction
                    if supertrend.iloc[i] == lower_band.iloc[i]:
                        direction.iloc[i] = 1
                    else:
                        direction.iloc[i] = -1
            
            df['supertrend'] = supertrend
            df['supertrend_direction'] = direction
            df['atr'] = atr
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return df
            
    def calculate_ema(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate EMA with improved logic"""
        try:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            return df
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return df
            
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI with improved logic"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return df
            
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX with improved logic"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = minus_dm.abs()
            
            atr = true_range.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(period).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return df
            
    def calculate_volume_ratio(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate volume ratio with improved logic"""
        try:
            df['volume_ma'] = df['volume'].rolling(period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            return df
        except Exception as e:
            logger.error(f"Error calculating volume ratio: {e}")
            return df
            
    def calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate momentum with improved logic"""
        try:
            df['momentum'] = df['close'].pct_change(period)
            return df
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return df
            
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate trading signal with improved logic"""
        try:
            if len(df) < 50:
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0,
                    'strategy': self.name,
                    'reason': 'insufficient_data'
                }
            
            # Calculate indicators
            df = self.calculate_supertrend(df)
            df = self.calculate_ema(df)
            df = self.calculate_rsi(df)
            df = self.calculate_adx(df)
            df = self.calculate_volume_ratio(df)
            df = self.calculate_momentum(df)
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check for valid data
            if pd.isna(latest['supertrend']) or pd.isna(latest['ema_20']) or pd.isna(latest['rsi']):
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0,
                    'strategy': self.name,
                    'reason': 'invalid_indicator_data'
                }
            
            # Initialize confidence
            confidence = 0
            
            # SuperTrend analysis
            supertrend_bullish = latest['close'] > latest['supertrend']
            supertrend_bearish = latest['close'] < latest['supertrend']
            
            # EMA analysis
            ema_bullish = latest['close'] > latest['ema_20']
            ema_bearish = latest['close'] < latest['ema_20']
            
            # RSI analysis
            rsi_bullish = latest['rsi'] > self.rsi_bullish
            rsi_bearish = latest['rsi'] < self.rsi_bearish
            
            # ADX analysis
            adx_strong = latest['adx'] > self.adx_threshold if not pd.isna(latest['adx']) else True
            
            # Volume analysis
            volume_ok = latest['volume_ratio'] > self.volume_filter if not pd.isna(latest['volume_ratio']) else True
            
            # Momentum analysis
            momentum_bullish = latest['momentum'] > self.momentum_bullish if not pd.isna(latest['momentum']) else True
            momentum_bearish = latest['momentum'] < self.momentum_bearish if not pd.isna(latest['momentum']) else True
            
            # ATR analysis
            atr_ok = latest['atr'] > self.atr_filter if not pd.isna(latest['atr']) else True
            
            # Generate signals
            if supertrend_bullish and ema_bullish and rsi_bullish and adx_strong and volume_ok and momentum_bullish and atr_ok:
                confidence = 50
                confidence += 20 if rsi_bullish else 0
                confidence += 15 if adx_strong else 0
                confidence += 10 if volume_ok else 0
                confidence += 10 if momentum_bullish else 0
                confidence += 5 if atr_ok else 0
                
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'confidence': min(confidence, 100),
                    'strategy': self.name,
                    'reason': 'supertrend_ema_bullish_alignment'
                }
                
            elif supertrend_bearish and ema_bearish and rsi_bearish and adx_strong and volume_ok and momentum_bearish and atr_ok:
                confidence = 50
                confidence += 20 if rsi_bearish else 0
                confidence += 15 if adx_strong else 0
                confidence += 10 if volume_ok else 0
                confidence += 10 if momentum_bearish else 0
                confidence += 5 if atr_ok else 0
                
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'confidence': min(confidence, 100),
                    'strategy': self.name,
                    'reason': 'supertrend_ema_bearish_alignment'
                }
            
            # Partial signals
            elif supertrend_bullish and ema_bullish:
                confidence = 30
                confidence += 10 if rsi_bullish else 0
                confidence += 5 if volume_ok else 0
                
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'confidence': min(confidence, 100),
                    'strategy': self.name,
                    'reason': 'partial_bullish_signal'
                }
                
            elif supertrend_bearish and ema_bearish:
                confidence = 30
                confidence += 10 if rsi_bearish else 0
                confidence += 5 if volume_ok else 0
                
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'confidence': min(confidence, 100),
                    'strategy': self.name,
                    'reason': 'partial_bearish_signal'
                }
            
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0,
                'strategy': self.name,
                'reason': 'no_signal_conditions_met'
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0,
                'strategy': self.name,
                'reason': f'error: {str(e)}'
            }
