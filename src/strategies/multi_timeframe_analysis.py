"""
Multi-Timeframe Analysis Strategy
Combines 1h, 4h, and 1d timeframes for higher confidence signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class MultiTimeframeAnalysis:
    def __init__(self):
        self.name = "multi_timeframe_analysis"
        self.min_confidence_threshold = 30
        self.timeframes = ['1h', '4h', '1d']
        self.weights = {'1d': 0.5, '4h': 0.3, '1h': 0.2}  # Daily has highest weight
        
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            if len(df) < 20:
                return 0.0
                
            # Calculate EMAs
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean()
            
            # Calculate trend strength
            latest = df.iloc[-1]
            
            # EMA alignment
            ema_alignment = 0
            if latest['ema_20'] > latest['ema_50'] > latest['ema_200']:
                ema_alignment = 1  # Bullish
            elif latest['ema_20'] < latest['ema_50'] < latest['ema_200']:
                ema_alignment = -1  # Bearish
                
            # Price vs EMAs
            price_vs_ema = 0
            if latest['close'] > latest['ema_20']:
                price_vs_ema += 1
            if latest['close'] > latest['ema_50']:
                price_vs_ema += 1
            if latest['close'] > latest['ema_200']:
                price_vs_ema += 1
            elif latest['close'] < latest['ema_20']:
                price_vs_ema -= 1
            if latest['close'] < latest['ema_50']:
                price_vs_ema -= 1
            if latest['close'] < latest['ema_200']:
                price_vs_ema -= 1
                
            # Combine signals
            trend_strength = (ema_alignment * 0.6 + price_vs_ema * 0.4) / 3
            
            return max(-1, min(1, trend_strength))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
            
    def calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum using RSI and MACD"""
        try:
            if len(df) < 20:
                return 0.0
                
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            # Get latest values
            latest_rsi = rsi.iloc[-1]
            latest_macd = macd.iloc[-1]
            latest_signal = signal.iloc[-1]
            
            # Calculate momentum
            momentum = 0
            
            # RSI momentum
            if latest_rsi > 70:
                momentum += 1
            elif latest_rsi > 50:
                momentum += 0.5
            elif latest_rsi < 30:
                momentum -= 1
            elif latest_rsi < 50:
                momentum -= 0.5
                
            # MACD momentum
            if latest_macd > latest_signal:
                momentum += 0.5
            else:
                momentum -= 0.5
                
            return max(-1, min(1, momentum))
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
            
    def calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """Calculate volume confirmation"""
        try:
            if len(df) < 20:
                return 0.0
                
            # Calculate volume moving average
            df['volume_ma'] = df['volume'].rolling(20).mean()
            
            # Calculate volume ratio
            latest_volume = df['volume'].iloc[-1]
            latest_volume_ma = df['volume_ma'].iloc[-1]
            
            if latest_volume_ma > 0:
                volume_ratio = latest_volume / latest_volume_ma
                
                # Volume confirmation
                if volume_ratio > 1.5:
                    return 1.0
                elif volume_ratio > 1.2:
                    return 0.5
                elif volume_ratio > 0.8:
                    return 0.0
                else:
                    return -0.5
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volume confirmation: {e}")
            return 0.0
            
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """Analyze a single timeframe"""
        try:
            if len(df) < 50:
                return {
                    'trend_strength': 0.0,
                    'momentum': 0.0,
                    'volume_confirmation': 0.0,
                    'weight': self.weights.get(timeframe, 0.2)
                }
                
            trend_strength = self.calculate_trend_strength(df)
            momentum = self.calculate_momentum(df)
            volume_confirmation = self.calculate_volume_confirmation(df)
            
            return {
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volume_confirmation': volume_confirmation,
                'weight': self.weights.get(timeframe, 0.2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return {
                'trend_strength': 0.0,
                'momentum': 0.0,
                'volume_confirmation': 0.0,
                'weight': self.weights.get(timeframe, 0.2)
            }
            
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate multi-timeframe signal"""
        try:
            if len(df) < 50:
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0,
                    'strategy': self.name,
                    'reason': 'insufficient_data'
                }
            
            # Analyze current timeframe (1h)
            current_analysis = self.analyze_timeframe(df, '1h')
            
            # For now, we'll use the current timeframe analysis
            # In a real implementation, you'd fetch 4h and 1d data
            
            trend_strength = current_analysis['trend_strength']
            momentum = current_analysis['momentum']
            volume_confirmation = current_analysis['volume_confirmation']
            
            # Calculate combined signal
            combined_signal = (trend_strength * 0.5 + momentum * 0.3 + volume_confirmation * 0.2)
            
            # Generate signal
            if combined_signal > 0.3:
                confidence = min(100, 40 + abs(combined_signal) * 60)
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'confidence': confidence,
                    'strategy': self.name,
                    'reason': 'multi_timeframe_bullish'
                }
            elif combined_signal < -0.3:
                confidence = min(100, 40 + abs(combined_signal) * 60)
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'confidence': confidence,
                    'strategy': self.name,
                    'reason': 'multi_timeframe_bearish'
                }
            else:
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0,
                    'strategy': self.name,
                    'reason': 'no_clear_signal'
                }
                
        except Exception as e:
            logger.error(f"Error generating multi-timeframe signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0,
                'strategy': self.name,
                'reason': f'error: {str(e)}'
            }
