"""
FIXED SuperTrend EMA Strategy - Addresses the confidence 0 issue
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy

logger = logging.getLogger(__name__)

class SupertrendEmaFixed(Strategy):
    """
    FIXED Multi-timeframe Supertrend + EMA strategy with proper signal generation.
    """

    # Minimum candles required for analysis
    min_candles = 60

    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        params = params or {}
        self.ema_period = params.get("ema_period", 21)  # More standard period
        self.supertrend_period = params.get("supertrend_period", 10)
        self.supertrend_multiplier = params.get("supertrend_multiplier", 3.0)
        self.timeframe_data = timeframe_data or {}
        self.min_confidence_threshold = params.get("min_confidence_threshold", 25)  # Lower threshold
        self.atr_period = params.get("atr_period", 14)
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 15)
        self.volume_threshold = params.get("volume_threshold", 0.8)  # More lenient
        super().__init__("supertrend_ema", params)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators."""
        df = df.copy()
        
        # Basic EMA
        df["ema"] = df["close"].ewm(span=self.ema_period).mean()
        df["price_to_ema_ratio"] = (df["close"] / df["ema"] - 1) * 100
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(self.atr_period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean().fillna(0)
        df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1)
        
        # EMA slope for momentum
        df['ema_slope'] = df['ema'].diff(5).fillna(0)
        
        # Simple Supertrend calculation
        df = self._add_supertrend(df)
        
        return df

    def _add_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Supertrend indicator."""
        # Calculate basic upper and lower bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (self.supertrend_multiplier * df['atr'])
        lower_band = hl2 - (self.supertrend_multiplier * df['atr'])
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                # Previous values
                prev_supertrend = supertrend.iloc[i-1]
                prev_direction = direction.iloc[i-1]
                
                # Current values
                current_upper = upper_band.iloc[i]
                current_lower = lower_band.iloc[i]
                current_close = df['close'].iloc[i]
                
                # Calculate new Supertrend
                if prev_direction == 1:  # Previous was bullish
                    if current_close <= current_lower:
                        supertrend.iloc[i] = current_lower
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = min(current_upper, prev_supertrend)
                        direction.iloc[i] = 1
                else:  # Previous was bearish
                    if current_close >= current_upper:
                        supertrend.iloc[i] = current_upper
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = max(current_lower, prev_supertrend)
                        direction.iloc[i] = -1
        
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        
        return df

    def analyze_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED vectorized analysis with proper signal generation.
        """
        # Ensure we have enough data
        if len(df) < self.min_candles:
            return pd.DataFrame()
        
        # Add indicators
        df = self.add_indicators(df)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        # Basic signal detection
        price_above_supertrend = df['close'] > df['supertrend']
        price_above_ema = df['close'] > df['ema']
        price_below_supertrend = df['close'] < df['supertrend']
        price_below_ema = df['close'] < df['ema']
        
        # Crossover detection
        bullish_crossover = (
            price_above_supertrend & 
            price_above_ema & 
            (df['close'].shift(1) <= df['supertrend'].shift(1))
        )
        
        bearish_crossover = (
            price_below_supertrend & 
            price_below_ema & 
            (df['close'].shift(1) >= df['supertrend'].shift(1))
        )
        
        # Apply filters (more lenient)
        atr_ok = df['atr'] > 0.01  # Very low threshold
        volume_ok = df['volume_ratio'] > self.volume_threshold
        rsi_ok = (df['rsi'] > 30) & (df['rsi'] < 70)  # Not oversold/overbought
        
        # Combine conditions
        bullish_signals = bullish_crossover & atr_ok & volume_ok & rsi_ok
        bearish_signals = bearish_crossover & atr_ok & volume_ok & rsi_ok
        
        # Calculate confidence scores
        confidence_scores = pd.Series(0, index=df.index)
        
        # Base confidence for signals
        confidence_scores.loc[bullish_signals] = 50  # Base confidence
        confidence_scores.loc[bearish_signals] = 50  # Base confidence
        
        # Add bonus for strong conditions
        strong_bullish = bullish_signals & (df['volume_ratio'] > 1.5) & (df['rsi'] > 50)
        strong_bearish = bearish_signals & (df['volume_ratio'] > 1.5) & (df['rsi'] < 50)
        
        confidence_scores.loc[strong_bullish] += 25
        confidence_scores.loc[strong_bearish] += 25
        
        # Apply confidence threshold
        valid_signals = confidence_scores >= self.min_confidence_threshold
        
        # Generate final signals
        signal_mask = (bullish_signals | bearish_signals) & valid_signals
        
        if not signal_mask.any():
            return pd.DataFrame()
        
        # Create signals DataFrame
        signals_df = df.loc[signal_mask].copy()
        signals_df['signal'] = np.where(bullish_signals.loc[signal_mask], 'BUY CALL', 'BUY PUT')
        signals_df['price'] = signals_df['close']
        signals_df['confidence_score'] = confidence_scores[signal_mask]
        
        # Calculate stop loss and targets
        signals_df['stop_loss'] = 1.5 * signals_df['atr']
        signals_df['target1'] = 2.0 * signals_df['atr']
        signals_df['target2'] = 3.0 * signals_df['atr']
        signals_df['target3'] = 4.0 * signals_df['atr']
        signals_df['position_multiplier'] = 1.0
        
        # Add reasoning
        signals_df['reasoning'] = 'Supertrend + EMA alignment with volume confirmation'
        
        return signals_df[['timestamp', 'signal', 'price', 'confidence_score', 
                          'stop_loss', 'target1', 'target2', 'target3', 
                          'position_multiplier', 'reasoning']]

    def validate(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """FIXED validation method."""
        try:
            if len(df) < self.min_candles:
                return False, {'reason': 'insufficient_data', 'required': self.min_candles, 'actual': len(df)}
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False, {'reason': 'missing_columns', 'missing': missing_cols}
            
            # Check for valid data
            if df['close'].isna().all():
                return False, {'reason': 'no_valid_price_data'}
            
            if df['volume'].isna().all():
                return False, {'reason': 'no_valid_volume_data'}
            
            return True, {'status': 'valid', 'candles': len(df)}
            
        except Exception as e:
            return False, {'reason': 'validation_error', 'error': str(e)}

    def get_signal(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """FIXED signal generation method."""
        try:
            # Validate input
            is_valid, validation_info = self.validate(df)
            if not is_valid:
                return {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'reasoning': f"Validation failed: {validation_info.get('reason', 'unknown')}"
                }
            
            # Get signals
            signals_df = self.analyze_vectorized(df)
            
            if signals_df.empty:
                return {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'reasoning': 'no signal conditions met'
                }
            
            # Get the latest signal
            latest_signal = signals_df.iloc[-1]
            
            return {
                'signal': latest_signal['signal'],
                'confidence': int(latest_signal['confidence_score']),
                'reasoning': latest_signal['reasoning'],
                'stop_loss': float(latest_signal['stop_loss']),
                'target1': float(latest_signal['target1']),
                'target2': float(latest_signal['target2']),
                'target3': float(latest_signal['target3'])
            }
            
        except Exception as e:
            logger.error(f"Error in get_signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reasoning': f'error: {str(e)}'
            }
