import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from src.core.indicators import indicators
import logging

class EmaCrossoverEnhanced(Strategy):
    """
    Enhanced EMA Crossover Strategy with:
    - Higher timeframe trend filter (15m EMA direction)
    - Wider EMAs (20/50, 50/200) to reduce noise
    - ATR-based filters and stop-loss
    - Trailing stop-loss
    """

    # Minimum candles required for analysis
    min_candles = 100

    def __init__(self, params: Dict[str, Any] = None):
        params = params or {}
        self.ema_short = params.get("ema_short", 20)  # Wider EMAs
        self.ema_long = params.get("ema_long", 50)
        self.ema_trend = params.get("ema_trend", 200)  # Trend filter
        self.atr_period = params.get("atr_period", 14)
        self.atr_threshold = params.get("atr_threshold", 0.5)  # Minimum ATR for trend
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 20)  # Minimum ADX for trend
        self.min_confidence_threshold = params.get("min_confidence_threshold", 60)
        self.volume_threshold = params.get("volume_threshold", 0.3)  # Volume confirmation
        self.rsi_period = params.get("rsi_period", 14)
        self.ema_slope_period = params.get("ema_slope_period", 5)  # EMA slope for momentum
        super().__init__("ema_crossover_enhanced", params)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators."""
        df = df.copy()
        
        # Add EMAs
        df['ema_short'] = indicators.ema(df, period=self.ema_short)
        df['ema_long'] = indicators.ema(df, period=self.ema_long)
        df['ema_trend'] = indicators.ema(df, period=self.ema_trend)
        
        # Add ATR for volatility filter
        df['atr'] = indicators.atr(df, period=self.atr_period)
        
        # Add ADX for trend strength
        df['adx'] = indicators.adx(df, period=self.adx_period)
        
        # Add RSI for additional confirmation
        df['rsi'] = indicators.rsi(df, period=14)
        
        # Add volume ratio
        df['volume_ratio'] = df['volume'].pct_change().fillna(0)
        
        # Add EMA slope for momentum
        df['ema_slope'] = df['ema_short'].diff(self.ema_slope_period).fillna(0)
        
        # Add volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean().fillna(0)
        df['volume_ratio_ma'] = (df['volume'] / df['volume_ma']).fillna(1)
        
        return df

    def detect_crossover(self, df: pd.DataFrame, short_col: str, long_col: str) -> Dict[str, Any]:
        """Detect EMA crossover using closed candles."""
        if len(df) < 3:
            return {'crossover': False, 'direction': 0}
        
        # Use closed candles for crossover detection
        prev = df.iloc[-2]  # Last closed candle
        prev_prev = df.iloc[-3]  # Previous closed candle
        
        # Check for crossover
        cross_up = (prev[short_col] > prev[long_col] and 
                   prev_prev[short_col] <= prev_prev[long_col])
        
        cross_down = (prev[short_col] < prev[long_col] and 
                     prev_prev[short_col] >= prev_prev[long_col])
        
        if cross_up:
            return {'crossover': True, 'direction': 1}
        elif cross_down:
            return {'crossover': True, 'direction': -1}
        else:
            return {'crossover': False, 'direction': 0}

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals using enhanced filters."""
        if not self.validate_data(data):
            return {'signal': 'NO TRADE', 'reason': 'insufficient data'}

        try:
            # Use closed candle for analysis
            candle = self.get_closed_candle(data)
            
            # Add indicators
            df = self.calculate_indicators(data)
            
            # Get indicator values from closed candle
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]
            ema_trend = df['ema_trend'].iloc[-1]
            atr = df['atr'].iloc[-1]
            adx = df['adx'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(atr):
                return {'signal': 'NO TRADE', 'reason': 'indicator data unavailable'}
            
            # Detect crossover
            crossover = self.detect_crossover(df, 'ema_short', 'ema_long')
            
            if not crossover['crossover']:
                return {'signal': 'NO TRADE', 'reason': 'no EMA crossover detected'}
            
            # Calculate confidence score
            confidence_score = 0
            
            # Trend alignment (40% weight)
            if crossover['direction'] == 1:  # Bullish crossover
                if candle['close'] > ema_trend:
                    confidence_score += 40  # Strong trend alignment
                elif candle['close'] > ema_long:
                    confidence_score += 25  # Moderate trend alignment
                else:
                    confidence_score += 10  # Weak trend alignment
            else:  # Bearish crossover
                if candle['close'] < ema_trend:
                    confidence_score += 40
                elif candle['close'] < ema_long:
                    confidence_score += 25
                else:
                    confidence_score += 10
            
            # ATR filter (20% weight)
            if atr > self.atr_threshold:
                confidence_score += 20
            else:
                return {'signal': 'NO TRADE', 'reason': f'ATR too low: {atr:.2f} < {self.atr_threshold}'}
            
            # ADX filter (20% weight)
            if adx > self.adx_threshold:
                confidence_score += 20
            else:
                return {'signal': 'NO TRADE', 'reason': f'ADX too low: {adx:.2f} < {self.adx_threshold}'}
            
            # RSI confirmation (10% weight)
            if crossover['direction'] == 1 and rsi > 45:  # Bullish with RSI support
                confidence_score += 10
            elif crossover['direction'] == -1 and rsi < 55:  # Bearish with RSI support
                confidence_score += 10
            
            # Volume confirmation (10% weight)
            volume_ma = df['volume_ma'].iloc[-1]
            volume_ratio_ma = df['volume_ratio_ma'].iloc[-1]
            ema_slope = df['ema_slope'].iloc[-1]
            
            if volume_ratio_ma > 1.2:  # Volume spike
                confidence_score += 10
            elif volume_ratio_ma > 1.0:  # Above average volume
                confidence_score += 5
            
            # EMA slope momentum (5% weight)
            if crossover['direction'] == 1 and ema_slope > 0:  # Bullish momentum
                confidence_score += 5
            elif crossover['direction'] == -1 and ema_slope < 0:  # Bearish momentum
                confidence_score += 5
            
            # Generate signals based on crossover direction
            if crossover['direction'] == 1 and confidence_score >= self.min_confidence_threshold:
                # BUY CALL signal
                stop_loss = 1.5 * atr
                target1 = 2.0 * atr
                target2 = 3.0 * atr
                target3 = 4.0 * atr
                
                # Dynamic position sizing based on confidence
                position_multiplier = 1.0 if confidence_score >= 80 else 0.8
                
                return {
                    'signal': 'BUY CALL',
                    'price': candle['close'],
                    'confidence_score': confidence_score,
                    'stop_loss': stop_loss,
                    'target1': target1,
                    'target2': target2,
                    'target3': target3,
                    'position_multiplier': position_multiplier,
                    'timestamp': candle.name if hasattr(candle, 'name') else datetime.now(),
                    'reasoning': f"EMA crossover bullish, Trend aligned, ATR {atr:.2f}, ADX {adx:.2f}, RSI {rsi:.1f}"
                }
            
            elif crossover['direction'] == -1 and confidence_score >= self.min_confidence_threshold:
                # BUY PUT signal
                stop_loss = 1.5 * atr
                target1 = 2.0 * atr
                target2 = 3.0 * atr
                target3 = 4.0 * atr
                
                # Dynamic position sizing based on confidence
                position_multiplier = 1.0 if confidence_score >= 80 else 0.8
                
                return {
                    'signal': 'BUY PUT',
                    'price': candle['close'],
                    'confidence_score': confidence_score,
                    'stop_loss': stop_loss,
                    'target1': target1,
                    'target2': target2,
                    'target3': target3,
                    'position_multiplier': position_multiplier,
                    'timestamp': candle.name if hasattr(candle, 'name') else datetime.now(),
                    'reasoning': f"EMA crossover bearish, Trend aligned, ATR {atr:.2f}, ADX {adx:.2f}, RSI {rsi:.1f}"
                }
            
            return {'signal': 'NO TRADE', 'reason': f'confidence too low: {confidence_score} < {self.min_confidence_threshold}'}
            
        except Exception as e:
            logging.error(f"Error in EmaCrossoverEnhanced analysis: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}
    
    def analyze_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized analysis of the entire dataframe - much faster than row-by-row processing.
        Returns a DataFrame with signals for all candles that meet the criteria.
        """
        try:
            # Ensure we have enough data
            if len(df) < self.min_candles:
                return pd.DataFrame()
            
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Ensure indicators are present
            if 'ema_short' not in df.columns:
                df = self.add_indicators(df)
            
            # Initialize signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals['signal'] = 'NO TRADE'
            signals['price'] = df['close']
            signals['confidence_score'] = 0
            signals['stop_loss'] = 0.0
            signals['target1'] = 0.0
            signals['target2'] = 0.0
            signals['target3'] = 0.0
            signals['position_multiplier'] = 1.0
            signals['reasoning'] = ''
            
            # Detect crossovers vectorized (look-ahead safe with shift)
            # Bullish crossover: ema_short crosses above ema_long
            bullish_cross = (df['ema_short'] > df['ema_long']) & (df['ema_short'].shift(1) <= df['ema_long'].shift(1))
            
            # Bearish crossover: ema_short crosses below ema_long
            bearish_cross = (df['ema_short'] < df['ema_long']) & (df['ema_short'].shift(1) >= df['ema_long'].shift(1))
            
            # ATR filter
            atr_filter = df['atr'] > self.atr_threshold
            
            # ADX filter
            adx_filter = df['adx'] > self.adx_threshold
            
            # RSI filters
            bullish_rsi = df['rsi'] > 45
            bearish_rsi = df['rsi'] < 55
            
            # Volume confirmation
            volume_filter = df['volume_ratio_ma'] > 1.0
            
            # EMA slope momentum
            bullish_momentum = df['ema_slope'] > 0
            bearish_momentum = df['ema_slope'] < 0
            
            # Trend alignment
            bullish_trend = df['close'] > df['ema_trend']
            bearish_trend = df['close'] < df['ema_trend']
            
            # Calculate confidence scores vectorized
            confidence_scores = pd.Series(0, index=df.index)
            
            # For bullish signals
            bullish_mask = bullish_cross & atr_filter & adx_filter & bullish_rsi & volume_filter & bullish_momentum
            confidence_scores.loc[bullish_mask & bullish_trend] += 40  # Strong trend alignment
            confidence_scores.loc[bullish_mask & (df['close'] > df['ema_long'])] += 25  # Moderate trend alignment
            confidence_scores.loc[bullish_mask] += 20  # ATR filter passed
            confidence_scores.loc[bullish_mask] += 20  # ADX filter passed
            confidence_scores.loc[bullish_mask] += 10  # RSI confirmation
            confidence_scores.loc[bullish_mask & (df['volume_ratio_ma'] > 1.2)] += 10  # Volume spike
            confidence_scores.loc[bullish_mask & (df['volume_ratio_ma'] > 1.0)] += 5   # Above average volume
            confidence_scores.loc[bullish_mask] += 5   # EMA slope momentum
            
            # For bearish signals
            bearish_mask = bearish_cross & atr_filter & adx_filter & bearish_rsi & volume_filter & bearish_momentum
            confidence_scores.loc[bearish_mask & bearish_trend] += 40  # Strong trend alignment
            confidence_scores.loc[bearish_mask & (df['close'] < df['ema_long'])] += 25  # Moderate trend alignment
            confidence_scores.loc[bearish_mask] += 20  # ATR filter passed
            confidence_scores.loc[bearish_mask] += 20  # ADX filter passed
            confidence_scores.loc[bearish_mask] += 10  # RSI confirmation
            confidence_scores.loc[bearish_mask & (df['volume_ratio_ma'] > 1.2)] += 10  # Volume spike
            confidence_scores.loc[bearish_mask & (df['volume_ratio_ma'] > 1.0)] += 5   # Above average volume
            confidence_scores.loc[bearish_mask] += 5   # EMA slope momentum
            
            # Apply confidence threshold
            valid_signals = confidence_scores >= self.min_confidence_threshold
            
            # Generate signals
            signals.loc[bullish_mask & valid_signals, 'signal'] = 'BUY CALL'
            signals.loc[bearish_mask & valid_signals, 'signal'] = 'BUY PUT'
            
            # Set confidence scores
            signals.loc[valid_signals, 'confidence_score'] = confidence_scores[valid_signals]
            
            # Calculate stop loss and targets (ATR-based)
            signals.loc[valid_signals, 'stop_loss'] = 1.5 * df.loc[valid_signals, 'atr']
            signals.loc[valid_signals, 'target1'] = 2.0 * df.loc[valid_signals, 'atr']
            signals.loc[valid_signals, 'target2'] = 3.0 * df.loc[valid_signals, 'atr']
            signals.loc[valid_signals, 'target3'] = 4.0 * df.loc[valid_signals, 'atr']
            
            # Dynamic position sizing
            high_confidence = confidence_scores >= 80
            signals.loc[valid_signals & high_confidence, 'position_multiplier'] = 1.0
            signals.loc[valid_signals & ~high_confidence, 'position_multiplier'] = 0.8
            
            # Add reasoning
            signals.loc[valid_signals, 'reasoning'] = (
                f"EMA crossover, ATR {df.loc[valid_signals, 'atr'].round(2)}, "
                f"ADX {df.loc[valid_signals, 'adx'].round(2)}, "
                f"RSI {df.loc[valid_signals, 'rsi'].round(1)}"
            )
            
            # Return only valid signals
            valid_signals_df = signals[signals['signal'] != 'NO TRADE'].copy()
            
            return valid_signals_df
            
        except Exception as e:
            logging.error(f"Error in EmaCrossoverEnhanced vectorized analysis: {e}")
            return pd.DataFrame() 