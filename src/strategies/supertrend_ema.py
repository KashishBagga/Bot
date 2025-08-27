import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from indicators.supertrend import get_supertrend_instance


class SupertrendEma(Strategy):
    """
    Multi-timeframe Supertrend + EMA strategy with signal confirmation across 3min, 15min, and 30min charts.
    """

    # Minimum candles required for analysis
    min_candles = 60

    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        params = params or {}
        self.ema_period = params.get("ema_period", 20)
        self.supertrend_period = params.get("supertrend_period", 10)
        self.supertrend_multiplier = params.get("supertrend_multiplier", 3.0)
        self.timeframe_data = timeframe_data or {}
        self._supertrend_instances = {}
        self.min_confidence_threshold = params.get("min_confidence_threshold", 60)
        self.atr_period = params.get("atr_period", 14)
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 20)
        self.volume_threshold = params.get("volume_threshold", 1.2)
        super().__init__("supertrend_ema", params)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema"] = df["close"].ewm(span=self.ema_period).mean()
        df["price_to_ema_ratio"] = (df["close"] / df["ema"] - 1) * 100
        
        # Add enhanced indicators
        from src.core.indicators import indicators
        df['atr'] = indicators.atr(df, period=self.atr_period)
        df['adx'] = indicators.adx(df, period=self.adx_period)
        df['rsi'] = indicators.rsi(df, period=14)
        
        # Add volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean().fillna(0)
        df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1)
        
        # Add EMA slope for momentum
        df['ema_slope'] = df['ema'].diff(5).fillna(0)
        
        return df

    def _get_supertrend_instance(self, timeframe: str):
        key = f"{self.__class__.__name__}_{timeframe}"
        if key not in self._supertrend_instances:
            self._supertrend_instances[key] = get_supertrend_instance(
                key, period=self.supertrend_period, multiplier=self.supertrend_multiplier
            )
        return self._supertrend_instances[key]

    def _evaluate_timeframe(self, df: pd.DataFrame, timeframe: str, ts: datetime) -> Optional[Dict[str, Any]]:
        df = df[df.index <= ts].copy()
        if df.empty or len(df) < 20:  # Need at least 20 candles for indicators
            return None

        try:
            # Use closed candle for analysis
            candle = self.get_closed_candle(df)
            st_instance = self._get_supertrend_instance(timeframe)
            st_data = st_instance.update(candle)

            ema = df["close"].ewm(span=self.ema_period).mean().iloc[-1]
            price_above_ema = candle["close"] > ema
        except (IndexError, ValueError):
            # Not enough data for indicators
            return None

        return {
            "supertrend": st_data["direction"],
            "ema_trend": 1 if price_above_ema else -1,
            "ema": ema,
            "candle": candle,
            "st_data": st_data
        }

    def safe_signal_time(self, val):
        return val if isinstance(val, (pd.Timestamp, datetime)) else datetime.now()

    def to_ist_str(self, val):
        if isinstance(val, (pd.Timestamp, datetime)):
            ist_dt = val + timedelta(hours=5, minutes=30)
            return ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        return None

    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics based on future data."""
        if future_data is None or future_data.empty:
            return {
                "outcome": "Pending",
                "pnl": 0.0,
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": "",
                "exit_time": None
            }
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        if signal == "BUY CALL":
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(candle.get('time', None))
                if candle['low'] <= (entry_price - stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price - stop_loss:.2f}"
                    exit_time = current_time
                    break
                if targets_hit == 0 and candle['high'] >= (entry_price + target):
                    targets_hit = 1
                    pnl = target
                    if candle['high'] >= (entry_price + target2):
                        targets_hit = 2
                        pnl = target2
                        if candle['high'] >= (entry_price + target3):
                            targets_hit = 3
                            pnl = target3
                            outcome = "Win"
                            exit_time = current_time
                            break
                    if targets_hit == 1:
                        outcome = "Win"
                        exit_time = current_time
                        break
        elif signal == "BUY PUT":
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(candle.get('time', None))
                if candle['high'] >= (entry_price + stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price + stop_loss:.2f}"
                    exit_time = current_time
                    break
                if targets_hit == 0 and candle['low'] <= (entry_price - target):
                    targets_hit = 1
                    pnl = target
                    if candle['low'] <= (entry_price - target2):
                        targets_hit = 2
                        pnl = target2
                        if candle['low'] <= (entry_price - target3):
                            targets_hit = 3
                            pnl = target3
                            outcome = "Win"
                            exit_time = current_time
                            break
                    if targets_hit == 1:
                        outcome = "Win"
                        exit_time = current_time
                        break
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        }

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
            if 'ema' not in df.columns:
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
            
            # Calculate Supertrend for the entire dataframe
            from indicators.supertrend import get_supertrend_instance
            st_instance = get_supertrend_instance(
                "supertrend_ema_vectorized", 
                period=self.supertrend_period, 
                multiplier=self.supertrend_multiplier
            )
            
            # Process each candle to get Supertrend data
            st_directions = []
            st_values = []
            
            for i in range(len(df)):
                candle = df.iloc[i]
                st_result = st_instance.update(candle)
                if st_result is not None:
                    st_value, st_direction = st_result
                    st_values.append(st_value)
                    st_directions.append(st_direction)
                else:
                    st_values.append(candle['close'])
                    st_directions.append(0)
            
            df['supertrend_direction'] = st_directions
            df['supertrend_value'] = st_values
            
            # Detect Supertrend crossovers (look-ahead safe with shift)
            # Bullish: price crosses above Supertrend
            bullish_cross = (df['close'] > df['supertrend_value']) & (df['close'].shift(1) <= df['supertrend_value'].shift(1))
            
            # Bearish: price crosses below Supertrend
            bearish_cross = (df['close'] < df['supertrend_value']) & (df['close'].shift(1) >= df['supertrend_value'].shift(1))
            
            # EMA trend alignment
            bullish_ema = df['close'] > df['ema']
            bearish_ema = df['close'] < df['ema']
            
            # ATR filter
            atr_filter = df['atr'] > 0.5  # Minimum ATR threshold
            
            # ADX filter
            adx_filter = df['adx'] > self.adx_threshold
            
            # RSI filters
            bullish_rsi = df['rsi'] > 45
            bearish_rsi = df['rsi'] < 55
            
            # Volume confirmation
            volume_filter = df['volume_ratio'] > 1.0
            
            # EMA slope momentum
            bullish_momentum = df['ema_slope'] > 0
            bearish_momentum = df['ema_slope'] < 0
            
            # Calculate confidence scores vectorized
            confidence_scores = pd.Series(0, index=df.index)
            
            # For bullish signals
            bullish_mask = bullish_cross & bullish_ema & atr_filter & adx_filter & bullish_rsi & volume_filter & bullish_momentum
            confidence_scores.loc[bullish_mask] += 40  # Supertrend + EMA alignment
            confidence_scores.loc[bullish_mask] += 20  # ATR filter passed
            confidence_scores.loc[bullish_mask] += 20  # ADX filter passed
            confidence_scores.loc[bullish_mask] += 10  # RSI confirmation
            confidence_scores.loc[bullish_mask & (df['volume_ratio'] > 1.2)] += 10  # Volume spike
            
            # For bearish signals
            bearish_mask = bearish_cross & bearish_ema & atr_filter & adx_filter & bearish_rsi & volume_filter & bearish_momentum
            confidence_scores.loc[bearish_mask] += 40  # Supertrend + EMA alignment
            confidence_scores.loc[bearish_mask] += 20  # ATR filter passed
            confidence_scores.loc[bearish_mask] += 20  # ADX filter passed
            confidence_scores.loc[bearish_mask] += 10  # RSI confirmation
            confidence_scores.loc[bearish_mask & (df['volume_ratio'] > 1.2)] += 10  # Volume spike
            
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
                f"Supertrend + EMA, ATR {df.loc[valid_signals, 'atr'].round(2)}, "
                f"ADX {df.loc[valid_signals, 'adx'].round(2)}, "
                f"RSI {df.loc[valid_signals, 'rsi'].round(1)}"
            )
            
            # Return only valid signals
            valid_signals_df = signals[signals['signal'] != 'NO TRADE'].copy()
            
            return valid_signals_df
            
        except Exception as e:
            logging.error(f"Error in SupertrendEma vectorized analysis: {e}")
            return pd.DataFrame()

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals using closed candles."""
        if not self.validate_data(data):
            return {'signal': 'NO TRADE', 'reason': 'insufficient data'}

        try:
            # Use closed candle for analysis
            candle = self.get_closed_candle(data)
            
            # Add indicators
            df = self.calculate_indicators(data)
            
            # Get SuperTrend data
            st_instance = self._get_supertrend_instance("5min")
            st_data = st_instance.update(candle)
            
            if st_data is None or st_data[1] is None:
                return {'signal': 'NO TRADE', 'reason': 'insufficient supertrend data'}
            
            supertrend_direction = st_data[1]
            
            # Get EMAs and other indicators
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(rsi) or pd.isna(macd) or pd.isna(atr):
                return {'signal': 'NO TRADE', 'reason': 'indicator data unavailable'}
            
            # ENHANCED FILTERS FOR NIFTY50 - VERY RELAXED VERSION FOR TESTING
            
            # 1. ATR-based volatility filter - very relaxed
            if atr < 0.05:  # Very low threshold for testing
                return {'signal': 'NO TRADE', 'reason': f'ATR too low: {atr:.2f} < 0.05'}
            
            # 2. Volume confirmation - very relaxed
            if volume_ratio < 0.1:  # Very low threshold for testing
                return {'signal': 'NO TRADE', 'reason': f'volume too low: {volume_ratio:.2f} < 0.1'}
            
            # BUY CALL conditions with very relaxed filters
            if (supertrend_direction == 1 and  # SuperTrend uptrend
                ema_9 > ema_21 and  # EMA crossover
                20 < rsi < 85 and  # Very wide RSI range
                macd > macd_signal):  # MACD bullish
                
                # Calculate enhanced confidence score with proper weighting
                trend_strength = 35 if ema_9 > ema_21 * 1.005 else 25  # EMA alignment
                rsi_strength = 20 if 40 < rsi < 70 else 10  # RSI confirmation
                macd_strength = 15 if macd > macd_signal * 1.05 else 8  # MACD momentum
                volume_strength = 15 if volume_ratio > 1.2 else 8  # Volume confirmation
                atr_strength = 10 if atr > 0.5 else 5  # Volatility filter
                adx_strength = 5 if df['adx'].iloc[-1] > self.adx_threshold else 0  # Trend strength
                
                confidence_score = trend_strength + rsi_strength + macd_strength + volume_strength + atr_strength + adx_strength
                
                if confidence_score >= 30:  # Very low threshold for testing
                    # ENHANCED RISK MANAGEMENT
                    stop_loss = 1.5 * atr  # ATR-based stop-loss
                    target1 = 2.0 * atr   # First target
                    
                    # Dynamic position sizing based on confidence
                    position_multiplier = 1.0 if confidence_score >= 80 else 0.8
                    
                    return {
                        'signal': 'BUY CALL',
                        'price': candle['close'],
                        'confidence_score': confidence_score,
                        'stop_loss': stop_loss,
                        'target1': target1,
                        'target2': 3.0 * atr,
                        'target3': 4.0 * atr,
                        'position_multiplier': position_multiplier,
                        'timestamp': candle.get('timestamp', datetime.now()),
                        'reasoning': f"SuperTrend uptrend, EMA crossover, RSI {rsi:.1f}, MACD bullish, ATR {atr:.2f}, Volume {volume_ratio:.2f}"
                    }
            
            # BUY PUT conditions with very relaxed filters
            elif (supertrend_direction == -1 and  # SuperTrend downtrend
                  ema_9 < ema_21 and  # EMA crossover
                  15 < rsi < 80 and  # Very wide RSI range
                  macd < macd_signal):  # MACD bearish
                
                # Calculate enhanced confidence score with proper weighting
                trend_strength = 35 if ema_9 < ema_21 * 0.995 else 25  # EMA alignment
                rsi_strength = 20 if 30 < rsi < 60 else 10  # RSI confirmation
                macd_strength = 15 if macd < macd_signal * 0.95 else 8  # MACD momentum
                volume_strength = 15 if volume_ratio > 1.2 else 8  # Volume confirmation
                atr_strength = 10 if atr > 0.5 else 5  # Volatility filter
                adx_strength = 5 if df['adx'].iloc[-1] > self.adx_threshold else 0  # Trend strength
                
                confidence_score = trend_strength + rsi_strength + macd_strength + volume_strength + atr_strength + adx_strength
                
                if confidence_score >= 30:  # Very low threshold for testing
                    # ENHANCED RISK MANAGEMENT
                    stop_loss = 1.5 * atr  # ATR-based stop-loss
                    target1 = 2.0 * atr   # First target
                    
                    # Dynamic position sizing based on confidence
                    position_multiplier = 1.0 if confidence_score >= 80 else 0.8
                    
                    return {
                        'signal': 'BUY PUT',
                        'price': candle['close'],
                        'confidence_score': confidence_score,
                        'stop_loss': stop_loss,
                        'target1': target1,
                        'target2': 3.0 * atr,
                        'target3': 4.0 * atr,
                        'position_multiplier': position_multiplier,
                        'timestamp': candle.get('timestamp', datetime.now()),
                        'reasoning': f"SuperTrend downtrend, EMA crossover, RSI {rsi:.1f}, MACD bearish, ATR {atr:.2f}, Volume {volume_ratio:.2f}"
                    }
            
            return {'signal': 'NO TRADE', 'reason': 'no signal conditions met'}
            
        except Exception as e:
            logging.error(f"Error in SupertrendEma analysis: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}

    def analyze_strategy(self, df: pd.DataFrame, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Strategy-specific analysis with pre-calculated indicators and market conditions - OPTIMIZED VERSION"""
        try:
            # Get previous candle for SuperTrend calculation
            prev_candle = None
            if len(df) > 1:
                prev_candle = df.iloc[-2].to_dict()
            
            # Get SuperTrend data
            st_instance = self._get_supertrend_instance("5min")
            st_data = st_instance.update(candle, prev_candle)
            
            if st_data is None or st_data[1] is None:
                return {'signal': 'NO TRADE', 'reason': 'insufficient supertrend data'}
            
            supertrend_direction = st_data[1]
            
            # Get EMAs and other indicators from pre-calculated data
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else ema_21
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(rsi) or pd.isna(macd) or pd.isna(atr):
                return {'signal': 'NO TRADE', 'reason': 'indicator data unavailable'}
            
            # MARKET CONDITION ANALYSIS - CRITICAL FILTER
            if 'market_tradeable' in df.columns and 'market_condition' in df.columns:
                market_tradeable = df['market_tradeable'].iloc[-1]
                market_condition = df['market_condition'].iloc[-1]
                adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20
                trend_direction = df['trend_direction'].iloc[-1] if 'trend_direction' in df.columns else 0
                
                # CRITICAL: Check if market is tradeable
                if not market_tradeable:
                    return {'signal': 'NO TRADE', 'reason': f"market not tradeable: {df['market_reason'].iloc[-1] if 'market_reason' in df.columns else 'unknown'}"}
                
                # CRITICAL: Only trade in trending markets - REJECT RANGING MARKETS
                if market_condition == 'RANGING':
                    return {'signal': 'NO TRADE', 'reason': f"market ranging: ADX {adx:.1f} < 25"}
                
                # CRITICAL: Only trade in trending markets - REJECT INSUFFICIENT DATA
                if market_condition == 'INSUFFICIENT_DATA':
                    return {'signal': 'NO TRADE', 'reason': f"insufficient data for market analysis"}
            else:
                # If market condition data is not available, don't trade
                return {'signal': 'NO TRADE', 'reason': 'market condition data not available'}
            
            # ENHANCED FILTERS - OPTIMIZED FOR BETTER SIGNAL GENERATION
            if atr < 0.3:  # Moderate volatility requirement
                return {'signal': 'NO TRADE', 'reason': f'ATR too low: {atr:.2f} < 0.3'}
            
            if volume_ratio < 0.8:  # Moderate volume requirement
                return {'signal': 'NO TRADE', 'reason': f'volume too low: {volume_ratio:.2f} < 0.8'}
            
            # IMPROVEMENT 1: EMA SLOPE FILTER - Only trade when EMA slope is strong
            if len(df) >= 10:
                ema_9_slope = (ema_9 - df['ema_9'].iloc[-5]) / df['ema_9'].iloc[-5] * 100
                ema_21_slope = (ema_21 - df['ema_21'].iloc[-5]) / df['ema_21'].iloc[-5] * 100
            else:
                ema_9_slope = 0
                ema_21_slope = 0
            
            # IMPROVEMENT 2: VOLUME CONFIRMATION - Require volume spike on crossover
            volume_confirmed = volume_ratio > 1.0
            
            # BUY CALL conditions with IMPROVED FILTERS
            if (supertrend_direction == 1 and  # SuperTrend uptrend
                ema_9 > ema_21 and  # EMA crossover
                ema_9_slope > 0.1 and  # EMA slope is positive (IMPROVEMENT)
                25 < rsi < 75 and  # Relaxed RSI range
                macd > macd_signal and  # MACD bullish
                volume_confirmed and  # Volume confirmation (IMPROVEMENT)
                market_condition['trend_direction'] == 1):  # Market trend aligns
                
                # Calculate enhanced confidence score with market condition bonus
                trend_strength = 30 if ema_9 > ema_21 * 1.005 else 20
                rsi_strength = 25 if 40 < rsi < 70 else 15
                macd_strength = 20 if macd > macd_signal * 1.05 else 10
                volume_strength = 15 if volume_ratio > 1.2 else 8
                atr_strength = 10 if atr > 0.5 else 5
                slope_strength = 10 if ema_9_slope > 0.2 else 5  # EMA slope bonus
                
                # Market condition bonus
                market_bonus = 0
                if market_condition['condition'] == 'STRONG_UPTREND':
                    market_bonus = 20
                elif market_condition['condition'] == 'WEAK_UPTREND':
                    market_bonus = 10
                
                confidence_score = trend_strength + rsi_strength + macd_strength + volume_strength + atr_strength + slope_strength + market_bonus
                
                if confidence_score >= 50:  # Lower threshold for more signals
                    # IMPROVEMENT 3: BETTER RISK MANAGEMENT
                    stop_loss = 1.5 * atr  # ATR-based stop-loss
                    target1 = 2.5 * atr   # First target (1:1.67)
                    target2 = 4.0 * atr   # Second target (1:2.67)
                    
                    return {
                        'signal': 'BUY CALL',
                        'price': candle['close'],
                        'confidence': confidence_score,
                        'stop_loss': stop_loss,
                        'target': target1,
                        'target2': target2,
                        'timestamp': candle.get('timestamp', datetime.now()),
                        'reason': f"SuperTrend uptrend, EMA crossover, Slope: {ema_9_slope:.2f}%, RSI {rsi:.1f}, MACD bullish, Volume: {volume_ratio:.2f}, Market: {market_condition['condition']}"
                    }
            
            # BUY PUT conditions with IMPROVED FILTERS
            elif (supertrend_direction == -1 and  # SuperTrend downtrend
                  ema_9 < ema_21 and  # EMA crossover
                  ema_9_slope < -0.1 and  # EMA slope is negative (IMPROVEMENT)
                  25 < rsi < 75 and  # Relaxed RSI range
                  macd < macd_signal and  # MACD bearish
                  volume_confirmed and  # Volume confirmation (IMPROVEMENT)
                  market_condition['trend_direction'] == -1):  # Market trend aligns
                
                # Calculate enhanced confidence score with market condition bonus
                trend_strength = 30 if ema_9 < ema_21 * 0.995 else 20
                rsi_strength = 25 if 30 < rsi < 60 else 15
                macd_strength = 20 if macd < macd_signal * 0.95 else 10
                volume_strength = 15 if volume_ratio > 1.2 else 8
                atr_strength = 10 if atr > 0.5 else 5
                slope_strength = 10 if ema_9_slope < -0.2 else 5  # EMA slope bonus
                
                # Market condition bonus
                market_bonus = 0
                if market_condition['condition'] == 'STRONG_DOWNTREND':
                    market_bonus = 20
                elif market_condition['condition'] == 'WEAK_DOWNTREND':
                    market_bonus = 10
                
                confidence_score = trend_strength + rsi_strength + macd_strength + volume_strength + atr_strength + slope_strength + market_bonus
                
                if confidence_score >= 50:  # Lower threshold for more signals
                    # IMPROVEMENT 3: BETTER RISK MANAGEMENT
                    stop_loss = 1.5 * atr  # ATR-based stop-loss
                    target1 = 2.5 * atr   # First target (1:1.67)
                    target2 = 4.0 * atr   # Second target (1:2.67)
                    
                    return {
                        'signal': 'BUY PUT',
                        'price': candle['close'],
                        'confidence': confidence_score,
                        'stop_loss': stop_loss,
                        'target': target1,
                        'target2': target2,
                        'timestamp': candle.get('timestamp', datetime.now()),
                        'reason': f"SuperTrend downtrend, EMA crossover, Slope: {ema_9_slope:.2f}%, RSI {rsi:.1f}, MACD bearish, Volume: {volume_ratio:.2f}, Market: {market_condition['condition']}"
                    }
            
            return {'signal': 'NO TRADE', 'reason': 'no conditions met'}
            
        except Exception as e:
            return {'signal': 'ERROR', 'reason': f'strategy error: {str(e)}'}

# Optional legacy adapter
def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    df = pd.DataFrame([candle])
    strategy = SupertrendEma()
    df = strategy.add_indicators(df)
    return strategy.analyze(df.iloc[-1], len(df) - 1, df)
