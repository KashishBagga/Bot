"""
EMA Crossover strategy.
Trading strategy based on crossing of exponential moving averages.
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from src.core.indicators import indicators
from db import log_strategy_sql

class EmaCrossover(Strategy):
    """
    Multi-timeframe EMA Crossover strategy with signal confirmation across 3min, 15min, and 30min charts.
    """
    
    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize the EMA Crossover strategy.
        
        Args:
            params: Strategy parameters
            timeframe_data: Dictionary of timeframes with their respective data
        """
        super().__init__("ema_crossover", params)
        self.fast_ema = params.get("fast_ema", 9) if params else 9
        self.slow_ema = params.get("slow_ema", 21) if params else 21
        self.timeframe_data = timeframe_data or {}
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Add the fast and slow EMAs
        fast_period = self.fast_ema
        slow_period = self.slow_ema
        
        data['ema_fast'] = indicators.ema(data, period=fast_period)
        data['ema_slow'] = indicators.ema(data, period=slow_period)
        
        # For backward compatibility
        data['ema_9'] = data['ema_fast']
        data['ema_21'] = data['ema_slow']
        
        # Calculate crossover strength (percentage difference between EMAs)
        # Avoid division by zero
        data['crossover_strength'] = data.apply(
            lambda row: 100 * (row['ema_fast'] - row['ema_slow']) / row['ema_slow'] 
            if row['ema_slow'] != 0 else 0, axis=1
        )
        
        # Determine momentum based on the slope of the fast EMA
        data['ema_fast_change'] = data['ema_fast'].pct_change(5) * 100  # 5-period percent change
        
        return data
    
    def safe_signal_time(self, val):
        return val if isinstance(val, (pd.Timestamp, datetime)) else datetime.now()
    
    def to_ist_str(self, val):
        if isinstance(val, (pd.Timestamp, datetime)):
            ist_dt = val + timedelta(hours=5, minutes=30)
            return ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        return None
    
    def analyze(self, candle: pd.Series, index: int, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Analyze data and generate trading signals.
        
        Args:
            candle: Current candle data
            index: Current index in the dataframe
            df: Full dataframe with indicators
            future_data: Optional future candles for performance tracking
            
        Returns:
            Dict[str, Any]: Signal data
        """
        # Ensure DataFrame has DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            else:
                raise ValueError("DataFrame must have a datetime index or 'time' column")

        if index >= len(df):
            return None

        # If we have timeframe data, use multi-timeframe analysis
        if self.timeframe_data:
            return self.analyze_multi_timeframe(candle, index, df, future_data)
        
        # Fall back to single timeframe analysis
        return self.analyze_single_timeframe(df.iloc[index:index+1], future_data)

    def analyze_single_timeframe(self, data: pd.DataFrame, future_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze single timeframe data for EMA crossover signals."""
        if data.empty:
            return None
            
        candle = data.iloc[0]
        
        # Get indicator values
        ema_9 = candle.get('ema_fast', candle.get('ema_9', 0))
        ema_21 = candle.get('ema_slow', candle.get('ema_21', 0))
        rsi = candle.get('rsi', 50)
        macd = candle.get('macd', 0)
        macd_signal = candle.get('macd_signal', 0)
        volume_ratio = candle.get('volume_ratio', 1.0)
        atr = candle.get('atr', candle['close'] * 0.01)
        
        # OPTIMIZATION: Enhanced signal generation with better filters
        signal = "NO TRADE"
        confidence_score = 0
        
        # Check for valid EMA values
        if ema_9 <= 0 or ema_21 <= 0:
            return None
            
        # Calculate EMA crossover strength
        crossover_strength = abs(ema_9 - ema_21) / ema_21 * 100
        
        # OPTIMIZATION: Improved signal conditions
        if ema_9 > ema_21 and crossover_strength > 0.1:  # Bullish crossover
            # Additional bullish filters
            if (rsi > 40 and rsi < 80 and  # RSI not overbought
                macd > macd_signal and     # MACD bullish
                volume_ratio > 0.8):       # Decent volume
                
                signal = "BUY CALL"
                # OPTIMIZATION: Better confidence calculation
                confidence_score = 60 + min(20, crossover_strength * 10) + min(10, (rsi - 40) / 2)
                
        elif ema_9 < ema_21 and crossover_strength > 0.1:  # Bearish crossover
            # Additional bearish filters
            if (rsi < 60 and rsi > 20 and  # RSI not oversold
                macd < macd_signal and     # MACD bearish
                volume_ratio > 0.8):       # Decent volume
                
                signal = "BUY PUT"
                # OPTIMIZATION: Better confidence calculation
                confidence_score = 60 + min(20, crossover_strength * 10) + min(10, (60 - rsi) / 2)
        
        # OPTIMIZATION: Minimum confidence threshold
        if confidence_score < 50:
            return None
            
        # Determine confidence level
        if confidence_score >= 90:
            confidence = "Very High"
        elif confidence_score >= 85:
            confidence = "High"
        elif confidence_score >= 70:
            confidence = "Medium"
        elif confidence_score >= 50:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        # OPTIMIZATION: Improved risk-reward ratios based on confidence
        if confidence_score >= 80:
            stop_loss = int(round(1.5 * atr))
            target1 = int(round(3.0 * atr))  # 2:1 R:R
            target2 = int(round(4.5 * atr))  # 3:1 R:R
            target3 = int(round(6.0 * atr))  # 4:1 R:R
        elif confidence_score >= 70:
            stop_loss = int(round(2.0 * atr))
            target1 = int(round(3.5 * atr))  # 1.75:1 R:R
            target2 = int(round(5.0 * atr))  # 2.5:1 R:R
            target3 = int(round(6.5 * atr))  # 3.25:1 R:R
        else:  # 50-69
            stop_loss = int(round(2.5 * atr))
            target1 = int(round(4.0 * atr))  # 1.6:1 R:R
            target2 = int(round(5.5 * atr))  # 2.2:1 R:R
            target3 = int(round(7.0 * atr))  # 2.8:1 R:R
        
        # OPTIMIZATION: Reduced position size for better risk management
        position_multiplier = 0.5 if confidence_score >= 80 else 0.3
        
        # Calculate performance if we have future data
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            price = candle['close']
            
            # Check future prices to see if targets or stop loss were hit
            if signal == "BUY CALL":
                # For buy calls, check if price went up to targets or down to stop loss
                max_future_price = future_data['high'].max()
                min_future_price = future_data['low'].min()
                
                # Check if stop loss was hit
                if min_future_price <= (price - stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss * position_multiplier
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price - stop_loss}"
                else:
                    outcome = "Win"
                    # Check which targets were hit
                    if max_future_price >= (price + target1):
                        targets_hit += 1
                        pnl += target1 * position_multiplier
                    if max_future_price >= (price + target2):
                        targets_hit += 1
                        pnl += (target2 - target1) * position_multiplier
                    if max_future_price >= (price + target3):
                        targets_hit += 1
                        pnl += (target3 - target2) * position_multiplier
                    
            elif signal == "BUY PUT":
                # For buy puts, check if price went down to targets or up to stop loss
                max_future_price = future_data['high'].max()
                min_future_price = future_data['low'].min()
                
                # Check if stop loss was hit
                if max_future_price >= (price + stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss * position_multiplier
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price + stop_loss}"
                else:
                    outcome = "Win"
                    # Check which targets were hit
                    if min_future_price <= (price - target1):
                        targets_hit += 1
                        pnl += target1 * position_multiplier
                    if min_future_price <= (price - target2):
                        targets_hit += 1
                        pnl += (target2 - target1) * position_multiplier
                    if min_future_price <= (price - target3):
                        targets_hit += 1
                        pnl += (target3 - target2) * position_multiplier
        
        # Build reasoning string
        price_reason = f"EMA crossover: {ema_9:.1f} {'>' if signal == 'BUY CALL' else '<'} {ema_21:.1f}"
        price_reason += f", Strength: {crossover_strength:.2f}%"
        price_reason += f", Volume: {volume_ratio:.1f}x"
        price_reason += f", MACD: {macd:.2f} vs {macd_signal:.2f}"
        price_reason += f", RSI: {rsi:.1f}"
        price_reason += f", Confidence: {confidence_score}"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "price": candle['close'],
            "stop_loss": stop_loss,
            "target": target1,
            "target2": target2,
            "target3": target3,
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "reasoning": price_reason
        }
        
    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics based on future data with trailing stop after target1, and let profits run after target3."""
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
            highest_price = entry_price
            trailing_sl = None
            target1_hit = target2_hit = target3_hit = False
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(idx)
                # Check if stop loss was hit before target1
                if not target1_hit and candle['low'] <= (entry_price - stop_loss):
                    outcome = "Loss"
                    pnl = -1.0 * stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price - stop_loss:.2f}"
                    exit_time = current_time
                    break
                # Check if target1 is hit
                if not target1_hit and candle['high'] >= (entry_price + target):
                    target1_hit = True
                    targets_hit = 1
                    highest_price = max(highest_price, candle['high'])
                    trailing_sl = highest_price - stop_loss
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                    continue
                # Check if target2 is hit
                if target1_hit and not target2_hit and candle['high'] >= (entry_price + target2):
                    target2_hit = True
                    targets_hit = 2
                    highest_price = max(highest_price, candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    pnl = target2
                    exit_time = current_time
                # Check if target3 is hit
                if target2_hit and not target3_hit and candle['high'] >= (entry_price + target3):
                    target3_hit = True
                    targets_hit = 3
                    highest_price = max(highest_price, candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    pnl = target3
                    exit_time = current_time
                    break
                # After target1, always trail SL at highest_price - stop_loss
                if target1_hit:
                    highest_price = max(highest_price, candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    # If price hits trailing SL, exit
                    if candle['low'] <= trailing_sl:
                        outcome = "Win"
                        pnl = trailing_sl - entry_price
                        failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                        exit_time = current_time
                        break
        elif signal == "BUY PUT":
            lowest_price = entry_price
            trailing_sl = None
            target1_hit = target2_hit = target3_hit = False
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(idx)
                # Check if stop loss was hit before target1
                if not target1_hit and candle['high'] >= (entry_price + stop_loss):
                    outcome = "Loss"
                    pnl = -1.0 * stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price + stop_loss:.2f}"
                    exit_time = current_time
                    break
                # Check if target1 is hit
                if not target1_hit and candle['low'] <= (entry_price - target):
                    target1_hit = True
                    targets_hit = 1
                    lowest_price = min(lowest_price, candle['low'])
                    trailing_sl = lowest_price + stop_loss
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                    continue
                # Check if target2 is hit
                if target1_hit and not target2_hit and candle['low'] <= (entry_price - target2):
                    target2_hit = True
                    targets_hit = 2
                    lowest_price = min(lowest_price, candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    pnl = target2
                    exit_time = current_time
                # Check if target3 is hit
                if target2_hit and not target3_hit and candle['low'] <= (entry_price - target3):
                    target3_hit = True
                    targets_hit = 3
                    lowest_price = min(lowest_price, candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    pnl = target3
                    exit_time = current_time
                    break
                # After target1, always trail SL at lowest_price + stop_loss
                if target1_hit:
                    lowest_price = min(lowest_price, candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    # If price hits trailing SL, exit
                    if candle['high'] >= trailing_sl:
                        outcome = "Win"
                        pnl = entry_price - trailing_sl
                        failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                        exit_time = current_time
                        break
        # Defensive IST conversion for exit_time
        exit_time_str = self.to_ist_str(exit_time) or (str(exit_time) if exit_time is not None else None)
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time_str
        }
        
    def _evaluate_timeframe(self, df: pd.DataFrame, timeframe: str, ts: datetime) -> Optional[Dict[str, Any]]:
        """Evaluate a specific timeframe for signal confirmation."""
        df = df[df.index <= ts].copy()
        if df.empty or len(df) < 25:  # Need at least 25 candles for EMA indicators
            return None

        # Add indicators to this timeframe data
        try:
            df = self.add_indicators(df)
        except (IndexError, ValueError):
            # Not enough data for indicators
            return None
            
        candle = df.iloc[-1]

        # Check for crossover signals in this timeframe
        signal_direction = 0
        
        # BUY CALL on bullish crossover
        if (candle['ema_fast'] > candle['ema_slow'] and 
            candle['close'] > candle['ema_fast'] and 
            candle.get('crossover_strength', 0) > 0):
            signal_direction = 1
            
        # BUY PUT on bearish crossover
        elif (candle['ema_fast'] < candle['ema_slow'] and 
              candle['close'] < candle['ema_fast'] and 
              candle.get('crossover_strength', 0) < 0):
            signal_direction = -1

        return {
            "signal_direction": signal_direction,
            "ema_fast": candle['ema_fast'],
            "ema_slow": candle['ema_slow'],
            "crossover_strength": candle.get('crossover_strength', 0),
            "rsi": candle.get('rsi', 50),
            "candle": candle
        }

    def analyze_multi_timeframe(self, candle: pd.Series, index: int, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Multi-timeframe analysis method."""
        # Ensure DataFrame has DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            else:
                raise ValueError("DataFrame must have a datetime index or 'time' column")

        if index >= len(df):
            return None

        ts = df.index[index]
        
        # If we have timeframe data, use multi-timeframe analysis
        if self.timeframe_data:
            results = []
            timeframes = {
                "3min": df, 
                "15min": self.timeframe_data.get("15min"), 
                "30min": self.timeframe_data.get("30min")
            }
            
            for tf, tf_df in timeframes.items():
                if tf_df is None or tf_df.empty:
                    # If any timeframe is missing, fall back to single timeframe
                    break
                tf_result = self._evaluate_timeframe(tf_df, tf, ts)
                if tf_result is None:
                    break
                results.append(tf_result)
            
            # If we have all timeframe results, use multi-timeframe logic
            if len(results) == 3:
                bullish_votes = sum(1 for r in results if r["signal_direction"] == 1)
                bearish_votes = sum(1 for r in results if r["signal_direction"] == -1)
                
                if bullish_votes >= 2:
                    signal = "BUY CALL"
                    confidence = "High" if bullish_votes == 3 else "Medium"
                elif bearish_votes >= 2:
                    signal = "BUY PUT"
                    confidence = "High" if bearish_votes == 3 else "Medium"
                else:
                    return None  # No clear signal across timeframes
                
                # Use base timeframe (3min) candle for calculations
                base_candle = results[0]["candle"]
                
                # Calculate performance if future data is available
                outcome = "Pending"
                pnl = 0.0
                targets_hit = 0
                stoploss_count = 0
                failure_reason = ""
                exit_time = ""
                
                if future_data is not None and not future_data.empty:
                    atr = base_candle.get('atr', 0)
                    stop_loss = round(atr * 1.0, 2) if atr > 0 else 0
                    target = round(atr * 1.5, 2) if atr > 0 else 0
                    target2 = round(atr * 2.0, 2) if atr > 0 else 0
                    target3 = round(atr * 2.5, 2) if atr > 0 else 0
                    
                    result = self.calculate_performance(
                        signal, base_candle['close'], stop_loss, target, target2, target3, future_data
                    )
                    outcome = result['outcome']
                    pnl = result['pnl']
                    targets_hit = result['targets_hit']
                    stoploss_count = result['stoploss_count']
                    failure_reason = result['failure_reason']
                    exit_time = result['exit_time']
                
                return {
                    "signal": signal,
                    "price": base_candle['close'],
                    "confidence": confidence,
                    "trade_type": "Intraday",
                    "ema_fast": base_candle['ema_fast'],
                    "ema_slow": base_candle['ema_slow'],
                    "crossover_strength": base_candle.get('crossover_strength', 0),
                    "stop_loss": round(base_candle.get('atr', 0) * 1.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target": round(base_candle.get('atr', 0) * 1.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target2": round(base_candle.get('atr', 0) * 2.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target3": round(base_candle.get('atr', 0) * 2.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "rsi": base_candle.get('rsi', 50),
                    "rsi_reason": f"Multi-timeframe confirmation ({bullish_votes if signal == 'BUY CALL' else bearish_votes}/3 timeframes)",
                    "macd_reason": "",
                    "price_reason": f"EMA crossover confirmed across {bullish_votes if signal == 'BUY CALL' else bearish_votes} timeframes",
                    "outcome": outcome,
                    "pnl": pnl,
                    "targets_hit": targets_hit,
                    "stoploss_count": stoploss_count,
                    "failure_reason": failure_reason,
                    "exit_time": exit_time
                }
        
        # Fall back to single timeframe analysis if multi-timeframe data not available
        return self.analyze_single_timeframe(df.iloc[index:index+1], future_data)

# Backward compatibility function
def run_strategy(candle, index_name, future_data=None, crossover_strength=None, momentum=None):
    """Legacy wrapper function for backward compatibility with function-based approach."""
    strategy = EmaCrossover({
        'crossover_strength': crossover_strength,
        'momentum': momentum
    })
    
    # The framework passes individual candles, but we need the full DataFrame context
    # This is a limitation of the function-based approach
    # Return NO TRADE for now to avoid errors
    return {
        "signal": "NO TRADE",
        "price": candle.get('close', 0) if hasattr(candle, 'get') else candle['close'],
        "confidence": "N/A",
        "reason": "Function-based approach not supported - use class-based approach instead"
    }
