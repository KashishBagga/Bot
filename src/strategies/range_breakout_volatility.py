"""
Range Breakout Volatility strategy.
Trading strategy based on price breakouts from established ranges with volatility filters.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from db import log_strategy_sql
import re

class RangeBreakoutVolatility(Strategy):
    """
    Multi-timeframe Range Breakout with Volatility strategy with signal confirmation across 3min, 15min, and 30min charts.
    """
    
    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including range_period, breakout_threshold, volatility_threshold
            timeframe_data: Dictionary of timeframes with their respective data
        """
        params = params or {}
        # Reduced range period from 10 to 5 for more frequent ranges
        self.range_period = params.get('range_period', 5)
        # Reduced breakout threshold from 0.3 to 0.5
        self.breakout_threshold = params.get('breakout_threshold', 0.5)
        # Reduced volatility threshold from 1.2 to 1.0
        self.volatility_threshold = params.get('volatility_threshold', 1.2)
        # Added flag to include signals even without volatility confirmation
        self.require_volatility = params.get('require_volatility', False)
        self.timeframe_data = timeframe_data or {}
        super().__init__("range_breakout_volatility", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Calculate range indicators
        period = self.range_period
        
        # Calculate range high and low
        if 'range_high' not in data.columns:
            data['range_high'] = data['high'].rolling(period).max().shift(1)
            data['range_low'] = data['low'].rolling(period).min().shift(1)
            data['range_width'] = (data['range_high'] - data['range_low']) / data['close'] * 100
            
        # Calculate breakout size
        if 'breakout_size' not in data.columns:
            data['breakout_size'] = np.where(
                data['close'] > data['range_high'],
                (data['close'] - data['range_high']) / data['close'] * 100,
                np.where(
                    data['close'] < data['range_low'],
                    (data['range_low'] - data['close']) / data['close'] * 100,
                    0
                )
            )
            
        # Calculate ATR-based volatility if not present
        if 'atr' not in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            # Calculate ATR
            data['tr'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )
            data['atr'] = data['tr'].rolling(14).mean()
            
        # Calculate volatility metrics
        if 'atr' in data.columns:
            data['atr_ratio'] = data['atr'] / data['atr'].rolling(10).mean()
            
            # Calculate ATR as percentage of price
            data['atr_percent'] = data['atr'] / data['close'] * 100
            
            # Calculate volatility rank (percentile of current ATR vs history)
            data['volatility_rank'] = data['atr_percent'].rolling(50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
            )
            
        return data
    
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
                current_time = self.safe_signal_time(candle.get('time', None))
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
                current_time = self.safe_signal_time(candle.get('time', None))
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
        exit_time_str = None
        if isinstance(exit_time, (pd.Timestamp, datetime)):
            ist_dt = exit_time + timedelta(hours=5, minutes=30)
            exit_time_str = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        elif exit_time is not None:
            exit_time_str = str(exit_time)
        
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time_str
        }
    
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
        """Single timeframe analysis method."""
        # Ensure 'time' column exists and is valid, and set as index
        if 'time' in data.columns:
            data = data.copy()
            data.loc[:, 'time'] = pd.to_datetime(data['time'], errors='coerce')
            data = data.set_index('time')
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Ensure indicators are calculated
        data = self.add_indicators(data)
        
        # Get the latest candle
        candle = data.iloc[-1]
        
        # Set default values
        signal = "NO TRADE"
        confidence = "Low"
        trade_type = "Intraday"
        rsi_reason = macd_reason = price_reason = ""
        
        # Performance tracking variables
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        # Get parameters
        breakout_threshold = self.breakout_threshold
        volatility_threshold = self.volatility_threshold
        
        # Get breakout size
        breakout_size = candle.get('breakout_size', 0)
        
        # Get volatility metrics
        atr = candle.get('atr', 0)
        atr_ratio = candle.get('atr_ratio', 1.0)
        volatility_rank = candle.get('volatility_rank', 50.0)
        
        # Calculate stop loss and targets based on ATR
        stop_loss = round(atr * 1.0, 2) if atr > 0 else 0
        target = round(atr * 1.5, 2) if atr > 0 else 0
        target2 = round(atr * 2.0, 2) if atr > 0 else 0
        target3 = round(atr * 2.5, 2) if atr > 0 else 0
        
        # Check for upside breakout
        if candle['close'] > candle.get('range_high', float('inf')):
            # Check if we meet breakout size threshold
            if breakout_size >= breakout_threshold:
                # Check if we need volatility confirmation
                if not self.require_volatility or atr_ratio >= volatility_threshold:
                    signal = "BUY CALL"
                    confidence = "Medium"
                    price_reason = f"Upside range breakout: {breakout_size:.2f}%"
                    
                    if atr_ratio >= volatility_threshold:
                        price_reason += f" with volatility ratio: {atr_ratio:.2f}x"
                        confidence = "High"
                    
                    if volatility_rank > 70:
                        confidence = "High"
                        price_reason += f", High volatility rank: {volatility_rank:.1f}"
                
        # Check for downside breakout
        elif candle['close'] < candle.get('range_low', 0):
            # Check if we meet breakout size threshold
            if breakout_size >= breakout_threshold:
                # Check if we need volatility confirmation
                if not self.require_volatility or atr_ratio >= volatility_threshold:
                    signal = "BUY PUT"
                    confidence = "Medium"
                    price_reason = f"Downside range breakdown: {breakout_size:.2f}%"
                    
                    if atr_ratio >= volatility_threshold:
                        price_reason += f" with volatility ratio: {atr_ratio:.2f}x"
                        confidence = "High"
                    
                    if volatility_rank > 70:
                        confidence = "High"
                        price_reason += f", High volatility rank: {volatility_rank:.1f}"
            
        # RSI confirmation
        if signal != "NO TRADE" and 'rsi' in candle:
            if signal == "BUY CALL" and candle['rsi'] > 60:
                rsi_reason = f"Strong RSI momentum: {candle['rsi']:.1f}"
                confidence = "High"
            elif signal == "BUY PUT" and candle['rsi'] < 40:
                rsi_reason = f"Weak RSI momentum: {candle['rsi']:.1f}"
                confidence = "High"
        
        # Calculate performance metrics if a signal was generated and future data is available
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            entry_price = candle['close']
            performance = self.calculate_performance(
                signal=signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                target2=target2,
                target3=target3,
                future_data=future_data
            )
            
            # Update performance metrics
            outcome = performance["outcome"]
            pnl = performance["pnl"]
            targets_hit = performance["targets_hit"]
            stoploss_count = performance["stoploss_count"]
            failure_reason = performance["failure_reason"]
            exit_time = performance["exit_time"]
        
        # Create the signal data dictionary
        signal_data = {
            "signal": signal,
            "price": candle['close'],
            "atr": atr,
            "volatility_rank": volatility_rank,
            "range_width": candle.get('range_width', 0),
            "breakout_size": breakout_size,
            "confidence": confidence,
            "price_reason": price_reason,
            "trade_type": trade_type,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": self.to_ist_str(exit_time) or (str(exit_time) if exit_time is not None else None)
        }
        
        return signal_data
    
    def _evaluate_timeframe(self, df: pd.DataFrame, timeframe: str, ts: datetime) -> Optional[Dict[str, Any]]:
        """Evaluate a specific timeframe for signal confirmation."""
        df = df[df.index <= ts].copy()
        if df.empty or len(df) < 20:  # Need at least 20 candles for indicators
            return None

        # Add indicators to this timeframe data
        try:
            df = self.add_indicators(df)
        except (IndexError, ValueError):
            # Not enough data for indicators
            return None
            
        candle = df.iloc[-1]

        # Check for range breakout signals in this timeframe
        signal_direction = 0
        breakout_size = candle.get('breakout_size', 0)
        atr_ratio = candle.get('atr_ratio', 1.0)
        
        # BUY CALL on upward breakout
        if (breakout_size > self.breakout_threshold and 
            candle['close'] > candle['range_high'] and
            (not self.require_volatility or atr_ratio >= self.volatility_threshold)):
            signal_direction = 1
            
        # BUY PUT on downward breakout
        elif (breakout_size > self.breakout_threshold and 
              candle['close'] < candle['range_low'] and
              (not self.require_volatility or atr_ratio >= self.volatility_threshold)):
            signal_direction = -1

        return {
            "signal_direction": signal_direction,
            "breakout_size": breakout_size,
            "atr_ratio": atr_ratio,
            "range_high": candle.get('range_high', 0),
            "range_low": candle.get('range_low', 0),
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
                    "breakout_size": base_candle.get('breakout_size', 0),
                    "atr_ratio": base_candle.get('atr_ratio', 1.0),
                    "range_high": base_candle.get('range_high', 0),
                    "range_low": base_candle.get('range_low', 0),
                    "stop_loss": round(base_candle.get('atr', 0) * 1.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target": round(base_candle.get('atr', 0) * 1.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target2": round(base_candle.get('atr', 0) * 2.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target3": round(base_candle.get('atr', 0) * 2.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "rsi": base_candle.get('rsi', 50),
                    "rsi_reason": f"Multi-timeframe confirmation ({bullish_votes if signal == 'BUY CALL' else bearish_votes}/3 timeframes)",
                    "macd_reason": "",
                    "price_reason": f"Range breakout confirmed across {bullish_votes if signal == 'BUY CALL' else bearish_votes} timeframes",
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
def run_strategy(candle, prev_candle=None, index_name=None, future_data=None, range_period=5, breakout_threshold=0.1, volatility_threshold=1.0, require_volatility=False):
    """Legacy wrapper function for backward compatibility with function-based approach."""
    strategy = RangeBreakoutVolatility({
        'range_period': range_period,
        'breakout_threshold': breakout_threshold,
        'volatility_threshold': volatility_threshold,
        'require_volatility': require_volatility
    })
    
    # Create a single-row DataFrame from the candle
    if not isinstance(candle, pd.DataFrame):
        import pandas as pd
        data = pd.DataFrame([candle])
    else:
        data = candle

    # Only set 'time' as index if it is a valid datetime
    if 'time' in data.columns:
        data = data.copy()
        data.loc[:, 'time'] = pd.to_datetime(data['time'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(data['time']):
            data = data.set_index('time')
    return strategy.analyze(data, index_name, future_data)
