"""
Breakout RSI strategy.
Trading strategy based on price breakouts with RSI confirmation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from db import log_strategy_sql
import re

class BreakoutRsi(Strategy):
    """
    Multi-timeframe Breakout RSI strategy with signal confirmation across 3min, 15min, and 30min charts.
    """
    
    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including breakout_strength, rsi_alignment, lookback_period
            timeframe_data: Optional dictionary of timeframes with market data
        """
        params = params or {}
        # Reduced breakout strength from 0.1 to 0.01 (even more lenient)
        self.breakout_strength = params.get('breakout_strength', 0.01)
        # Reduced RSI alignment threshold from 40 to 30
        self.rsi_alignment = params.get('rsi_alignment', 30)
        # Reduced lookback period from 5 to 3
        self.lookback_period = params.get('lookback_period', 3)
        # Added parameter for requiring RSI confirmation (disabled by default)
        self.require_rsi = params.get('require_rsi', False)
        self.breakout_threshold = params.get("breakout_threshold", 0.01) if params else 0.01
        self.rsi_overbought = params.get("rsi_overbought", 60) if params else 60
        self.rsi_oversold = params.get("rsi_oversold", 40) if params else 40
        self.timeframe_data = timeframe_data or {}
        super().__init__("breakout_rsi", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Calculate previous high and low
        period = self.lookback_period
        if 'prev_high' not in data.columns:
            data['prev_high'] = data['high'].rolling(period).max().shift(1)
            data['prev_low'] = data['low'].rolling(period).min().shift(1)
            
        # Calculate breakout strength
        if 'breakout_strength' not in data.columns:
            data['breakout_strength'] = np.where(
                data['close'] > data['prev_high'],
                (data['close'] - data['prev_high']) / data['close'] * 100,
                np.where(
                    data['close'] < data['prev_low'],
                    (data['prev_low'] - data['close']) / data['close'] * 100,
                    0
                )
            )
            
        # RSI alignment check
        if 'rsi' in data.columns:
            data['rsi_alignment'] = np.where(
                data['close'] > data['prev_high'],
                data['rsi'],  # Higher is better for upside breakouts
                np.where(
                    data['close'] < data['prev_low'],
                    100 - data['rsi'],  # Lower RSI is better for downside breakouts
                    50  # Neutral when no breakout
                )
            )
            
        return data
    
    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for future candles."""
        if future_data is None or future_data.empty:
            return {
                "outcome": "Pending", 
                "pnl": 0.0, 
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": "",
                "exit_time": None
            }
        
        # Initialize performance metrics
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        # For BUY CALL, check if future prices went up to targets or down to stop loss
        if signal == "BUY CALL":
            # Process each future candle chronologically
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(candle.get('time', None))
                
                # Check if stop loss was hit
                if candle['low'] <= (entry_price - stop_loss):
                    outcome = "Loss"
                    pnl = -1.0 * stop_loss  # Negative value for stop loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price - stop_loss:.2f}"
                    exit_time = current_time
                    break  # Exit the loop as trade is closed
                
                # Check which targets were hit
                if targets_hit == 0 and candle['high'] >= (entry_price + target):
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                
                if targets_hit == 1 and candle['high'] >= (entry_price + target2):
                    targets_hit = 2
                    pnl += (target2 - target)
                
                if targets_hit == 2 and candle['high'] >= (entry_price + target3):
                    targets_hit = 3
                    pnl += (target3 - target2)
                    exit_time = current_time
                    break  # Exit the loop as all targets are hit
        
        # For BUY PUT, check if future prices went down to targets or up to stop loss
        elif signal == "BUY PUT":
            # Process each future candle chronologically
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(candle.get('time', None))
                
                # Check if stop loss was hit
                if candle['high'] >= (entry_price + stop_loss):
                    outcome = "Loss"
                    pnl = -1.0 * stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price + stop_loss:.2f}"
                    exit_time = current_time
                    break  # Exit the loop as trade is closed
                
                # Check which targets were hit
                if targets_hit == 0 and candle['low'] <= (entry_price - target):
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                
                if targets_hit == 1 and candle['low'] <= (entry_price - target2):
                    targets_hit = 2
                    pnl += (target2 - target)
                
                if targets_hit == 2 and candle['low'] <= (entry_price - target3):
                    targets_hit = 3
                    pnl += (target3 - target2)
                    exit_time = current_time
                    break  # Exit the loop as all targets are hit
        
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
        rsi_reason = price_reason = ""
        
        # Performance tracking variables
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        # Get parameters
        breakout_strength_threshold = self.breakout_strength
        rsi_alignment_threshold = self.rsi_alignment
        
        # Calculate ATR for stops and targets if available
        atr = candle.get('atr', 0)
        stop_loss = round(atr * 1.0, 2) if atr > 0 else 0
        target = round(atr * 1.5, 2) if atr > 0 else 0
        target2 = round(atr * 2.0, 2) if atr > 0 else 0
        target3 = round(atr * 2.5, 2) if atr > 0 else 0
        
        # Check for breakouts - BUY CALL on upside breakout
        prev_high = candle.get('prev_high', float('-inf'))
        prev_low = candle.get('prev_low', float('inf'))
        
        # Make breakout detection more lenient - check if we have valid prev_high/prev_low
        if pd.isna(prev_high) or prev_high == float('-inf'):
            prev_high = candle['close'] * 0.999  # Use a very close value if no prev_high
        if pd.isna(prev_low) or prev_low == float('inf'):
            prev_low = candle['close'] * 1.001  # Use a very close value if no prev_low
            
        if candle['close'] > prev_high:
            # Very lenient breakout condition - any movement above prev_high
            signal = "BUY CALL"
            confidence = "Medium"
            price_reason = f"Breakout above resistance: {candle.get('breakout_strength', 0):.2f}%"
            
            # RSI confirmation enhances confidence
            if candle.get('rsi', 0) > 50:
                confidence = "High"
                rsi_reason = f"Strong RSI momentum: {candle.get('rsi', 0):.1f}"
                
        # Check for breakouts - BUY PUT on downside breakout
        elif candle['close'] < prev_low:
            # Very lenient breakout condition - any movement below prev_low
            signal = "BUY PUT"
            confidence = "Medium"
            price_reason = f"Breakdown below support: {candle.get('breakout_strength', 0):.2f}%"
            
            # RSI confirmation enhances confidence
            if candle.get('rsi', 0) < 50:
                confidence = "High"
                rsi_reason = f"Weak RSI momentum: {candle.get('rsi', 0):.1f}"
        
        # Alternative condition: Simple RSI-based signals when no clear breakout
        elif signal == "NO TRADE":
            # BUY CALL on strong RSI momentum (lowered threshold)
            if candle.get('rsi', 0) > 60:
                signal = "BUY CALL"
                confidence = "Low"
                rsi_reason = f"Strong RSI momentum: {candle.get('rsi', 0):.1f}"
                price_reason = "RSI-based signal (no clear breakout)"
            # BUY PUT on weak RSI momentum (raised threshold)
            elif candle.get('rsi', 0) < 40:
                signal = "BUY PUT"
                confidence = "Low"
                rsi_reason = f"Weak RSI momentum: {candle.get('rsi', 0):.1f}"
                price_reason = "RSI-based signal (no clear breakout)"
        
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
            "rsi": candle.get('rsi', 0),
            "confidence": confidence,
            "trade_type": trade_type,
            "breakout_strength": candle.get('breakout_strength', 0),
            "rsi_alignment": candle.get('rsi_alignment', 0),
            "stop_loss": stop_loss,
            "target": target,
            "target2": target2,
            "target3": target3,
            "rsi_reason": rsi_reason,
            "price_reason": price_reason,
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
        if df.empty:
            return None

        # Add indicators to this timeframe data
        df = self.add_indicators(df)
        candle = df.iloc[-1]

        # Check for breakout signals with RSI confirmation in this timeframe
        signal_direction = 0
        breakout_size = candle.get('breakout_size', 0)
        rsi = candle.get('rsi', 50)
        
        # BUY CALL on upward breakout with RSI confirmation
        if (breakout_size > self.breakout_threshold and 
            candle['close'] > candle.get('resistance', candle['close']) and
            (not self.require_rsi or rsi > self.rsi_overbought)):
            signal_direction = 1
            
        # BUY PUT on downward breakout with RSI confirmation
        elif (breakout_size > self.breakout_threshold and 
              candle['close'] < candle.get('support', candle['close']) and
              (not self.require_rsi or rsi < self.rsi_oversold)):
            signal_direction = -1

        return {
            "signal_direction": signal_direction,
            "breakout_size": breakout_size,
            "rsi": rsi,
            "resistance": candle.get('resistance', 0),
            "support": candle.get('support', 0),
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
                    "rsi": base_candle.get('rsi', 50),
                    "resistance": base_candle.get('resistance', 0),
                    "support": base_candle.get('support', 0),
                    "stop_loss": round(base_candle.get('atr', 0) * 1.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target": round(base_candle.get('atr', 0) * 1.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target2": round(base_candle.get('atr', 0) * 2.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target3": round(base_candle.get('atr', 0) * 2.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "rsi_reason": f"Multi-timeframe confirmation ({bullish_votes if signal == 'BUY CALL' else bearish_votes}/3 timeframes)",
                    "macd_reason": "",
                    "price_reason": f"Breakout with RSI confirmed across {bullish_votes if signal == 'BUY CALL' else bearish_votes} timeframes",
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
def run_strategy(candle, prev_candle=None, index_name=None, future_data=None, breakout_strength=0.1, rsi_alignment=40, lookback_period=5, require_rsi=False):
    """Legacy wrapper function for backward compatibility with function-based approach."""
    strategy = BreakoutRsi({
        'breakout_strength': breakout_strength,
        'rsi_alignment': rsi_alignment,
        'lookback_period': lookback_period,
        'require_rsi': require_rsi
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
