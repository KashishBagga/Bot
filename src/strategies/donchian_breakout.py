"""
Donchian Channel Breakout strategy.
Trading strategy based on breakouts from Donchian channels.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from db import log_strategy_sql

class DonchianBreakout(Strategy):
    """
    Multi-timeframe Donchian Channel Breakout strategy with signal confirmation across 3min, 15min, and 30min charts.
    """
    
    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including channel_period, breakout_strength, volume_trigger
            timeframe_data: Optional dictionary of timeframes with their corresponding data
        """
        super().__init__("donchian_breakout", params)
        self.channel_period = params.get("channel_period", 10) if params else 10
        self.breakout_strength = params.get("breakout_strength", 0.1) if params else 0.1
        self.volume_trigger = params.get("volume_trigger", 1.0) if params else 1.0
        self.timeframe_data = timeframe_data or {}
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Calculate Donchian Channels if not already present
        if 'donchian_upper' not in data.columns:
            period = self.channel_period
            data['donchian_upper'] = data['high'].rolling(period).max()
            data['donchian_lower'] = data['low'].rolling(period).min()
            data['donchian_middle'] = (data['donchian_upper'] + data['donchian_lower']) / 2
            
        # Calculate channel width as percentage of price
        if 'channel_width' not in data.columns:
            data['channel_width'] = (data['donchian_upper'] - data['donchian_lower']) / data['close'] * 100
            
        # Calculate breakout size
        data['prev_upper'] = data['donchian_upper'].shift(1)
        data['prev_lower'] = data['donchian_lower'].shift(1)
        data['breakout_size'] = np.where(
            data['close'] > data['prev_upper'],
            (data['close'] - data['prev_upper']) / data['close'] * 100,
            np.where(
                data['close'] < data['prev_lower'],
                (data['prev_lower'] - data['close']) / data['close'] * 100,
                0
            )
        )
        
        # Calculate volume ratio
        if 'volume' in data.columns:
            data['avg_volume'] = data['volume'].rolling(10).mean()
            data['volume_ratio'] = data['volume'] / data['avg_volume']
        else:
            data['volume_ratio'] = 1.0
            
        return data
    
    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics based on future data.
        
        Args:
            signal: The trading signal (BUY CALL or BUY PUT)
            entry_price: The price at signal generation
            stop_loss: The stop loss price
            target: The first target price
            target2: The second target price
            target3: The third target price
            future_data: Future candles after signal generation
            
        Returns:
            Dict containing outcome, pnl, targets_hit, stoploss_count, and failure_reason
        """
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
        exit_time = None  # Initialize exit_time here
        
        # Calculate stop loss and target prices
        if signal == "BUY CALL":
            stop_loss_price = entry_price - stop_loss
            target1_price = entry_price + target
            target2_price = entry_price + target2
            target3_price = entry_price + target3
            
            # Process each future candle chronologically
            for i, candle in future_data.iterrows():
                # Check if stop loss is hit first
                if candle['low'] <= stop_loss_price:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                    exit_time = self.safe_signal_time(candle.get('time', None))
                    break  # Exit the loop as trade is closed
                
                # Check which targets are hit
                if targets_hit == 0 and candle['high'] >= target1_price:
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                
                if targets_hit == 1 and candle['high'] >= target2_price:
                    targets_hit = 2
                    pnl += (target2 - target)
                
                if targets_hit == 2 and candle['high'] >= target3_price:
                    targets_hit = 3
                    pnl += (target3 - target2)
                
        elif signal == "BUY PUT":
            stop_loss_price = entry_price + stop_loss
            target1_price = entry_price - target
            target2_price = entry_price - target2
            target3_price = entry_price - target3
            
            # Process each future candle chronologically
            for i, candle in future_data.iterrows():
                # Check if stop loss is hit first
                if candle['high'] >= stop_loss_price:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                    exit_time = self.safe_signal_time(candle.get('time', None))
                    break  # Exit the loop as trade is closed
                
                # Check which targets are hit
                if targets_hit == 0 and candle['low'] <= target1_price:
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                
                if targets_hit == 1 and candle['low'] <= target2_price:
                    targets_hit = 2
                    pnl += (target2 - target)
                
                if targets_hit == 2 and candle['low'] <= target3_price:
                    targets_hit = 3
                    pnl += (target3 - target2)
                
        # Defensive IST conversion for exit_time
        exit_time_str = None
        if isinstance(exit_time, (pd.Timestamp, datetime)):
            ist_dt = exit_time + timedelta(hours=5, minutes=30)
            exit_time_str = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        elif exit_time is not None:
            exit_time_str = str(exit_time)
        return {
            "outcome": outcome,
            "pnl": round(pnl, 2),
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
        exit_time = None  # Initialize exit_time here
        
        # Get parameters
        breakout_strength = self.breakout_strength
        volume_trigger = self.volume_trigger
        
        # Calculate ATR for stops and targets
        atr = candle.get('atr', 0)
        stop_loss = round(atr * 1.0, 2) if atr > 0 else 0
        target = round(atr * 1.5, 2) if atr > 0 else 0
        target2 = round(atr * 2.0, 2) if atr > 0 else 0
        target3 = round(atr * 2.5, 2) if atr > 0 else 0
        
        # Check for breakouts
        # BUY CALL on upper channel breakout - simplified condition
        if candle['close'] > candle['prev_upper']:
            
            signal = "BUY CALL"
            confidence = "Medium"
            price_reason = f"Breakout above Donchian channel: {candle.get('breakout_size', 0):.2f}% with volume ratio: {candle.get('volume_ratio', 1.0):.2f}x"
            
            # Add breakout_size and volume checks for confidence
            if candle.get('breakout_size', 0) >= breakout_strength and candle.get('volume_ratio', 1.0) >= volume_trigger:
                confidence = "High"
            
            if candle.get('rsi', 0) > 60:
                confidence = "High"
                rsi_reason = f"Strong RSI momentum: {candle.get('rsi', 0):.1f}"
                
        # BUY PUT on lower channel breakout - simplified condition
        elif candle['close'] < candle['prev_lower']:
            
            signal = "BUY PUT"
            confidence = "Medium"
            price_reason = f"Breakdown below Donchian channel: {candle.get('breakout_size', 0):.2f}% with volume ratio: {candle.get('volume_ratio', 1.0):.2f}x"
            
            # Add breakout_size and volume checks for confidence
            if candle.get('breakout_size', 0) >= breakout_strength and candle.get('volume_ratio', 1.0) >= volume_trigger:
                confidence = "High"
            
            if candle.get('rsi', 0) < 40:
                confidence = "High"
                rsi_reason = f"Strong RSI momentum: {candle.get('rsi', 0):.1f}"
                
        # Calculate performance metrics with trailing stop logic
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            if signal == "BUY CALL":
                stop_loss_price = candle['close'] - stop_loss
                target1_price = candle['close'] + target
                target2_price = candle['close'] + target2
                target3_price = candle['close'] + target3

                highest_price = candle['close']
                trailing_sl = None
                target1_hit = target2_hit = target3_hit = False
                for i, future_candle in future_data.iterrows():
                    if not target1_hit and future_candle['low'] <= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        break
                    if not target1_hit and future_candle['high'] >= target1_price:
                        target1_hit = True
                        targets_hit = 1
                        pnl = target
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = highest_price - stop_loss
                        outcome = "Win"
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        continue
                    if target1_hit and not target2_hit and future_candle['high'] >= target2_price:
                        target2_hit = True
                        targets_hit = 2
                        pnl += (target2 - target)
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        continue
                    if target2_hit and not target3_hit and future_candle['high'] >= target3_price:
                        target3_hit = True
                        targets_hit = 3
                        pnl += (target3 - target2)
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        continue
                    if target1_hit:
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        if future_candle['low'] <= trailing_sl:
                            outcome = "Win"
                            pnl = trailing_sl - candle['close']
                            failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                            exit_time = self.safe_signal_time(future_candle.get('time', None))
                            break
            elif signal == "BUY PUT":
                stop_loss_price = candle['close'] + stop_loss
                target1_price = candle['close'] - target
                target2_price = candle['close'] - target2
                target3_price = candle['close'] - target3

                lowest_price = candle['close']
                trailing_sl = None
                target1_hit = target2_hit = target3_hit = False
                for i, future_candle in future_data.iterrows():
                    if not target1_hit and future_candle['high'] >= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        break
                    if not target1_hit and future_candle['low'] <= target1_price:
                        target1_hit = True
                        targets_hit = 1
                        pnl = target
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = lowest_price + stop_loss
                        outcome = "Win"
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        continue
                    if target1_hit and not target2_hit and future_candle['low'] <= target2_price:
                        target2_hit = True
                        targets_hit = 2
                        pnl += (target2 - target)
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        continue
                    if target2_hit and not target3_hit and future_candle['low'] <= target3_price:
                        target3_hit = True
                        targets_hit = 3
                        pnl += (target3 - target2)
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        exit_time = self.safe_signal_time(future_candle.get('time', None))
                        continue
                    if target1_hit:
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        if future_candle['high'] >= trailing_sl:
                            outcome = "Win"
                            pnl = candle['close'] - trailing_sl
                            failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                            exit_time = self.safe_signal_time(future_candle.get('time', None))
                            break
        
        # Create the signal data dictionary
        signal_data = {
            "signal": signal,
            "price": candle['close'],
            "confidence": confidence,
            "trade_type": trade_type,
            "channel_width": candle.get('channel_width', 0),
            "breakout_size": candle.get('breakout_size', 0),
            "volume_ratio": candle.get('volume_ratio', 0),
            "stop_loss": stop_loss,
            "target": target,
            "target2": target2,
            "target3": target3,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
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
        if df.empty or len(df) < 20:  # Need at least 20 candles for indicators
            return None

        # Add indicators to this timeframe data
        try:
            df = self.add_indicators(df)
        except (IndexError, ValueError):
            # Not enough data for indicators
            return None
            
        candle = df.iloc[-1]

        # Check for breakouts in this timeframe
        signal_direction = 0
        
        # BUY CALL on upper channel breakout
        if candle['close'] > candle['prev_upper']:
            signal_direction = 1
            
        # BUY PUT on lower channel breakout
        elif candle['close'] < candle['prev_lower']:
            signal_direction = -1

        return {
            "signal_direction": signal_direction,
            "channel_width": candle.get('channel_width', 0),
            "breakout_size": candle.get('breakout_size', 0),
            "volume_ratio": candle.get('volume_ratio', 0),
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
                    "channel_width": base_candle.get('channel_width', 0),
                    "breakout_size": base_candle.get('breakout_size', 0),
                    "volume_ratio": base_candle.get('volume_ratio', 0),
                    "stop_loss": round(base_candle.get('atr', 0) * 1.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target": round(base_candle.get('atr', 0) * 1.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target2": round(base_candle.get('atr', 0) * 2.0, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "target3": round(base_candle.get('atr', 0) * 2.5, 2) if base_candle.get('atr', 0) > 0 else 0,
                    "rsi": base_candle.get('rsi', 50),
                    "rsi_reason": f"Multi-timeframe confirmation ({bullish_votes if signal == 'BUY CALL' else bearish_votes}/3 timeframes)",
                    "macd_reason": "",
                    "price_reason": f"Donchian breakout confirmed across {bullish_votes if signal == 'BUY CALL' else bearish_votes} timeframes",
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
def run_strategy(candle, prev_candle=None, index_name=None, future_data=None, channel_period=10, breakout_strength=0.1, volume_trigger=1.0):
    """Legacy wrapper function for backward compatibility with function-based approach."""
    strategy = DonchianBreakout({
        'channel_period': channel_period,
        'breakout_strength': breakout_strength,
        'volume_trigger': volume_trigger
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
