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
        data['crossover_strength'] = 100 * (data['ema_fast'] - data['ema_slow']) / data['ema_slow']
        
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
        """Single timeframe analysis method."""
        # Ensure 'time' column exists and is valid, and set as index
        if 'time' in data.columns:
            data = data.copy()
            data.loc[:, 'time'] = pd.to_datetime(data['time'], errors='coerce')
            data = data.set_index('time')
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Calculate indicators if they haven't been calculated yet
        if 'ema_fast' not in data.columns:
            data = self.add_indicators(data)
        
        # Get the latest candle
        candle = data.iloc[-1]
        
        # Set default values
        signal = "NO TRADE"  # Changed from "None" to "NO TRADE" for consistency
        confidence = "Low"
        trade_type = "Intraday"
        rsi_reason = macd_reason = price_reason = ""
        
        # Performance tracking variables
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        outcome = "Pending"
        failure_reason = ""
        
        # Determine momentum
        momentum = None
        if candle['ema_fast_change'] > 0.5:
            momentum = "Strong bullish"
        elif candle['ema_fast_change'] > 0.2:
            momentum = "Bullish"
        elif candle['ema_fast_change'] < -0.5:
            momentum = "Strong bearish"
        elif candle['ema_fast_change'] < -0.2:
            momentum = "Bearish"
        
        # Calculate ATR-based stop loss and targets
        atr = candle['atr'] if 'atr' in candle else abs(candle['ema_fast'] - candle['ema_slow']) * 2
        stop_loss = int(round(atr))
        target1 = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))
        
        # Check for bullish signal (fast EMA above slow EMA)
        if candle['ema_fast'] > candle['ema_slow'] and candle['close'] > candle['ema_fast']:
            signal = "BUY CALL"
            confidence = "High" if candle['crossover_strength'] > 0.5 else "Medium"
            price_reason = f"EMA{self.fast_ema} crossed above EMA{self.slow_ema} (Strength: {candle['crossover_strength']:.2f}%)"
            if momentum:
                price_reason += f", {momentum} momentum"
        
        # Check for bearish signal (fast EMA below slow EMA)
        elif candle['ema_fast'] < candle['ema_slow'] and candle['close'] < candle['ema_fast']:
            signal = "BUY PUT"
            confidence = "High" if abs(candle['crossover_strength']) > 0.5 else "Medium"
            price_reason = f"EMA{self.fast_ema} crossed below EMA{self.slow_ema} (Strength: {candle['crossover_strength']:.2f}%)"
            if momentum:
                price_reason += f", {momentum} momentum"
        
        # If we have a trade signal and future data, calculate performance
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
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price - stop_loss}"
                else:
                    outcome = "Win"
                    # Check which targets were hit
                    if max_future_price >= (price + target1):
                        targets_hit += 1
                        pnl += target1
                    if max_future_price >= (price + target2):
                        targets_hit += 1
                        pnl += (target2 - target1)
                    if max_future_price >= (price + target3):
                        targets_hit += 1
                        pnl += (target3 - target2)
                    
            elif signal == "BUY PUT":
                # For buy puts, check if price went down to targets or up to stop loss
                max_future_price = future_data['high'].max()
                min_future_price = future_data['low'].min()
                
                # Check if stop loss was hit
                if max_future_price >= (price + stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price + stop_loss}"
                else:
                    outcome = "Win"
                    # Check which targets were hit
                    if min_future_price <= (price - target1):
                        targets_hit += 1
                        pnl += target1
                    if min_future_price <= (price - target2):
                        targets_hit += 1
                        pnl += (target2 - target1)
                    if min_future_price <= (price - target3):
                        targets_hit += 1
                        pnl += (target3 - target2)
        
        # --- Outcome logic fix ---
        # If a trade was simulated and outcome is set, keep it as 'Win' or 'Loss'
        # Otherwise, set outcome to 'No Trade' or 'Data Missing' as appropriate
        if signal == "NO TRADE":
            outcome = "No Trade"
        elif (signal.startswith("BUY") and (future_data is None or future_data.empty)):
            outcome = "Data Missing"
        elif outcome not in ["Win", "Loss"]:
            outcome = "No Trade"
        
        # Create the signal data dictionary
        signal_data = {
            "signal": signal,
            "price": candle['close'],
            "rsi": candle.get('rsi', 0),
            "macd": candle.get('macd', 0),
            "macd_signal": candle.get('macd_signal', 0),
            "ema_20": candle.get('ema', 0),
            "ema_fast": candle['ema_fast'],
            "ema_slow": candle['ema_slow'],
            "ema_9": candle['ema_9'],  # For backward compatibility
            "ema_21": candle['ema_21'],  # For backward compatibility
            "atr": atr,
            "stop_loss": stop_loss,
            "target": target1,
            "target2": target2,
            "target3": target3,
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "price_reason": price_reason,
            "trade_type": trade_type,
            "crossover_strength": candle['crossover_strength'],
            "momentum": momentum,
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": None  # Initialize exit_time as None
        }
        
        # If future data is available, try to determine exit time
        if future_data is not None and not future_data.empty and signal != "NO TRADE":
            exit_time = None
            
            # For BUY CALL scenario
            if signal == "BUY CALL":
                price = candle['close']
                stop_loss_price = price - stop_loss
                target1_price = price + target1
                target2_price = price + target2
                target3_price = price + target3
                
                # Iterate through future candles chronologically
                for idx, future_candle in future_data.iterrows():
                    # Get timestamp in the correct format
                    current_time = self.safe_signal_time(future_candle.get('time', None))
                    
                    # Check stop loss first (exit on low price)
                    if future_candle['low'] <= stop_loss_price:
                        exit_time = current_time
                        break
                    
                    # Check targets (exit on the highest target reached)
                    highest_target_reached = 0
                    if future_candle['high'] >= target3_price:
                        highest_target_reached = 3
                    elif future_candle['high'] >= target2_price:
                        highest_target_reached = 2
                    elif future_candle['high'] >= target1_price:
                        highest_target_reached = 1
                    
                    if highest_target_reached > 0:
                        exit_time = current_time
                        if highest_target_reached == 3:  # If highest target reached, we're done
                            break
            
            # For BUY PUT scenario
            elif signal == "BUY PUT":
                price = candle['close']
                stop_loss_price = price + stop_loss
                target1_price = price - target1
                target2_price = price - target2
                target3_price = price - target3
                
                # Iterate through future candles chronologically
                for idx, future_candle in future_data.iterrows():
                    # Get timestamp in the correct format
                    current_time = self.safe_signal_time(future_candle.get('time', None))
                    
                    # Check stop loss first (exit on high price)
                    if future_candle['high'] >= stop_loss_price:
                        exit_time = current_time
                        break
                    
                    # Check targets (exit on the lowest target reached)
                    lowest_target_reached = 0
                    if future_candle['low'] <= target3_price:
                        lowest_target_reached = 3
                    elif future_candle['low'] <= target2_price:
                        lowest_target_reached = 2
                    elif future_candle['low'] <= target1_price:
                        lowest_target_reached = 1
                    
                    if lowest_target_reached > 0:
                        exit_time = current_time
                        if lowest_target_reached == 3:  # If lowest target reached, we're done
                            break
            
            # Update the signal data with the exit time
            signal_data["exit_time"] = self.to_ist_str(exit_time) or (str(exit_time) if exit_time is not None else None)
        
        return signal_data
        
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
    
    # Create a single-row DataFrame from the candle
    if not isinstance(candle, pd.DataFrame):
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
