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

class BreakoutRsi(Strategy):
    """Trading strategy implementation for Breakout with RSI confirmation.
    
    Generates signals based on price breakouts with RSI confirmation.
    Buy Call signals when price breaks resistance with RSI above threshold.
    Buy Put signals when price breaks support with RSI below threshold.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including breakout_strength, rsi_alignment, lookback_period
        """
        params = params or {}
        # Reduced breakout strength from 0.5 to 0.1
        self.breakout_strength = params.get('breakout_strength', 0.1)
        # Reduced RSI alignment threshold from 50 to 40
        self.rsi_alignment = params.get('rsi_alignment', 40)
        # Reduced lookback period from 10 to 5
        self.lookback_period = params.get('lookback_period', 5)
        # Added parameter for requiring RSI confirmation
        self.require_rsi = params.get('require_rsi', False)
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
                # Get candle timestamp for exit_time
                current_time = None
                if isinstance(idx, pd.Timestamp):
                    current_time = idx.strftime("%Y-%m-%d %H:%M:%S")
                elif hasattr(candle, 'name') and isinstance(candle.name, pd.Timestamp):
                    current_time = candle.name.strftime("%Y-%m-%d %H:%M:%S")
                elif 'time' in candle and candle['time'] is not None:
                    if isinstance(candle['time'], pd.Timestamp):
                        current_time = candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        current_time = str(candle['time'])
                
                # Convert timestamp to IST if it's not None
                if current_time:
                    try:
                        # Convert to datetime object first
                        dt_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                        # Add IST offset (+5:30)
                        ist_dt = dt_obj + timedelta(hours=5, minutes=30)
                        current_time = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        # If conversion fails, use the original timestamp
                        pass
                
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
                    break  # Exit the loop as all targets are hit
        
        # For BUY PUT, check if future prices went down to targets or up to stop loss
        elif signal == "BUY PUT":
            # Process each future candle chronologically
            for idx, candle in future_data.iterrows():
                # Get candle timestamp for exit_time
                current_time = None
                if isinstance(idx, pd.Timestamp):
                    current_time = idx.strftime("%Y-%m-%d %H:%M:%S")
                elif hasattr(candle, 'name') and isinstance(candle.name, pd.Timestamp):
                    current_time = candle.name.strftime("%Y-%m-%d %H:%M:%S")
                elif 'time' in candle and candle['time'] is not None:
                    if isinstance(candle['time'], pd.Timestamp):
                        current_time = candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        current_time = str(candle['time'])
                
                # Convert timestamp to IST if it's not None
                if current_time:
                    try:
                        # Convert to datetime object first
                        dt_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                        # Add IST offset (+5:30)
                        ist_dt = dt_obj + timedelta(hours=5, minutes=30)
                        current_time = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        # If conversion fails, use the original timestamp
                        pass
                
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
                    break  # Exit the loop as all targets are hit
        
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        }
    
    def analyze(self, data: pd.DataFrame, index_name: str = None, future_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze data and generate trading signals.
        
        Args:
            data: Market data with indicators
            index_name: Name of the index or symbol being analyzed
            future_data: Optional future candles for performance tracking
            
        Returns:
            Dict[str, Any]: Signal data
        """
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
        if candle['close'] > candle.get('prev_high', float('-inf')):
            # Check if breakout size meets threshold
            if candle.get('breakout_strength', 0) >= breakout_strength_threshold:
                # Check RSI alignment if required
                if not self.require_rsi or candle.get('rsi_alignment', 0) > rsi_alignment_threshold:
                    signal = "BUY CALL"
                    confidence = "Medium"
                    price_reason = f"Breakout above resistance: {candle.get('breakout_strength', 0):.2f}%"
                    
                    # RSI confirmation enhances confidence
                    if candle.get('rsi', 0) > 50 and candle.get('rsi_alignment', 0) > rsi_alignment_threshold:
                        confidence = "High"
                        rsi_reason = f"Strong RSI momentum: {candle.get('rsi', 0):.1f}"
                
        # Check for breakouts - BUY PUT on downside breakout
        elif candle['close'] < candle.get('prev_low', float('inf')):
            # Check if breakout size meets threshold
            if candle.get('breakout_strength', 0) >= breakout_strength_threshold:
                # Check RSI alignment if required
                if not self.require_rsi or candle.get('rsi_alignment', 0) > rsi_alignment_threshold:
                    signal = "BUY PUT"
                    confidence = "Medium"
                    price_reason = f"Breakdown below support: {candle.get('breakout_strength', 0):.2f}%"
                    
                    # RSI confirmation enhances confidence
                    if candle.get('rsi', 0) < 50 and candle.get('rsi_alignment', 0) > rsi_alignment_threshold:
                        confidence = "High"
                        rsi_reason = f"Weak RSI momentum: {candle.get('rsi', 0):.1f}"
        
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
            "exit_time": exit_time
        }
        
        # If index_name is provided, log to database
        if index_name and signal != "NO TRADE":
            db_signal_data = signal_data.copy()
            
            # Get signal_time
            signal_time = None
            if hasattr(candle, 'name') and isinstance(candle.name, pd.Timestamp):
                # If candle has a timestamp index
                signal_time = candle.name.strftime("%Y-%m-%d %H:%M:%S")
            elif 'time' in data.columns and len(data) > 0:
                # If time is a column in the dataframe
                signal_time = data.iloc[-1]['time'].strftime("%Y-%m-%d %H:%M:%S") 
            else:
                # Fallback to current time if no timestamp is available
                signal_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert signal_time to IST (+5:30)
            try:
                # Parse the signal time string to a datetime object
                dt_obj = datetime.strptime(signal_time, "%Y-%m-%d %H:%M:%S")
                # Add 5 hours and 30 minutes to convert to IST
                ist_dt = dt_obj + timedelta(hours=5, minutes=30)
                db_signal_data["signal_time"] = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                # Use original time if conversion fails
                print(f"Failed to convert signal_time to IST: {e}")
                db_signal_data["signal_time"] = signal_time
                
            db_signal_data["index_name"] = index_name
            log_strategy_sql('breakout_rsi', db_signal_data)
        
        return signal_data

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
        # Import pandas here to avoid circular imports
        import pandas as pd
        data = pd.DataFrame([candle])
    else:
        data = candle
        
    return strategy.analyze(data, index_name, future_data)
