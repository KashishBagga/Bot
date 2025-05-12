"""
EMA Crossover strategy.
Trading strategy based on crossing of exponential moving averages.
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from src.core.strategy import Strategy
from src.core.indicators import indicators
from db import log_strategy_sql

class EmaCrossover(Strategy):
    """Trading strategy based on EMA crossover signals.
    
    Generates signals based on the crossover of fast EMA (default 9) and slow EMA (default 21).
    Buy Call signals when fast EMA crosses above slow EMA and price is above fast EMA.
    Buy Put signals when fast EMA crosses below slow EMA and price is below fast EMA.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the EMA Crossover strategy.
        
        Args:
            params: Strategy parameters
        """
        super().__init__("ema_crossover", params)
        
        # Set default parameters if not provided
        if not self.params.get('fast_ema'):
            self.params['fast_ema'] = 9
        if not self.params.get('slow_ema'):
            self.params['slow_ema'] = 21
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Add the fast and slow EMAs
        fast_period = self.params.get('fast_ema', 9)
        slow_period = self.params.get('slow_ema', 21)
        
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
    
    def analyze(self, data: pd.DataFrame, index_name: str = None, future_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze data and generate trading signals.
        
        Args:
            data: Market data with indicators
            index_name: Name of the index or symbol being analyzed
            future_data: Optional future candles for performance tracking
            
        Returns:
            Dict[str, Any]: Signal data
        """
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
            price_reason = f"EMA{self.params['fast_ema']} crossed above EMA{self.params['slow_ema']} (Strength: {candle['crossover_strength']:.2f}%)"
            if momentum:
                price_reason += f", {momentum} momentum"
        
        # Check for bearish signal (fast EMA below slow EMA)
        elif candle['ema_fast'] < candle['ema_slow'] and candle['close'] < candle['ema_fast']:
            signal = "BUY PUT"
            confidence = "High" if abs(candle['crossover_strength']) > 0.5 else "Medium"
            price_reason = f"EMA{self.params['fast_ema']} crossed below EMA{self.params['slow_ema']} (Strength: {candle['crossover_strength']:.2f}%)"
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
                    outcome = "Failure"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price - stop_loss}"
                else:
                    outcome = "Success"
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
                    outcome = "Failure"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price + stop_loss}"
                else:
                    outcome = "Success"
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
                    current_time = None
                    if isinstance(idx, pd.Timestamp):
                        current_time = idx.strftime("%Y-%m-%d %H:%M:%S")
                    elif 'time' in future_candle and future_candle['time'] is not None:
                        if isinstance(future_candle['time'], pd.Timestamp):
                            current_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            current_time = str(future_candle['time'])
                    
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
                    current_time = None
                    if isinstance(idx, pd.Timestamp):
                        current_time = idx.strftime("%Y-%m-%d %H:%M:%S")
                    elif 'time' in future_candle and future_candle['time'] is not None:
                        if isinstance(future_candle['time'], pd.Timestamp):
                            current_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            current_time = str(future_candle['time'])
                    
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
            signal_data["exit_time"] = exit_time
        
        # If index_name is provided, log to database
        if index_name and signal != "NO TRADE":
            db_signal_data = signal_data.copy()
            # Use the actual candle time for signal_time instead of current time
            if hasattr(candle, 'name') and isinstance(candle.name, pd.Timestamp):
                # If candle has a timestamp index
                db_signal_data["signal_time"] = candle.name.strftime("%Y-%m-%d %H:%M:%S")
            elif 'time' in data.columns and len(data) > 0:
                # If time is a column in the dataframe
                db_signal_data["signal_time"] = data.iloc[-1]['time'].strftime("%Y-%m-%d %H:%M:%S") 
            else:
                # Fallback to current time if no timestamp is available
                db_signal_data["signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            db_signal_data["index_name"] = index_name
            log_strategy_sql('ema_crossover', db_signal_data)
        
        return signal_data
        
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
        
    return strategy.analyze(data, index_name, future_data)
