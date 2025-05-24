"""
SupertrendEma strategy.
Trading strategy implementation.
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from src.core.strategy import Strategy
from db import log_strategy_sql

class SupertrendEma(Strategy):
    """Trading strategy implementation for Supertrend with EMA confirmation.
    
    Generates signals based on Supertrend indicator and EMA confirmation.
    Buy Call signals when price is above EMA and Supertrend is positive.
    Buy Put signals when price is below EMA and Supertrend is negative.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including price_to_ema_ratio
        """
        params = params or {}
        self.price_to_ema_ratio = params.get('price_to_ema_ratio', None)
        super().__init__("supertrend_ema", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Ensure we have EMA-20
        if 'ema_20' not in data.columns:
            data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
        
        # Calculate price to EMA ratio
        if 'price_to_ema_ratio' not in data.columns:
            data['price_to_ema_ratio'] = (data['close'] / data['ema_20'] - 1) * 100
            
        # Calculate Supertrend if not already present
        if 'supertrend' not in data.columns and 'atr' in data.columns:
            # Basic implementation of Supertrend
            # In a real implementation, this would be more complex
            factor = 3
            upper_band = data['high'] + factor * data['atr']
            lower_band = data['low'] - factor * data['atr']
            
            # Initialize the supertrend column
            data['supertrend'] = 0
            
            # First value is set based on the first closing price
            first_idx = data.index[0]
            data.loc[first_idx, 'supertrend'] = 1 if data.loc[first_idx, 'close'] > lower_band.iloc[0] else -1
            
            # Calculate supertrend direction
            for i in range(1, len(data)):
                curr_idx = data.index[i]
                prev_idx = data.index[i-1]
                
                if data.loc[prev_idx, 'supertrend'] == 1:
                    # Previously uptrend
                    if data.loc[curr_idx, 'close'] < lower_band.iloc[i]:
                        data.loc[curr_idx, 'supertrend'] = -1  # New downtrend
                    else:
                        data.loc[curr_idx, 'supertrend'] = 1  # Continue uptrend
                else:
                    # Previously downtrend
                    if data.loc[curr_idx, 'close'] > upper_band.iloc[i]:
                        data.loc[curr_idx, 'supertrend'] = 1  # New uptrend
                    else:
                        data.loc[curr_idx, 'supertrend'] = -1  # Continue downtrend
                        
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
            index_name: Optional index name for database logging
            future_data: Optional future data for performance tracking
            
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
        price_reason = ""
        
        # Initialize performance metrics
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        # Calculate price to EMA ratio if not already available
        price_to_ema_ratio = candle.get('price_to_ema_ratio', 
                                      self.price_to_ema_ratio if self.price_to_ema_ratio is not None 
                                      else (candle['close'] / candle['ema_20'] - 1) * 100)
        
        # Get supertrend value - in this simplified implementation, we'll assume the supertrend
        # is 1 when price is above EMA-20 and -1 when price is below EMA-20
        # In a real implementation, this would come from the supertrend indicator
        supertrend_value = candle.get('supertrend', 1 if candle['close'] > candle['ema_20'] else -1)
        
        # Signal generation based on Supertrend and EMA
        if candle['close'] > candle['ema_20'] and supertrend_value > 0:
            signal = "BUY CALL"
            # Higher confidence when price is further above EMA (showing momentum)
            if price_to_ema_ratio and price_to_ema_ratio > 0.5:
                confidence = "High"
                price_reason = f"Price strongly above EMA ({price_to_ema_ratio:.2f}%) with positive Supertrend"
            else:
                confidence = "Medium"
                price_reason = f"Price above EMA ({price_to_ema_ratio:.2f}%) with positive Supertrend"
                
        elif candle['close'] < candle['ema_20'] and supertrend_value < 0:
            signal = "BUY PUT"
            # Higher confidence when price is further below EMA (showing momentum)
            if price_to_ema_ratio and price_to_ema_ratio < -0.5:
                confidence = "High"
                price_reason = f"Price strongly below EMA ({price_to_ema_ratio:.2f}%) with negative Supertrend"
            else:
                confidence = "Medium"
                price_reason = f"Price below EMA ({price_to_ema_ratio:.2f}%) with negative Supertrend"
        
        # Calculate ATR-based stop loss and targets
        atr = candle['atr'] if 'atr' in candle else abs(candle['high'] - candle['low'])
        stop_loss = int(round(atr))
        target1 = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))
        
        # Calculate performance metrics if a signal was generated and future data is available
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            entry_price = candle['close']
            performance = self.calculate_performance(
                signal=signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target1,
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
        
        # Return the signal data
        signal_data = {
            "signal": signal,
            "price": candle['close'],
            "ema_20": candle['ema_20'],
            "atr": atr,
            "price_to_ema_ratio": price_to_ema_ratio,
            "supertrend": supertrend_value,
            "stop_loss": stop_loss,
            "target": target1,
            "target2": target2,
            "target3": target3,
            "confidence": confidence,
            "price_reason": price_reason,
            "trade_type": trade_type,
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
            log_strategy_sql('supertrend_ema', db_signal_data)
        
        return signal_data
