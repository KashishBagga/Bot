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
                "failure_reason": ""
            }
        
        # Initialize performance metrics
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        
        # For BUY CALL, check if future prices went up to targets or down to stop loss
        if signal == "BUY CALL":
            max_future_price = future_data['high'].max()
            min_future_price = future_data['low'].min()
            
            # Check if stop loss was hit
            if min_future_price <= (entry_price - stop_loss):
                outcome = "Loss"
                pnl = -1.0 * stop_loss  # Negative value for stop loss
                stoploss_count = 1
                failure_reason = f"Stop loss hit at {entry_price - stop_loss:.2f}"
            else:
                outcome = "Win"
                # Check which targets were hit
                if max_future_price >= (entry_price + target):
                    targets_hit += 1
                    pnl += 1.0 * target
                if max_future_price >= (entry_price + target2):
                    targets_hit += 1
                    pnl += 1.0 * (target2 - target)
                if max_future_price >= (entry_price + target3):
                    targets_hit += 1
                    pnl += 1.0 * (target3 - target2)
        
        # For BUY PUT, check if future prices went down to targets or up to stop loss
        elif signal == "BUY PUT":
            max_future_price = future_data['high'].max()
            min_future_price = future_data['low'].min()
            
            # Check if stop loss was hit
            if max_future_price >= (entry_price + stop_loss):
                outcome = "Loss"
                pnl = -1.0 * stop_loss
                stoploss_count = 1
                failure_reason = f"Stop loss hit at {entry_price + stop_loss:.2f}"
            else:
                outcome = "Win"
                # Check which targets were hit
                if min_future_price <= (entry_price - target):
                    targets_hit += 1
                    pnl += 1.0 * target
                if min_future_price <= (entry_price - target2):
                    targets_hit += 1
                    pnl += 1.0 * (target2 - target)
                if min_future_price <= (entry_price - target3):
                    targets_hit += 1
                    pnl += 1.0 * (target3 - target2)
        
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason
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
            "failure_reason": failure_reason
        }
        
        # If index_name is provided, log to database
        if index_name and signal != "NO TRADE":
            db_signal_data = signal_data.copy()
            db_signal_data["signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db_signal_data["index_name"] = index_name
            log_strategy_sql('supertrend_ema', db_signal_data)
        
        return signal_data
