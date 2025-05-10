"""
Breakout RSI strategy.
Trading strategy based on price breakouts with RSI confirmation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
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
        
        return {
            "outcome": outcome,
            "pnl": round(pnl, 2),
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason
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
        rsi_reason = macd_reason = price_reason = ""
        
        # Performance tracking variables
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        
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
            "macd_reason": macd_reason,
            "price_reason": price_reason,
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
