"""
Range Breakout Volatility strategy.
Trading strategy based on price breakouts from established ranges with volatility filters.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from src.core.strategy import Strategy
from db import log_strategy_sql

class RangeBreakoutVolatility(Strategy):
    """Trading strategy implementation for Range Breakout with Volatility.
    
    Generates signals based on price breakouts from established ranges with volatility filters.
    Buy Call signals when price breaks above range with increasing volatility.
    Buy Put signals when price breaks below range with increasing volatility.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including range_period, breakout_threshold, volatility_threshold
        """
        params = params or {}
        # Reduced range period from 10 to 5 for more frequent ranges
        self.range_period = params.get('range_period', 5)
        # Reduced breakout threshold from 0.3 to 0.1
        self.breakout_threshold = params.get('breakout_threshold', 0.1)
        # Reduced volatility threshold from 1.2 to 1.0
        self.volatility_threshold = params.get('volatility_threshold', 1.0)
        # Added flag to include signals even without volatility confirmation
        self.require_volatility = params.get('require_volatility', False)
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
            "failure_reason": failure_reason
        }
        
        # If index_name is provided, log to database
        if index_name and signal != "NO TRADE":
            db_signal_data = signal_data.copy()
            db_signal_data["signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db_signal_data["index_name"] = index_name
            log_strategy_sql('range_breakout_volatility', db_signal_data)
        
        return signal_data

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
        # Import pandas here to avoid circular imports
        import pandas as pd
        data = pd.DataFrame([candle])
    else:
        data = candle
        
    return strategy.analyze(data, index_name, future_data)
