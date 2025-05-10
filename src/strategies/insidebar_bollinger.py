"""
Inside Bar Bollinger Bands strategy.
Trading strategy based on inside bars and Bollinger Bands.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from src.core.strategy import Strategy
from db import log_strategy_sql

class InsidebarBollinger(Strategy):
    """Trading strategy implementation for Inside Bar Bollinger Bands.
    
    Generates signals based on inside bars and their position relative to Bollinger Bands.
    Buy Call signals when an inside bar forms near the lower Bollinger Band.
    Buy Put signals when an inside bar forms near the upper Bollinger Band.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including bollinger_width, price_to_band_ratio, inside_bar_size
        """
        params = params or {}
        self.bollinger_width = params.get('bollinger_width', None)
        self.price_to_band_ratio = params.get('price_to_band_ratio', None)
        self.inside_bar_size = params.get('inside_bar_size', None)
        # Added proximity threshold parameter (percent of price)
        self.proximity_threshold = params.get('proximity_threshold', 0.5)
        # Added flag to check for near bands instead of strict crossing
        self.near_bands = params.get('near_bands', True)
        # Added parameter to relax inside bar condition
        self.partial_inside = params.get('partial_inside', True)
        super().__init__("insidebar_bollinger", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Calculate Bollinger Band indicators if not already present
        if 'bollinger_width' not in data.columns and 'bollinger_upper' in data.columns and 'bollinger_lower' in data.columns:
            data['bollinger_width'] = (data['bollinger_upper'] - data['bollinger_lower']) / data['close'] * 100
            
        if 'price_to_band_ratio' not in data.columns and 'bollinger_upper' in data.columns and 'bollinger_lower' in data.columns:
            # 0 = at lower band, 1 = at upper band, 0.5 = at middle band
            data['price_to_band_ratio'] = (data['close'] - data['bollinger_lower']) / (data['bollinger_upper'] - data['bollinger_lower'])
            
        # Calculate inside bar indicator
        data['prev_high'] = data['high'].shift(1)
        data['prev_low'] = data['low'].shift(1)
        
        # Traditional inside bar (completely inside)
        data['is_inside'] = (data['high'] < data['prev_high']) & (data['low'] > data['prev_low'])
        
        # Partial inside bar (high OR low is inside previous bar)
        data['is_partial_inside'] = ((data['high'] < data['prev_high']) | (data['low'] > data['prev_low']))
        
        # Calculate inside bar size as percentage of previous bar
        data['inside_bar_size'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low']) * 100
            
        # Calculate proximity to Bollinger Bands as percentage of price
        if 'bollinger_lower' in data.columns and 'bollinger_upper' in data.columns:
            data['lower_band_proximity'] = (data['close'] - data['bollinger_lower']) / data['close'] * 100
            data['upper_band_proximity'] = (data['bollinger_upper'] - data['close']) / data['close'] * 100
            
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
        rsi_reason = price_reason = ""
        option_chain_confirmation = "No"
        
        # Performance tracking variables
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        
        # Check if current bar is an inside bar (or partial inside bar if enabled)
        is_inside = candle['is_inside'] if not self.partial_inside else candle['is_partial_inside']
        
        # Get bollinger width or use the computed one
        bollinger_width = self.bollinger_width if self.bollinger_width is not None else candle.get('bollinger_width', None)
        
        # Get price to band ratio or use the computed one
        price_to_band_ratio = self.price_to_band_ratio if self.price_to_band_ratio is not None else candle.get('price_to_band_ratio', None)
        
        # Get inside bar size or use the computed one
        inside_bar_size = self.inside_bar_size if self.inside_bar_size is not None else candle.get('inside_bar_size', None)
        
        # Get proximity to bands
        lower_proximity = candle.get('lower_band_proximity', 100)  # Default to large value if not available
        upper_proximity = candle.get('upper_band_proximity', 100)  # Default to large value if not available
        
        # Calculate ATR for stops and targets if available
        atr = candle.get('atr', 0)
        stop_loss = round(atr * 1.0, 2) if atr > 0 else 0
        target = round(atr * 1.5, 2) if atr > 0 else 0
        target2 = round(atr * 2.0, 2) if atr > 0 else 0
        target3 = round(atr * 2.5, 2) if atr > 0 else 0
        
        # Generate signals based on inside bar and Bollinger Bands
        # BUY CALL - either directly below lower band or near lower band based on settings
        if is_inside and (
            (not self.near_bands and candle['close'] < candle['bollinger_lower']) or 
            (self.near_bands and lower_proximity < self.proximity_threshold)
        ):
            signal = "BUY CALL"
            confidence = "Medium"
            price_reason = "Inside bar near lower Bollinger Band"
            if bollinger_width:
                price_reason += f", Bollinger width: {bollinger_width:.2f}%"
            if inside_bar_size:
                price_reason += f", Inside bar size: {inside_bar_size:.2f}%"
                
        # BUY PUT - either directly above upper band or near upper band based on settings
        elif is_inside and (
            (not self.near_bands and candle['close'] > candle['bollinger_upper']) or 
            (self.near_bands and upper_proximity < self.proximity_threshold)
        ):
            signal = "BUY PUT"
            confidence = "Medium"
            price_reason = "Inside bar near upper Bollinger Band"
            if bollinger_width:
                price_reason += f", Bollinger width: {bollinger_width:.2f}%"
            if inside_bar_size:
                price_reason += f", Inside bar size: {inside_bar_size:.2f}%"
        
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
            "bollinger_width": bollinger_width if bollinger_width is not None else 0,
            "price_to_band_ratio": price_to_band_ratio if price_to_band_ratio is not None else 0,
            "inside_bar_size": inside_bar_size if inside_bar_size is not None else 0,
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "price_reason": price_reason,
            "trade_type": trade_type,
            "option_chain_confirmation": option_chain_confirmation,
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
            log_strategy_sql('insidebar_bollinger', db_signal_data)
        
        return signal_data

# Backward compatibility function
def run_strategy(candle, prev_candle, index_name, future_data=None, bollinger_width=None, price_to_band_ratio=None, inside_bar_size=None, partial_inside=True, near_bands=True):
    """Legacy wrapper function for backward compatibility with function-based approach."""
    strategy = InsidebarBollinger({
        'bollinger_width': bollinger_width,
        'price_to_band_ratio': price_to_band_ratio,
        'inside_bar_size': inside_bar_size,
        'partial_inside': partial_inside,
        'near_bands': near_bands,
        'proximity_threshold': 0.5
    })
    
    # Create a single-row DataFrame from the candle
    if not isinstance(candle, pd.DataFrame):
        # Import pandas here to avoid circular imports
        import pandas as pd
        data = pd.DataFrame([candle])
        # Add prev candle data
        data['prev_high'] = prev_candle['high']
        data['prev_low'] = prev_candle['low']
        data['is_inside'] = (candle['high'] < prev_candle['high']) and (candle['low'] > prev_candle['low'])
        data['is_partial_inside'] = ((candle['high'] < prev_candle['high']) or (candle['low'] > prev_candle['low']))
    else:
        data = candle
        
    return strategy.analyze(data, index_name, future_data)
