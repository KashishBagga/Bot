"""
Donchian Channel Breakout strategy.
Trading strategy based on breakouts from Donchian channels.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from src.core.strategy import Strategy
from db import log_strategy_sql

class DonchianBreakout(Strategy):
    """Trading strategy implementation for Donchian Channel Breakout.
    
    Generates signals based on price breakouts from Donchian channels.
    Buy Call signals when price breaks above the upper Donchian channel.
    Buy Put signals when price breaks below the lower Donchian channel.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including channel_period, breakout_strength, volume_trigger
        """
        params = params or {}
        # Reduced channel period from 20 to 10 for more frequent channels
        self.channel_period = params.get('channel_period', 10)
        # Reduced breakout strength requirement from 0.5 to 0.1
        self.breakout_strength = params.get('breakout_strength', 0.1)
        # Reduced volume trigger from 1.5 to 1.0
        self.volume_trigger = params.get('volume_trigger', 1.0)
        super().__init__("donchian_breakout", params)
    
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
            "failure_reason": failure_reason
        }
        
        # If index_name is provided, log to database
        if index_name and signal != "NO TRADE":
            db_signal_data = signal_data.copy()
            db_signal_data["signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db_signal_data["index_name"] = index_name
            log_strategy_sql('donchian_breakout', db_signal_data)
        
        return signal_data

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
        
    return strategy.analyze(data, index_name, future_data)
