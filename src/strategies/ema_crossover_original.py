"""
EMA Crossover Original strategy.
This is the original implementation of the EMA Crossover strategy.
"""
import pandas as pd
from typing import Dict, Any
from src.core.strategy import Strategy

class EmaCrossoverOriginal(Strategy):
    """Trading strategy implementation for EMA Crossover (original version).
    
    Generates signals based on the crossover of EMA9 and EMA21.
    Buy Call signals when EMA9 crosses above EMA21 and price is above EMA9.
    Buy Put signals when EMA9 crosses below EMA21 and price is below EMA9.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including crossover_strength and momentum
        """
        params = params or {}
        self.crossover_strength = params.get('crossover_strength', None)
        self.momentum = params.get('momentum', None)
        super().__init__("ema_crossover_original", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Ensure we have the required EMAs
        if 'ema_9' not in data.columns:
            data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
        if 'ema_21' not in data.columns:
            data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
        if 'ema_20' not in data.columns:
            data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
            
        # Calculate crossover strength if needed
        if 'crossover_strength' not in data.columns:
            # Calculate difference between EMA9 and EMA21 as percentage of price
            data['crossover_strength'] = (data['ema_9'] - data['ema_21']) / data['close'] * 100
            
        return data
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals.
        
        Args:
            data: Market data with indicators
            
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
        
        # Calculate ATR-based stop loss and targets
        atr = candle['atr'] if 'atr' in candle else abs(candle['ema_9'] - candle['ema_21']) * 2
        stop_loss = int(round(atr))
        target1 = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))
        
        # Check for EMA crossover conditions
        crossover_strength = candle.get('crossover_strength', self.crossover_strength)
        
        if candle['ema_9'] > candle['ema_21'] and candle['close'] > candle['ema_9']:
            signal = "BUY CALL"
            confidence = "High" if crossover_strength and crossover_strength > 0.5 else "Medium"
            price_reason = f"EMA9 crossed above EMA21"
            if crossover_strength:
                price_reason += f" (Strength: {crossover_strength:.2f}%)"
            if self.momentum:
                price_reason += f", {self.momentum} momentum"
                
        elif candle['ema_9'] < candle['ema_21'] and candle['close'] < candle['ema_9']:
            signal = "BUY PUT"
            confidence = "High" if crossover_strength and crossover_strength > 0.5 else "Medium"
            price_reason = f"EMA9 crossed below EMA21"
            if crossover_strength:
                price_reason += f" (Strength: {crossover_strength:.2f}%)"
            if self.momentum:
                price_reason += f", {self.momentum} momentum"
        
        # Return the signal data
        return {
            "signal": signal,
            "price": candle['close'],
            "ema_9": candle['ema_9'],
            "ema_21": candle['ema_21'],
            "ema_20": candle['ema_20'],
            "crossover_strength": crossover_strength,
            "momentum": self.momentum,
            "atr": atr,
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "target3": target3,
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "price_reason": price_reason,
            "trade_type": trade_type
        } 