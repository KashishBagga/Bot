"""
EMA Crossover Original strategy.
This is the original implementation of the EMA Crossover strategy.
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from src.core.strategy import Strategy
from db import log_strategy_sql

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
        rsi_reason = macd_reason = price_reason = ""
        
        # Initialize performance metrics
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        
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
        
        # Create signal data with performance metrics
        signal_data = {
            "signal": signal,
            "price": candle['close'],
            "ema_9": candle['ema_9'],
            "ema_21": candle['ema_21'],
            "ema_20": candle['ema_20'],
            "crossover_strength": crossover_strength,
            "momentum": self.momentum,
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
            log_strategy_sql('ema_crossover_original', db_signal_data)
        
        return signal_data 