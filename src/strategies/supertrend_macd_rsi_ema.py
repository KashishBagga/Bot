"""
Supertrend MACD RSI EMA strategy.
Trading strategy combining Supertrend, MACD, RSI, and EMA indicators.
"""
import pandas as pd
from typing import Dict, Any
from src.core.strategy import Strategy
from src.core.indicators import indicators

class SupertrendMacdRsiEma(Strategy):
    """Trading strategy combining multiple technical indicators."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the Supertrend MACD RSI EMA strategy.
        
        Args:
            params: Strategy parameters
        """
        super().__init__("supertrend_macd_rsi_ema", params)
        
        # Set default parameters if not provided
        if not self.params.get('supertrend_period'):
            self.params['supertrend_period'] = 10
        if not self.params.get('supertrend_multiplier'):
            self.params['supertrend_multiplier'] = 3.0
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Add Supertrend indicator
        period = self.params.get('supertrend_period', 10)
        multiplier = self.params.get('supertrend_multiplier', 3.0)
        
        supertrend_data = indicators.supertrend(data, period=period, multiplier=multiplier)
        data['supertrend'] = supertrend_data['supertrend']
        data['supertrend_direction'] = supertrend_data['direction']
        
        # Calculate body and range for candle analysis
        data['body'] = abs(data['close'] - data['open'])
        data['full_range'] = data['high'] - data['low']
        data['body_ratio'] = data['body'] / data['full_range'].replace(0, float('nan'))
        
        return data
    
    def analyze(self, data: pd.DataFrame, index_name: str = None, future_data = None) -> Dict[str, Any]:
        """Analyze data and generate trading signals.
        
        Args:
            data: Market data with indicators
            index_name: Name of the index or symbol being analyzed
            future_data: Future candles for performance calculation
            
        Returns:
            Dict[str, Any]: Signal data
        """
        # Calculate indicators if they haven't been calculated yet
        if 'supertrend' not in data.columns:
            data = self.calculate_indicators(data)
        
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
        exit_time = None
        
        # Check if the candle is valid for analysis
        if candle['full_range'] == 0:
            return {
                "signal": "NO TRADE",
                "price": candle['close'],
                "rsi": candle['rsi'],
                "macd": candle['macd'],
                "macd_signal": candle['macd_signal'],
                "ema_20": candle['ema'],
                "atr": candle['atr'],
                "confidence": "Low",
                "rsi_reason": "",
                "macd_reason": "",
                "price_reason": "Invalid candle with zero range",
                "trade_type": trade_type,
                "outcome": outcome,
                "pnl": pnl,
                "targets_hit": targets_hit,
                "stoploss_count": stoploss_count,
                "failure_reason": failure_reason,
                "exit_time": exit_time
            }
        
        # Check for bullish signal (RSI, MACD, EMA)
        if (candle['rsi'] > 65 and  # Relaxed from 65
            candle['macd'] > candle['macd_signal'] and  # MACD above signal line
            candle['close'] > candle['ema'] * 0.1 and  # Close near or above EMA
            candle['supertrend_direction'] > 0):  # Supertrend is bullish (1 for uptrend)
            
            signal = "BUY CALL"
            confidence = "High" if candle['rsi'] > 70 else "Medium"
            
            # Provide reasons for the signal
            rsi_reason = f"RSI {candle['rsi']:.2f} > 65"
            macd_reason = f"MACD {candle['macd']:.2f} > Signal {candle['macd_signal']:.2f}"
            price_reason = f"Price {candle['close']:.2f} > EMA {candle['ema']:.2f}, Supertrend bullish"
        
        # Check for bearish signal (RSI, MACD, EMA)
        elif (candle['rsi'] < 35 and  # Relaxed from 35
              candle['macd'] < candle['macd_signal'] and  # MACD below signal line
              candle['close'] < candle['ema'] * 1.01 and  # Close near or below EMA
              candle['supertrend_direction'] < 0):  # Supertrend is bearish (-1 for downtrend)
            
            signal = "BUY PUT"
            confidence = "High" if candle['rsi'] < 30 else "Medium"
            
            # Provide reasons for the signal
            rsi_reason = f"RSI {candle['rsi']:.2f} < 45"
            macd_reason = f"MACD {candle['macd']:.2f} < Signal {candle['macd_signal']:.2f}"
            price_reason = f"Price {candle['close']:.2f} < EMA {candle['ema']:.2f}, Supertrend bearish"
        
        # Calculate ATR-based stop loss and targets
        atr = candle['atr']
        stop_loss = int(round(atr))
        target = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))
        
        # Calculate performance if we have a signal and future data
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            # Calculate performance metrics
            if signal == "BUY CALL":
                stop_loss_price = candle['close'] - stop_loss
                target1_price = candle['close'] + target
                target2_price = candle['close'] + target2
                target3_price = candle['close'] + target3
                
                # Process each future candle chronologically
                for i, future_candle in future_data.iterrows():
                    # Check if stop loss is hit first
                    if future_candle['low'] <= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        # Set exit_time when stop loss is hit
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        break  # Exit the loop as trade is closed
                    
                    # Check which targets are hit
                    if targets_hit == 0 and future_candle['high'] >= target1_price:
                        targets_hit = 1
                        pnl = target
                        outcome = "Win"
                        # Set exit_time when first target is hit
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    if targets_hit == 1 and future_candle['high'] >= target2_price:
                        targets_hit = 2
                        pnl += (target2 - target)
                    
                    if targets_hit == 2 and future_candle['high'] >= target3_price:
                        targets_hit = 3
                        pnl += (target3 - target2)
            
            elif signal == "BUY PUT":
                stop_loss_price = candle['close'] + stop_loss
                target1_price = candle['close'] - target
                target2_price = candle['close'] - target2
                target3_price = candle['close'] - target3
                
                # Process each future candle chronologically
                for i, future_candle in future_data.iterrows():
                    # Check if stop loss is hit first
                    if future_candle['high'] >= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        # Set exit_time when stop loss is hit
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        break  # Exit the loop as trade is closed
                    
                    # Check which targets are hit
                    if targets_hit == 0 and future_candle['low'] <= target1_price:
                        targets_hit = 1
                        pnl = target
                        outcome = "Win"
                        # Set exit_time when first target is hit
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    if targets_hit == 1 and future_candle['low'] <= target2_price:
                        targets_hit = 2
                        pnl += (target2 - target)
                    
                    if targets_hit == 2 and future_candle['low'] <= target3_price:
                        targets_hit = 3
                        pnl += (target3 - target2)
        
        # Return the signal data
        return {
            "signal": signal,
            "price": candle['close'],
            "rsi": candle['rsi'],
            "macd": candle['macd'],
            "macd_signal": candle['macd_signal'],
            "ema_20": candle['ema'],
            "atr": atr,
            "supertrend": candle['supertrend'],
            "supertrend_direction": candle['supertrend_direction'],
            "stop_loss": stop_loss,
            "target": target,
            "target2": target2,
            "target3": target3,
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "price_reason": price_reason,
            "trade_type": trade_type,
            "option_chain_confirmation": "Yes" if confidence == "High" else "No",
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        } 