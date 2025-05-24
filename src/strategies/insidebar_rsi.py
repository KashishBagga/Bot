"""
InsidebarRsi strategy.
Trading strategy implementation.
"""
import pandas as pd
from typing import Dict, Any
from src.core.strategy import Strategy
from src.core.indicators import indicators
from src.models.database import db

class InsidebarRsi(Strategy):
    """Trading strategy implementation for Inside Bar with RSI confirmation."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters
        """
        default_params = {
            'rsi_overbought': 60,
            'rsi_extreme_overbought': 70,
            'rsi_oversold': 40,
            'rsi_extreme_oversold': 30
        }
        
        # Use provided params or defaults
        if params:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        else:
            params = default_params
            
        super().__init__("insidebar_rsi", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Identify inside bars
        data['prev_high'] = data['high'].shift(1)
        data['prev_low'] = data['low'].shift(1)
        data['is_inside'] = (data['high'] < data['prev_high']) & (data['low'] > data['prev_low'])
        
        # Calculate RSI levels based on thresholds
        def get_rsi_level(rsi):
            if rsi < self.params['rsi_extreme_oversold']:
                return "Extreme Oversold"
            elif rsi < self.params['rsi_oversold']:
                return "Oversold"
            elif rsi < self.params['rsi_overbought']:
                return "Neutral"
            elif rsi < self.params['rsi_extreme_overbought']:
                return "Overbought"
            else:
                return "Extreme Overbought"
        
        # Apply the function to create an RSI level column
        data['rsi_level'] = data['rsi'].apply(get_rsi_level)
        
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
        if 'is_inside' not in data.columns:
            data = self.calculate_indicators(data)
        
        # Get the latest candle
        candle = data.iloc[-1]
        
        # Set default values
        signal = "NO TRADE"
        confidence = "Low"
        trade_type = "Intraday"
        rsi_reason = price_reason = ""
        
        # Performance tracking variables
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        # Check if we have an inside bar
        is_inside = candle['is_inside']
        rsi_level = candle['rsi_level']
        
        # Implement strategy logic
        if is_inside and candle['rsi'] > self.params['rsi_overbought']:
            signal = "BUY CALL"
            confidence = "High" if candle['rsi'] > self.params['rsi_extreme_overbought'] else "Medium"
            rsi_reason = f"RSI {candle['rsi']:.2f} > {self.params['rsi_overbought']} ({rsi_level})"
            price_reason = "Inside bar pattern"
        elif is_inside and candle['rsi'] < self.params['rsi_oversold']:
            signal = "BUY PUT"
            confidence = "High" if candle['rsi'] < self.params['rsi_extreme_oversold'] else "Medium"
            rsi_reason = f"RSI {candle['rsi']:.2f} < {self.params['rsi_oversold']} ({rsi_level})"
            price_reason = "Inside bar pattern"
        
        # Calculate stops and targets based on ATR
        atr = candle['atr']
        stop_loss = int(round(atr))
        target1 = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))
        
        # Calculate performance if we have a signal and future data
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            # Calculate performance metrics with trailing stop logic
            if signal == "BUY CALL":
                stop_loss_price = candle['close'] - stop_loss
                target1_price = candle['close'] + target1
                target2_price = candle['close'] + target2
                target3_price = candle['close'] + target3
                
                highest_price = candle['close']
                trailing_sl = None
                target1_hit = target2_hit = target3_hit = False
                for i, future_candle in future_data.iterrows():
                    # Check if stop loss is hit first
                    if not target1_hit and future_candle['low'] <= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        break
                    # Check which targets are hit
                    if not target1_hit and future_candle['high'] >= target1_price:
                        target1_hit = True
                        targets_hit = 1
                        pnl = target1
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = highest_price - stop_loss
                        outcome = "Win"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        continue
                    if target1_hit and not target2_hit and future_candle['high'] >= target2_price:
                        target2_hit = True
                        targets_hit = 2
                        pnl += (target2 - target1)
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    if target2_hit and not target3_hit and future_candle['high'] >= target3_price:
                        target3_hit = True
                        targets_hit = 3
                        pnl += (target3 - target2)
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    # After target1, always trail SL at highest_price - stop_loss
                    if target1_hit:
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        # If price hits trailing SL, exit
                        if future_candle['low'] <= trailing_sl:
                            outcome = "Win"
                            pnl = trailing_sl - candle['close']
                            failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                            if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                                exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                            elif 'time' in future_data.columns:
                                exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                            break
            elif signal == "BUY PUT":
                stop_loss_price = candle['close'] + stop_loss
                target1_price = candle['close'] - target1
                target2_price = candle['close'] - target2
                target3_price = candle['close'] - target3
                
                lowest_price = candle['close']
                trailing_sl = None
                target1_hit = target2_hit = target3_hit = False
                for i, future_candle in future_data.iterrows():
                    # Check if stop loss is hit first
                    if not target1_hit and future_candle['high'] >= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        break
                    # Check which targets are hit
                    if not target1_hit and future_candle['low'] <= target1_price:
                        target1_hit = True
                        targets_hit = 1
                        pnl = target1
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = lowest_price + stop_loss
                        outcome = "Win"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        continue
                    if target1_hit and not target2_hit and future_candle['low'] <= target2_price:
                        target2_hit = True
                        targets_hit = 2
                        pnl += (target2 - target1)
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    if target2_hit and not target3_hit and future_candle['low'] <= target3_price:
                        target3_hit = True
                        targets_hit = 3
                        pnl += (target3 - target2)
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    # After target1, always trail SL at lowest_price + stop_loss
                    if target1_hit:
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        # If price hits trailing SL, exit
                        if future_candle['high'] >= trailing_sl:
                            outcome = "Win"
                            pnl = candle['close'] - trailing_sl
                            failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                            if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                                exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                            elif 'time' in future_data.columns:
                                exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                            break
        
        # Return the signal data
        return {
            "signal": signal,
            "price": candle['close'],
            "rsi": candle['rsi'],
            "rsi_level": rsi_level,
            "ema_20": candle['ema'] if 'ema' in candle else 0,
            "atr": candle['atr'],
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "price_reason": price_reason,
            "trade_type": trade_type,
            "stop_loss": stop_loss,
            "target": target1,
            "target2": target2,
            "target3": target3,
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        }
