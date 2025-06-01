"""
Supertrend EMA strategy.
Trading strategy based on Supertrend indicator with EMA confirmation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from db import log_strategy_sql
from indicators.supertrend import Supertrend, calculate_supertrend_live, get_supertrend_instance

class SupertrendEma(Strategy):
    """Trading strategy implementation for Supertrend with EMA confirmation.
    
    Generates signals based on Supertrend indicator with EMA confirmation.
    Buy Call signals when price is above EMA and Supertrend is bullish.
    Buy Put signals when price is below EMA and Supertrend is bearish.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters including ema_period, supertrend_period, supertrend_multiplier
        """
        params = params or {}
        self.ema_period = params.get('ema_period', 20)
        self.supertrend_period = params.get('supertrend_period', 10)
        self.supertrend_multiplier = params.get('supertrend_multiplier', 3.0)
        self.price_to_ema_threshold = params.get('price_to_ema_threshold', 0.5)
        super().__init__("supertrend_ema", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Add EMA_20 if not present
        if 'ema_20' not in data.columns:
            data['ema_20'] = data['close'].ewm(span=self.ema_period).mean()
        
        # Calculate price to EMA ratio
        if 'price_to_ema_ratio' not in data.columns:
            data['price_to_ema_ratio'] = (data['close'] / data['ema_20'] - 1) * 100
            
        return data
    
    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame, index_name: str = None) -> Dict[str, Any]:
        """Calculate performance metrics for future candles."""
        if future_data is None or future_data.empty:
            return {
                "outcome": "Pending", 
                "pnl": 0.0, 
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": "",
                "exit_time": None
            }
        
        # Initialize performance metrics
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        trailing_stop = None
        max_profit = 0.0
        
        # Get supertrend instance for this index
        supertrend_instance = get_supertrend_instance(index_name) if index_name else None
        
        # Process each future candle chronologically
        for idx, future_candle in future_data.iterrows():
            # Get candle timestamp for exit_time
            current_time = None
            if isinstance(idx, pd.Timestamp):
                current_time = idx.strftime("%Y-%m-%d %H:%M:%S")
            elif hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                current_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
            elif 'time' in future_candle and future_candle['time'] is not None:
                if isinstance(future_candle['time'], pd.Timestamp):
                    current_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                else:
                    current_time = str(future_candle['time'])
            
            # Convert timestamp to IST if it's not None
            if current_time:
                try:
                    dt_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                    ist_dt = dt_obj + timedelta(hours=5, minutes=30)
                    current_time = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            high = future_candle['high']
            low = future_candle['low']
            close = future_candle['close']
            
            # For BUY CALL signals
            if signal == "BUY CALL":
                # Track maximum profit potential
                current_profit = high - entry_price
                max_profit = max(max_profit, current_profit)
                
                # Check if stop loss was hit before any targets
                if low <= (entry_price - stop_loss) and targets_hit == 0:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price - stop_loss:.2f}"
                    exit_time = current_time
                    break
                
                # Check targets in sequence
                if targets_hit == 0 and high >= (entry_price + target):
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                    trailing_stop = entry_price + (target * 0.5)  # Trail at 50% of first target
                
                if targets_hit == 1 and high >= (entry_price + target2):
                    targets_hit = 2
                    pnl = target2
                    trailing_stop = entry_price + target  # Trail at first target level
                
                if targets_hit == 2 and high >= (entry_price + target3):
                    targets_hit = 3
                    pnl = target3
                    break
                
                # Trailing stop logic
                if targets_hit >= 1:
                    # Update trailing stop to protect profits
                    if trailing_stop is not None:
                        new_trail = high - (stop_loss * 0.75)  # Use 75% of initial stop loss
                        trailing_stop = max(trailing_stop, new_trail)
                        
                        if low <= trailing_stop:
                            outcome = "Win"
                            pnl = trailing_stop - entry_price
                            failure_reason = f"Trailing SL hit at {trailing_stop:.2f}"
                            exit_time = current_time
                            break
                    
                    # Check for trend reversal
                    if supertrend_instance:
                        st_future = supertrend_instance.update(future_candle)
                        if (future_candle.get('ema_20') and 
                            (close < future_candle['ema_20'] or st_future['direction'] < 0)):
                            outcome = "Win"
                            pnl = close - entry_price
                            failure_reason = "Trend reversal (EMA/Supertrend)"
                            exit_time = current_time
                            break
            
            # For BUY PUT signals
            elif signal == "BUY PUT":
                # Track maximum profit potential
                current_profit = entry_price - low
                max_profit = max(max_profit, current_profit)
                
                # Check if stop loss was hit before any targets
                if high >= (entry_price + stop_loss) and targets_hit == 0:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price + stop_loss:.2f}"
                    exit_time = current_time
                    break
                
                # Check targets in sequence
                if targets_hit == 0 and low <= (entry_price - target):
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                    trailing_stop = entry_price - (target * 0.5)  # Trail at 50% of first target
                
                if targets_hit == 1 and low <= (entry_price - target2):
                    targets_hit = 2
                    pnl = target2
                    trailing_stop = entry_price - target  # Trail at first target level
                
                if targets_hit == 2 and low <= (entry_price - target3):
                    targets_hit = 3
                    pnl = target3
                    break
                
                # Trailing stop logic
                if targets_hit >= 1:
                    # Update trailing stop to protect profits
                    if trailing_stop is not None:
                        new_trail = low + (stop_loss * 0.75)  # Use 75% of initial stop loss
                        trailing_stop = min(trailing_stop, new_trail)
                        
                        if high >= trailing_stop:
                            outcome = "Win"
                            pnl = entry_price - trailing_stop
                            failure_reason = f"Trailing SL hit at {trailing_stop:.2f}"
                            exit_time = current_time
                            break
                    
                    # Check for trend reversal
                    if supertrend_instance:
                        st_future = supertrend_instance.update(future_candle)
                        if (future_candle.get('ema_20') and 
                            (close > future_candle['ema_20'] or st_future['direction'] > 0)):
                            outcome = "Win"
                            pnl = entry_price - close
                            failure_reason = "Trend reversal (EMA/Supertrend)"
                            exit_time = current_time
                            break
        
        # If still pending but had targets hit, consider it a win
        if outcome == "Pending":
            if targets_hit > 0:
                outcome = "Win"
            elif max_profit >= target:  # Could have hit target but missed
                outcome = "Loss"
                pnl = -stop_loss
                failure_reason = "Missed profit opportunity"
        
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        }
    
    def analyze(self, data: pd.DataFrame, index_name: str = None, future_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze data and generate trading signals."""
        if data.empty or len(data) < 1:
            return {"signal": "NO TRADE"}
        
        # Get the latest candle and convert to dictionary
        candle_series = data.iloc[-1]
        candle = candle_series.to_dict()
        
        # Initialize signal variables
        signal = "NO TRADE"
        confidence = "Low"
        trade_type = "Intraday"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        outcome = "Pending"
        failure_reason = ""
        
        # Get supertrend instance and calculate supertrend
        supertrend_instance = get_supertrend_instance(index_name) if index_name else None
        if not supertrend_instance:
            return {"signal": "NO TRADE"}
        
        supertrend_data = supertrend_instance.update(candle)
        
        # Calculate price to EMA ratio
        price_to_ema_ratio = 0
        if 'ema_20' in candle:
            price_to_ema_ratio = (candle['close'] / candle['ema_20'] - 1) * 100
        
        supertrend_value = supertrend_data['value']
        supertrend_direction = supertrend_data['direction']
        
        # Generate signals based on EMA and Supertrend
        if 'ema_20' in candle:
            if candle['close'] > candle['ema_20'] and supertrend_direction > 0:
                signal = "BUY CALL"
                if (price_to_ema_ratio > self.price_to_ema_threshold and 
                    candle['close'] > supertrend_value and 
                    candle['close'] > supertrend_data['upperband']):
                    confidence = "High"
                else:
                    confidence = "Medium"
            elif candle['close'] < candle['ema_20'] and supertrend_direction < 0:
                signal = "BUY PUT"
                if (price_to_ema_ratio < -self.price_to_ema_threshold and 
                    candle['close'] < supertrend_value and 
                    candle['close'] < supertrend_data['lowerband']):
                    confidence = "High"
                else:
                    confidence = "Medium"
        
        # Calculate targets and stop loss if signal is generated
        stop_loss = 0
        target1 = 0
        target2 = 0
        target3 = 0
        
        if signal != "NO TRADE":
            price = candle['close']
            
            # Calculate ATR-based stop loss and targets
            atr = candle.get('atr', 0)
            if atr == 0:  # Fallback if ATR is not available
                atr = (candle['high'] - candle['low']) * 0.5
            
            # More conservative stop loss and targets
            stop_loss = atr * 1.0  # 1.0 ATR for stop loss
            target1 = atr * 1.5    # 1.5 ATR for first target
            target2 = atr * 2.0    # 2.0 ATR for second target
            target3 = atr * 2.5    # 2.5 ATR for third target
            
            # Calculate performance if future data is available
            if future_data is not None and not future_data.empty:
                performance = self.calculate_performance(
                    signal, price, stop_loss, target1, target2, target3, future_data, index_name
                )
                outcome = performance["outcome"]
                pnl = performance["pnl"]
                targets_hit = performance["targets_hit"]
                stoploss_count = performance["stoploss_count"]
                failure_reason = performance["failure_reason"]
        
        # Prepare signal data for logging
        signal_data = {
            "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "index_name": index_name,
            "signal": signal,
            "price": candle['close'],
            "ema_20": candle.get('ema_20', 0),
            "atr": candle.get('atr', 0),
            "price_to_ema_ratio": price_to_ema_ratio,
            "confidence": confidence,
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
            "supertrend_value": supertrend_value,
            "supertrend_direction": supertrend_direction,
            "supertrend_upperband": supertrend_data['upperband'],
            "supertrend_lowerband": supertrend_data['lowerband']
        }
        
        # Log to database
        log_strategy_sql('supertrend_ema', signal_data)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "trade_type": trade_type,
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "supertrend_value": supertrend_value,
            "supertrend_direction": supertrend_direction,
            "supertrend_upperband": supertrend_data['upperband'],
            "supertrend_lowerband": supertrend_data['lowerband']
        }

# Legacy function for backward compatibility
def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    """Legacy function wrapper for backward compatibility."""
    # Create a DataFrame from the single candle
    df = pd.DataFrame([candle])
    
    # Create strategy instance
    strategy = SupertrendEma()
    
    # Add indicators
    df = strategy.add_indicators(df)
    
    # Analyze
    result = strategy.analyze(df, index_name, future_data)
    
    return result
