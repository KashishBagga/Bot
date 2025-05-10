#!/usr/bin/env python3
"""
Direct performance tracking test for all trading strategies.
This script creates specific test cases for each strategy to validate
the performance tracking logic for Win, Loss, and Pending outcomes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import all strategies
from src.strategies.breakout_rsi import BreakoutRsi
from src.strategies.donchian_breakout import DonchianBreakout
from src.strategies.insidebar_bollinger import InsidebarBollinger
from src.strategies.range_breakout_volatility import RangeBreakoutVolatility

def create_base_data(days=10, base_price=100):
    """Create a base dataset with standard columns."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    data = pd.DataFrame({
        'date': dates,
        'open': [base_price] * days,
        'high': [base_price * 1.01] * days,
        'low': [base_price * 0.99] * days,
        'close': [base_price] * days,
        'volume': [1000] * days,
        'atr': [1.0] * days,
        'rsi': [50] * days,
        'bollinger_upper': [base_price * 1.02] * days,
        'bollinger_lower': [base_price * 0.98] * days,
        'bollinger_middle': [base_price] * days
    })
    
    return data

def create_future_data_win_scenario(days=5, base_price=100, signal="BUY CALL"):
    """Create future data for a winning scenario."""
    dates = [datetime.now() + timedelta(days=i) for i in range(1, days+1)]
    
    if signal == "BUY CALL":
        # Progressive price increase to hit targets
        prices = [base_price * (1 + (i * 0.01)) for i in range(1, days+1)]
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]
    else:  # BUY PUT
        # Progressive price decrease to hit targets
        prices = [base_price * (1 - (i * 0.01)) for i in range(1, days+1)]
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]
        
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices
    })
    
    return data

def create_future_data_loss_scenario(days=5, base_price=100, signal="BUY CALL"):
    """Create future data for a losing scenario."""
    dates = [datetime.now() + timedelta(days=i) for i in range(1, days+1)]
    
    if signal == "BUY CALL":
        # First day hits stop loss
        prices = [base_price * 0.97] + [base_price * 0.95] * (days - 1)
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]
    else:  # BUY PUT
        # First day hits stop loss
        prices = [base_price * 1.03] + [base_price * 1.05] * (days - 1)
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]
        
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices
    })
    
    return data

def setup_breakout_rsi_data(signal="BUY CALL"):
    """Create test data for the BreakoutRsi strategy."""
    data = create_base_data()
    
    # Set up the last candle for a breakout
    if signal == "BUY CALL":
        # Set the previous high (at index -2) to be below current close
        data.loc[len(data)-2, 'high'] = data.iloc[-1]['close'] * 0.98
        # Set RSI for a bullish scenario
        data.loc[len(data)-1, 'rsi'] = 65
    else:  # BUY PUT
        # Set the previous low (at index -2) to be above current close
        data.loc[len(data)-2, 'low'] = data.iloc[-1]['close'] * 1.02
        # Set RSI for a bearish scenario
        data.loc[len(data)-1, 'rsi'] = 35
    
    # Create additional fields for the BreakoutRsi strategy
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    return data

def setup_donchian_breakout_data(signal="BUY CALL"):
    """Create test data for the DonchianBreakout strategy."""
    data = create_base_data()
    
    # Calculate Donchian channels
    data['donchian_upper'] = data['high'].rolling(5).max().shift(1)
    data['donchian_lower'] = data['low'].rolling(5).min().shift(1)
    data['donchian_middle'] = (data['donchian_upper'] + data['donchian_lower']) / 2
    
    # Set up the last candle for a breakout
    if signal == "BUY CALL":
        # Make sure the close is above the upper Donchian channel
        data.loc[len(data)-1, 'donchian_upper'] = data.iloc[-1]['close'] * 0.98
        # Make sure prev_upper is set to match donchian_upper for the breakout check
        data.loc[len(data)-1, 'close'] = data.iloc[-1]['donchian_upper'] * 1.02
        # Set RSI for a bullish scenario
        data.loc[len(data)-1, 'rsi'] = 65
    else:  # BUY PUT
        # Make sure the close is below the lower Donchian channel
        data.loc[len(data)-1, 'donchian_lower'] = data.iloc[-1]['close'] * 1.02
        # Make sure prev_lower is set to match donchian_lower for the breakout check
        data.loc[len(data)-1, 'close'] = data.iloc[-1]['donchian_lower'] * 0.98
        # Set RSI for a bearish scenario
        data.loc[len(data)-1, 'rsi'] = 35
    
    # Add other required fields
    data['prev_upper'] = data['donchian_upper']
    data['prev_lower'] = data['donchian_lower']
    data['channel_width'] = (data['donchian_upper'] - data['donchian_lower']) / data['close'] * 100
    data['breakout_size'] = 0.5  # Set a reasonable breakout size
    data['volume_ratio'] = 1.5  # Set a volume spike
    
    return data

def setup_insidebar_bollinger_data(signal="BUY CALL"):
    """Create test data for the InsidebarBollinger strategy."""
    data = create_base_data()
    
    # Calculate inside bar indicators
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['is_inside'] = (data['high'] < data['prev_high']) & (data['low'] > data['prev_low'])
    data['is_partial_inside'] = (data['high'] < data['prev_high']) | (data['low'] > data['prev_low'])
    data['inside_bar_size'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low']) * 100
    
    # Set up the last candle for a signal
    if signal == "BUY CALL":
        # Set up an inside bar
        data.loc[len(data)-1, 'high'] = data.iloc[-2]['high'] * 0.99
        data.loc[len(data)-1, 'low'] = data.iloc[-2]['low'] * 1.01
        # Position close near lower band
        data.loc[len(data)-1, 'bollinger_lower'] = data.iloc[-1]['close'] * 0.99
        data.loc[len(data)-1, 'close'] = data.iloc[-1]['bollinger_lower'] * 1.001
        # Force is_inside to True
        data.loc[len(data)-1, 'is_inside'] = True
        data.loc[len(data)-1, 'is_partial_inside'] = True
    else:  # BUY PUT
        # Set up an inside bar
        data.loc[len(data)-1, 'high'] = data.iloc[-2]['high'] * 0.99
        data.loc[len(data)-1, 'low'] = data.iloc[-2]['low'] * 1.01
        # Position close near upper band
        data.loc[len(data)-1, 'bollinger_upper'] = data.iloc[-1]['close'] * 1.01
        data.loc[len(data)-1, 'close'] = data.iloc[-1]['bollinger_upper'] * 0.999
        # Force is_inside to True
        data.loc[len(data)-1, 'is_inside'] = True
        data.loc[len(data)-1, 'is_partial_inside'] = True
    
    # Calculate additional Bollinger Band indicators
    data['bollinger_width'] = (data['bollinger_upper'] - data['bollinger_lower']) / data['close'] * 100
    data['price_to_band_ratio'] = (data['close'] - data['bollinger_lower']) / (data['bollinger_upper'] - data['bollinger_lower'])
    data['lower_band_proximity'] = (data['close'] - data['bollinger_lower']) / data['close'] * 100
    data['upper_band_proximity'] = (data['bollinger_upper'] - data['close']) / data['close'] * 100
    
    return data

def setup_range_breakout_volatility_data(signal="BUY CALL"):
    """Create test data for the RangeBreakoutVolatility strategy."""
    data = create_base_data()
    
    # Calculate range indicators
    data['range_high'] = data['high'].rolling(5).max().shift(1)
    data['range_low'] = data['low'].rolling(5).min().shift(1)
    data['range_width'] = (data['range_high'] - data['range_low']) / data['close'] * 100
    
    # Set up the last candle for a breakout
    if signal == "BUY CALL":
        # Make sure the range high is below current close for upside breakout
        data.loc[len(data)-1, 'range_high'] = data.iloc[-1]['close'] * 0.98
        # Set RSI for a bullish scenario
        data.loc[len(data)-1, 'rsi'] = 65
    else:  # BUY PUT
        # Make sure the range low is above current close for downside breakout
        data.loc[len(data)-1, 'range_low'] = data.iloc[-1]['close'] * 1.02
        # Set RSI for a bearish scenario
        data.loc[len(data)-1, 'rsi'] = 35
    
    # Calculate additional volatility indicators
    data['breakout_size'] = 0.5  # Set a reasonable breakout size
    data['atr_ratio'] = 1.2  # Increased volatility
    data['volatility_rank'] = 75  # High volatility rank
    
    return data

def test_strategy_with_scenario(strategy_class, strategy_name, setup_data_func, signal_type, scenario):
    """Test a strategy with a specific scenario."""
    print(f"\n==== Testing {strategy_name} with {signal_type}, scenario: {scenario} ====")
    
    # Create a strategy instance
    strategy = strategy_class()
    
    # Create test data
    data = setup_data_func(signal=signal_type)
    
    # Get current price
    current_price = data.iloc[-1]['close']
    
    # Create future data
    if scenario == "win":
        future_data = create_future_data_win_scenario(base_price=current_price, signal=signal_type)
    elif scenario == "loss":
        future_data = create_future_data_loss_scenario(base_price=current_price, signal=signal_type)
    else:
        future_data = None
    
    # Analyze the data
    signal_data = strategy.analyze(data, future_data=future_data)
    
    # Print results
    print(f"Generated Signal: {signal_data['signal']}")
    if signal_data['signal'] != "NO TRADE":
        print(f"Outcome: {signal_data['outcome']}")
        print(f"PnL: {signal_data['pnl']}")
        print(f"Targets Hit: {signal_data['targets_hit']}")
        print(f"Stop Loss Count: {signal_data['stoploss_count']}")
        print(f"Failure Reason: {signal_data['failure_reason']}")
    else:
        print("No trade signal generated.")
    
    return signal_data

def main():
    """Test performance tracking for all strategies with manually constructed scenarios."""
    print("===== DIRECT TESTING OF PERFORMANCE TRACKING FOR ALL STRATEGIES =====")
    
    # Test BreakoutRsi strategy
    print("\n>>>>> Testing BreakoutRsi Strategy <<<<<")
    test_strategy_with_scenario(BreakoutRsi, "BreakoutRsi", setup_breakout_rsi_data, "BUY CALL", "win")
    test_strategy_with_scenario(BreakoutRsi, "BreakoutRsi", setup_breakout_rsi_data, "BUY CALL", "loss")
    test_strategy_with_scenario(BreakoutRsi, "BreakoutRsi", setup_breakout_rsi_data, "BUY PUT", "win")
    test_strategy_with_scenario(BreakoutRsi, "BreakoutRsi", setup_breakout_rsi_data, "BUY PUT", "loss")
    
    # Test DonchianBreakout strategy
    print("\n>>>>> Testing DonchianBreakout Strategy <<<<<")
    test_strategy_with_scenario(DonchianBreakout, "DonchianBreakout", setup_donchian_breakout_data, "BUY CALL", "win")
    test_strategy_with_scenario(DonchianBreakout, "DonchianBreakout", setup_donchian_breakout_data, "BUY CALL", "loss")
    test_strategy_with_scenario(DonchianBreakout, "DonchianBreakout", setup_donchian_breakout_data, "BUY PUT", "win")
    test_strategy_with_scenario(DonchianBreakout, "DonchianBreakout", setup_donchian_breakout_data, "BUY PUT", "loss")
    
    # Test InsidebarBollinger strategy
    print("\n>>>>> Testing InsidebarBollinger Strategy <<<<<")
    test_strategy_with_scenario(InsidebarBollinger, "InsidebarBollinger", setup_insidebar_bollinger_data, "BUY CALL", "win")
    test_strategy_with_scenario(InsidebarBollinger, "InsidebarBollinger", setup_insidebar_bollinger_data, "BUY CALL", "loss")
    test_strategy_with_scenario(InsidebarBollinger, "InsidebarBollinger", setup_insidebar_bollinger_data, "BUY PUT", "win")
    test_strategy_with_scenario(InsidebarBollinger, "InsidebarBollinger", setup_insidebar_bollinger_data, "BUY PUT", "loss")
    
    # Test RangeBreakoutVolatility strategy
    print("\n>>>>> Testing RangeBreakoutVolatility Strategy <<<<<")
    test_strategy_with_scenario(RangeBreakoutVolatility, "RangeBreakoutVolatility", setup_range_breakout_volatility_data, "BUY CALL", "win")
    test_strategy_with_scenario(RangeBreakoutVolatility, "RangeBreakoutVolatility", setup_range_breakout_volatility_data, "BUY CALL", "loss")
    test_strategy_with_scenario(RangeBreakoutVolatility, "RangeBreakoutVolatility", setup_range_breakout_volatility_data, "BUY PUT", "win")
    test_strategy_with_scenario(RangeBreakoutVolatility, "RangeBreakoutVolatility", setup_range_breakout_volatility_data, "BUY PUT", "loss")
    
    print("\n===== PERFORMANCE TRACKING TESTS COMPLETED =====")

if __name__ == "__main__":
    main() 