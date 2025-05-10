#!/usr/bin/env python3
"""
Test script to verify the complete flow of strategies, from analyze to performance tracking.
This script creates test data that will generate signals and then verifies that 
performance metrics are properly calculated and returned in the signal data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import all strategies
from src.strategies.breakout_rsi import BreakoutRsi
from src.strategies.donchian_breakout import DonchianBreakout
from src.strategies.insidebar_bollinger import InsidebarBollinger
from src.strategies.range_breakout_volatility import RangeBreakoutVolatility

def create_breakout_test_data(signal_type="BUY CALL", days=30):
    """Create test data specifically for breakout_rsi strategy."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Start with a price of 100
    close_prices = [100]
    for i in range(1, days):
        # Random daily change between -0.5% and +0.5%
        change = np.random.uniform(-0.5, 0.5)
        close_prices.append(close_prices[-1] * (1 + change/100))
    
    # Create high and low prices
    high_prices = [price * (1 + np.random.uniform(0, 0.5)/100) for price in close_prices]
    low_prices = [price * (1 - np.random.uniform(0, 0.5)/100) for price in close_prices]
    
    # Create open prices
    open_prices = [(h + l) / 2 for h, l in zip(high_prices, low_prices)]
    
    # ATR values (simulated)
    atr = [1.0] * days
    
    # Create RSI values
    if signal_type == "BUY CALL":
        rsi = [55] * days  # Bullish bias
    else:
        rsi = [45] * days  # Bearish bias
    
    # Make the last day a strong breakout
    if signal_type == "BUY CALL":
        # Make the previous high lower than current close for upside breakout
        high_prices[-2] = close_prices[-1] * 0.995
        close_prices[-1] = close_prices[-1] * 1.01  # 1% higher
        rsi[-1] = 65  # Strong bullish momentum
    else:
        # Make the previous low higher than current close for downside breakout
        low_prices[-2] = close_prices[-1] * 1.005
        close_prices[-1] = close_prices[-1] * 0.99  # 1% lower
        rsi[-1] = 35  # Strong bearish momentum
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': [1000] * days,
        'atr': atr,
        'rsi': rsi
    })
    
    return df

def create_donchian_test_data(signal_type="BUY CALL", days=30):
    """Create test data specifically for donchian_breakout strategy."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Start with a price of 100
    close_prices = [100]
    for i in range(1, days):
        # Random daily change between -0.5% and +0.5%
        change = np.random.uniform(-0.5, 0.5)
        close_prices.append(close_prices[-1] * (1 + change/100))
    
    # Create high and low prices
    high_prices = [price * (1 + np.random.uniform(0, 0.5)/100) for price in close_prices]
    low_prices = [price * (1 - np.random.uniform(0, 0.5)/100) for price in close_prices]
    
    # Create open prices
    open_prices = [(h + l) / 2 for h, l in zip(high_prices, low_prices)]
    
    # ATR values (simulated)
    atr = [1.0] * days
    
    # Create volume (increasing for the signal)
    volume = [1000] * days
    volume[-1] = 1500  # 50% volume increase on signal day
    
    # Make the last day a breakout beyond Donchian channel
    if signal_type == "BUY CALL":
        # Set previous 10 days to establish a channel
        max_prev = max(high_prices[-11:-1])
        close_prices[-1] = max_prev * 1.02  # Price closes 2% above the channel
        high_prices[-1] = close_prices[-1] * 1.005
    else:
        # Set previous 10 days to establish a channel
        min_prev = min(low_prices[-11:-1])
        close_prices[-1] = min_prev * 0.98  # Price closes 2% below the channel
        low_prices[-1] = close_prices[-1] * 0.995
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'atr': atr
    })
    
    return df

def create_insidebar_test_data(signal_type="BUY CALL", days=30):
    """Create test data specifically for insidebar_bollinger strategy."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Start with a price of 100
    close_prices = [100]
    for i in range(1, days):
        # Random daily change between -0.5% and +0.5%
        change = np.random.uniform(-0.5, 0.5)
        close_prices.append(close_prices[-1] * (1 + change/100))
    
    # Create high and low prices
    high_prices = [price * (1 + np.random.uniform(0, 0.5)/100) for price in close_prices]
    low_prices = [price * (1 - np.random.uniform(0, 0.5)/100) for price in close_prices]
    
    # Create open prices
    open_prices = [(h + l) / 2 for h, l in zip(high_prices, low_prices)]
    
    # ATR values (simulated)
    atr = [1.0] * days
    
    # Create Bollinger Bands (20-period SMA ± 2 standard deviations)
    # Simplified for test purposes
    sma = close_prices[-1]
    std = close_prices[-1] * 0.02  # 2% of price as standard deviation
    
    bollinger_upper = [close_prices[-1] + 2 * std] * days
    bollinger_lower = [close_prices[-1] - 2 * std] * days
    bollinger_middle = [close_prices[-1]] * days
    
    # Make the last 2 days match inside bar pattern
    if signal_type == "BUY CALL":
        # Previous day with wide range
        high_prices[-2] = close_prices[-1] * 1.03
        low_prices[-2] = close_prices[-1] * 0.97
        
        # Current day with inside bar near lower band
        high_prices[-1] = high_prices[-2] * 0.99  # Inside the previous day
        low_prices[-1] = low_prices[-2] * 1.01  # Inside the previous day
        close_prices[-1] = bollinger_lower[-1] * 1.005  # Close near lower band
    else:
        # Previous day with wide range
        high_prices[-2] = close_prices[-1] * 1.03
        low_prices[-2] = close_prices[-1] * 0.97
        
        # Current day with inside bar near upper band
        high_prices[-1] = high_prices[-2] * 0.99  # Inside the previous day
        low_prices[-1] = low_prices[-2] * 1.01  # Inside the previous day
        close_prices[-1] = bollinger_upper[-1] * 0.995  # Close near upper band
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': [1000] * days,
        'atr': atr,
        'bollinger_upper': bollinger_upper,
        'bollinger_lower': bollinger_lower,
        'bollinger_middle': bollinger_middle
    })
    
    return df

def create_range_breakout_test_data(signal_type="BUY CALL", days=30):
    """Create test data specifically for range_breakout_volatility strategy."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Start with a price of 100
    close_prices = [100]
    for i in range(1, days):
        # Random daily change between -0.5% and +0.5%
        change = np.random.uniform(-0.5, 0.5)
        close_prices.append(close_prices[-1] * (1 + change/100))
    
    # Create high and low prices
    high_prices = [price * (1 + np.random.uniform(0, 0.5)/100) for price in close_prices]
    low_prices = [price * (1 - np.random.uniform(0, 0.5)/100) for price in close_prices]
    
    # Create open prices
    open_prices = [(h + l) / 2 for h, l in zip(high_prices, low_prices)]
    
    # ATR values (simulated)
    atr = [1.0] * days
    volatility = [1.0] * days  # Normalized volatility
    
    # Create a range pattern followed by a breakout
    if signal_type == "BUY CALL":
        # Last 5 days in a range
        range_high = close_prices[-1] * 1.01
        range_low = close_prices[-1] * 0.99
        
        for i in range(6, 2, -1):
            high_prices[-i] = range_high
            low_prices[-i] = range_low
            close_prices[-i] = (range_high + range_low) / 2
        
        # Breakout on the last day
        close_prices[-1] = range_high * 1.02
        high_prices[-1] = close_prices[-1] * 1.005
        volatility[-1] = 1.2  # Increased volatility
    else:
        # Last 5 days in a range
        range_high = close_prices[-1] * 1.01
        range_low = close_prices[-1] * 0.99
        
        for i in range(6, 2, -1):
            high_prices[-i] = range_high
            low_prices[-i] = range_low
            close_prices[-i] = (range_high + range_low) / 2
        
        # Breakout on the last day
        close_prices[-1] = range_low * 0.98
        low_prices[-1] = close_prices[-1] * 0.995
        volatility[-1] = 1.2  # Increased volatility
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': [1000] * days,
        'atr': atr,
        'volatility': volatility
    })
    
    return df

def create_future_data(current_price, days=5, scenario="win", signal_type="BUY CALL"):
    """Create future data for testing performance tracking."""
    dates = [datetime.now() + timedelta(days=i) for i in range(1, days+1)]
    
    if scenario == "win":
        if signal_type == "BUY CALL":
            # Bullish scenario - price rises hitting targets
            close_prices = [current_price * (1 + 0.01 * i) for i in range(1, days+1)]
            high_prices = [p * 1.02 for p in close_prices]  # High enough to hit targets
            low_prices = [p * 0.99 for p in close_prices]   # Not low enough to hit stop loss
        else:  # BUY PUT
            # Bearish scenario - price falls hitting targets
            close_prices = [current_price * (1 - 0.01 * i) for i in range(1, days+1)]
            high_prices = [p * 1.01 for p in close_prices]  # Not high enough to hit stop loss
            low_prices = [p * 0.98 for p in close_prices]   # Low enough to hit targets
    else:  # loss
        if signal_type == "BUY CALL":
            # Price drops hitting stop loss on day 2
            close_prices = [current_price * 0.999] + [current_price * 0.97] + [current_price * 0.95] * (days - 2)
            high_prices = [p * 1.01 for p in close_prices]  # Not high enough to hit targets
            low_prices = [p * 0.97 for p in close_prices]   # Low enough to hit stop loss on day 2
        else:  # BUY PUT
            # Price rises hitting stop loss on day 2
            close_prices = [current_price * 1.001] + [current_price * 1.03] + [current_price * 1.05] * (days - 2)
            high_prices = [p * 1.03 for p in close_prices]  # High enough to hit stop loss on day 2
            low_prices = [p * 0.99 for p in close_prices]   # Not low enough to hit targets
    
    # Create open prices
    open_prices = [(h + l) / 2 for h, l in zip(high_prices, low_prices)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    
    return df

def test_strategy(strategy_class, name, test_data_func, signal_type="BUY CALL", scenario="win"):
    """Test a strategy end-to-end, from analyze to performance tracking."""
    print(f"\n==== Testing {name} Strategy with {signal_type}, scenario: {scenario} ====")
    
    # Create a strategy instance
    strategy = strategy_class()
    
    # Create test data that should generate a signal
    data = test_data_func(signal_type=signal_type)
    
    # Get current price from the last candle
    current_price = data.iloc[-1]['close']
    print(f"Entry Price: {current_price:.2f}")
    
    # Create future data based on the scenario
    future_data = create_future_data(
        current_price=current_price, 
        scenario=scenario, 
        signal_type=signal_type
    )
    
    # Analyze the data and get the signal
    signal_data = strategy.analyze(data, index_name="TEST", future_data=future_data)
    
    # Print results
    print(f"Generated Signal: {signal_data['signal']}")
    
    if signal_data['signal'] == signal_type:
        print(f"✅ Signal {signal_type} was generated correctly")
        
        # Print performance metrics
        print(f"Outcome: {signal_data['outcome']}")
        print(f"PnL: {signal_data['pnl']}")
        print(f"Targets Hit: {signal_data['targets_hit']}")
        print(f"Stop Loss Count: {signal_data['stoploss_count']}")
        
        # Check expectations for the scenario
        if scenario == "win":
            expected_outcome = "Win"
            expected_pnl_sign = "positive"
            expected_targets_hit = "at least 1"
            expected_stoploss_count = 0
        else:  # loss
            expected_outcome = "Loss"
            expected_pnl_sign = "negative"
            expected_targets_hit = 0
            expected_stoploss_count = 1
        
        # Verify outcome
        if signal_data['outcome'] == expected_outcome:
            print(f"✅ Outcome is {expected_outcome} as expected")
        else:
            print(f"❌ Outcome is {signal_data['outcome']}, expected {expected_outcome}")
        
        # Verify PnL
        if (expected_pnl_sign == "positive" and signal_data['pnl'] > 0) or \
           (expected_pnl_sign == "negative" and signal_data['pnl'] < 0):
            print(f"✅ PnL is {signal_data['pnl']} ({expected_pnl_sign}) as expected")
        else:
            print(f"❌ PnL is {signal_data['pnl']}, expected {expected_pnl_sign}")
        
        # Verify targets hit
        if (expected_targets_hit == "at least 1" and signal_data['targets_hit'] >= 1) or \
           (expected_targets_hit == 0 and signal_data['targets_hit'] == 0):
            print(f"✅ Targets hit is {signal_data['targets_hit']} as expected")
        else:
            print(f"❌ Targets hit is {signal_data['targets_hit']}, expected {expected_targets_hit}")
        
        # Verify stop loss count
        if signal_data['stoploss_count'] == expected_stoploss_count:
            print(f"✅ Stop loss count is {signal_data['stoploss_count']} as expected")
        else:
            print(f"❌ Stop loss count is {signal_data['stoploss_count']}, expected {expected_stoploss_count}")
    else:
        print(f"❌ Expected signal {signal_type}, but got {signal_data['signal']}")
    
    return signal_data

def main():
    """Run tests for all strategies with different scenarios."""
    print("===== TESTING COMPLETE STRATEGY FLOW WITH PERFORMANCE TRACKING =====")
    
    # Strategy class, name, and test data function pairs
    strategies = [
        (BreakoutRsi, "BreakoutRsi", create_breakout_test_data),
        (DonchianBreakout, "DonchianBreakout", create_donchian_test_data),
        (InsidebarBollinger, "InsidebarBollinger", create_insidebar_test_data),
        (RangeBreakoutVolatility, "RangeBreakoutVolatility", create_range_breakout_test_data)
    ]
    
    # Test all strategies with "win" scenario for BUY CALL
    print("\n>>>>> Testing WIN scenario for BUY CALL signals <<<<<")
    for strategy_class, name, test_data_func in strategies:
        test_strategy(strategy_class, name, test_data_func, signal_type="BUY CALL", scenario="win")
    
    # Test all strategies with "loss" scenario for BUY CALL
    print("\n>>>>> Testing LOSS scenario for BUY CALL signals <<<<<")
    for strategy_class, name, test_data_func in strategies:
        test_strategy(strategy_class, name, test_data_func, signal_type="BUY CALL", scenario="loss")
    
    # Test all strategies with "win" scenario for BUY PUT
    print("\n>>>>> Testing WIN scenario for BUY PUT signals <<<<<")
    for strategy_class, name, test_data_func in strategies:
        test_strategy(strategy_class, name, test_data_func, signal_type="BUY PUT", scenario="win")
    
    # Test all strategies with "loss" scenario for BUY PUT
    print("\n>>>>> Testing LOSS scenario for BUY PUT signals <<<<<")
    for strategy_class, name, test_data_func in strategies:
        test_strategy(strategy_class, name, test_data_func, signal_type="BUY PUT", scenario="loss")
    
    print("\n===== COMPLETE STRATEGY FLOW TESTS COMPLETED =====")

if __name__ == "__main__":
    main() 