#!/usr/bin/env python3
"""
Test script to verify performance tracking for all strategies.
This script directly tests the calculate_performance method for each strategy.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import all strategies
from src.strategies.breakout_rsi import BreakoutRsi
from src.strategies.donchian_breakout import DonchianBreakout
from src.strategies.insidebar_bollinger import InsidebarBollinger
from src.strategies.range_breakout_volatility import RangeBreakoutVolatility

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
            # Price drops hitting stop loss on first candle
            close_prices = [current_price * 0.97] + [current_price * 0.95] * (days - 1)
            high_prices = [p * 1.01 for p in close_prices]  # Not high enough to hit targets
            low_prices = [p * 0.95 for p in close_prices]   # Low enough to hit stop loss
        else:  # BUY PUT
            # Price rises hitting stop loss on first candle
            close_prices = [current_price * 1.03] + [current_price * 1.05] * (days - 1)
            high_prices = [p * 1.05 for p in close_prices]  # High enough to hit stop loss
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

def test_performance_direct(strategy_class, name, signal_type="BUY CALL", scenario="win"):
    """Test the calculate_performance method directly on a strategy."""
    print(f"\n==== Testing {name} calculate_performance with {signal_type}, scenario: {scenario} ====")
    
    # Create a strategy instance
    strategy = strategy_class()
    
    # Fixed values for testing
    entry_price = 100.0
    atr = 1.0  # ATR value for calculating stop loss and targets
    
    # Calculate stop loss and targets based on ATR
    stop_loss = round(atr * 1.0, 2)  # 1.0 ATR
    target = round(atr * 1.5, 2)     # 1.5 ATR
    target2 = round(atr * 2.0, 2)    # 2.0 ATR
    target3 = round(atr * 2.5, 2)    # 2.5 ATR
    
    # Create future data
    future_data = create_future_data(entry_price, scenario=scenario, signal_type=signal_type)
    
    # Call calculate_performance directly
    performance = strategy.calculate_performance(
        signal=signal_type,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target=target,
        target2=target2,
        target3=target3,
        future_data=future_data
    )
    
    # Print the performance metrics
    print(f"Entry Price: {entry_price}")
    print(f"Stop Loss: {stop_loss}")
    print(f"Target 1: {target}")
    print(f"Target 2: {target2}")
    print(f"Target 3: {target3}")
    print(f"Outcome: {performance['outcome']}")
    print(f"PnL: {performance['pnl']}")
    print(f"Targets Hit: {performance['targets_hit']}")
    print(f"Stop Loss Count: {performance['stoploss_count']}")
    print(f"Failure Reason: {performance['failure_reason']}")
    
    # Verify results
    if scenario == "win":
        assert performance['outcome'] == "Win"
        assert performance['pnl'] > 0
        assert performance['targets_hit'] > 0
        assert performance['stoploss_count'] == 0
    else:  # loss
        assert performance['outcome'] == "Loss"
        assert performance['pnl'] < 0
        assert performance['targets_hit'] == 0
        assert performance['stoploss_count'] == 1
    
    return performance

def main():
    """Run direct tests for all strategies."""
    print("===== DIRECT TESTING OF PERFORMANCE CALCULATION =====")
    
    # List of strategies to test
    strategies = [
        (BreakoutRsi, "BreakoutRsi"),
        (DonchianBreakout, "DonchianBreakout"),
        (InsidebarBollinger, "InsidebarBollinger"),
        (RangeBreakoutVolatility, "RangeBreakoutVolatility")
    ]
    
    # Test all strategies with "win" scenario for BUY CALL
    print("\n>>>>> Testing WIN scenario for BUY CALL signals <<<<<")
    for strategy_class, name in strategies:
        test_performance_direct(strategy_class, name, signal_type="BUY CALL", scenario="win")
    
    # Test all strategies with "loss" scenario for BUY CALL
    print("\n>>>>> Testing LOSS scenario for BUY CALL signals <<<<<")
    for strategy_class, name in strategies:
        test_performance_direct(strategy_class, name, signal_type="BUY CALL", scenario="loss")
    
    # Test all strategies with "win" scenario for BUY PUT
    print("\n>>>>> Testing WIN scenario for BUY PUT signals <<<<<")
    for strategy_class, name in strategies:
        test_performance_direct(strategy_class, name, signal_type="BUY PUT", scenario="win")
    
    # Test all strategies with "loss" scenario for BUY PUT
    print("\n>>>>> Testing LOSS scenario for BUY PUT signals <<<<<")
    for strategy_class, name in strategies:
        test_performance_direct(strategy_class, name, signal_type="BUY PUT", scenario="loss")
    
    print("\n===== PERFORMANCE CALCULATION TESTS COMPLETED =====")

if __name__ == "__main__":
    main() 