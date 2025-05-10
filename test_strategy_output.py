#!/usr/bin/env python3
"""
Test script to verify strategy outputs including performance metrics.
This script creates sample data, runs each strategy, and displays the full output.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import all strategies
from src.strategies.breakout_rsi import BreakoutRsi, run_strategy as run_breakout_rsi
from src.strategies.donchian_breakout import DonchianBreakout, run_strategy as run_donchian_breakout
from src.strategies.insidebar_bollinger import InsidebarBollinger, run_strategy as run_insidebar_bollinger
from src.strategies.range_breakout_volatility import RangeBreakoutVolatility, run_strategy as run_range_breakout_volatility

def create_sample_data(signal_type="BUY CALL", days=30, current_price=100):
    """Create sample market data with indicators for testing."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Generate price data
    if signal_type == "BUY CALL":
        # Bullish trend
        close_prices = [current_price * (1 - 0.2 * (i/days)) for i in range(days)]
    else:
        # Bearish trend
        close_prices = [current_price * (1 + 0.2 * (i/days)) for i in range(days)]
    
    # Create high and low prices with some volatility
    high_prices = [price * (1 + np.random.uniform(0.01, 0.03)) for price in close_prices]
    low_prices = [price * (1 - np.random.uniform(0.01, 0.03)) for price in close_prices]
    open_prices = [price * (1 + np.random.uniform(-0.01, 0.01)) for price in close_prices]
    
    # Create volume
    volume = [np.random.uniform(1000, 10000) for _ in range(days)]
    
    # Create common indicators
    # ATR
    tr_values = []
    for i in range(days):
        if i == 0:
            tr = high_prices[i] - low_prices[i]
        else:
            tr_1 = high_prices[i] - low_prices[i]
            tr_2 = abs(high_prices[i] - close_prices[i-1])
            tr_3 = abs(low_prices[i] - close_prices[i-1])
            tr = max(tr_1, tr_2, tr_3)
        tr_values.append(tr)
    
    atr_period = 14
    atr = []
    for i in range(days):
        if i < atr_period:
            atr.append(np.mean(tr_values[:i+1]))
        else:
            atr.append(np.mean(tr_values[i-atr_period+1:i+1]))
    
    # RSI
    rsi = []
    for i in range(days):
        if signal_type == "BUY CALL":
            rsi.append(np.random.uniform(50, 70))
        else:
            rsi.append(np.random.uniform(30, 50))
    
    # Bollinger Bands
    sma_period = 20
    sma = []
    for i in range(days):
        if i < sma_period:
            sma.append(np.mean(close_prices[:i+1]))
        else:
            sma.append(np.mean(close_prices[i-sma_period+1:i+1]))
    
    # Standard deviation
    std = []
    for i in range(days):
        if i < sma_period:
            std.append(np.std(close_prices[:i+1]))
        else:
            std.append(np.std(close_prices[i-sma_period+1:i+1]))
    
    # Bollinger Bands
    bollinger_upper = [sma[i] + 2 * std[i] for i in range(days)]
    bollinger_lower = [sma[i] - 2 * std[i] for i in range(days)]
    bollinger_middle = sma
    
    # MACD
    ema12_period = 12
    ema26_period = 26
    signal_period = 9
    
    ema12 = []
    ema26 = []
    for i in range(days):
        if i == 0:
            ema12.append(close_prices[i])
            ema26.append(close_prices[i])
        else:
            k12 = 2 / (ema12_period + 1)
            k26 = 2 / (ema26_period + 1)
            ema12.append(close_prices[i] * k12 + ema12[i-1] * (1-k12))
            ema26.append(close_prices[i] * k26 + ema26[i-1] * (1-k26))
    
    macd = [ema12[i] - ema26[i] for i in range(days)]
    
    signal = []
    for i in range(days):
        if i == 0:
            signal.append(macd[i])
        else:
            k = 2 / (signal_period + 1)
            signal.append(macd[i] * k + signal[i-1] * (1-k))
    
    macd_hist = [macd[i] - signal[i] for i in range(days)]
    
    # Create the DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'atr': atr,
        'rsi': rsi,
        'bollinger_upper': bollinger_upper,
        'bollinger_lower': bollinger_lower,
        'bollinger_middle': bollinger_middle,
        'macd': macd,
        'macd_signal': signal,
        'macd_hist': macd_hist
    })
    
    # Set specific values for the last candle to generate signals
    if signal_type == "BUY CALL":
        # For BreakoutRsi - set previous high below current close
        df.loc[len(df)-2, 'high'] = df.iloc[-1]['close'] * 0.98
        
        # For DonchianBreakout - price breaks above the upper channel
        df['donchian_upper'] = df['high'].rolling(10).max().shift(1)
        df['donchian_lower'] = df['low'].rolling(10).min().shift(1)
        df.loc[len(df)-1, 'donchian_upper'] = df.iloc[-1]['close'] * 0.98
        
        # For InsidebarBollinger - inside bar near lower band
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['is_inside'] = (df['high'] < df['prev_high']) & (df['low'] > df['prev_low'])
        df.loc[len(df)-2, 'high'] = df.iloc[-1]['high'] * 1.1  # Previous high > current high
        df.loc[len(df)-1, 'high'] = df.iloc[-2]['high'] * 0.95  # Current high < previous high
        df.loc[len(df)-1, 'is_inside'] = True
        df.loc[len(df)-1, 'bollinger_lower'] = df.iloc[-1]['close'] * 0.99  # Close near lower band
        
        # For RangeBreakoutVolatility - price breaks above range high
        df['range_high'] = df['high'].rolling(5).max().shift(1)
        df['range_low'] = df['low'].rolling(5).min().shift(1)
        df.loc[len(df)-1, 'range_high'] = df.iloc[-1]['close'] * 0.98
        
    else:  # BUY PUT
        # For BreakoutRsi - set previous low above current close
        df.loc[len(df)-2, 'low'] = df.iloc[-1]['close'] * 1.02
        
        # For DonchianBreakout - price breaks below the lower channel
        df['donchian_upper'] = df['high'].rolling(10).max().shift(1)
        df['donchian_lower'] = df['low'].rolling(10).min().shift(1)
        df.loc[len(df)-1, 'donchian_lower'] = df.iloc[-1]['close'] * 1.02
        
        # For InsidebarBollinger - inside bar near upper band
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['is_inside'] = (df['high'] < df['prev_high']) & (df['low'] > df['prev_low'])
        df.loc[len(df)-2, 'low'] = df.iloc[-1]['low'] * 0.9  # Previous low < current low
        df.loc[len(df)-1, 'low'] = df.iloc[-2]['low'] * 1.05  # Current low > previous low
        df.loc[len(df)-1, 'is_inside'] = True
        df.loc[len(df)-1, 'bollinger_upper'] = df.iloc[-1]['close'] * 1.01  # Close near upper band
        
        # For RangeBreakoutVolatility - price breaks below range low
        df['range_high'] = df['high'].rolling(5).max().shift(1)
        df['range_low'] = df['low'].rolling(5).min().shift(1)
        df.loc[len(df)-1, 'range_low'] = df.iloc[-1]['close'] * 1.02
    
    # Additional indicators for strategies
    # For InsidebarBollinger
    df['is_partial_inside'] = df['is_inside'].copy()
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['close'] * 100
    df['price_to_band_ratio'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
    df['inside_bar_size'] = 90  # Assume inside bar is 90% of previous bar
    df['lower_band_proximity'] = (df['close'] - df['bollinger_lower']) / df['close'] * 100
    df['upper_band_proximity'] = (df['bollinger_upper'] - df['close']) / df['close'] * 100
    
    # For DonchianBreakout
    df['prev_upper'] = df['donchian_upper']
    df['prev_lower'] = df['donchian_lower']
    df['channel_width'] = (df['donchian_upper'] - df['donchian_lower']) / df['close'] * 100
    df['breakout_size'] = 0.5  # Set a reasonable breakout size
    df['volume_ratio'] = 1.5  # Set a volume spike
    
    # For RangeBreakoutVolatility
    df['range_width'] = (df['range_high'] - df['range_low']) / df['close'] * 100
    df['breakout_size'] = 0.5
    df['atr_ratio'] = 1.2
    df['volatility_rank'] = 75
    
    # For BreakoutRsi
    df['breakout_strength'] = 0.5
    df['rsi_alignment'] = 65 if signal_type == "BUY CALL" else 35
    
    return df

def create_future_data(current_price, days=5, scenario="win", signal_type="BUY CALL"):
    """Create future data for testing performance metrics."""
    dates = [datetime.now() + timedelta(days=i) for i in range(1, days+1)]
    
    if scenario == "win":
        if signal_type == "BUY CALL":
            # Bullish scenario - price rises hitting targets
            close_prices = [current_price * (1 + 0.01 * i) for i in range(1, days+1)]
        else:  # BUY PUT
            # Bearish scenario - price falls hitting targets
            close_prices = [current_price * (1 - 0.01 * i) for i in range(1, days+1)]
    else:  # loss
        if signal_type == "BUY CALL":
            # Price drops hitting stop loss
            close_prices = [current_price * 0.97] + [current_price * 0.95] * (days - 1)
        else:  # BUY PUT
            # Price rises hitting stop loss
            close_prices = [current_price * 1.03] + [current_price * 1.05] * (days - 1)
    
    # Create high and low prices
    if signal_type == "BUY CALL" and scenario == "win":
        # For BUY CALL win, high prices should reach target levels
        high_prices = [p * 1.02 for p in close_prices]
        low_prices = [p * 0.99 for p in close_prices]
    elif signal_type == "BUY PUT" and scenario == "win":
        # For BUY PUT win, low prices should reach target levels
        high_prices = [p * 1.01 for p in close_prices]
        low_prices = [p * 0.98 for p in close_prices]
    elif signal_type == "BUY CALL" and scenario == "loss":
        # For BUY CALL loss, low prices should hit stop loss
        high_prices = [p * 1.01 for p in close_prices]
        low_prices = [p * 0.95 for p in close_prices]  # Make sure low hits stop loss
    else:  # BUY PUT loss
        # For BUY PUT loss, high prices should hit stop loss
        high_prices = [p * 1.05 for p in close_prices]  # Make sure high hits stop loss
        low_prices = [p * 0.99 for p in close_prices]
    
    open_prices = [(h + l) / 2 for h, l in zip(high_prices, low_prices)]
    
    future_df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    
    return future_df

def print_dict(d, indent=0):
    """Print a dictionary with nice formatting."""
    for key, value in d.items():
        if key == "signal_time":
            continue
        if isinstance(value, dict):
            print(' ' * indent + str(key) + ':')
            print_dict(value, indent + 4)
        else:
            print(' ' * indent + f"{key}: {value}")

def test_strategy(strategy_name, strategy_class, run_function, signal_type="BUY CALL", scenario="win"):
    """Test a strategy and print its output."""
    print(f"\n===== Testing {strategy_name} with {signal_type}, scenario: {scenario} =====")
    
    # Create sample data
    df = create_sample_data(signal_type=signal_type)
    current_price = df.iloc[-1]['close']
    
    # Create future data
    future_df = create_future_data(current_price, scenario=scenario, signal_type=signal_type)
    
    # Create strategy instance
    strategy = strategy_class()
    
    # Run the strategy via class method
    signal_data_class = strategy.analyze(df, future_data=future_df)
    
    # Run the strategy via function (backward compatibility)
    signal_data_func = run_function(df.iloc[-1], df.iloc[-2], "TEST", future_data=future_df)
    
    print("\nOutput from class method:")
    print_dict(signal_data_class)
    
    print("\nOutput from function method:")
    print_dict(signal_data_func)
    
    return signal_data_class, signal_data_func

def main():
    """Run tests for each strategy and scenario."""
    strategies = [
        ("BreakoutRsi", BreakoutRsi, run_breakout_rsi),
        ("DonchianBreakout", DonchianBreakout, run_donchian_breakout),
        ("InsidebarBollinger", InsidebarBollinger, run_insidebar_bollinger),
        ("RangeBreakoutVolatility", RangeBreakoutVolatility, run_range_breakout_volatility)
    ]
    
    for strategy_name, strategy_class, run_function in strategies:
        # Test BUY CALL scenarios
        test_strategy(strategy_name, strategy_class, run_function, "BUY CALL", "win")
        test_strategy(strategy_name, strategy_class, run_function, "BUY CALL", "loss")
        
        # Test BUY PUT scenarios
        test_strategy(strategy_name, strategy_class, run_function, "BUY PUT", "win")
        test_strategy(strategy_name, strategy_class, run_function, "BUY PUT", "loss")

if __name__ == "__main__":
    main() 