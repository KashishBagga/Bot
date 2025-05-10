#!/usr/bin/env python3
"""
Test script to verify performance metrics are properly tracked when logging new signals.
This script creates test data with future candles, runs strategies, and ensures the
resulting signal contains properly calculated performance metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

# Import all strategies
from src.strategies.breakout_rsi import BreakoutRsi
from src.strategies.donchian_breakout import DonchianBreakout
from src.strategies.insidebar_bollinger import InsidebarBollinger
from src.strategies.range_breakout_volatility import RangeBreakoutVolatility
from src.strategies.ema_crossover import EmaCrossover
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.strategies.insidebar_rsi import InsidebarRsi

# Map strategy class names to their database table names
STRATEGY_DB_TABLES = {
    'BreakoutRsi': 'breakout_rsi',
    'DonchianBreakout': 'donchian_breakout',
    'InsidebarBollinger': 'insidebar_bollinger',
    'RangeBreakoutVolatility': 'range_breakout_volatility',
    'EmaCrossover': 'ema_crossover',
    'SupertrendEma': 'supertrend_ema',
    'SupertrendMacdRsiEma': 'supertrend_macd_rsi_ema',
    'InsidebarRsi': 'insidebar_rsi'
}

def create_sample_data(strategy_name=None, signal_type="BUY CALL", days=10, current_price=100):
    """Create sample market data for strategy testing.
    
    Args:
        strategy_name: Name of the strategy class to create data for
        signal_type: Target signal to generate (BUY CALL or BUY PUT)
        days: Number of days of data to generate
        current_price: Current price to use as a reference
        
    Returns:
        DataFrame with market data tailored for the strategy
    """
    # Create base data
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Generate price data with a trend
    if signal_type == "BUY CALL":
        # Bullish trend
        close_prices = [current_price * (1 - 0.01 * i) for i in range(days)]
    else:
        # Bearish trend
        close_prices = [current_price * (1 + 0.01 * i) for i in range(days)]
    
    # Create high, low, and open prices
    high_prices = [price * (1 + np.random.uniform(0.005, 0.015)) for price in close_prices]
    low_prices = [price * (1 - np.random.uniform(0.005, 0.015)) for price in close_prices]
    open_prices = [price * (1 + np.random.uniform(-0.005, 0.005)) for price in close_prices]
    
    # Create volume data
    volume = [np.random.uniform(1000, 10000) for _ in range(days)]
    
    # Create ATR values (simplified)
    atr = [1.0] * days
    
    # Create RSI values (simplified)
    if signal_type == "BUY CALL":
        rsi = [60 + np.random.uniform(-5, 15) for _ in range(days)]  # 55-75 range for buy calls
    else:
        rsi = [40 + np.random.uniform(-15, 5) for _ in range(days)]  # 25-45 range for buy puts
    
    # Create Bollinger Bands (simplified)
    bollinger_upper = [p * 1.02 for p in close_prices]
    bollinger_lower = [p * 0.98 for p in close_prices]
    bollinger_middle = close_prices
    
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
        'bollinger_middle': bollinger_middle
    })
    
    # Create EMA indicators for crossover strategies
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # For EmaCrossover and EmaCrossoverOriginal strategies
    if 'EmaCrossover' in strategy_name:
        if signal_type == "BUY CALL":
            # Make fast EMA cross above slow EMA in the last candle
            df.loc[len(df)-2, 'ema_9'] = df.iloc[-2]['ema_21'] * 0.99  # Just below in the previous candle
            df.loc[len(df)-1, 'ema_9'] = df.iloc[-1]['ema_21'] * 1.01  # Just above in the current candle
            # Make sure price is above fast EMA
            df.loc[len(df)-1, 'close'] = df.iloc[-1]['ema_9'] * 1.01
        else:  # BUY PUT
            # Make fast EMA cross below slow EMA in the last candle
            df.loc[len(df)-2, 'ema_9'] = df.iloc[-2]['ema_21'] * 1.01  # Just above in the previous candle
            df.loc[len(df)-1, 'ema_9'] = df.iloc[-1]['ema_21'] * 0.99  # Just below in the current candle
            # Make sure price is below fast EMA
            df.loc[len(df)-1, 'close'] = df.iloc[-1]['ema_9'] * 0.99
            
        # Calculate crossover strength
        df['crossover_strength'] = (df['ema_9'] - df['ema_21']) / df['close'] * 100
        df['ema_fast'] = df['ema_9']
        df['ema_slow'] = df['ema_21']
        df['ema_fast_change'] = df['ema_fast'].pct_change() * 100
    
    # For SupertrendEma strategy
    if strategy_name == "SupertrendEma":
        # Calculate price to EMA ratio
        df['price_to_ema_ratio'] = (df['close'] / df['ema_20'] - 1) * 100
        
        # Add supertrend indicator (simplified)
        df['supertrend'] = 1 if signal_type == "BUY CALL" else -1
        
        # Make sure price is on the right side of EMA
        if signal_type == "BUY CALL":
            df.loc[len(df)-1, 'close'] = df.iloc[-1]['ema_20'] * 1.01  # Price above EMA
        else:
            df.loc[len(df)-1, 'close'] = df.iloc[-1]['ema_20'] * 0.99  # Price below EMA
    
    # For SupertrendMacdRsiEma strategy
    if strategy_name == "SupertrendMacdRsiEma":
        # Add MACD
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Add Supertrend direction
        df['supertrend_direction'] = 1 if signal_type == "BUY CALL" else -1
        
        # Make sure RSI, MACD, and price are aligned with signal
        if signal_type == "BUY CALL":
            df.loc[len(df)-1, 'rsi'] = 60  # RSI > 55
            df.loc[len(df)-1, 'macd'] = df.iloc[-1]['macd_signal'] * 1.1  # MACD above signal
            df.loc[len(df)-1, 'close'] = df.iloc[-1]['ema_20'] * 1.01  # Price above EMA
        else:
            df.loc[len(df)-1, 'rsi'] = 40  # RSI < 45
            df.loc[len(df)-1, 'macd'] = df.iloc[-1]['macd_signal'] * 0.9  # MACD below signal
            df.loc[len(df)-1, 'close'] = df.iloc[-1]['ema_20'] * 0.99  # Price below EMA
            
        # Add columns for candle analysis
        df['ema'] = df['ema_20']
        df['body'] = abs(df['close'] - df['open'])
        df['full_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / df['full_range']
    
    # For InsidebarRsi strategy
    if strategy_name == "InsidebarRsi":
        # Add prev_high and prev_low
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        
        # Create inside bar pattern in the last candle
        df.loc[len(df)-1, 'high'] = df.iloc[-2]['high'] * 0.99
        df.loc[len(df)-1, 'low'] = df.iloc[-2]['low'] * 1.01
        
        # Set RSI level for proper signal
        df['rsi_level'] = 'overbought' if signal_type == "BUY PUT" else 'oversold'
        
        # Adjust RSI
        if signal_type == "BUY CALL":
            df.loc[len(df)-1, 'rsi'] = 30  # Oversold for buy call
        else:
            df.loc[len(df)-1, 'rsi'] = 70  # Overbought for buy put
    
    # Strategy-specific adjustments for the other strategies remain the same
    if strategy_name == "BreakoutRsi":
        if signal_type == "BUY CALL":
            # For BreakoutRsi
            df.loc[len(df)-2, 'high'] = df.iloc[-1]['close'] * 0.99  # previous high below current close
        else:
            # For BreakoutRsi
            df.loc[len(df)-2, 'low'] = df.iloc[-1]['close'] * 1.01  # previous low above current close
    
    elif strategy_name == "DonchianBreakout":
        # Create Donchian Channel
        df['donchian_upper'] = df['high'].rolling(5).max().shift(1)
        df['donchian_lower'] = df['low'].rolling(5).min().shift(1)
        
        if signal_type == "BUY CALL":
            df.loc[len(df)-1, 'donchian_upper'] = df.iloc[-1]['close'] * 0.99
        else:
            df.loc[len(df)-1, 'donchian_lower'] = df.iloc[-1]['close'] * 1.01
    
    elif strategy_name == "InsidebarBollinger":
        # Create inside bar pattern
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['is_inside'] = (df['high'] < df['prev_high']) & (df['low'] > df['prev_low'])
        df.loc[len(df)-1, 'is_inside'] = True
        
        if signal_type == "BUY CALL":
            df.loc[len(df)-1, 'bollinger_lower'] = df.iloc[-1]['close'] * 0.99
        else:
            df.loc[len(df)-1, 'bollinger_upper'] = df.iloc[-1]['close'] * 1.01
    
    elif strategy_name == "RangeBreakoutVolatility":
        # Create range high and low
        df['range_high'] = df['high'].rolling(5).max().shift(1)
        df['range_low'] = df['low'].rolling(5).min().shift(1)
        
        if signal_type == "BUY CALL":
            df.loc[len(df)-1, 'range_high'] = df.iloc[-1]['close'] * 0.99
        else:
            df.loc[len(df)-1, 'range_low'] = df.iloc[-1]['close'] * 1.01
    
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

def check_db_for_signals(table, latest_signals=3):
    """Check the database for the latest signals to confirm performance metrics."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"""
            SELECT id, signal_time, signal, price, outcome, pnl, targets_hit, stoploss_count, failure_reason
            FROM {table}
            ORDER BY id DESC
            LIMIT {latest_signals}
        """)
        signals = cursor.fetchall()
        
        if signals:
            print(f"\nLatest {len(signals)} signals in {table}:")
            for signal in signals:
                id, time, signal_type, price, outcome, pnl, targets_hit, stoploss_count, failure = signal
                print(f"ID: {id}, Time: {time}, Signal: {signal_type}, Price: {price}")
                print(f"  Outcome: {outcome}, PnL: {pnl}, Targets Hit: {targets_hit}, SL Count: {stoploss_count}")
                if failure:
                    print(f"  Failure Reason: {failure}")
                print("---")
        else:
            print(f"No signals found in {table}")
            
    except Exception as e:
        print(f"Error querying {table}: {e}")
    
    conn.close()

def test_strategy_logging(strategy_class, strategy_name, signal_type="BUY CALL", scenario="win"):
    """Test a strategy end-to-end, analyzing data and logging signal with performance metrics."""
    print(f"\n==== Testing {strategy_name} with {signal_type}, scenario: {scenario} ====")
    
    # Get the correct database table name
    db_table = STRATEGY_DB_TABLES.get(strategy_name, strategy_name.lower())
    
    # Create a strategy instance
    strategy = strategy_class()
    
    # Create sample data tailored for the strategy and future data
    data = create_sample_data(strategy_name=strategy_name, signal_type=signal_type)
    current_price = data.iloc[-1]['close']
    future_data = create_future_data(current_price, scenario=scenario, signal_type=signal_type)
    
    # Check the analyze method's signature
    import inspect
    try:
        analyze_sig = inspect.signature(strategy.analyze)
        analyze_params = analyze_sig.parameters
        
        # Run the strategy with appropriate parameters
        if 'index_name' in analyze_params and 'future_data' in analyze_params:
            # Full signature with index_name and future_data
            signal_data = strategy.analyze(data, index_name=f"TEST_{strategy_name}", future_data=future_data)
        elif 'future_data' in analyze_params:
            # Signature with future_data but no index_name
            signal_data = strategy.analyze(data, future_data=future_data)
        else:
            # Basic signature with just data
            signal_data = strategy.analyze(data)
            
            # If the strategy doesn't accept future_data, we need to log to the database manually
            # for strategies that don't have logging in their analyze method
            if signal_data['signal'] != "NO TRADE":
                signal_data_copy = signal_data.copy()
                signal_data_copy["signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                signal_data_copy["index_name"] = f"TEST_{strategy_name}"
                
                # Calculate performance metrics if possible
                if has_calculate_performance_method(strategy):
                    entry_price = signal_data['price']
                    stop_loss = signal_data.get('stop_loss', 1.0)
                    target = signal_data.get('target', 1.5) 
                    target2 = signal_data.get('target2', 2.0)
                    target3 = signal_data.get('target3', 2.5)
                    
                    performance = strategy.calculate_performance(
                        signal=signal_data['signal'],
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        target2=target2,
                        target3=target3,
                        future_data=future_data
                    )
                    
                    # Update signal_data with performance metrics
                    signal_data_copy.update(performance)
                
                # Log to database
                from db import log_strategy_sql
                log_strategy_sql(db_table, signal_data_copy)
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return None
    
    # Display the generated signal
    print(f"Generated Signal: {signal_data['signal']}")
    
    if signal_data['signal'] == signal_type:
        print("✅ Signal generated correctly")
        
        # Display performance metrics if they exist
        outcome = signal_data.get('outcome', 'Not tracked')
        pnl = signal_data.get('pnl', 0)
        targets_hit = signal_data.get('targets_hit', 0)
        stoploss_count = signal_data.get('stoploss_count', 0)
        
        print(f"Outcome: {outcome}")
        print(f"PnL: {pnl}")
        print(f"Targets Hit: {targets_hit}")
        print(f"Stop Loss Count: {stoploss_count}")
        
        # Verify expected outcomes for strategies that track performance
        if outcome != 'Not tracked' and outcome != 'Pending':
            if scenario == "win":
                expected_outcome = "Win"
                expected_pnl_sign = "positive"
            else:
                expected_outcome = "Loss"
                expected_pnl_sign = "negative"
            
            # Check if metrics match expectations
            outcome_correct = outcome == expected_outcome
            pnl_correct = (expected_pnl_sign == "positive" and pnl > 0) or (expected_pnl_sign == "negative" and pnl < 0)
            
            if outcome_correct and pnl_correct:
                print("✅ Performance metrics calculated correctly")
            else:
                print("❌ Performance metrics incorrect")
                if not outcome_correct:
                    print(f"  Expected outcome: {expected_outcome}, Got: {outcome}")
                if not pnl_correct:
                    print(f"  Expected PnL sign: {expected_pnl_sign}, Got: {pnl}")
        else:
            print("⚠️ This strategy doesn't track performance in the same way")
    else:
        print(f"❌ Expected signal {signal_type}, got {signal_data['signal']}")
    
    # Check what's stored in the database
    print("\nChecking database entries:")
    check_db_for_signals(db_table)
    
    return signal_data

def has_calculate_performance_method(strategy):
    """Check if a strategy has the expected calculate_performance method."""
    try:
        method = getattr(strategy, 'calculate_performance', None)
        if method is None:
            return False
            
        # Try to get the method's signature
        import inspect
        sig = inspect.signature(method)
        params = sig.parameters
        
        # Check if the method has the expected parameters
        required_params = ['signal', 'entry_price', 'stop_loss', 'target', 'future_data']
        return all(param in params for param in required_params)
    except:
        return False

def test_performance_direct(strategy_class, strategy_name, signal_type="BUY CALL", scenario="win"):
    """Test the calculate_performance method directly on a strategy."""
    print(f"\n==== Testing {strategy_name} calculate_performance directly ====")
    
    # Create a strategy instance
    strategy = strategy_class()
    
    # Check if the strategy has a calculate_performance method with the expected signature
    if not has_calculate_performance_method(strategy):
        print(f"⚠️ {strategy_name} doesn't have a compatible calculate_performance method.")
        print("  Performance metrics are calculated differently in this strategy.")
        return None
    
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
    
    try:
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
        print(f"Outcome: {performance['outcome']}")
        print(f"PnL: {performance['pnl']}")
        print(f"Targets Hit: {performance['targets_hit']}")
        print(f"Stop Loss Count: {performance['stoploss_count']}")
        print(f"Failure Reason: {performance['failure_reason']}")
        
        # Verify expectations
        if scenario == "win":
            expected_outcome = "Win"
            expected_pnl_sign = "positive"
        else:
            expected_outcome = "Loss" 
            expected_pnl_sign = "negative"
        
        outcome_correct = performance['outcome'] == expected_outcome
        pnl_correct = (expected_pnl_sign == "positive" and performance['pnl'] > 0) or (expected_pnl_sign == "negative" and performance['pnl'] < 0)
        
        if outcome_correct and pnl_correct:
            print("✅ Direct performance calculation works correctly")
        else:
            print("❌ Performance calculation has issues")
        
        return performance
    except Exception as e:
        print(f"❌ Error testing performance calculation: {e}")
        return None

def main():
    """Run comprehensive tests for performance tracking."""
    print("===== TESTING PERFORMANCE TRACKING IN NEW SIGNAL LOGGING =====")
    
    # List of strategies to test
    strategies = [
        (BreakoutRsi, "BreakoutRsi"),
        (DonchianBreakout, "DonchianBreakout"),
        (InsidebarBollinger, "InsidebarBollinger"),
        (RangeBreakoutVolatility, "RangeBreakoutVolatility"),
        (EmaCrossover, "EmaCrossover"),
        (SupertrendEma, "SupertrendEma"),
        (SupertrendMacdRsiEma, "SupertrendMacdRsiEma"),
        (InsidebarRsi, "InsidebarRsi")
    ]
    
    # Test direct performance calculation first for strategies that support it
    for strategy_class, name in strategies:
        # Create an instance to check if it has calculate_performance method
        strategy = strategy_class()
        if has_calculate_performance_method(strategy):
            test_performance_direct(strategy_class, name, "BUY CALL", "win")
            test_performance_direct(strategy_class, name, "BUY CALL", "loss")
    
    # Now test end-to-end with signal logging for all strategies
    for strategy_class, name in strategies:
        # Test win scenario with BUY CALL
        test_strategy_logging(strategy_class, name, "BUY CALL", "win")
        
        # Test loss scenario with BUY CALL
        test_strategy_logging(strategy_class, name, "BUY CALL", "loss")
    
    print("\n===== PERFORMANCE TRACKING TESTS COMPLETED =====")

if __name__ == "__main__":
    main() 