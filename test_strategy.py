import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import ta.momentum
import matplotlib.pyplot as plt
from strategies.supertrend_macd_rsi_ema import generate_strategy_signal, execute_supertrend_macd_rsi_ema_strategy
from indicators.ema import calculate_ema
from indicators.macd import calculate_macd
from indicators.rsi import calculate_rsi

def create_sample_data():
    """Create sample data with required fields to test the strategy."""
    # Create date range
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)  # for reproducibility
    close = np.random.normal(100, 5, 100)
    close = np.cumsum(np.random.normal(0, 1, 100)) + 100  # Random walk
    
    # Create larger up and down movements to trigger signals
    for i in range(10, 90, 20):
        close[i:i+10] = close[i-1] * 1.10  # 10% up trend
    for i in range(50, 90, 20):
        close[i:i+10] = close[i-1] * 0.90  # 10% down trend
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': dates,
        'open': close - np.random.normal(0, 1, 100),
        'high': close + np.random.normal(1, 0.5, 100),
        'low': close - np.random.normal(1, 0.5, 100),
        'close': close,
        'volume': np.random.normal(1000000, 200000, 100)
    })
    
    # Ensure high > low
    df['high'] = np.maximum(df['high'], df['open'])
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['open'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    # Calculate indicators
    df['ema_20'] = calculate_ema(df['close'], span=20)
    macd, macd_signal = calculate_macd(df)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['rsi'] = calculate_rsi(df)
    
    # Calculate ATR
    df['tr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    
    # Add some modifications to ensure we hit signal conditions
    # Create a candle with RSI > 55, MACD > Signal, and price > EMA for BUY CALL
    df.loc[50, 'rsi'] = 60
    df.loc[50, 'macd'] = 2.0
    df.loc[50, 'macd_signal'] = 1.0
    df.loc[50, 'close'] = df.loc[50, 'ema_20'] * 1.02
    
    # Create a candle with RSI < 45, MACD < Signal, and price < EMA for BUY PUT
    df.loc[70, 'rsi'] = 40
    df.loc[70, 'macd'] = -2.0
    df.loc[70, 'macd_signal'] = -1.0
    df.loc[70, 'close'] = df.loc[70, 'ema_20'] * 0.98
    
    return df

def test_strategy():
    """Test the strategy with sample data."""
    print("Creating sample data...")
    df = create_sample_data()
    
    print("\nSample data statistics:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df[['time', 'open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal']].head())
    
    print("\nLast few rows:")
    print(df[['time', 'open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal']].tail())
    
    # Check the modified candles that should trigger signals
    print("\nModified candles that should trigger signals:")
    print("BUY CALL trigger candle (index 50):")
    print(df.iloc[50][['time', 'rsi', 'macd', 'macd_signal', 'close', 'ema_20']])
    print("BUY PUT trigger candle (index 70):")
    print(df.iloc[70][['time', 'rsi', 'macd', 'macd_signal', 'close', 'ema_20']])
    
    # Manually check signal conditions with direct debugging
    print("\nManually checking signal conditions:")
    for idx in [50, 70]:
        candle = df.iloc[idx]
        print(f"\nTesting candle at index {idx}, time: {candle['time'].date()}")
        
        # BUY CALL condition checks
        rsi_condition = candle['rsi'] > 55
        macd_condition = candle['macd'] > candle['macd_signal']
        price_condition = candle['close'] > candle['ema_20'] * 0.99
        
        print(f"BUY CALL conditions: RSI>55: {rsi_condition}, MACD>Signal: {macd_condition}, Price>EMA*0.99: {price_condition}")
        
        # BUY PUT condition checks
        rsi_put_condition = candle['rsi'] < 45
        macd_put_condition = candle['macd'] < candle['macd_signal']
        price_put_condition = candle['close'] < candle['ema_20'] * 1.01
        
        print(f"BUY PUT conditions: RSI<45: {rsi_put_condition}, MACD<Signal: {macd_put_condition}, Price<EMA*1.01: {price_put_condition}")
        
        # Call the actual strategy function
        signal, confidence = generate_strategy_signal(candle)
        print(f"Generated signal: {signal}, Confidence: {confidence}")
    
    # Test strategy with sample data
    print("\nExecuting strategy...")
    execute_supertrend_macd_rsi_ema_strategy(df, "SAMPLE_INDEX", 50)
    
    # Plot key indicators for visual inspection
    plt.figure(figsize=(12, 8))
    
    # Plot price and EMA
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['close'], label='Close')
    plt.plot(df['time'], df['ema_20'], label='EMA 20')
    plt.legend()
    plt.title('Price and EMA')
    
    # Plot MACD
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['macd'], label='MACD')
    plt.plot(df['time'], df['macd_signal'], label='Signal')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.title('MACD')
    
    # Plot RSI
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['rsi'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.legend()
    plt.title('RSI')
    
    plt.tight_layout()
    plt.savefig('strategy_debug.png')
    print("Saved indicator plot to 'strategy_debug.png'")

if __name__ == "__main__":
    test_strategy() 