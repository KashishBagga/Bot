#!/usr/bin/env python3
# Debug script for testing strategy logging in backtesting

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from db import log_strategy_sql, setup_sqlite, setup_backtesting_table
from strategies.insidebar_rsi import strategy_insidebar_rsi
from strategies.supertrend_ema import strategy_supertrend_ema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_data(days=30):
    """Create mock OHLCV data for testing"""
    logger.info("Creating mock data...")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random price data
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    base_price = 18000
    
    # Create price movements with some randomness but trending behavior
    price_changes = np.random.normal(0, 100, len(dates))  # Daily changes with std dev of 100
    # Add a slight upward trend
    trend = np.linspace(0, 200, len(dates))
    price_changes = price_changes + trend
    
    # Calculate prices
    closes = base_price + np.cumsum(price_changes)
    
    # Generate OHLC with realistic relationships
    highs = closes + np.random.uniform(50, 150, len(dates))
    lows = closes - np.random.uniform(50, 150, len(dates))
    opens = lows + (highs - lows) * np.random.random(len(dates))
    
    # Create volume with some randomness
    volumes = np.random.uniform(1000000, 5000000, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Calculate technical indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['atr'] = calculate_atr(df)
    
    logger.info(f"Created mock data with {len(df)} rows")
    return df

def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    if down == 0:
        return np.ones_like(prices) * 100
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        if down == 0:
            rsi[i] = 100
        else:
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

def calculate_ema(prices, window):
    """Calculate EMA"""
    return pd.Series(prices).ewm(span=window, adjust=False).mean().values

def calculate_atr(df, window=14):
    """Calculate ATR"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=window).mean()
    return atr.fillna(tr)

def test_strategy_logging_with_mock_data():
    """Test strategy logging with mock data"""
    logger.info("Setting up database...")
    setup_sqlite()
    setup_backtesting_table()
    
    logger.info("Running strategy tests...")
    df = create_mock_data()
    
    # Test for inside bar pattern
    for i in range(1, len(df) - 10):  # Leave some data for future price checking
        current_candle = df.iloc[i].copy()
        prev_candle = df.iloc[i-1].copy()
        future_data = df.iloc[i+1:i+10]
        
        logger.info(f"Testing insidebar_rsi for candle at {current_candle['time']}")
        # Call the strategy function
        try:
            result = strategy_insidebar_rsi(current_candle, prev_candle, "NIFTY50", future_data)
            logger.info(f"Strategy result: {result['signal']}")
        except Exception as e:
            logger.error(f"Error in insidebar_rsi: {e}")
            
        logger.info(f"Testing supertrend_ema for candle at {current_candle['time']}")
        # Call the strategy function
        try:
            result = strategy_supertrend_ema(current_candle, "NIFTY50", future_data)
            logger.info(f"Strategy result: {result['signal']}")
        except Exception as e:
            logger.error(f"Error in supertrend_ema: {e}")
    
    logger.info("Testing completed!")

if __name__ == "__main__":
    logger.info("Starting debug backtesting...")
    test_strategy_logging_with_mock_data()
    logger.info("Debug backtesting completed.") 