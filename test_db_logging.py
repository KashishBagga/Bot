#!/usr/bin/env python3
# Test script to check if strategy-specific tables are being created correctly

import sqlite3
import os
from datetime import datetime
from db import log_strategy_sql

def test_strategy_logging():
    """
    Test the strategy-specific logging functionality
    """
    print("Testing strategy-specific database logging...")
    
    # List of strategies to test
    strategies = [
        'insidebar_rsi',
        'supertrend_ema',
        'breakout_rsi',
        'ema_crossover',
        'donchian_breakout',
        'insidebar_bollinger',
        'range_breakout_volatility',
        'supertrend_macd_rsi_ema'
    ]
    
    # Create a sample signal data structure
    sample_signal_data = {
        "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "index_name": "NIFTY50",
        "signal": "BUY CALL",
        "price": 21000.0,
        "rsi": 65.5,
        "macd": 25.0,
        "macd_signal": 20.0,
        "ema_20": 20950.0,
        "atr": 250.0,
        "confidence": "Medium",
        "trade_type": "Intraday",
        "stop_loss": 250,
        "target": 375,
        "target2": 500,
        "target3": 625,
        "outcome": "Success",
        "pnl": 500.0,
        "targets_hit": 2,
        "stoploss_count": 0,
        "failure_reason": ""
    }
    
    # Test each strategy
    for strategy in strategies:
        print(f"\nTesting logging for {strategy}...")
        try:
            log_strategy_sql(strategy, sample_signal_data)
            
            # Verify the table was created and the data was inserted
            conn = sqlite3.connect("trading_signals.db")
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{strategy}'")
            table_exists = cursor.fetchone()
            
            if table_exists:
                print(f"✅ Table {strategy} exists")
                
                # Check if data was inserted
                cursor.execute(f"SELECT COUNT(*) FROM {strategy}")
                count = cursor.fetchone()[0]
                print(f"✅ Table {strategy} has {count} records")
                
                # Check table structure
                cursor.execute(f"PRAGMA table_info({strategy})")
                columns = cursor.fetchall()
                print(f"✅ Table {strategy} has {len(columns)} columns")
                
                for col in columns[:5]:  # Show first 5 columns
                    print(f"   - {col[1]} ({col[2]})")
            else:
                print(f"❌ Table {strategy} does not exist")
                
            conn.close()
                
        except Exception as e:
            print(f"❌ Error testing {strategy}: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_strategy_logging() 