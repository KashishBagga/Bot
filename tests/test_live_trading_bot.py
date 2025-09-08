#!/usr/bin/env python3
"""
Test Live Trading Bot
Simple test script to verify all components work correctly
"""

import sys
import sqlite3
import pandas as pd
from datetime import datetime
import os

def test_database_setup():
    """Test database setup and table creation"""
    
    try:
        # Import the bot
        from live_trading_bot import LiveTradingBot
        
        # Create bot instance
        bot = LiveTradingBot()
        
        # Check if database file exists
        if os.path.exists("trading_signals.db"):
        else:
            return False
        
        # Check tables
        conn = sqlite3.connect("trading_signals.db")
        cursor = conn.cursor()
        
        # Check if tables exist
        tables = ['live_signals', 'live_trade_executions', 'daily_trading_summary']
        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            result = cursor.fetchone()
            if result:
            else:
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        return False

def test_strategy_import():
    """Test strategy imports"""
    
    try:
        from src.strategies.insidebar_rsi import InsidebarRsi
        from src.strategies.ema_crossover import EmaCrossover
        from src.strategies.supertrend_ema import SupertrendEma
        from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
        
        strategies = {
            'insidebar_rsi': InsidebarRsi(),
            'ema_crossover': EmaCrossover(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }
        
        for name, strategy in strategies.items():
        
        return True
        
    except Exception as e:
        return False

def test_technical_indicators():
    """Test technical indicators"""
    
    try:
        from live_trading_bot import LiveTradingBot
        
        bot = LiveTradingBot()
        
        # Create sample data
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104] * 10,  # More data for indicators
            'high': [105, 106, 107, 108, 109] * 10,
            'low': [99, 100, 101, 102, 103] * 10,
            'close': [104, 105, 106, 107, 108] * 10,
            'volume': [1000, 1100, 1200, 1300, 1400] * 10
        })
        
        # Add indicators using the bot's method
        data_with_indicators = bot.add_technical_indicators(data)
        
        # Check if indicators were added
        expected_indicators = ['ema_9', 'ema_21', 'rsi', 'macd', 'macd_signal']
        for indicator in expected_indicators:
            if indicator in data_with_indicators.columns:
            else:
                return False
        
        return True
        
    except Exception as e:
        return False

def test_market_data_generation():
    """Test market data generation"""
    
    try:
        from live_trading_bot import LiveTradingBot
        
        bot = LiveTradingBot()
        
        # Test market data generation (this will be simulated data)
        symbols = ['NSE_NIFTYBANK_INDEX', 'NSE_NIFTY50_INDEX']  # Updated to match data directory names
        for symbol in symbols:
            data = bot.get_market_data(symbol)
            
            if data is not None and len(data) > 0:
            else:
        
        return True
        
    except Exception as e:
        return False

def test_signal_processing():
    """Test signal processing"""
    
    try:
        from live_trading_bot import LiveTradingBot
        
        bot = LiveTradingBot()
        
        # Create test signal
        test_signal = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': 'test_strategy',
            'symbol': 'NSE_NIFTYBANK_INDEX',
            'signal': 'BUY CALL',
            'confidence_score': 75,
            'price': 45000.0,
            'target': 45500.0,
            'stop_loss': 44500.0,
            'status': 'GENERATED'
        }
        
        # Process signal
        bot.process_signals([test_signal])
        
        # Check if signal was stored
        conn = sqlite3.connect("trading_signals.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM live_signals WHERE strategy = 'test_strategy'")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            return True
        else:
            return False
        
    except Exception as e:
        return False

def test_daily_summary():
    """Test daily summary functionality"""
    
    try:
        from view_daily_trading_summary import DailyTradingSummaryViewer
        
        viewer = DailyTradingSummaryViewer()
        
        # Test viewer initialization
        if viewer.db_path:
        else:
            return False
        
        return True
        
    except Exception as e:
        return False

def main():
    """Run all tests"""
    
    tests = [
        test_database_setup,
        test_strategy_import,
        test_technical_indicators,
        test_market_data_generation,
        test_signal_processing,
        test_daily_summary
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
    
    
    if failed == 0:
    else:
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 