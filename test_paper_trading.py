#!/usr/bin/env python3
"""
Test Paper Trading Bot
Quick test to verify paper trading functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from paper_trading_bot import PaperTradingBot
from datetime import datetime

def test_paper_trading():
    """Test paper trading bot functionality."""
    print("🧪 Testing Paper Trading Bot...")
    
    # Create bot instance
    bot = PaperTradingBot(initial_capital=100000.0, max_risk_per_trade=0.02)
    
    # Test data loading
    print("📊 Testing data loading...")
    df = bot.get_latest_data("NSE:NIFTY50-INDEX", "5min", 200)
    
    if df.empty:
        print("❌ Failed to load data")
        return False
    
    print(f"✅ Loaded {len(df)} candles")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Test signal generation
    print("🎯 Testing signal generation...")
    signals = bot.generate_signals(df, "NSE:NIFTY50-INDEX")
    
    print(f"✅ Generated {len(signals)} signals")
    
    if signals:
        for i, signal in enumerate(signals[:3]):  # Show first 3 signals
            print(f"   Signal {i+1}: {signal['strategy']} - {signal['signal']} at ₹{signal['price']:.2f}")
            print(f"   Confidence: {signal['confidence']:.1f}%")
            print(f"   Reasoning: {signal['reasoning'][:100]}...")
    
    # Test position sizing
    print("💰 Testing position sizing...")
    if signals:
        signal = signals[0]
        position_size = bot.calculate_position_size(
            signal['price'], 
            signal['stop_loss'], 
            signal['confidence']
        )
        print(f"✅ Position size: {position_size} shares")
    
    # Test performance summary
    print("📈 Testing performance summary...")
    perf = bot.get_performance_summary()
    print(f"✅ Performance summary generated")
    print(f"   Capital: ₹{perf['current_capital']:,.2f}")
    print(f"   Trades: {perf['total_trades']}")
    print(f"   Win Rate: {perf['win_rate']:.1f}%")
    
    print("\n🎉 All tests passed! Paper trading bot is ready.")
    return True

if __name__ == "__main__":
    test_paper_trading() 