#!/usr/bin/env python3
"""
Test Live Paper Trading System
Quick test to validate the paper trading functionality
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from live_paper_trading import LivePaperTradingSystem
from src.models.option_contract import StrikeSelection

def test_live_paper_trading():
    """Test live paper trading system."""
    print("🚀 Testing Live Paper Trading System")
    print("=" * 50)
    
    # 1. Initialize trading system
    print("\n1. Initializing live paper trading system...")
    try:
        trading_system = LivePaperTradingSystem(
            initial_capital=100000.0,
            max_risk_per_trade=0.02,
            confidence_cutoff=40.0,
            exposure_limit=0.6,
            max_daily_loss_pct=0.03,
            commission_bps=1.0,
            slippage_bps=5.0,
            expiry_type="weekly",
            strike_selection=StrikeSelection.ATM,
            delta_target=0.30,
            symbols=['NSE:NIFTY50-INDEX'],
            strategies=['ema_crossover_enhanced'],
            data_provider="paper"
        )
        
        print("✅ Live paper trading system initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # 2. Test market hours check
    print("\n2. Testing market hours check...")
    try:
        current_time = datetime.now()
        is_open = trading_system._is_market_open(current_time)
        print(f"✅ Market hours check: {'OPEN' if is_open else 'CLOSED'}")
    except Exception as e:
        print(f"❌ Market hours check failed: {e}")
        return False
    
    # 3. Test signal generation
    print("\n3. Testing signal generation...")
    try:
        # Get recent index data
        index_data = trading_system._get_recent_index_data('NSE:NIFTY50-INDEX')
        if index_data is not None and not index_data.empty:
            signals = trading_system._generate_signals(index_data)
            print(f"✅ Generated {len(signals)} signals")
        else:
            print("⚠️ No index data available for signal generation")
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False
    
    # 4. Test trade opening logic
    print("\n4. Testing trade opening logic...")
    try:
        # Create a sample signal
        sample_signal = {
            'timestamp': datetime.now(),
            'strategy': 'ema_crossover_enhanced',
            'signal': 'BUY CALL',
            'price': 25000.0,
            'confidence': 60.0,
            'reasoning': 'Test signal'
        }
        
        should_open = trading_system._should_open_trade(sample_signal)
        print(f"✅ Trade opening logic: {'ALLOWED' if should_open else 'BLOCKED'}")
    except Exception as e:
        print(f"❌ Trade opening logic failed: {e}")
        return False
    
    # 5. Test performance report
    print("\n5. Testing performance report...")
    try:
        report = trading_system.get_performance_report()
        print(f"✅ Performance report generated:")
        print(f"   Capital: ₹{report['current_capital']:,.2f}")
        print(f"   Total Trades: {report['total_trades']}")
        print(f"   Win Rate: {report['win_rate']:.1f}%")
    except Exception as e:
        print(f"❌ Performance report failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All live paper trading tests passed!")
    print("✅ System is ready for live paper trading")
    
    return True

def test_short_paper_trading_session():
    """Test a short paper trading session."""
    print("\n🚀 Testing Short Paper Trading Session")
    print("=" * 50)
    
    try:
        # Initialize trading system
        trading_system = LivePaperTradingSystem(
            initial_capital=100000.0,
            max_risk_per_trade=0.02,
            confidence_cutoff=40.0,
            symbols=['NSE:NIFTY50-INDEX'],
            strategies=['ema_crossover_enhanced'],
            data_provider="paper"
        )
        
        # Start trading for 2 minutes
        print("⏰ Starting 2-minute paper trading session...")
        trading_system.start_trading()
        
        import time
        time.sleep(120)  # Run for 2 minutes
        
        # Stop trading
        trading_system.stop_trading()
        
        # Print performance report
        trading_system.print_performance_report()
        
        print("✅ Short paper trading session completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Short paper trading session failed: {e}")
        return False

if __name__ == "__main__":
    # Run basic tests
    if test_live_paper_trading():
        # Run short session test
        test_short_paper_trading_session()
    else:
        print("❌ Basic tests failed, skipping session test") 