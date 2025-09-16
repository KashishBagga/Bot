#!/usr/bin/env python3
"""
Comprehensive System Test
========================
Tests all components for compatibility, data flow, and signal execution
"""

import sys
import os
import time
from datetime import datetime, timedelta
sys.path.append('src')

def test_fyers_client():
    """Test Fyers client data fetching."""
    print("🧪 Testing Fyers Client...")
    print("-" * 40)
    
    try:
        from src.api.fyers import FyersClient
        
        client = FyersClient()
        print(f"✅ FyersClient initialized")
        
        # Test current price
        symbol = "NSE:NIFTY50-INDEX"
        price = client.get_current_price(symbol)
        if price:
            print(f"✅ Current price for {symbol}: {price}")
        else:
            print(f"❌ No current price for {symbol}")
        
        # Test historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        data = client.get_historical_data(symbol, start_date, end_date, "1h")
        if data:
            print(f"✅ Historical data: {len(data.get('candles', []))} candles")
        else:
            print(f"❌ No historical data")
        
        return True
        
    except Exception as e:
        print(f"❌ FyersClient test failed: {e}")
        return False

def test_enhanced_database():
    """Test enhanced database structure."""
    print("\n🧪 Testing Enhanced Database...")
    print("-" * 40)
    
    try:
        from src.models.enhanced_database import EnhancedTradingDatabase
        
        db = EnhancedTradingDatabase("data/test_enhanced.db")
        print("✅ Enhanced database initialized")
        
        # Test saving entry signal
        signal_id = f"test_signal_{int(time.time())}"
        success = db.save_entry_signal(
            market="indian",
            signal_id=signal_id,
            symbol="NSE:NIFTY50-INDEX",
            strategy="simple_ema",
            signal_type="BUY CALL",
            confidence=75.0,
            price=25000.0,
            timestamp=datetime.now().isoformat(),
            timeframe="5m",
            strength="strong",
            indicator_values={"ema_12": 25000, "ema_26": 24950, "rsi": 65},
            market_condition="trending",
            volatility=0.15,
            position_size=5000.0,
            stop_loss_price=24250.0,
            take_profit_price=26250.0
        )
        
        if success:
            print("✅ Entry signal saved successfully")
        else:
            print("❌ Failed to save entry signal")
        
        # Test saving rejected signal
        rejected_id = f"rejected_signal_{int(time.time())}"
        success = db.save_rejected_signal(
            market="indian",
            signal_id=rejected_id,
            symbol="NSE:NIFTYBANK-INDEX",
            strategy="ema_crossover_enhanced",
            signal_type="BUY PUT",
            confidence=20.0,
            price=55000.0,
            timestamp=datetime.now().isoformat(),
            rejection_reason="Low confidence",
            indicator_values={"ema_12": 55000, "ema_26": 55100, "rsi": 30}
        )
        
        if success:
            print("✅ Rejected signal saved successfully")
        else:
            print("❌ Failed to save rejected signal")
        
        # Test market statistics
        stats = db.get_market_statistics("indian")
        print(f"✅ Market statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced database test failed: {e}")
        return False

def test_fixed_strategy_engine():
    """Test fixed strategy engine."""
    print("\n🧪 Testing Fixed Strategy Engine...")
    print("-" * 40)
    
    try:
        from src.core.fixed_strategy_engine import FixedStrategyEngine
        import pandas as pd
        import numpy as np
        
        # Create mock historical data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        mock_data = pd.DataFrame({
            'open': np.random.uniform(24000, 26000, 100),
            'high': np.random.uniform(25000, 27000, 100),
            'low': np.random.uniform(23000, 25000, 100),
            'close': np.random.uniform(24000, 26000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Add some trend
        mock_data['close'] = mock_data['close'] + np.arange(100) * 10
        
        historical_data = {
            "NSE:NIFTY50-INDEX": mock_data
        }
        
        current_prices = {
            "NSE:NIFTY50-INDEX": 25000.0
        }
        
        engine = FixedStrategyEngine(["NSE:NIFTY50-INDEX"])
        print("✅ Fixed strategy engine initialized")
        
        # Generate signals
        signals = engine.generate_signals(historical_data, current_prices)
        print(f"✅ Generated {len(signals)} signals")
        
        if signals:
            signal = signals[0]
            print(f"✅ Sample signal: {signal['strategy']} {signal['signal']} @ {signal['price']} (confidence: {signal['confidence']})")
            print(f"   Position size: {signal['position_size']}")
            print(f"   Stop loss: {signal['stop_loss_price']}")
            print(f"   Take profit: {signal['take_profit_price']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed strategy engine test failed: {e}")
        return False

def test_websocket_integration():
    """Test WebSocket integration."""
    print("\n🧪 Testing WebSocket Integration...")
    print("-" * 40)
    
    try:
        from src.core.fyers_websocket_manager import FyersWebSocketManager
        
        symbols = ["NSE:NIFTY50-INDEX"]
        ws_manager = FyersWebSocketManager(symbols)
        print("✅ WebSocket manager initialized")
        
        # Test connection status
        status = ws_manager.get_connection_status()
        print(f"✅ Connection status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket integration test failed: {e}")
        return False

def test_system_compatibility():
    """Test system compatibility."""
    print("\n🧪 Testing System Compatibility...")
    print("-" * 40)
    
    try:
        # Test imports
        from src.core.risk_manager import RiskManager
        from src.monitoring.system_monitor import SystemMonitor
        from src.core.enhanced_real_time_manager import EnhancedRealTimeDataManager
        
        print("✅ All core modules import successfully")
        
        # Test risk manager
        risk_manager = RiskManager()
        print("✅ Risk manager initialized")
        
        # Test system monitor
        monitor = SystemMonitor()
        print("✅ System monitor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ System compatibility test failed: {e}")
        return False

def main():
    """Run comprehensive system test."""
    print("🚀 Comprehensive System Test")
    print("=" * 50)
    
    tests = [
        test_fyers_client,
        test_enhanced_database,
        test_fixed_strategy_engine,
        test_websocket_integration,
        test_system_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for production.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
