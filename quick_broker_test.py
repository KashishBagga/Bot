#!/usr/bin/env python3
"""
Quick Broker Connection Test
===========================

Essential pre-trading checks to ensure system is ready.
"""

import sys
import time
import logging
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fyers_token():
    """Test Fyers token validity"""
    try:
        from refresh_fyers_token import check_and_refresh_token
        print("🔍 Testing Fyers token...")
        
        start_time = time.time()
        access_token = check_and_refresh_token()
        refresh_time = time.time() - start_time
        
        if access_token:
            print(f"✅ Token valid (length: {len(access_token)}, refresh time: {refresh_time:.2f}s)")
            return True
        else:
            print("❌ Could not obtain access token")
            return False
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        return False

def test_historical_data():
    """Test historical data loading"""
    try:
        from src.data.local_data_loader import LocalDataLoader
        print("\n🔍 Testing historical data...")
        
        data_loader = LocalDataLoader()
        symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        
        for symbol in symbols:
            df = data_loader.load_data(symbol, '5min', 100)
            if df is not None and len(df) > 0:
                print(f"✅ {symbol}: {len(df)} candles loaded")
            else:
                print(f"❌ {symbol}: No data loaded")
                return False
        return True
    except Exception as e:
        print(f"❌ Historical data test failed: {e}")
        return False

def test_database():
    """Test database connectivity"""
    try:
        from src.models.unified_database import UnifiedDatabase
        print("\n🔍 Testing database...")
        
        db = UnifiedDatabase()
        db.init_database()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_strategy_generation():
    """Test strategy signal generation"""
    try:
        from src.data.local_data_loader import LocalDataLoader
        from simple_backtest import OptimizedBacktester
        print("\n🔍 Testing strategy generation...")
        
        data_loader = LocalDataLoader()
        backtester = OptimizedBacktester()
        
        df = data_loader.load_data('NSE:NIFTY50-INDEX', '5min', 200)
        if df is None or len(df) < 50:
            print("❌ Insufficient data for strategy testing")
            return False
        
        signals = backtester.run_enhanced_strategies_optimized(df, 'NSE:NIFTY50-INDEX', "test_session", disable_tqdm=True)
        print(f"✅ Strategy generation successful: {len(signals)} signals generated")
        return True
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False

def test_fyers_client():
    """Test Fyers client initialization"""
    try:
        from refresh_fyers_token import check_and_refresh_token
        from src.api.fyers import FyersClient
        print("\n🔍 Testing Fyers client...")
        
        access_token = check_and_refresh_token()
        if not access_token:
            print("❌ No access token available")
            return False
        
        fyers_client = FyersClient()
        fyers_client.access_token = access_token
        fyers_client.initialize_client()
        print("✅ Fyers client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Fyers client test failed: {e}")
        return False

def main():
    """Run all essential tests"""
    print("🚀 QUICK BROKER CONNECTION TEST")
    print("=" * 50)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Fyers Token", test_fyers_token),
        ("Historical Data", test_historical_data),
        ("Database", test_database),
        ("Strategy Generation", test_strategy_generation),
        ("Fyers Client", test_fyers_client)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for trading.")
        print("\n🚀 READY FOR TRADING COMMAND:")
        print("python3 live_paper_trading.py --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --data_provider fyers")
        return 0
    else:
        print(f"\n⚠️ {total-passed} TEST(S) FAILED. Please review before trading.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 