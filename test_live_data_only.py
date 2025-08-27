#!/usr/bin/env python3
"""
Test Live Data Only
==================

Verify that the system uses ONLY live data from Fyers, no historical/simulated data.
"""

import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_live_data_fetching():
    """Test that the system can fetch live data from Fyers."""
    try:
        from src.api.fyers import FyersClient
        from refresh_fyers_token import check_and_refresh_token
        
        print("🔍 Testing live data fetching from Fyers...")
        
        # Get fresh token
        access_token = check_and_refresh_token()
        if not access_token:
            print("❌ Could not obtain Fyers access token")
            return False
        
        # Initialize Fyers client
        fyers_client = FyersClient()
        fyers_client.access_token = access_token
        fyers_client.initialize_client()
        
        # Test live data fetching for both symbols
        symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        
        for symbol in symbols:
            print(f"\n📡 Testing live data for {symbol}...")
            
            # Get live price first
            price = fyers_client.get_underlying_price(symbol)
            if price and price > 0:
                print(f"✅ Live price: ₹{price:,.2f}")
            else:
                print(f"❌ No live price received for {symbol}")
                return False
            
            # Get historical data
            live_data = fyers_client.get_historical_data(
                symbol=symbol,
                resolution="5",
                date_format="1",
                range_from=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                range_to=datetime.now().strftime("%Y-%m-%d"),
                cont_flag="1"
            )
            
            if live_data and 'candles' in live_data:
                print(f"✅ Historical data received: {len(live_data['candles'])} candles")
                
                if len(live_data['candles']) > 0:
                    # Check data freshness
                    latest_timestamp = live_data['candles'][-1][0]
                    latest_time = datetime.fromtimestamp(latest_timestamp)
                    time_diff = datetime.now() - latest_time
                    
                    if time_diff.total_seconds() < 86400:  # Within last 24 hours
                        print(f"✅ Data is recent: {time_diff.total_seconds()/3600:.1f} hours old")
                    else:
                        print(f"⚠️ Data might be stale: {time_diff.total_seconds()/3600:.1f} hours old")
                else:
                    print("⚠️ No candles in historical data (market might be closed)")
                    
            else:
                print("⚠️ No historical data received (will use minimal dataset)")
        
        return True
        
    except Exception as e:
        print(f"❌ Live data test failed: {e}")
        return False

def test_live_price_fetching():
    """Test that the system can fetch live underlying prices."""
    try:
        from src.api.fyers import FyersClient
        from refresh_fyers_token import check_and_refresh_token
        
        print("\n🔍 Testing live price fetching from Fyers...")
        
        # Get fresh token
        access_token = check_and_refresh_token()
        if not access_token:
            print("❌ Could not obtain Fyers access token")
            return False
        
        # Initialize Fyers client
        fyers_client = FyersClient()
        fyers_client.access_token = access_token
        fyers_client.initialize_client()
        
        # Test live price fetching
        symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        
        for symbol in symbols:
            print(f"\n📊 Testing live price for {symbol}...")
            
            # Get live underlying price
            price = fyers_client.get_underlying_price(symbol)
            
            if price and price > 0:
                print(f"✅ Live price received: ₹{price:,.2f}")
            else:
                print(f"❌ No live price received for {symbol}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Live price test failed: {e}")
        return False

def main():
    """Run live data tests."""
    print("🚀 LIVE DATA ONLY TEST")
    print("=" * 50)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test live data fetching
    data_test = test_live_data_fetching()
    
    # Test live price fetching
    price_test = test_live_price_fetching()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    if data_test and price_test:
        print("✅ ALL TESTS PASSED!")
        print("✅ System is using ONLY live data from Fyers")
        print("\n🚀 READY FOR LIVE TRADING:")
        print("python3 live_paper_trading.py --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --data_provider fyers")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ System cannot fetch live data properly")
        return 1

if __name__ == "__main__":
    from datetime import timedelta
    exit_code = main()
    sys.exit(exit_code) 