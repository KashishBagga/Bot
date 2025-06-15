#!/usr/bin/env python3
"""
Real-time Data Connection Test
Test script to verify Fyers API connection and real-time data fetching
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.fyers import FyersClient
from src.data.market_data import MarketData
from optimized_live_trading_bot import OptimizedLiveTradingBot

def test_fyers_connection():
    """Test Fyers API connection"""
    print("🔍 Testing Fyers API Connection...")
    print("=" * 50)
    
    try:
        # Initialize Fyers client
        fyers_client = FyersClient()
        
        print("📡 Initializing Fyers client...")
        if fyers_client.initialize_client():
            print("✅ Fyers client initialized successfully")
            
            # Test profile access
            profile = fyers_client.get_profile()
            if profile and 'data' in profile:
                print(f"✅ Profile access successful: {profile['data'].get('name', 'Unknown')}")
            
            return fyers_client
        else:
            print("❌ Failed to initialize Fyers client")
            return None
            
    except Exception as e:
        print(f"❌ Error initializing Fyers: {e}")
        return None

def test_market_data(fyers_client):
    """Test market data fetching"""
    if not fyers_client:
        print("⚠️ Skipping market data test - no Fyers connection")
        return False
    
    print("\n📊 Testing Market Data Fetching...")
    print("=" * 50)
    
    try:
        market_data = MarketData(fyers_client.fyers)
        
        # Test symbols
        symbols = {
            'NIFTY50': 'NSE:NIFTY50-INDEX',
            'BANKNIFTY': 'NSE:BANKNIFTY-INDEX'
        }
        
        for name, symbol in symbols.items():
            print(f"\n📈 Fetching data for {name} ({symbol})...")
            
            df = market_data.fetch_fyers_candles(
                symbol=symbol,
                resolution="5",  # 5-minute candles
                days_back=1
            )
            
            if not df.empty:
                latest_price = df['close'].iloc[-1]
                latest_time = df.index[-1]
                print(f"✅ {name}: ₹{latest_price:.2f} (Last update: {latest_time})")
                print(f"   📊 Data points: {len(df)} candles")
            else:
                print(f"❌ No data received for {name}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return False

def test_live_bot_integration():
    """Test live bot integration with real-time data"""
    print("\n🤖 Testing Live Bot Integration...")
    print("=" * 50)
    
    try:
        # Initialize live bot
        bot = OptimizedLiveTradingBot()
        
        # Check connection status
        status = bot.get_connection_status()
        print(f"📡 Data Source: {status['data_source']}")
        print(f"📊 Connection: {status['message']}")
        
        if status['real_time_data']:
            print("\n🔍 Testing data fetch through bot...")
            
            # Test data fetching for both symbols
            for symbol in ['NIFTY50', 'BANKNIFTY']:
                data = bot.get_market_data(symbol, periods=10)
                if data is not None and not data.empty:
                    latest_price = data['close'].iloc[-1]
                    print(f"✅ {symbol}: ₹{latest_price:.2f} ({len(data)} candles)")
                else:
                    print(f"❌ Failed to get data for {symbol}")
        else:
            print("🔄 No real-time connection - testing fallback data...")
            # Test fallback data generation
            data = bot.get_market_data('NIFTY50', periods=5)
            if data is not None and not data.empty:
                latest_price = data['close'].iloc[-1]
                print(f"✅ Fallback data working - NIFTY50: ₹{latest_price:.2f}")
            else:
                print("❌ Fallback data generation failed")
        
        return status['real_time_data']
        
    except Exception as e:
        print(f"❌ Error testing live bot: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 REAL-TIME DATA CONNECTION TEST")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now()}")
    
    # Test 1: Fyers connection
    fyers_client = test_fyers_connection()
    
    # Test 2: Market data fetching
    market_data_success = test_market_data(fyers_client)
    
    # Test 3: Live bot integration
    bot_integration_success = test_live_bot_integration()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Fyers Connection: {'PASS' if fyers_client else 'FAIL'}")
    print(f"📊 Market Data: {'PASS' if market_data_success else 'FAIL'}")
    print(f"🤖 Bot Integration: {'PASS' if bot_integration_success else 'FAIL'}")
    
    if fyers_client and market_data_success and bot_integration_success:
        print("\n🎉 ALL TESTS PASSED - Real-time data is working!")
        print("🚀 Your live bot is ready to use real market data")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("📋 Check your Fyers API credentials in .env file")
        print("🔄 Bot will use simulated data as fallback")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 