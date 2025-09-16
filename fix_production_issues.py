#!/usr/bin/env python3
"""
Fix Production Issues - Critical Data Integration
"""

import os
import sys
sys.path.append('src')

def fix_fyers_api_integration():
    """Fix Fyers API integration issues"""
    print("🔧 FIXING FYERS API INTEGRATION")
    print("=" * 50)
    
    # Check if credentials are set
    client_id = os.getenv('FYERS_CLIENT_ID')
    access_token = os.getenv('FYERS_ACCESS_TOKEN')
    secret_key = os.getenv('FYERS_SECRET_KEY')
    
    if not client_id or not access_token or not secret_key:
        print("❌ CRITICAL: Fyers API credentials not configured!")
        print("Please set the following environment variables:")
        print("  export FYERS_CLIENT_ID='your_client_id'")
        print("  export FYERS_ACCESS_TOKEN='your_access_token'")
        print("  export FYERS_SECRET_KEY='your_secret_key'")
        print("  export FYERS_REDIRECT_URI='http://localhost:8080'")
        return False
    
    print("✅ Fyers API credentials found")
    
    # Test API connection
    try:
        from src.api.fyers import FyersClient
        client = FyersClient()
        
        # Test with a simple symbol
        test_symbol = "NSE:NIFTY50-INDEX"
        price = client.get_current_price(test_symbol)
        
        if price:
            print(f"✅ API working: {test_symbol} = ₹{price:,.2f}")
            return True
        else:
            print(f"❌ API not returning data for {test_symbol}")
            print("Possible issues:")
            print("  1. Market is closed")
            print("  2. Invalid symbol format")
            print("  3. API rate limiting")
            print("  4. Authentication issues")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def fix_websocket_connection():
    """Fix WebSocket connection issues"""
    print("\n🔧 FIXING WEBSOCKET CONNECTION")
    print("=" * 50)
    
    try:
        from src.core.fyers_websocket_manager import get_websocket_manager
        
        symbols = ["NSE:NIFTY50-INDEX"]
        ws_manager = get_websocket_manager(symbols)
        
        print("✅ WebSocket manager initialized")
        
        # Start WebSocket
        ws_manager.start()
        print("🚀 WebSocket started")
        
        # Wait for connection
        import time
        for i in range(15):  # Wait up to 15 seconds
            if ws_manager.is_connected:
                print(f"✅ WebSocket connected after {i+1} seconds")
                
                # Test live data
                live_data = ws_manager.get_live_data("NSE:NIFTY50-INDEX")
                if live_data:
                    print(f"✅ Live data: ₹{live_data.ltp:,.2f}")
                    ws_manager.stop()
                    return True
                else:
                    print("❌ No live data received")
                    ws_manager.stop()
                    return False
            
            time.sleep(1)
        
        print("❌ WebSocket connection timeout")
        ws_manager.stop()
        return False
        
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def fix_historical_data_api():
    """Fix historical data API issues"""
    print("\n🔧 FIXING HISTORICAL DATA API")
    print("=" * 50)
    
    try:
        from src.api.fyers import FyersClient
        from datetime import datetime, timedelta
        
        client = FyersClient()
        
        # Test historical data with proper date format
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"Testing historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        hist_data = client.get_historical_data(
            symbol="NSE:NIFTY50-INDEX",
            start_date=start_date,
            end_date=end_date,
            interval="1h"
        )
        
        if hist_data and 'candles' in hist_data and len(hist_data['candles']) > 0:
            print(f"✅ Historical data working: {len(hist_data['candles'])} candles")
            latest = hist_data['candles'][-1]
            print(f"   Latest: Open={latest[1]}, High={latest[2]}, Low={latest[3]}, Close={latest[4]}")
            return True
        else:
            print("❌ No historical data returned")
            return False
            
    except Exception as e:
        print(f"❌ Historical data test failed: {e}")
        return False

def create_production_config():
    """Create production configuration file"""
    print("\n🔧 CREATING PRODUCTION CONFIGURATION")
    print("=" * 50)
    
    config_content = '''# Production Trading System Configuration
# Copy this to .env file and configure with your real credentials

# Fyers API Configuration (REQUIRED)
FYERS_CLIENT_ID=your_real_client_id_here
FYERS_ACCESS_TOKEN=your_real_access_token_here
FYERS_SECRET_KEY=your_real_secret_key_here
FYERS_REDIRECT_URI=http://localhost:8080
FYERS_RESPONSE_TYPE=code
FYERS_GRANT_TYPE=authorization_code

# Trading Configuration
SIGNAL_COOLDOWN_MINUTES=5
MAX_SIGNALS_PER_CYCLE=5
CONFIDENCE_THRESHOLD=50.0

# Risk Management
MAX_RISK_PER_TRADE=0.02
DAILY_LOSS_LIMIT=0.05
EMERGENCY_STOP_LOSS=0.10

# System Configuration
LOG_LEVEL=INFO
PERFORMANCE_LOG_INTERVAL=300
API_RETRY_ATTEMPTS=3
API_TIMEOUT=30

# Market Hours (IST)
MARKET_OPEN_TIME=09:15
MARKET_CLOSE_TIME=15:30
EOD_EXIT_TIME=15:25
'''
    
    with open('production.env.example', 'w') as f:
        f.write(config_content)
    
    print("✅ Production configuration template created: production.env.example")
    print("📝 Please copy to .env and configure with your real credentials")

def main():
    """Main function to fix all production issues"""
    print("🚨 PRODUCTION ISSUES DIAGNOSIS AND FIXES")
    print("=" * 60)
    
    # Check API credentials
    api_working = fix_fyers_api_integration()
    
    # Check WebSocket
    websocket_working = fix_websocket_connection()
    
    # Check historical data
    historical_working = fix_historical_data_api()
    
    # Create production config
    create_production_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PRODUCTION READINESS SUMMARY")
    print("=" * 60)
    
    print(f"🔌 Fyers API: {'✅ WORKING' if api_working else '❌ NOT WORKING'}")
    print(f"📡 WebSocket: {'✅ WORKING' if websocket_working else '❌ NOT WORKING'}")
    print(f"📊 Historical Data: {'✅ WORKING' if historical_working else '❌ NOT WORKING'}")
    
    if api_working and websocket_working and historical_working:
        print("\n🎉 ALL SYSTEMS WORKING - READY FOR PRODUCTION!")
    else:
        print("\n⚠️ CRITICAL ISSUES FOUND - NOT PRODUCTION READY")
        print("\nRequired fixes:")
        if not api_working:
            print("  1. Configure Fyers API credentials")
        if not websocket_working:
            print("  2. Fix WebSocket connection")
        if not historical_working:
            print("  3. Fix historical data API")
        
        print("\n📝 Next steps:")
        print("  1. Copy production.env.example to .env")
        print("  2. Configure with your real Fyers API credentials")
        print("  3. Test during market hours")
        print("  4. Run this script again to verify")

if __name__ == "__main__":
    main()
