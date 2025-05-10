#!/usr/bin/env python3
"""
Test script for Fyers API symbol lookup
"""
import os
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
import json

# Load environment variables
load_dotenv()

# Get Fyers API credentials from environment variables
client_id = os.getenv("FYERS_CLIENT_ID")
auth_code = os.getenv("FYERS_AUTH_CODE")

# Get access token directly from environment variable
access_token = os.getenv("FYERS_ACCESS_TOKEN")

if not access_token:
    print("⚠️ No FYERS_ACCESS_TOKEN found in .env file!")
    print("Please run test_fyers.py first, then add the access_token to your .env file as FYERS_ACCESS_TOKEN=your_token")
    exit(1)

print(f"Using access token: {access_token[:15]}...")

# Initialize Fyers model with token
fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")

# Test symbols to try
symbols = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTY-INDEX",
    "NSE:NIFTY50",
    "NSE:NIFTY",
    "NSE:BANKNIFTY-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:BANKNIFTY",
    "NSE:NIFTYBANK"
]

# Test each symbol
for symbol in symbols:
    print(f"\nTesting symbol: {symbol}")
    
    # Try to get quotes
    try:
        data = {"symbols": symbol}
        quotes = fyers.quotes(data)
        print(f"Quotes response: {json.dumps(quotes, indent=2)}")
        
        if quotes.get('s') == 'ok':
            print(f"✅ Symbol {symbol} is valid")
        else:
            print(f"❌ Symbol {symbol} is invalid: {quotes.get('message')}")
    except Exception as e:
        print(f"❌ Error with symbol {symbol}: {e}")
    
    # Try to get market depth
    try:
        data = {"symbol": symbol, "ohlcv_flag": 1}
        depth = fyers.depth(data)
        print(f"Depth response: {json.dumps(depth, indent=2)}")
    except Exception as e:
        print(f"❌ Error getting depth for {symbol}: {e}")

# Try to get historical data
print("\n\nTesting historical data API...")
for symbol in symbols:
    print(f"\nTesting historical data for symbol: {symbol}")
    
    # Try different resolutions
    for resolution in ["1", "5", "15", "D", "W", "M"]:
        try:
            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": "1672531200",  # 2023-01-01
                "range_to": "1703980800",    # 2023-12-31
                "cont_flag": "1"
            }
            print(f"\nTrying resolution: {resolution}")
            print(f"Request data: {data}")
            
            history = fyers.history(data)
            
            if isinstance(history, dict) and 'code' in history and history['code'] != 200:
                print(f"❌ Error: {history.get('message')} (Code: {history.get('code')})")
            elif history.get('candles'):
                print(f"✅ Success! Got {len(history['candles'])} candles")
            else:
                print(f"❌ No candles returned: {history}")
        except Exception as e:
            print(f"❌ Exception: {e}") 