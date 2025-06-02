#!/usr/bin/env python3
"""
Parquet Data Store Setup Script
Downloads and stores historical market data in parquet files for all timeframes.
Designed for 5+ years of data with efficient storage and retrieval.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.parquet_data_store import ParquetDataStore
from all_strategies import initialize_fyers_client
from dotenv import load_dotenv

def setup_parquet_data(years_back: int = 5, timeframes: List[str] = None, symbols: dict = None):
    """Set up parquet data store with historical data.
    
    Args:
        years_back: Years of historical data to fetch and store
        timeframes: List of timeframes to fetch (None for all)
        symbols: Symbols to process
    """
    load_dotenv()
    
    # Default symbols
    symbols = symbols or {
        "NSE:NIFTY50-INDEX": "NIFTY50",
        "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
    }
    
    # Default timeframes (all available)
    if timeframes is None:
        timeframes = ['1min', '3min', '5min', '15min', '30min', '1hour', '4hour', '1day']
    
    print(f"🚀 Setting up Parquet Data Store with DIRECT API FETCHING")
    print(f"📊 Each timeframe will be fetched separately from Fyers API (most accurate):")
    print(f"  • {', '.join(timeframes)}")
    print(f"📅 Period: {years_back} years of historical data")
    print(f"📈 Symbols: {list(symbols.values())}")
    print(f"💾 Storage: Compressed parquet files with technical indicators")
    print(f"✨ Advantage: Real market data for each timeframe, not calculated")
    print()
    
    # Confirm the operation
    if years_back >= 3:
        estimated_time = len(timeframes) * len(symbols) * 2  # Rough estimate
        print(f"⚠️  This will fetch {years_back} years of data for {len(timeframes)} timeframes.")
        print(f"   Estimated time: {estimated_time}-{estimated_time*2} minutes depending on API speed.")
        confirmation = input("Do you want to continue? (y/N): ").strip().lower()
        if confirmation not in ['y', 'yes']:
            print("❌ Setup cancelled")
            return False
    
    # Initialize components
    data_store = ParquetDataStore()
    fyers = initialize_fyers_client()
    
    # Test API connection
    print(f"🔧 Testing API connection...")
    try:
        test_data = {
            "symbol": "NSE:NIFTY50-INDEX",
            "resolution": "5",  # Test with 5-minute data
            "date_format": "1",
            "range_from": "2025-05-30",
            "range_to": "2025-06-02",
            "cont_flag": "1"
        }
        test_response = fyers.history(test_data)
        if isinstance(test_response, dict) and test_response.get('code') == 200:
            print(f"✅ API connection successful")
        else:
            print(f"❌ API connection failed: {test_response}")
            return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    # Fetch and store data
    start_time = datetime.now()
    data_store.fetch_and_store_data(fyers, symbols, timeframes, years_back)
    duration = datetime.now() - start_time
    
    print(f"\n✅ Parquet data store setup completed in {duration.total_seconds():.1f} seconds")
    
    # Show final summary
    print(f"\n" + "="*60)
    print(f"🎉 PARQUET DATA STORE READY!")
    print(f"="*60)
    
    info = data_store.get_storage_info()
    print(f"📊 Total symbols: {info['total_symbols']}")
    print(f"💾 Total storage: {info['total_size_mb']} MB")
    print(f"📁 Storage location: {info['storage_directory']}")
    
    print(f"\n🎯 Available timeframes for each symbol:")
    for symbol_info in info['symbols']:
        timeframes = symbol_info['timeframes']
        print(f"  • {symbol_info['name']}: {len(timeframes)} timeframes ({', '.join(timeframes)})")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Run backtests: python3 backtesting_parquet.py --days 30 --timeframe 5min")
    print(f"  2. Check data info: python3 backtesting_parquet.py --data-info")
    print(f"  3. Load specific data: data_store.load_data('NSE:NIFTY50-INDEX', '15min', days_back=30)")
    
    return True

def show_data_info():
    """Show information about stored parquet data."""
    data_store = ParquetDataStore()
    info = data_store.get_storage_info()
    
    print("📊 Parquet Data Store Information:")
    print(f"📁 Storage directory: {info['storage_directory']}")
    print(f"📊 Total symbols: {info['total_symbols']}")
    print(f"💾 Total storage: {info['total_size_mb']} MB")
    
    if info['symbols']:
        print("\n📈 Stored Symbols:")
        for symbol_info in info['symbols']:
            print(f"\n  • {symbol_info['name']} ({symbol_info['symbol']})")
            print(f"    📅 Period: {symbol_info['date_range']}")
            print(f"    📊 Base candles: {symbol_info['base_candles_count']:,}")
            print(f"    🎯 Timeframes: {len(symbol_info['timeframes'])}")
            print(f"      Available: {', '.join(symbol_info['timeframes'])}")
            print(f"    💾 Size: {symbol_info['size_mb']} MB")
            print(f"    🕒 Last updated: {symbol_info['last_updated']}")
    
    return info

def clear_parquet_data(symbol: str = None):
    """Clear parquet data."""
    data_store = ParquetDataStore()
    
    if symbol:
        print(f"🗑️ Clearing data for {symbol}...")
    else:
        print(f"🗑️ Clearing ALL parquet data...")
        confirmation = input("Are you sure you want to delete all data? (y/N): ").strip().lower()
        if confirmation not in ['y', 'yes']:
            print("❌ Clear operation cancelled")
            return
    
    data_store.clear_data(symbol)

def main():
    parser = argparse.ArgumentParser(description='Set up parquet data store for efficient backtesting')
    parser.add_argument('--years', type=int, default=5, help='Years of data to store (default: 5)')
    parser.add_argument('--timeframes', type=str, help='Timeframes to process, comma separated (default: 1min,3min,5min,15min,30min,1hour,4hour,1day)')
    parser.add_argument('--symbols', type=str, help='Symbols to process, comma separated (default: NIFTY50,BANKNIFTY)')
    parser.add_argument('--data-info', action='store_true', help='Show stored data information')
    parser.add_argument('--clear-data', type=str, nargs='?', const='ALL', help='Clear data (optionally specify symbol)')
    
    args = parser.parse_args()
    
    # Handle info operations
    if args.data_info:
        show_data_info()
        return
    
    if args.clear_data:
        symbol = None if args.clear_data == 'ALL' else args.clear_data
        clear_parquet_data(symbol)
        return
    
    # Process symbols if provided
    symbols = None
    if args.symbols:
        symbol_list = args.symbols.split(',')
        symbols = {}
        for symbol in symbol_list:
            symbol = symbol.strip().upper()
            if symbol == 'NIFTY50':
                symbols["NSE:NIFTY50-INDEX"] = "NIFTY50"
            elif symbol == 'BANKNIFTY':
                symbols["NSE:NIFTYBANK-INDEX"] = "BANKNIFTY"
            else:
                # Assume it's a stock
                symbols[f"NSE:{symbol}-EQ"] = symbol
    
    # Validate timeframes
    if args.timeframes:
        timeframes = args.timeframes.split(',')
    else:
        timeframes = None
    
    # Set up parquet data store
    success = setup_parquet_data(
        years_back=args.years,
        timeframes=timeframes,
        symbols=symbols
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 