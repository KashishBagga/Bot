#!/usr/bin/env python3
"""
Run backtesting with real Fyers market data
"""
import sys
import traceback
import sqlite3
import os
from datetime import datetime, timedelta

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        print("Checking required dependencies...")
        import fyers_apiv3
        import pandas
        import ta
        import backoff
        import numpy
        import dotenv
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install all required dependencies with:")
        print("pip install fyers-apiv3 requests pandas numpy ta SQLAlchemy schedule pytz python-dotenv backoff")
        return False

def check_db_tables():
    """Check if required database tables exist"""
    print("Checking database tables...")
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Found tables: {tables}")
    
    conn.close()

def check_fyers_credentials():
    """Check if Fyers API credentials are properly set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check essential credentials
    print("Checking Fyers API credentials...")
    client_id = os.getenv("FYERS_CLIENT_ID")
    access_token = os.getenv("FYERS_ACCESS_TOKEN")
    
    if not client_id or not access_token:
        print("❌ Missing Fyers API credentials")
        print("Please run test_fyers.py first to set up your credentials")
        return False
    
    print(f"✅ Found Fyers credentials for client ID {client_id}")
    return True

def run_backtesting():
    """Run the backtesting script with error handling"""
    try:
        # Check dependencies first
        if not check_dependencies():
            return False
            
        # Check Fyers credentials
        if not check_fyers_credentials():
            return False
            
        # Check database tables
        check_db_tables()
        
        print("\n🚀 Starting backtesting with real Fyers data...")
        
        # Import here after dependency check
        import backtesting
        
        # Run the backtesting
        print("✅ Backtesting completed successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import required module: {e}")
        print("Make sure all required packages are installed")
        return False
    except Exception as e:
        print(f"❌ Backtesting failed with error: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        return False

def run_strategy_test(strategy_name=None):
    """Run test for a specific strategy with real Fyers data"""
    try:
        # Check dependencies
        if not check_dependencies():
            return False
            
        # Check Fyers credentials
        if not check_fyers_credentials():
            return False
        
        # Import here after checks
        from backtesting import fetch_candles, generate_signal_all
        from fyers_apiv3 import fyersModel
        from dotenv import load_dotenv
        import pandas as pd
        
        load_dotenv()
        
        # Get Fyers API credentials
        client_id = os.getenv("FYERS_CLIENT_ID")
        access_token = os.getenv("FYERS_ACCESS_TOKEN")
        
        print(f"🔄 Initializing Fyers client...")
        fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")
        
        # Strategy to test
        if not strategy_name:
            strategy_name = input("Enter strategy name to test (e.g., strategy_range_breakout_volatility): ")
        
        # Symbols to test
        symbols = {
            "NSE:NIFTY50-INDEX": "NIFTY50",
            "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
        }
        
        # Use a recent time period (last 30 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"📊 Testing strategy '{strategy_name}' from {start_date} to {end_date}")
        
        for symbol, index_name in symbols.items():
            try:
                print(f"\n📈 Processing {index_name}...")
                
                # Fetch data from Fyers
                df = fetch_candles(symbol, fyers, resolution="5", range_from=start_date, range_to=end_date)
                
                if df.empty:
                    print(f"❗ No candles available for {index_name}")
                    continue
                
                print(f"✅ Successfully fetched {len(df)} candles for {index_name}")
                
                # Calculate indicators and run strategy
                print(f"🔄 Analyzing with strategy '{strategy_name}'...")
                
                # Pre-calculate basic indicators
                from ta.trend import EMAIndicator
                from ta.momentum import RSIIndicator
                from ta.trend import MACD
                from ta.volatility import AverageTrueRange, BollingerBands
                
                # EMA
                df['ema_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
                df['ema_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
                df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
                
                # RSI
                df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
                
                # MACD
                macd = MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_diff'] = macd.macd_diff()
                
                # ATR
                df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
                
                # Bollinger Bands
                bb = BollingerBands(df['close'])
                df['bollinger_upper'] = bb.bollinger_hband()
                df['bollinger_lower'] = bb.bollinger_lband()
                df['bollinger_mid'] = bb.bollinger_mavg()
                
                # Display sample data
                print("\n📊 Sample Data:")
                print(df[['time', 'open', 'high', 'low', 'close', 'ema_20', 'rsi', 'atr']].tail(5))
                
                # Import our strategy modules
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
                # For running our new class-based strategies from src
                if os.path.exists(f"src/strategies/{strategy_name}.py"):
                    print(f"🔄 Using class-based strategy from src/strategies/{strategy_name}.py")
                    from src.strategies import get_strategy_class
                    strategy_class = get_strategy_class(strategy_name)
                    if strategy_class:
                        strategy = strategy_class()
                        df = strategy.add_indicators(df)
                        
                        # Generate signals for last 10 candles
                        for i in range(max(0, len(df)-10), len(df)):
                            candle_data = df.iloc[i:i+1]
                            signal_result = strategy.analyze(candle_data)
                            print(f"Time: {df.iloc[i]['time']} - Signal: {signal_result['signal']} (Price: {signal_result['price']:.2f})")
                    else:
                        print(f"❌ Strategy class '{strategy_name}' not found in src/strategies")
                
                # For running our old function-based strategies
                elif os.path.exists(f"strategies/{strategy_name}.py"):
                    print(f"🔄 Using function-based strategy from strategies/{strategy_name}.py")
                    # Import dynamically
                    exec(f"from strategies.{strategy_name} import {strategy_name}")
                    for i in range(max(0, len(df)-10), len(df)):
                        candle = df.iloc[i]
                        # Call strategy function
                        result = eval(f"{strategy_name}(candle, index_name)")
                        print(f"Time: {candle['time']} - Signal: {result['signal']} (Price: {candle['close']:.2f})")
                else:
                    print(f"❌ Strategy file for '{strategy_name}' not found")
                    
                print(f"\n✅ Strategy test completed for {index_name}")
                
            except Exception as e:
                print(f"❌ Error processing {index_name}: {e}")
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtesting or test a specific strategy')
    parser.add_argument('--full', action='store_true', help='Run full backtesting with all strategies')
    parser.add_argument('--strategy', type=str, help='Name of the specific strategy to test')
    
    args = parser.parse_args()
    
    if args.full:
        success = run_backtesting()
    else:
        success = run_strategy_test(args.strategy)
        
    sys.exit(0 if success else 1) 