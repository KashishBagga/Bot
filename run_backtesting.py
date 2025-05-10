#!/usr/bin/env python3
"""
Run backtesting with real Fyers market data
"""
import sys
import traceback
import sqlite3
import os
import csv
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

def log_to_database(strategy_name, symbol, signals):
    """Save strategy signals to the SQLite database in the appropriate strategy table
    
    Args:
        strategy_name: Name of the strategy
        symbol: Symbol being analyzed (e.g., NIFTY50)
        signals: List of signal dictionaries containing strategy results
        
    Returns:
        int: Number of records inserted
    """
    from db import log_strategy_sql
    
    print(f"🔄 Saving {len(signals)} signals to database table '{strategy_name}'...")
    
    records_saved = 0
    for signal_info in signals:
        # Prepare signal data with appropriate fields for database
        timestamp = signal_info.get('time')
        signal_data = {
            'signal_time': timestamp,
            'index_name': symbol,
            'signal': signal_info.get('signal', 'NO TRADE'),
            'price': signal_info.get('price', 0),
            'confidence': signal_info.get('confidence', 'Low'),
            'trade_type': 'Backtest',
            'outcome': 'Pending',  # Backtest outcomes would be determined by future price action
        }
        
        # Include any other fields that are present in the signal_info
        for key, value in signal_info.items():
            if key not in ['time'] and key not in signal_data:
                signal_data[key] = value
        
        # Log to the database using the strategy-specific function
        log_strategy_sql(strategy_name, signal_data)
        records_saved += 1
    
    print(f"✅ Saved {records_saved} signals to database table '{strategy_name}'")
    return records_saved

def run_strategy_test(strategy_name=None, days_back=30, resolution="5", save_to_db=True, symbols=None):
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
        if not symbols:
            symbols = {
                "NSE:NIFTY50-INDEX": "NIFTY50",
                "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
            }
        
        # Use a recent time period (last X days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        print(f"📊 Testing strategy '{strategy_name}' from {start_date} to {end_date} with {resolution} min candles")
        
        results = {}
        
        for symbol, index_name in symbols.items():
            try:
                print(f"\n📈 Processing {index_name}...")
                
                # Fetch data from Fyers
                df = fetch_candles(symbol, fyers, resolution=resolution, range_from=start_date, range_to=end_date)
                
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
                
                # Track all signals for CSV export
                all_signals = []
                signal_count = {"NO TRADE": 0, "BUY CALL": 0, "BUY PUT": 0}
                
                # For running our new class-based strategies from src
                if os.path.exists(f"src/strategies/{strategy_name}.py"):
                    print(f"🔄 Using class-based strategy from src/strategies/{strategy_name}.py")
                    from src.strategies import get_strategy_class
                    strategy_class = get_strategy_class(strategy_name)
                    if strategy_class:
                        strategy = strategy_class()
                        df = strategy.add_indicators(df)
                        
                        # Generate signals for all candles in the dataframe
                        candle_count = len(df)
                        print(f"Generating signals for {candle_count} candles...")
                        
                        # For very large datasets, just print samples and count signals
                        signal_samples = min(10, candle_count)
                        print_interval = max(1, candle_count // 10)  # Print 10% of signals
                        
                        # Process all candles
                        for i in range(candle_count):
                            candle_data = df.iloc[i:i+1]
                            signal_result = strategy.analyze(candle_data)
                            
                            # Safe access to values - some strategies might return different keys
                            price = signal_result.get('price', df.iloc[i]['close'])
                            signal = signal_result.get('signal', 'NO TRADE')
                            confidence = signal_result.get('confidence', 'Low')
                            
                            # Count signal types
                            signal_count[signal] = signal_count.get(signal, 0) + 1
                            
                            # Record the signal
                            signal_info = {
                                'time': df.iloc[i]['time'].strftime('%Y-%m-%d %H:%M:%S'),
                                'signal': signal,
                                'price': price,
                                'confidence': confidence
                            }
                            all_signals.append(signal_info)
                            
                            # Print some samples (last few and some distributed across the dataset)
                            if i >= candle_count - signal_samples or i % print_interval == 0:
                                # print(f"Time: {df.iloc[i]['time']} - Signal: {signal} (Price: {price:.2f})")
                                pass
                    else:
                        print(f"❌ Strategy class '{strategy_name}' not found in src/strategies")
                
                # For running our old function-based strategies
                elif os.path.exists(f"strategies/{strategy_name}.py"):
                    print(f"🔄 Using function-based strategy from strategies/{strategy_name}.py")
                    # Import dynamically
                    exec(f"from strategies.{strategy_name} import {strategy_name}")
                    
                    # Process all candles
                    candle_count = len(df)
                    print(f"Generating signals for {candle_count} candles...")
                    
                    # For very large datasets, just print samples and count signals
                    signal_samples = min(10, candle_count)
                    print_interval = max(1, candle_count // 10)  # Print 10% of signals
                    
                    for i in range(candle_count):
                        candle = df.iloc[i]
                        # Call strategy function
                        result = eval(f"{strategy_name}(candle, index_name)")
                        
                        # Extract signal
                        signal = result.get('signal', 'NO TRADE')
                        price = candle['close']
                        confidence = result.get('confidence', 'Low')
                        
                        # Count signal types
                        signal_count[signal] = signal_count.get(signal, 0) + 1
                        
                        # Record the signal
                        signal_info = {
                            'time': candle['time'].strftime('%Y-%m-%d %H:%M:%S'),
                            'signal': signal,
                            'price': price,
                            'confidence': confidence
                        }
                        all_signals.append(signal_info)
                        
                        # Print some samples
                        if i >= candle_count - signal_samples or i % print_interval == 0:
                            # print(f"Time: {candle['time']} - Signal: {signal} (Price: {price:.2f})")
                else:
                    print(f"❌ Strategy file for '{strategy_name}' not found")
                    continue
                    
                # Display signal distribution
                print(f"\n📊 Signal Distribution:")
                for signal_type, count in signal_count.items():
                    percentage = (count / candle_count) * 100 if candle_count > 0 else 0
                    print(f"{signal_type}: {count} ({percentage:.1f}%)")
                
                # Save signals to database if requested (replacing the CSV saving)
                if save_to_db and all_signals:
                    records_saved = log_to_database(strategy_name, index_name, all_signals)
                    results[index_name] = {
                        'candles': candle_count,
                        'signals': signal_count,
                        'records_saved': records_saved
                    }
                else:
                    results[index_name] = {
                        'candles': candle_count,
                        'signals': signal_count
                    }
                
                print(f"\n✅ Strategy test completed for {index_name}")
                
            except Exception as e:
                print(f"❌ Error processing {index_name}: {e}")
                traceback.print_exc()
        
        # Overall summary
        print("\n📋 Overall Summary:")
        for symbol, result in results.items():
            print(f"{symbol}:")
            print(f"  Candles analyzed: {result['candles']}")
            print("  Signal distribution:")
            for signal_type, count in result['signals'].items():
                percentage = (count / result['candles']) * 100 if result['candles'] > 0 else 0
                print(f"    {signal_type}: {count} ({percentage:.1f}%)")
            if 'records_saved' in result:
                print(f"  Results saved to database: {result['records_saved']} records")
        
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
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--resolution', type=str, default='5', help='Candle resolution in minutes (1, 5, 15, 30, 60, D)')
    parser.add_argument('--save-db', action='store_true', help='Save results to database (default: True)', default=True)
    parser.add_argument('--no-save', action='store_true', help="Don't save results to database")
    parser.add_argument('--symbols', type=str, help='Symbols to test, comma separated (default: NIFTY50,BANKNIFTY)')
    
    args = parser.parse_args()
    
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
    
    if args.full:
        success = run_backtesting()
    else:
        # If --no-save is specified, don't save to database
        save_to_db = not args.no_save if args.no_save else args.save_db
        
        success = run_strategy_test(
            strategy_name=args.strategy,
            days_back=args.days,
            resolution=args.resolution,
            save_to_db=save_to_db,
            symbols=symbols
        )
        
    sys.exit(0 if success else 1) 