#!/usr/bin/env python3
"""
Run all trading strategies at once and save results to database
"""
import sys
import traceback
import os
import time
from datetime import datetime, timedelta
import argparse
import backoff
import pandas as pd
import concurrent.futures
from collections import defaultdict
from src.core.multi_timeframe import prepare_multi_timeframe_data

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

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def fetch_candles(symbol, fyers, resolution="15", range_from=None, range_to=None):
    """
    Fetch historical candle data from Fyers API with improved error handling
    
    Args:
        symbol: The symbol to fetch data for (e.g., "NSE:NIFTY50-INDEX")
        fyers: The Fyers API client
        resolution: The timeframe resolution (1, 5, 15, D, W, M)
        range_from: Start date in YYYY-MM-DD format or timestamp
        range_to: End date in YYYY-MM-DD format or timestamp
    
    Returns:
        DataFrame with OHLCV data
    """
    # Set default date range if not provided - use a shorter recent period
    if not range_from:
        range_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not range_to:
        range_to = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching data for {symbol} from {range_from} to {range_to} with resolution {resolution}")
    
    # Try a simpler approach with direct date strings
    try:
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",  # Unix timestamp
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1"
        }
        
        print(f"Request data: {data}")
        response = fyers.history(data)
        
        # Check if response contains error
        if isinstance(response, dict) and 'code' in response and response['code'] != 200:
            error_code = response.get('code')
            error_message = response.get('message', 'Unknown error')
            print(f"API Error: {error_code} - {error_message}")
            
            # Try a shorter date range (last 7 days)
            print("Trying with a shorter date range (last 7 days)...")
            short_range_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            short_range_to = datetime.now().strftime('%Y-%m-%d')
            
            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": short_range_from,
                "range_to": short_range_to,
                "cont_flag": "1"
            }
            
            print(f"New request data: {data}")
            response = fyers.history(data)
            
            if isinstance(response, dict) and 'code' in response and response['code'] != 200:
                raise Exception(f"Failed to fetch data: {response.get('message')}")
        
        # Check if candles exist in the response
        candles = response.get('candles')
        if not candles:
            raise Exception(f"No candle data fetched for {symbol}")
            
        print(f"✅ Successfully fetched {len(candles)} candles for {symbol}")
        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    except Exception as e:
        print(f"❗ Error fetching candles: {e}")
        raise

def get_available_strategies():
    """Get all available strategies from src/strategies directory"""
    try:
        from src.strategies import get_available_strategies
        return get_available_strategies()
    except ImportError:
        print("❌ Could not import strategies module")
        return []

def initialize_fyers_client():
    from fyers_apiv3 import fyersModel
    from dotenv import load_dotenv
    load_dotenv()
    return fyersModel.FyersModel(
        token=os.getenv("FYERS_ACCESS_TOKEN"),
        client_id=os.getenv("FYERS_CLIENT_ID"),
        is_async=False,
        log_path=""
    )

def fetch_all_symbol_data(fyers, symbols, resolution, days_back):
    from ta.trend import EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    dataframes = {}
    multi_timeframe_dataframes = {}
    for symbol, name in symbols.items():
        try:
            print(f"\n📈 {name}")
            df = fetch_candles(symbol, fyers, resolution, start_date, end_date)
            if df.empty:
                print(f"❗ No data for {name}")
                continue

            df['ema_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema'] = df['ema_20']
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            bb = BollingerBands(df['close'])
            df['bollinger_upper'] = bb.bollinger_hband()
            df['bollinger_lower'] = bb.bollinger_lband()
            df['bollinger_mid'] = bb.bollinger_mavg()

            dataframes[name] = df
            # Prepare multi-timeframe data for each symbol
            multi_timeframe_dataframes[name] = prepare_multi_timeframe_data(df, base_resolution='3min')
        except Exception as e:
            print(f"❌ Error fetching {name}: {e}")
            traceback.print_exc()
    return dataframes, multi_timeframe_dataframes

def print_summary(results, duration):
    print("\n📊 Overall Execution Summary:")
    print(f"⏱ Duration: {duration:.2f} seconds")
    print(f"📦 Total strategies: {len(results)}")

    for strategy_name, strategy_results in results.items():
        print(f"\n📌 {strategy_name}:")

        # Handle top-level error
        if 'error' in strategy_results:
            print(f"  ❌ ERROR: {strategy_results['error']}")
            continue

        for index_name, stats in strategy_results.items():
            print(f"  ▶ {index_name}")

            # Handle nested error
            if isinstance(stats, dict) and 'error' in stats:
                print(f"    ❌ ERROR: {stats['error']}")
                continue

            candles = stats.get('candles', 0)
            print(f"    Candles: {candles}")

            signals = stats.get('signals', {})
            if signals:
                print("    Signal distribution:")
                for signal_type, count in signals.items():
                    percentage = (count / candles) * 100 if candles else 0
                    print(f"      {signal_type}: {count} ({percentage:.1f}%)")
            else:
                print("    ⚠️ No signals generated.")

            if 'records_saved' in stats:
                print(f"    Records saved: {stats['records_saved']}")

def run_strategy(strategy_name, dataframes, multi_timeframe_dataframes, save_to_db):
    """Run a single strategy on all provided dataframes.
    
    Args:
        strategy_name: Name of the strategy to run
        dataframes: Dict of dataframes keyed by index name
        multi_timeframe_dataframes: Dict of multi-timeframe dataframes keyed by index name
        save_to_db: Whether to save results to database
    
    Returns:
        Dict containing results for each index
    """
    from src.strategies import get_strategy_class
    if save_to_db:
        from db import log_strategy_sql
    else:
        def log_strategy_sql(*args, **kwargs):
            pass  # no-op if not saving to DB
    strategy_results = {}
    print(f"\n====== Running strategy: {strategy_name} ======")

    # Get the strategy class
    strategy_class = get_strategy_class(strategy_name)
    if not strategy_class:
        print(f"❌ Strategy class '{strategy_name}' not found")
        return {strategy_name: {"error": "Strategy class not found"}}

    # Process each index
    for index_name, df in dataframes.items():
        try:
            # Add indicators to the dataframe
            df_with_indicators = strategy_class().add_indicators(df.copy())
            candle_count = len(df_with_indicators)
            signal_count = defaultdict(int)
            all_signals = []
            # Prepare multi-timeframe data for this symbol
            timeframe_data = multi_timeframe_dataframes.get(index_name)
            # Initialize the strategy with timeframe_data if supported
            try:
                strategy = strategy_class(timeframe_data=timeframe_data)
            except TypeError:
                strategy = strategy_class()
            # Process each candle
            for i in range(candle_count):
                row = df_with_indicators.iloc[i]
                # For supertrend_ema, always use the new signature
                if strategy_name == 'supertrend_ema':
                    signal_result = strategy.analyze(row, i, df_with_indicators)
                elif hasattr(strategy, 'analyze') and 'timeframe_data' in strategy.__init__.__code__.co_varnames:
                    signal_result = strategy.analyze(row, i, df_with_indicators)
                else:
                    # Fallback for legacy strategies
                    candle_data = df_with_indicators.iloc[i:i+1]
                    future_data = df_with_indicators.iloc[i+1:i+11] if i + 1 < candle_count else None
                    analyze_kwargs = {"future_data": future_data}
                    if strategy_name not in ["insidebar_rsi", "supertrend_macd_rsi_ema"]:
                        analyze_kwargs["index_name"] = index_name
                    signal_result = strategy.analyze(candle_data, **analyze_kwargs)
                # If signal_result is None, treat as NO TRADE
                if signal_result is None:
                    signal_result = {'signal': 'NO TRADE'}
                signal = signal_result.get('signal', 'NO TRADE')
                signal_count[signal] += 1
                # Prepare signal info for database
                if isinstance(row.name, pd.Timestamp):
                    time_str = row.name.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = str(row.name)
                signal_info = {
                    'time': time_str,
                    'signal_time': time_str,
                    'signal': signal,
                    'price': signal_result.get('price', row['close']),
                    'confidence': signal_result.get('confidence', 'Low'),
                    'index_name': index_name
                }
                for key, value in signal_result.items():
                    if key not in signal_info:
                        signal_info[key] = value
                all_signals.append(signal_info)
            records_saved = 0
            if save_to_db:
                trading_signals = [s for s in all_signals if s['signal'] not in ['NO TRADE', None]]
                for signal_info in trading_signals:
                    try:
                        log_strategy_sql(strategy_name, signal_info)
                        records_saved += 1
                    except Exception as e:
                        print(f"❌ DB Save error: {e}")
            strategy_results[index_name] = {
                'candles': candle_count,
                'signals': dict(signal_count),
                'records_saved': records_saved if save_to_db else None
            }
        except Exception as e:
            print(f"❌ Error in {index_name}: {e}")
            traceback.print_exc()
            strategy_results[index_name] = {"error": str(e)}

    print(f"====== Completed: {strategy_name} ======\n")
    return strategy_results

def run_all_strategies(days_back=5, resolution="15", save_to_db=True, symbols=None):
    if not check_dependencies() or not check_fyers_credentials():
        return False

    strategies = get_available_strategies()
    if not strategies:
        print("❌ No strategies found")
        return False

    symbols = symbols or {
        "NSE:NIFTY50-INDEX": "NIFTY50",
        "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
    }

    print(f"🔄 Running {len(strategies)} strategies")
    fyers = initialize_fyers_client()
    dataframes, multi_timeframe_dataframes = fetch_all_symbol_data(fyers, symbols, resolution, days_back)

    if not dataframes:
        print("❌ No data available")
        return False

    results = {}
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_strategy, name, dataframes, multi_timeframe_dataframes, save_to_db): name
            for name in strategies
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = {"error": str(e)}

    duration = time.time() - start_time
    print_summary(results, duration)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all trading strategies and save results to database')
    parser.add_argument('--days', type=int, default=5, help='Number of days to backtest')
    parser.add_argument('--resolution', type=str, default='15', help='Candle resolution in minutes (1, 5, 15, 30, 60, D)')
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
    
    # Run all strategies
    success = run_all_strategies(
        days_back=args.days,
        resolution=args.resolution,
        save_to_db=not args.no_save,
        symbols=symbols
    )
    
    sys.exit(0 if success else 1) 