#!/usr/bin/env python3
"""
Parquet-Based Backtesting System
Ultra-fast backtesting using pre-stored parquet data with all timeframes readily available.
"""
import sys
import argparse
import time
import concurrent.futures
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.parquet_data_store import ParquetDataStore
from all_strategies import run_strategy, get_available_strategies
from dotenv import load_dotenv

def run_parquet_backtest(days_back: int = 30, timeframe: str = "5min", 
                        save_to_db: bool = True, symbols: List[str] = None,
                        strategies: List[str] = None, parallel: bool = True):
    """Run backtest using parquet data store.
    
    Args:
        days_back: Number of days to backtest
        timeframe: Timeframe to use for backtesting
        save_to_db: Whether to save results to database
        symbols: Specific symbols to test
        strategies: Specific strategies to test
        parallel: Whether to run strategies in parallel
    """
    load_dotenv()
    
    # Initialize data store
    data_store = ParquetDataStore()
    
    # Get available symbols
    available_symbols = data_store.get_available_symbols()
    if not available_symbols:
        print("❌ No data found in parquet store. Run setup_parquet_data.py first.")
        return False
    
    # Process symbols
    if symbols:
        symbol_list = []
        for symbol in symbols:
            if symbol.upper() == 'NIFTY50':
                symbol_list.append('NSE:NIFTY50-INDEX')
            elif symbol.upper() == 'BANKNIFTY':
                symbol_list.append('NSE:NIFTYBANK-INDEX')
            else:
                symbol_list.append(f'NSE:{symbol.upper()}-EQ')
        
        # Filter to available symbols
        symbols_to_test = [s for s in symbol_list if s in available_symbols]
        if not symbols_to_test:
            print(f"❌ None of the specified symbols found in data store.")
            print(f"Available symbols: {available_symbols}")
            return False
    else:
        symbols_to_test = available_symbols
    
    # Get available strategies
    all_strategies = get_available_strategies()
    if strategies:
        strategies_to_test = [s for s in strategies if s in all_strategies]
        if not strategies_to_test:
            print(f"❌ None of the specified strategies found.")
            print(f"Available strategies: {all_strategies}")
            return False
    else:
        strategies_to_test = all_strategies
    
    # Validate timeframe
    sample_symbol = symbols_to_test[0]
    available_timeframes = data_store.get_available_timeframes(sample_symbol)
    if timeframe not in available_timeframes:
        print(f"❌ Timeframe '{timeframe}' not available.")
        print(f"Available timeframes: {available_timeframes}")
        return False
    
    execution_mode = "Parallel" if parallel else "Sequential"
    print(f"🚀 Running Parquet-Based Backtest ({execution_mode}):")
    print(f"  📅 Period: Last {days_back} days")
    print(f"  ⏰ Timeframe: {timeframe}")
    print(f"  💿 Save to DB: {save_to_db}")
    print(f"  📈 Symbols: {len(symbols_to_test)} ({', '.join([s.split(':')[1].replace('-INDEX', '').replace('-EQ', '') for s in symbols_to_test])})")
    print(f"  🧠 Strategies: {len(strategies_to_test)} ({', '.join(strategies_to_test)})")
    print(f"  ⚡ Execution: {execution_mode}")
    
    start_time = time.time()
    
    # Load data for all symbols
    print(f"\n📊 Loading {timeframe} data...")
    dataframes = {}
    multi_timeframe_dataframes = {}
    
    for symbol in symbols_to_test:
        # Load primary timeframe data
        df = data_store.load_data(symbol, timeframe, days_back)
        if df.empty:
            print(f"⚠️ No data for {symbol} at {timeframe}")
            continue
        
        # Create display name
        if 'NIFTY50' in symbol:
            display_name = 'NIFTY50'
        elif 'NIFTYBANK' in symbol:
            display_name = 'BANKNIFTY'
        else:
            display_name = symbol.split(':')[1].replace('-EQ', '')
        
        dataframes[display_name] = df
        
        # Load multi-timeframe data (for strategies that need it)
        all_timeframes = data_store.get_available_timeframes(symbol)
        multi_tf_data = {}
        for tf in all_timeframes:
            tf_df = data_store.load_data(symbol, tf, days_back)
            if not tf_df.empty:
                multi_tf_data[tf] = tf_df
        
        multi_timeframe_dataframes[display_name] = multi_tf_data
        
        print(f"  ✅ {display_name}: {len(df)} candles ({timeframe})")
    
    if not dataframes:
        print("❌ No data loaded")
        return False
    
    data_load_time = time.time() - start_time
    print(f"📈 Data loaded in {data_load_time:.2f} seconds")
    
    # Run strategies
    print(f"\n🎯 Running {len(strategies_to_test)} strategies ({execution_mode})...")
    results = {}
    
    if parallel and len(strategies_to_test) > 1:
        # Parallel execution
        print_lock = threading.Lock()
        
        def run_strategy_wrapper(strategy_name):
            try:
                with print_lock:
                    print(f"  🔄 Running {strategy_name}...")
                result = run_strategy(strategy_name, dataframes, multi_timeframe_dataframes, save_to_db)
                with print_lock:
                    print(f"  ✅ {strategy_name} completed")
                return strategy_name, result
            except Exception as e:
                with print_lock:
                    print(f"  ❌ {strategy_name} failed: {e}")
                return strategy_name, {"error": str(e)}
        
        # Use ThreadPoolExecutor for parallel execution
        max_workers = min(len(strategies_to_test), 4)  # Limit to 4 parallel strategies
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_strategy = {executor.submit(run_strategy_wrapper, strategy): strategy 
                                for strategy in strategies_to_test}
            
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy_name, result = future.result()
                results[strategy_name] = result
    else:
        # Sequential execution
        for strategy_name in strategies_to_test:
            try:
                print(f"  🔄 Running {strategy_name}...")
                result = run_strategy(strategy_name, dataframes, multi_timeframe_dataframes, save_to_db)
                results[strategy_name] = result
                print(f"  ✅ {strategy_name} completed")
            except Exception as e:
                print(f"  ❌ {strategy_name} failed: {e}")
                results[strategy_name] = {"error": str(e)}
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n📊 Backtest Summary:")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"  📊 Data loading: {data_load_time:.2f}s")
    print(f"  🧠 Strategy execution: {total_time - data_load_time:.2f}s")
    print(f"📦 Strategies tested: {len(results)}")
    
    total_signals = 0
    for strategy_name, strategy_results in results.items():
        if 'error' in strategy_results:
            print(f"  ❌ {strategy_name}: ERROR")
            continue
            
        strategy_signals = 0
        for index_name, stats in strategy_results.items():
            if isinstance(stats, dict) and 'records_saved' in stats:
                records_saved = stats.get('records_saved', 0)
                strategy_signals += records_saved if records_saved is not None else 0
        
        total_signals += strategy_signals
        print(f"  📈 {strategy_name}: {strategy_signals} signals")
    
    print(f"\n🎉 Total signals generated: {total_signals}")
    
    # Performance stats
    total_candles = sum(len(df) for df in dataframes.values())
    candles_per_second = total_candles / total_time if total_time > 0 else 0
    speedup = f" (⚡ {total_time / (data_load_time + total_time - data_load_time):.1f}x faster)" if parallel else ""
    print(f"⚡ Performance: {total_candles:,} candles processed at {candles_per_second:,.0f} candles/second{speedup}")
    
    return True

def show_data_info():
    """Display parquet data store information."""
    data_store = ParquetDataStore()
    info = data_store.get_storage_info()
    
    print("📊 Parquet Data Store Information:")
    print(f"📁 Storage directory: {info['storage_directory']}")
    print(f"📊 Total symbols: {info['total_symbols']}")
    print(f"💾 Total storage: {info['total_size_mb']} MB")
    
    if info['symbols']:
        print("\n📈 Available Data:")
        for symbol_info in info['symbols']:
            print(f"\n  • {symbol_info['name']} ({symbol_info['symbol']})")
            print(f"    📅 Period: {symbol_info['date_range']}")
            print(f"    📊 Base candles: {symbol_info['base_candles_count']:,}")
            print(f"    🎯 Timeframes: {len(symbol_info['timeframes'])}")
            print(f"      {', '.join(symbol_info['timeframes'])}")
            print(f"    💾 Size: {symbol_info['size_mb']} MB")
    else:
        print("\n❌ No data found. Run setup_parquet_data.py to initialize.")

def show_timeframe_comparison():
    """Show data comparison across timeframes for a symbol."""
    data_store = ParquetDataStore()
    available_symbols = data_store.get_available_symbols()
    
    if not available_symbols:
        print("❌ No data available")
        return
    
    # Use first available symbol for demonstration
    symbol = available_symbols[0]
    timeframes = data_store.get_available_timeframes(symbol)
    
    if not timeframes:
        print(f"❌ No timeframes available for {symbol}")
        return
    
    print(f"📊 Timeframe Comparison for {symbol}:")
    print(f"{'Timeframe':<10} {'Candles':<15} {'Date Range':<25} {'Size'}")
    print("-" * 70)
    
    for tf in timeframes:
        df = data_store.load_data(symbol, tf, days_back=None)  # Load all data
        if not df.empty:
            candle_count = len(df)
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            date_range = f"{start_date} to {end_date}"
            
            # Estimate size
            file_path = data_store._get_timeframe_file(symbol, tf)
            size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
            
            print(f"{tf:<10} {candle_count:<15,} {date_range:<25} {size_mb:.2f} MB")

def benchmark_loading():
    """Benchmark data loading performance."""
    data_store = ParquetDataStore()
    available_symbols = data_store.get_available_symbols()
    
    if not available_symbols:
        print("❌ No data available for benchmarking")
        return
    
    symbol = available_symbols[0]
    timeframes = data_store.get_available_timeframes(symbol)
    
    print(f"⚡ Benchmarking data loading for {symbol}:")
    print(f"{'Timeframe':<10} {'Load Time':<12} {'Candles':<12} {'Speed'}")
    print("-" * 50)
    
    for tf in timeframes:
        start_time = time.time()
        df = data_store.load_data(symbol, tf, days_back=30)  # Last 30 days
        load_time = time.time() - start_time
        
        if not df.empty:
            candles = len(df)
            speed = candles / load_time if load_time > 0 else 0
            print(f"{tf:<10} {load_time:<12.4f}s {candles:<12,} {speed:,.0f}/s")

def main():
    parser = argparse.ArgumentParser(description='Run parquet-based backtesting')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    parser.add_argument('--timeframe', type=str, default='5min', help='Timeframe to use (default: 5min)')
    parser.add_argument('--no-save', action='store_true', help="Don't save results to database")
    parser.add_argument('--symbols', type=str, help='Symbols to test, comma separated (e.g., NIFTY50,BANKNIFTY)')
    parser.add_argument('--strategies', type=str, help='Strategies to test, comma separated')
    parser.add_argument('--data-info', action='store_true', help='Show data store information')
    parser.add_argument('--timeframe-comparison', action='store_true', help='Show timeframe comparison')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark data loading performance')
    parser.add_argument('--sequential', action='store_true', help='Run strategies sequentially instead of parallel')
    
    args = parser.parse_args()
    
    # Handle info operations
    if args.data_info:
        show_data_info()
        return
    
    if args.timeframe_comparison:
        show_timeframe_comparison()
        return
    
    if args.benchmark:
        benchmark_loading()
        return
    
    # Process symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Process strategies if provided
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]
    
    # Run backtest
    success = run_parquet_backtest(
        days_back=args.days,
        timeframe=args.timeframe,
        save_to_db=not args.no_save,
        symbols=symbols,
        strategies=strategies,
        parallel=not args.sequential
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 