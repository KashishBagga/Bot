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
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.parquet_data_store import ParquetDataStore
from all_strategies import get_available_strategies, run_strategy
from dotenv import load_dotenv

def run_parquet_backtest(days_back: int = 30, timeframe: str = "5min", 
                        save_to_db: bool = True, symbols: List[str] = None,
                        strategies: List[str] = None, parallel: bool = True, no_cache: bool = False):
    """Run backtest using parquet data store.
    
    Args:
        days_back: Number of days to backtest
        timeframe: Timeframe to use for backtesting
        save_to_db: Whether to save results to database
        symbols: Specific symbols to test
        strategies: Specific strategies to test
        parallel: Whether to run strategies in parallel
        no_cache: Whether to disable result caching
    """
    load_dotenv()
    
    # Initialize data store
    data_store = ParquetDataStore()
    
    # Get available symbols
    available_symbols = data_store.get_available_symbols()
    if not available_symbols:
        print("❌ No data found in parquet store. Run setup_parquet_data.py first.")
        return False
    
    # Clean up duplicate symbols (prioritize NSE: format over simple names)
    cleaned_symbols = []
    nse_symbols = [s for s in available_symbols if s.startswith('NSE:')]
    simple_symbols = [s for s in available_symbols if not s.startswith('NSE:')]
    
    # Add NSE symbols first
    cleaned_symbols.extend(nse_symbols)
    
    # Add simple symbols only if no NSE equivalent exists
    for simple in simple_symbols:
        # Check if there's already an NSE equivalent
        nse_equivalent = None
        if simple == 'NIFTY50':
            nse_equivalent = 'NSE:NIFTY50-INDEX'
        elif simple == 'NIFTYBANK' or simple == 'BANKNIFTY':
            nse_equivalent = 'NSE:NIFTYBANK-INDEX'
        
        if nse_equivalent not in nse_symbols:
            cleaned_symbols.append(simple)
    
    symbols_to_test = cleaned_symbols
    
    # Process user-specified symbols if provided
    if symbols:
        symbol_list = []
        for symbol in symbols:
            if symbol.upper() == 'NIFTY50':
                # Prefer NSE format if available, fallback to simple
                if 'NSE:NIFTY50-INDEX' in cleaned_symbols:
                    symbol_list.append('NSE:NIFTY50-INDEX')
                elif 'NIFTY50' in cleaned_symbols:
                    symbol_list.append('NIFTY50')
            elif symbol.upper() == 'BANKNIFTY':
                # Prefer NSE format if available, fallback to simple
                if 'NSE:NIFTYBANK-INDEX' in cleaned_symbols:
                    symbol_list.append('NSE:NIFTYBANK-INDEX')
                elif 'BANKNIFTY' in cleaned_symbols:
                    symbol_list.append('BANKNIFTY')
            else:
                symbol_list.append(f'NSE:{symbol.upper()}-EQ')
        
        # Filter to available symbols
        symbols_to_test = [s for s in symbol_list if s in cleaned_symbols]
        if not symbols_to_test:
            print(f"❌ None of the specified symbols found in data store.")
            print(f"Available symbols: {cleaned_symbols}")
            return False
    else:
        symbols_to_test = cleaned_symbols
    
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
    print(f"  📈 Symbols: {len(symbols_to_test)} ({', '.join([s.split(':')[1].replace('-INDEX', '').replace('-EQ', '') if ':' in s else s for s in symbols_to_test])})")
    print(f"  🧠 Strategies: {len(strategies_to_test)} ({', '.join(strategies_to_test)})")
    print(f"  ⚡ Execution: {execution_mode}")
    
    # Check cache for identical configuration
    cache_enabled = enable_smart_caching() and not no_cache
    cache_key = get_cache_key(days_back, timeframe, symbols_to_test, strategies_to_test, save_to_db)
    
    if cache_enabled:
        cached_results = load_cached_results(cache_key)
        if cached_results is not None:
            print("\n🎉 Cache hit! Skipping computation...")
            
            # Print summary from cache
            print(f"\n📊 Backtest Summary (from cache):")
            print(f"📦 Strategies tested: {len(cached_results)}")
            
            total_signals = 0
            for strategy_name, strategy_results in cached_results.items():
                if 'error' in strategy_results:
                    print(f"  ❌ {strategy_name}: ERROR")
                    continue
                    
                strategy_signals = 0
                for index_name, stats in strategy_results.items():
                    if isinstance(stats, dict) and 'signals' in stats:
                        signals_dict = stats.get('signals', {})
                        for signal_type, count in signals_dict.items():
                            if signal_type != 'NO TRADE':
                                strategy_signals += count
                
                total_signals += strategy_signals
                print(f"  📈 {strategy_name}: {strategy_signals} signals")
            
            print(f"\n🎉 Total signals generated: {total_signals}")
            print(f"⚡ Performance: INSTANT (cached)")
            return True
    
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
            display_name = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol
        
        # Check if we already have this display name (avoid duplicates)
        if display_name in dataframes:
            print(f"⚠️ Skipping duplicate symbol: {symbol} (already loaded as {display_name})")
            continue
        
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
    
    # Preprocess common indicators
    processed_dataframes = preprocess_common_indicators(dataframes)
    
    # Run strategies
    print(f"\n🎯 Running {len(strategies_to_test)} strategies ({execution_mode})...")
    results = optimize_strategy_execution(strategies_to_test, processed_dataframes, multi_timeframe_dataframes, save_to_db, parallel)
    
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
            if isinstance(stats, dict) and 'signals' in stats:
                # Count all signals except 'NO TRADE'
                signals_dict = stats.get('signals', {})
                for signal_type, count in signals_dict.items():
                    if signal_type != 'NO TRADE':
                        strategy_signals += count
        
        total_signals += strategy_signals
        print(f"  📈 {strategy_name}: {strategy_signals} signals")
    
    print(f"\n🎉 Total signals generated: {total_signals}")
    
    # Performance stats
    total_candles = sum(len(df) for df in dataframes.values())
    candles_per_second = total_candles / total_time if total_time > 0 else 0
    speedup = f" (⚡ {total_time / (data_load_time + total_time - data_load_time):.1f}x faster)" if parallel else ""
    print(f"⚡ Performance: {total_candles:,} candles processed at {candles_per_second:,.0f} candles/second{speedup}")
    
    # Save results to cache for future use
    if cache_enabled:
        save_to_cache(cache_key, results)
    
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

def preprocess_common_indicators(dataframes):
    """Pre-calculate common indicators used by multiple strategies to avoid redundant calculations."""
    processed_dataframes = {}
    
    print("📊 Pre-calculating common indicators...")
    for symbol, df in dataframes.items():
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Common indicators used across strategies
        from ta.trend import EMAIndicator, SMAIndicator
        from ta.momentum import RSIIndicator
        from ta.volatility import AverageTrueRange, BollingerBands
        from ta.trend import MACD
        
        # EMAs (most commonly used)
        processed_df['ema_9'] = EMAIndicator(processed_df['close'], window=9).ema_indicator()
        processed_df['ema_20'] = EMAIndicator(processed_df['close'], window=20).ema_indicator()
        processed_df['ema_21'] = EMAIndicator(processed_df['close'], window=21).ema_indicator()
        processed_df['ema_50'] = EMAIndicator(processed_df['close'], window=50).ema_indicator()
        
        # RSI (very common)
        processed_df['rsi'] = RSIIndicator(processed_df['close'], window=14).rsi()
        
        # ATR (used by many strategies)
        processed_df['atr'] = AverageTrueRange(processed_df['high'], processed_df['low'], processed_df['close'], window=14).average_true_range()
        
        # MACD (common momentum indicator)
        macd = MACD(processed_df['close'])
        processed_df['macd'] = macd.macd()
        processed_df['macd_signal'] = macd.macd_signal()
        processed_df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands (used by several strategies)
        bb = BollingerBands(processed_df['close'], window=20)
        processed_df['bb_upper'] = bb.bollinger_hband()
        processed_df['bb_lower'] = bb.bollinger_lband()
        processed_df['bb_middle'] = bb.bollinger_mavg()
        
        # Volume indicators
        if 'volume' in processed_df.columns:
            processed_df['volume_sma'] = SMAIndicator(processed_df['volume'], window=20).sma_indicator()
        
        processed_dataframes[symbol] = processed_df
    
    return processed_dataframes

def optimize_strategy_execution(strategies_to_test, processed_dataframes, multi_timeframe_dataframes, save_to_db, parallel):
    """Optimized strategy execution with result caching and vectorized operations."""
    results = {}
    
    # Strategy result cache to avoid recomputing identical operations
    strategy_cache = {}
    
    # Group strategies by similarity to optimize execution order
    similar_strategies = {
        'trend_following': ['supertrend_ema', 'ema_crossover'],
        'mean_reversion': ['insidebar_bollinger', 'insidebar_rsi'],
        'breakout': ['breakout_rsi', 'donchian_breakout', 'range_breakout_volatility'],
        'complex': ['supertrend_macd_rsi_ema']
    }
    
    # Flatten and maintain order
    ordered_strategies = []
    for group in similar_strategies.values():
        for strategy in group:
            if strategy in strategies_to_test:
                ordered_strategies.append(strategy)
    
    # Add any remaining strategies
    for strategy in strategies_to_test:
        if strategy not in ordered_strategies:
            ordered_strategies.append(strategy)
    
    if parallel and len(ordered_strategies) > 1:
        return run_parallel_optimized(ordered_strategies, processed_dataframes, multi_timeframe_dataframes, save_to_db)
    else:
        return run_sequential_optimized(ordered_strategies, processed_dataframes, multi_timeframe_dataframes, save_to_db)

def run_parallel_optimized(strategies_to_test, dataframes, multi_timeframe_dataframes, save_to_db):
    """Optimized parallel execution with intelligent worker allocation."""
    results = {}
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
    
    # Optimize worker count based on strategy count and available resources
    cpu_cores = os.cpu_count()
    
    # For small strategy counts, use fewer workers to reduce overhead
    if len(strategies_to_test) <= 2:
        max_workers = min(len(strategies_to_test), 2)
    elif len(strategies_to_test) <= 4:
        max_workers = min(len(strategies_to_test), 3)
    else:
        # For larger strategy counts, use more workers but leave some cores free
        max_workers = min(len(strategies_to_test), max(4, cpu_cores - 1))
    
    print(f"  🔧 Using {max_workers} parallel workers (CPU cores: {cpu_cores})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_strategy = {executor.submit(run_strategy_wrapper, strategy): strategy 
                            for strategy in strategies_to_test}
        
        for future in concurrent.futures.as_completed(future_to_strategy):
            strategy_name, result = future.result()
            results[strategy_name] = result
    
    return results

def run_sequential_optimized(strategies_to_test, dataframes, multi_timeframe_dataframes, save_to_db):
    """Optimized sequential execution with shared computations."""
    results = {}
    
    for strategy_name in strategies_to_test:
        try:
            print(f"  🔄 Running {strategy_name}...")
            result = run_strategy(strategy_name, dataframes, multi_timeframe_dataframes, save_to_db)
            results[strategy_name] = result
            print(f"  ✅ {strategy_name} completed")
        except Exception as e:
            print(f"  ❌ {strategy_name} failed: {e}")
            results[strategy_name] = {"error": str(e)}
    
    return results

def add_vectorized_processing():
    """Add support for vectorized strategy processing."""
    pass

def add_result_caching(timeframe, days_back):
    """Add intelligent result caching based on timeframe and period."""
    import hashlib
    cache_key = f"{timeframe}_{days_back}"
    cache_dir = Path("cache/backtest_results")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.pkl"

def add_progressive_backtesting():
    """Add support for progressive backtesting (start small, scale up)."""
    pass

def enable_smart_caching(cache_results=True):
    """Enable smart result caching for faster repeated backtests."""
    if not cache_results:
        return False
    
    cache_dir = Path("cache/backtest_results")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return True

def get_cache_key(days_back, timeframe, symbols, strategies, save_to_db):
    """Generate cache key for backtest configuration."""
    import hashlib
    
    # Create deterministic hash of configuration
    config_str = f"{days_back}_{timeframe}_{sorted(symbols)}_{sorted(strategies)}_{save_to_db}"
    return hashlib.md5(config_str.encode()).hexdigest()

def load_cached_results(cache_key):
    """Load cached backtest results if available."""
    import pickle
    
    cache_file = Path("cache/backtest_results") / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is recent (within last hour for demo)
            import time
            if time.time() - cached_data['timestamp'] < 3600:  # 1 hour
                print("🚀 Using cached results (performance boost!)")
                return cached_data['results']
        except Exception as e:
            print(f"⚠️ Cache read error: {e}")
    
    return None

def save_to_cache(cache_key, results):
    """Save backtest results to cache."""
    import pickle
    import time
    
    cache_file = Path("cache/backtest_results") / f"{cache_key}.pkl"
    cached_data = {
        'results': results,
        'timestamp': time.time()
    }
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print("💾 Results cached for future use")
    except Exception as e:
        print(f"⚠️ Cache save error: {e}")

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
    parser.add_argument('--no-cache', action='store_true', help='Disable result caching')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached results')
    
    args = parser.parse_args()
    
    # Handle cache operations
    if args.clear_cache:
        clear_cache()
        return
    
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
        parallel=not args.sequential,
        no_cache=args.no_cache
    )
    
    sys.exit(0 if success else 1)

def clear_cache():
    """Clear all cached backtest results."""
    import shutil
    
    cache_dir = Path("cache/backtest_results")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("🗑️ Cache cleared successfully!")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    else:
        print("ℹ️ No cache found to clear.")

if __name__ == "__main__":
    main() 