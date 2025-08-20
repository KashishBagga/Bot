#!/usr/bin/env python3
"""
All Strategies - Parquet Only Version
Run all trading strategies using only parquet data (no API calls)
Designed for consistent backtesting with 20-year historical data
"""

import sys
import argparse
import time
import concurrent.futures
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import os
import pandas as pd # Added pandas import for pd.concat

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.parquet_data_store import ParquetDataStore
from src.models.backtesting_summary import BacktestingSummary
from dotenv import load_dotenv
import src.warning_filters  # noqa: F401

# Import strategy modules
def get_available_strategies():
    """Get all available trading strategies"""
    strategies_dir = Path(__file__).parent / "src" / "strategies"
    if not strategies_dir.exists():
        return []
    
    strategies = []
    for file in strategies_dir.glob("*.py"):
        if file.name != "__init__.py":
            strategy_name = file.stem
            strategies.append(strategy_name)
    
    return sorted(strategies)

def add_technical_indicators(df):
    """Add comprehensive technical indicators to dataframe"""
    if len(df) < 50:  # Need minimum data for indicators
        return df
    
    try:
        # EMA calculations (if not already present)
        if 'ema_9' not in df.columns:
            df['ema_9'] = df['close'].ewm(span=9).mean()
        if 'ema_21' not in df.columns:
            df['ema_21'] = df['close'].ewm(span=21).mean()
        if 'ema_50' not in df.columns:
            df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # RSI calculation (if not already present)
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD calculation (if not already present)
        if 'macd' not in df.columns:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands (if not already present)
        if 'bb_upper' not in df.columns:
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_middle'] = sma_20
        
        # ATR (if not already present)
        if 'atr' not in df.columns:
            prev_close = df['close'].shift(1)
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - prev_close).abs()
            tr3 = (df['low'] - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Additional indicators for comprehensive analysis
        # SMA calculations
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position indicators
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, pd.NA)
        df['candle_size'] = (df['high'] - df['low']) / df['close']
        
    except Exception as e:
        print(f"⚠️ Error adding indicators: {e}")
    
    return df

def run_strategy(strategy_name, dataframes, multi_timeframe_dataframes, save_to_db):
    """Run a single strategy on all provided dataframes"""
    try:
        # Import strategy module
        strategy_module = __import__(f"src.strategies.{strategy_name}", fromlist=[strategy_name])
        
        # Get strategy class - convert snake_case to CamelCase
        class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
        
        # Check if strategy class exists
        if not hasattr(strategy_module, class_name):
            return {"error": f"Strategy class {class_name} not found in {strategy_name}"}
        
        strategy_class = getattr(strategy_module, class_name)
        
        results = {}
        
        # Run strategy on each symbol
        for index_name, df in dataframes.items():
            if df.empty:
                continue
            
            try:
                # Add indicators if not present
                df_with_indicators = add_technical_indicators(df.copy())
                
                # Get multi-timeframe data for this symbol
                multi_tf_data = multi_timeframe_dataframes.get(index_name, {})
                
                # Instantiate strategy with multi-timeframe data
                strategy_instance = strategy_class(timeframe_data=multi_tf_data)
                
                # Add strategy-specific indicators
                df_with_indicators = strategy_instance.add_indicators(df_with_indicators)
                
                # Initialize results for this symbol
                symbol_results = {
                    'signals': {'BUY CALL': 0, 'BUY PUT': 0, 'BUY': 0, 'SELL': 0, 'NO TRADE': 0},
                    'total_profit_loss': 0.0,
                    'trades': [],
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_signals': 0
                }
                
                # Strategy-specific execution logic
                if strategy_name == 'insidebar_rsi':
                    # InsidebarRsi expects: analyze(data, index_name, future_data)
                    # We need to pass the full dataframe for each analysis
                    for i in range(50, len(df_with_indicators)):  # Skip first 50 for indicator warmup
                        try:
                            # Get data up to current point
                            current_data = df_with_indicators.iloc[:i+1]
                            
                            # Get future data for performance calculation
                            future_data = df_with_indicators.iloc[i+1:i+51] if i+1 < len(df_with_indicators) else None
                            
                            # Call strategy analyze method
                            result = strategy_instance.analyze(current_data, index_name, future_data)
                            
                            if result and isinstance(result, dict):
                                signal = result.get('signal', 'NO TRADE')
                                symbol_results['signals'][signal] = symbol_results['signals'].get(signal, 0) + 1
                                
                                # Track all signals (including NO TRADE for analysis)
                                symbol_results['total_signals'] += 1
                                
                                # Track actual trades (non-NO TRADE signals)
                                if signal not in ['NO TRADE', 'None', None]:
                                    pnl = result.get('pnl', 0.0)
                                    symbol_results['total_profit_loss'] += pnl
                                    
                                    trade_info = {
                                        'timestamp': df_with_indicators.index[i],
                                        'signal': signal,
                                        'price': result.get('price', 0),
                                        'profit_loss': pnl,
                                        'confidence': result.get('confidence', 'Unknown'),
                                        'outcome': result.get('outcome', 'Pending')
                                    }
                                    symbol_results['trades'].append(trade_info)
                                    
                                    if pnl > 0:
                                        symbol_results['winning_trades'] += 1
                                    symbol_results['total_trades'] += 1
                                    
                        except Exception as e:
                            print(f"⚠️ Error processing row {i} for {strategy_name} on {index_name}: {e}")
                            continue
                
                elif strategy_name in ['ema_crossover', 'supertrend_ema', 'supertrend_macd_rsi_ema']:
                    # These strategies expect: analyze(candle, index, df, future_data)
                    for i in range(50, len(df_with_indicators)):  # Skip first 50 for indicator warmup
                        try:
                            candle = df_with_indicators.iloc[i]
                            
                            # Get future data for performance calculation
                            future_data = df_with_indicators.iloc[i+1:i+51] if i+1 < len(df_with_indicators) else None
                            
                            # Call strategy analyze method
                            result = strategy_instance.analyze(candle, i, df_with_indicators, future_data)
                            
                            if result and isinstance(result, dict):
                                signal = result.get('signal', 'NO TRADE')
                                symbol_results['signals'][signal] = symbol_results['signals'].get(signal, 0) + 1
                                
                                # Track all signals (including NO TRADE for analysis)
                                symbol_results['total_signals'] += 1
                                
                                # Track actual trades (non-NO TRADE signals)
                                if signal not in ['NO TRADE', 'None', None]:
                                    pnl = result.get('pnl', 0.0)
                                    symbol_results['total_profit_loss'] += pnl
                                    
                                    trade_info = {
                                        'timestamp': df_with_indicators.index[i],
                                        'signal': signal,
                                        'price': result.get('price', 0),
                                        'profit_loss': pnl,
                                        'confidence': result.get('confidence', 'Unknown'),
                                        'outcome': result.get('outcome', 'Pending')
                                    }
                                    symbol_results['trades'].append(trade_info)
                                    
                                    if pnl > 0:
                                        symbol_results['winning_trades'] += 1
                                    symbol_results['total_trades'] += 1
                                    
                        except Exception as e:
                            print(f"⚠️ Error processing row {i} for {strategy_name} on {index_name}: {e}")
                            continue
                
                else:
                    # Generic strategy execution - try to determine method signature
                    import inspect
                    analyze_signature = inspect.signature(strategy_instance.analyze)
                    params = list(analyze_signature.parameters.keys())
                    
                    if 'data' in params and 'index_name' in params:
                        # Strategy expects full dataframe like insidebar_rsi
                        for i in range(50, len(df_with_indicators)):
                            try:
                                current_data = df_with_indicators.iloc[:i+1]
                                future_data = df_with_indicators.iloc[i+1:i+51] if i+1 < len(df_with_indicators) else None
                                
                                result = strategy_instance.analyze(current_data, index_name, future_data)
                                
                                if result and isinstance(result, dict):
                                    signal = result.get('signal', 'NO TRADE')
                                    symbol_results['signals'][signal] = symbol_results['signals'].get(signal, 0) + 1
                                    symbol_results['total_signals'] += 1
                                    
                                    if signal not in ['NO TRADE', 'None', None]:
                                        pnl = result.get('pnl', 0.0)
                                        symbol_results['total_profit_loss'] += pnl
                                        
                                        trade_info = {
                                            'timestamp': df_with_indicators.index[i],
                                            'signal': signal,
                                            'price': result.get('price', 0),
                                            'profit_loss': pnl,
                                            'confidence': result.get('confidence', 'Unknown'),
                                            'outcome': result.get('outcome', 'Pending')
                                        }
                                        symbol_results['trades'].append(trade_info)
                                        
                                        if pnl > 0:
                                            symbol_results['winning_trades'] += 1
                                        symbol_results['total_trades'] += 1
                                        
                            except Exception as e:
                                print(f"⚠️ Error processing row {i} for {strategy_name} on {index_name}: {e}")
                                continue
                    
                    elif 'candle' in params and 'index' in params:
                        # Strategy expects candle-by-candle analysis
                        for i in range(50, len(df_with_indicators)):
                            try:
                                candle = df_with_indicators.iloc[i]
                                future_data = df_with_indicators.iloc[i+1:i+51] if i+1 < len(df_with_indicators) else None
                                
                                result = strategy_instance.analyze(candle, i, df_with_indicators, future_data)
                                
                                if result and isinstance(result, dict):
                                    signal = result.get('signal', 'NO TRADE')
                                    symbol_results['signals'][signal] = symbol_results['signals'].get(signal, 0) + 1
                                    symbol_results['total_signals'] += 1
                                    
                                    if signal not in ['NO TRADE', 'None', None]:
                                        pnl = result.get('pnl', 0.0)
                                        symbol_results['total_profit_loss'] += pnl
                                        
                                        trade_info = {
                                            'timestamp': df_with_indicators.index[i],
                                            'signal': signal,
                                            'price': result.get('price', 0),
                                            'profit_loss': pnl,
                                            'confidence': result.get('confidence', 'Unknown'),
                                            'outcome': result.get('outcome', 'Pending')
                                        }
                                        symbol_results['trades'].append(trade_info)
                                        
                                        if pnl > 0:
                                            symbol_results['winning_trades'] += 1
                                        symbol_results['total_trades'] += 1
                                        
                            except Exception as e:
                                print(f"⚠️ Error processing row {i} for {strategy_name} on {index_name}: {e}")
                                continue
                    
                    else:
                        # Fallback - try single analysis on full dataframe
                        try:
                            result = strategy_instance.analyze(df_with_indicators, index_name, None)
                            
                            if result and isinstance(result, dict):
                                signal = result.get('signal', 'NO TRADE')
                                symbol_results['signals'][signal] = symbol_results['signals'].get(signal, 0) + 1
                                symbol_results['total_signals'] += 1
                                
                                if signal not in ['NO TRADE', 'None', None]:
                                    pnl = result.get('pnl', 0.0)
                                    symbol_results['total_profit_loss'] += pnl
                                    
                                    trade_info = {
                                        'timestamp': df_with_indicators.index[-1],
                                        'signal': signal,
                                        'price': result.get('price', 0),
                                        'profit_loss': pnl,
                                        'confidence': result.get('confidence', 'Unknown'),
                                        'outcome': result.get('outcome', 'Pending')
                                    }
                                    symbol_results['trades'].append(trade_info)
                                    
                                    if pnl > 0:
                                        symbol_results['winning_trades'] += 1
                                    symbol_results['total_trades'] += 1
                                    
                        except Exception as e:
                            print(f"⚠️ Error in fallback analysis for {strategy_name} on {index_name}: {e}")
                
                # Calculate win rate
                if symbol_results['total_trades'] > 0:
                    symbol_results['win_rate'] = (symbol_results['winning_trades'] / symbol_results['total_trades']) * 100
                
                results[index_name] = symbol_results
                
                # Print progress for this symbol
                # total_signals = sum(count for signal, count in symbol_results['signals'].items() if signal != 'NO TRADE')
                # print(f"  📊 {index_name}: {total_signals} signals from {symbol_results['total_signals']} analyses")
                
                # Save to database if requested
                if save_to_db:
                    save_strategy_results_to_db(strategy_name, index_name, symbol_results)
                
            except Exception as e:
                print(f"❌ Error running {strategy_name} on {index_name}: {e}")
                results[index_name] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ Error importing/running strategy {strategy_name}: {e}")
        return {"error": str(e)}

def save_strategy_results_to_db(strategy_name, symbol, results):
    """Save strategy results to database"""
    try:
        import sqlite3
        conn = sqlite3.connect('trading_signals.db')
        
        # Create table if not exists
        table_name = strategy_name.lower()
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                signal TEXT,
                confidence_score REAL,
                price REAL,
                reasoning TEXT,
                profit_loss REAL
            )
        ''')
        
        # Insert trade records
        for trade in results.get('trades', []):
            conn.execute(f'''
                INSERT INTO {table_name} 
                (timestamp, symbol, signal, confidence_score, price, reasoning, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'].isoformat(),
                symbol,
                trade['signal'],
                80.0,  # Default confidence
                trade['price'],
                f"Parquet backtest - {strategy_name}",
                trade['profit_loss']
            ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"⚠️ Error saving to database: {e}")

def print_summary(results, duration, days_back=None, timeframe=None, symbols_tested=None, strategies_tested=None):
    """Print comprehensive summary of backtest results"""
    print(f"\n{'='*80}")
    print(f"📊 PARQUET BACKTESTING SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"🧠 Strategies tested: {len(results)}")
    
    total_signals = 0
    total_profit_loss = 0.0
    successful_strategies = 0
    strategy_results_data = []
    
    for strategy_name, strategy_results in results.items():
        if 'error' in strategy_results:
            print(f"❌ {strategy_name}: ERROR - {strategy_results['error']}")
            continue
        
        successful_strategies += 1
        strategy_signals = 0
        strategy_profit = 0.0
        strategy_trades = 0
        strategy_win_rate = 0.0
        total_analyses = 0
        
        print(f"\n🎯 {strategy_name.upper()}:")
        
        for symbol, stats in strategy_results.items():
            if isinstance(stats, dict) and 'signals' in stats:
                signals_dict = stats.get('signals', {})
                
                # Count only actual trading signals (exclude NO TRADE)
                symbol_signals = sum(count for signal, count in signals_dict.items() if signal not in ['NO TRADE', 'None', None])
                
                # Get other stats
                symbol_profit = stats.get('total_profit_loss', 0.0)
                symbol_trades = stats.get('total_trades', 0)
                symbol_win_rate = stats.get('win_rate', 0.0)
                symbol_analyses = stats.get('total_signals', 0)
                
                strategy_signals += symbol_signals
                strategy_profit += symbol_profit
                strategy_trades += symbol_trades
                total_analyses += symbol_analyses
                
                # Print detailed breakdown for each symbol
                no_trade_count = signals_dict.get('NO TRADE', 0)
                buy_call_count = signals_dict.get('BUY CALL', 0)
                buy_put_count = signals_dict.get('BUY PUT', 0)
                buy_count = signals_dict.get('BUY', 0)
                sell_count = signals_dict.get('SELL', 0)
                
                # print(f"  📈 {symbol}: {symbol_signals} signals from {symbol_analyses} analyses")
                print(f"      🔍 Breakdown: BUY CALL({buy_call_count}), BUY PUT({buy_put_count}), BUY({buy_count}), SELL({sell_count}), NO TRADE({no_trade_count})")
                if symbol_signals > 0:
                    print(f"      💰 P&L: ₹{symbol_profit:.2f}, Win Rate: {symbol_win_rate:.1f}% ({symbol_trades} trades)")
                
                # Store data for database logging
                strategy_results_data.append({
                    'strategy_name': strategy_name,
                    'symbol': symbol,
                    'signals_count': symbol_signals,
                    'pnl': symbol_profit,
                    'win_rate': symbol_win_rate,
                    'total_trades': symbol_trades,
                    'total_analyses': symbol_analyses
                })
        
        if strategy_trades > 0:
            strategy_win_rate = sum(
                stats.get('winning_trades', 0) for stats in strategy_results.values() 
                if isinstance(stats, dict)
            ) / strategy_trades * 100
        
        total_signals += strategy_signals
        total_profit_loss += strategy_profit
        
        # print(f"  📊 Strategy Total: {strategy_signals} signals from {total_analyses} analyses, ₹{strategy_profit:.2f} P&L, {strategy_win_rate:.1f}% win rate")
    
    print(f"\n🎉 OVERALL RESULTS:")
    print(f"  ✅ Successful strategies: {successful_strategies}/{len(results)}")
    print(f"  📈 Total signals generated: {total_signals:,}")
    print(f"  💰 Total P&L: ₹{total_profit_loss:.2f}")
    print(f"  ⚡ Performance: {total_signals/duration:.0f} signals/second")
    
    # Additional debugging info
    if total_signals == 0:
        print(f"\n⚠️  DEBUG INFO:")
        print(f"  🔍 No signals generated - this indicates a problem with:")
        print(f"      • Strategy logic conditions may be too restrictive")
        print(f"      • Indicator calculations may be incorrect")
        print(f"      • Data quality issues (missing indicators, wrong timeframes)")
        print(f"      • Strategy parameter mismatches")
        print(f"  💡 Recommendation: Check individual strategy outputs and reduce signal thresholds")
    
    print(f"{'='*80}")
    
    # Log to backtesting summary database
    if days_back and timeframe and symbols_tested and strategies_tested:
        try:
            summary = BacktestingSummary()
            backtest_data = {
                'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'period_days': days_back,
                'timeframe': timeframe,
                'symbols': symbols_tested,
                'strategies': strategies_tested,
                'total_signals': total_signals,
                'total_pnl': total_profit_loss,
                'duration_seconds': duration,
                'performance_rate': total_signals/duration if duration > 0 else 0,
                'strategy_results': strategy_results_data
            }
            
            run_id = summary.log_backtest_run(backtest_data)
            print(f"📊 Backtest results logged to database (Run ID: {run_id})")
        except Exception as e:
            print(f"⚠️ Failed to log backtest results: {e}")
    
    print(f"\n🎉 Parquet backtesting completed!")
    print(f"📊 All data sourced from local parquet files (no API calls)")
    if total_signals > 0:
        print(f"🚀 Signal generation working correctly!")
    else:
        print(f"⚠️ Signal generation needs investigation - check strategy logic")

def run_all_strategies_parquet(days_back=30, timeframe="15min", save_to_db=True, 
                              symbols=None, strategies=None, parallel=True):
    """Run all strategies using parquet data only"""
    load_dotenv()
    
    # Initialize parquet data store
    data_store = ParquetDataStore()
    
    # Get available symbols
    available_symbols = data_store.get_available_symbols()
    if not available_symbols:
        print("❌ No parquet data found. Run setup_20_year_parquet_data.py first.")
        return False
    
    # Process symbols with better mapping
    if symbols:
        symbols_to_test = []
        for symbol in symbols:
            symbol_upper = symbol.upper()
            
            # Direct mapping for common symbols
            if symbol_upper == 'NIFTY50':
                # Check both possible formats
                candidates = ['NSE:NIFTY50-INDEX', 'NSE_NIFTY50_INDEX', 'NIFTY50']
                for candidate in candidates:
                    if candidate in available_symbols:
                        symbols_to_test.append(candidate)
                        break
            elif symbol_upper == 'BANKNIFTY':
                # Check both possible formats
                candidates = ['NSE:NIFTYBANK-INDEX', 'NSE_NIFTYBANK_INDEX', 'BANKNIFTY', 'NIFTYBANK']
                for candidate in candidates:
                    if candidate in available_symbols:
                        symbols_to_test.append(candidate)
                        break
            else:
                # Try to find matching symbols
                matching_symbols = [s for s in available_symbols if symbol_upper in s.upper()]
                if matching_symbols:
                    symbols_to_test.append(matching_symbols[0])
        
        symbols_to_test = list(set(symbols_to_test))  # Remove duplicates
    else:
        symbols_to_test = available_symbols
    
    if not symbols_to_test:
        print("❌ No valid symbols found in parquet data")
        print(f"Available symbols: {available_symbols}")
        return False
    
    # Get available strategies
    all_strategies = get_available_strategies()
    if strategies:
        strategies_to_test = [s for s in strategies if s in all_strategies]
        if not strategies_to_test:
            print(f"❌ No valid strategies found. Available: {all_strategies}")
            return False
    else:
        strategies_to_test = all_strategies
    
    # Validate timeframe
    sample_symbol = symbols_to_test[0]
    available_timeframes = data_store.get_available_timeframes(sample_symbol)
    if timeframe not in available_timeframes:
        print(f"❌ Timeframe '{timeframe}' not available for {sample_symbol}")
        print(f"Available timeframes: {available_timeframes}")
        return False
    
    print(f"🚀 Running Parquet-Only Backtesting:")
    print(f"  📅 Period: Last {days_back} days")
    print(f"  ⏰ Timeframe: {timeframe}")
    print(f"  💿 Save to DB: {save_to_db}")
    print(f"  📈 Symbols: {len(symbols_to_test)} ({', '.join([get_display_name(s) for s in symbols_to_test])})")
    print(f"  🧠 Strategies: {len(strategies_to_test)} ({', '.join(strategies_to_test)})")
    print(f"  ⚡ Execution: {'Parallel' if parallel else 'Sequential'}")
    
    start_time = time.time()
    
    # Load data for all symbols
    print(f"\n📊 Loading {timeframe} parquet data...")
    dataframes = {}
    multi_timeframe_dataframes = {}
    
    for symbol in symbols_to_test:
        # Load primary timeframe data
        df = data_store.load_data(symbol, timeframe, days_back)
        if df.empty:
            print(f"⚠️ No data for {symbol} at {timeframe}")
            continue
        
        # Create display name
        display_name = get_display_name(symbol)
        
        dataframes[display_name] = df
        
        # Load multi-timeframe data
        all_timeframes = data_store.get_available_timeframes(symbol)
        multi_tf_data = {}
        for tf in all_timeframes:
            tf_df = data_store.load_data(symbol, tf, days_back)
            if not tf_df.empty:
                multi_tf_data[tf] = tf_df
        
        multi_timeframe_dataframes[display_name] = multi_tf_data
        
        print(f"  ✅ {display_name}: {len(df)} candles ({timeframe})")
    
    if not dataframes:
        print("❌ No data loaded successfully")
        return False
    
    # Run strategies
    print(f"\n🧠 Running {len(strategies_to_test)} strategies...")
    results = {}
    
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_strategy, name, dataframes, multi_timeframe_dataframes, save_to_db): name
                for name in strategies_to_test
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    print(f"  ✅ {name} completed")
                except Exception as e:
                    print(f"  ❌ {name} failed: {e}")
                    results[name] = {"error": str(e)}
    else:
        for name in strategies_to_test:
            try:
                results[name] = run_strategy(name, dataframes, multi_timeframe_dataframes, save_to_db)
                print(f"  ✅ {name} completed")
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                results[name] = {"error": str(e)}
    
    duration = time.time() - start_time
    print_summary(results, duration, days_back, timeframe, symbols_to_test, strategies_to_test)
    
    return True

def get_display_name(symbol: str) -> str:
    """Convert symbol to display name"""
    if 'NIFTY50' in symbol:
        return 'NIFTY50'
    elif 'NIFTYBANK' in symbol:
        return 'BANKNIFTY'
    elif 'NIFTYFIN' in symbol:
        return 'NIFTYFIN'
    elif 'NIFTYIT' in symbol:
        return 'NIFTYIT'
    elif 'NIFTYPHARMA' in symbol:
        return 'NIFTYPHARMA'
    elif 'NIFTYMETAL' in symbol:
        return 'NIFTYMETAL'
    elif 'NIFTYAUTO' in symbol:
        return 'NIFTYAUTO'
    elif 'NIFTYREALTY' in symbol:
        return 'NIFTYREALTY'
    elif 'NIFTYFMCG' in symbol:
        return 'NIFTYFMCG'
    elif 'NIFTYENERGY' in symbol:
        return 'NIFTYENERGY'
    else:
        # Extract symbol name from NSE format or use as-is
        if ':' in symbol:
            return symbol.split(':')[1].replace('-INDEX', '').replace('-EQ', '')
        elif '_' in symbol:
            return symbol.replace('_', '').replace('NSE', '').replace('INDEX', '')
        else:
            return symbol

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run all trading strategies using parquet data only')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--timeframe', type=str, default='15min', help='Timeframe (1min, 5min, 15min, 30min, 1hour, 4hour, 1day)')
    parser.add_argument('--no-save', action='store_true', help="Don't save results to database")
    parser.add_argument('--symbols', type=str, help='Symbols to test, comma separated (default: all available)')
    parser.add_argument('--strategies', type=str, help='Strategies to test, comma separated (default: all available)')
    parser.add_argument('--sequential', action='store_true', help='Run strategies sequentially instead of parallel')
    
    args = parser.parse_args()
    
    # Process symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Process strategies if provided
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]
    
    # Run backtesting
    success = run_all_strategies_parquet(
        days_back=args.days,
        timeframe=args.timeframe,
        save_to_db=not args.no_save,
        symbols=symbols,
        strategies=strategies,
        parallel=not args.sequential
    )
    
    if success:
        print(f"\n🎉 Parquet backtesting completed successfully!")
        print(f"📊 All data sourced from local parquet files (no API calls)")
        print(f"🚀 Ready for production backtesting with 20-year data!")
    else:
        print(f"\n❌ Backtesting failed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 