#!/usr/bin/env python3
"""
Manual Backtest Runner
Quick backtest using existing parquet data
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from strategies.ema_crossover import EmaCrossover
from strategies.insidebar_rsi import InsidebarRsi  
from strategies.supertrend_ema import SupertrendEma
from strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma

def simple_backtest(strategy_class, symbol, timeframe, data_path, initial_capital=100000):
    """
    Run a simple backtest for a strategy
    """
    print(f"\n🔍 Running {strategy_class.__name__} on {symbol} ({timeframe})")
    
    try:
        # Load data
        data = pd.read_parquet(data_path)
        print(f"Data shape: {data.shape}, Date range: {data['time'].min()} to {data['time'].max()}")
        
        # Skip rows with insufficient data (first 50 rows to let EMAs stabilize)
        start_idx = 50
        if len(data) <= start_idx:
            print(f"Insufficient data: {len(data)} rows")
            return 0, 0, 0
        
        # Initialize strategy
        strategy = strategy_class()
        
        # Modify strategy thresholds for testing
        if hasattr(strategy, 'fast_ema'):
            # Lower the crossover strength threshold for testing
            original_threshold = 0.8
            test_threshold = 0.2  # Much lower threshold
            print(f"Using crossover threshold: {test_threshold} (original: {original_threshold})")
        
        # Generate signals
        signals = []
        
        try:
            if strategy_class.__name__ == "EmaCrossover":
                # Add indicators to the data
                data_with_indicators = strategy.add_indicators(data.copy())
                
                # Skip initial rows where EMAs are still stabilizing
                for i in range(start_idx, len(data_with_indicators)):
                    candle = data_with_indicators.iloc[i]
                    
                    # Skip if EMAs are zero or invalid
                    if (candle['ema_fast'] == 0 or candle['ema_slow'] == 0 or 
                        pd.isna(candle['ema_fast']) or pd.isna(candle['ema_slow'])):
                        continue
                    
                    # Manual signal generation with lower threshold
                    crossover_strength = abs(candle['crossover_strength']) if not pd.isna(candle['crossover_strength']) and candle['crossover_strength'] != float('inf') else 0
                    
                    signal = None
                    if (candle['ema_fast'] > candle['ema_slow'] and 
                        candle['close'] > candle['ema_fast'] and
                        crossover_strength > test_threshold):
                        signal = 'BUY'
                    elif (candle['ema_fast'] < candle['ema_slow'] and 
                          candle['close'] < candle['ema_fast'] and
                          crossover_strength > test_threshold):
                        signal = 'SELL'
                    
                    if signal:
                        signals.append({
                            'time': candle.name if hasattr(candle, 'name') else data_with_indicators.index[i],
                            'signal': signal,
                            'price': candle['close'],
                            'strength': crossover_strength
                        })
                        
            elif strategy_class.__name__ == "InsidebarRsi":
                # Use original analyze method for other strategies
                for i in range(start_idx, len(data)):
                    try:
                        result = strategy.analyze(data.iloc[i:i+10], i)  # Give more context rows
                        if result and result.get('signal') not in [None, 'NO TRADE', 'None']:
                            signals.append({
                                'time': data.iloc[i]['time'],
                                'signal': result['signal'],
                                'price': data.iloc[i]['close']
                            })
                    except Exception as e:
                        continue
                        
            else:  # SuperTrend strategies
                # Use original analyze method but with error handling
                for i in range(start_idx, min(len(data)-10, 1000)):  # Limit to avoid long processing
                    try:
                        candle = data.iloc[i]
                        result = strategy.analyze(candle, i, data)
                        if result and result.get('signal') not in [None, 'NO TRADE', 'None']:
                            signals.append({
                                'time': candle['time'],
                                'signal': result['signal'],
                                'price': candle['close']
                            })
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"Error generating signals: {e}")
            
        print(f"Signals generated: {len(signals)}")
        if signals:
            print(f"Sample signals: {signals[:3]}")
        
        # Simple trading simulation
        capital = initial_capital
        trades = []
        position = None
        
        for signal_data in signals:
            if position is None:  # Enter position
                position = {
                    'type': signal_data['signal'],
                    'entry_price': signal_data['price'],
                    'entry_time': signal_data['time'],
                    'quantity': int(capital / signal_data['price'])
                }
            else:  # Exit position (simplified - exit on opposite signal or after some bars)
                exit_price = signal_data['price']
                if position['type'] == 'BUY':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:  # SELL
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'success': pnl > 0
                })
                
                capital += pnl
                position = None
        
        # Calculate metrics
        total_return = ((capital - initial_capital) / initial_capital) * 100
        num_trades = len(trades)
        win_rate = (sum(1 for t in trades if t['success']) / num_trades * 100) if num_trades > 0 else 0
        
        return total_return, num_trades, win_rate
        
    except Exception as e:
        print(f"Error in backtest: {e}")
        return 0, 0, 0

def run_backtests():
    """Run backtests on all available data"""
    
    # Updated data files dictionary with correct path
    data_files = {
        'NSE_NIFTYBANK_INDEX': {
            '1min': 'data/parquet/NSE_NIFTYBANK_INDEX/1min.parquet',
            '5min': 'data/parquet/NSE_NIFTYBANK_INDEX/5min.parquet',
            '15min': 'data/parquet/NSE_NIFTYBANK_INDEX/15min.parquet',
        },
        'NSE_NIFTY50_INDEX': {
            '5min': 'data/parquet/NSE_NIFTY50_INDEX/5min.parquet',
        }
    }
    
    # Check which files actually exist
    available_files = {}
    for symbol, timeframes in data_files.items():
        available_files[symbol] = {}
        for timeframe, path in timeframes.items():
            if os.path.exists(path):
                available_files[symbol][timeframe] = path
                print(f"✅ Found: {path}")
            else:
                print(f"❌ Missing: {path}")
    
    # Strategies to test
    strategies = {
        'EMA Crossover': EmaCrossover,
        'InsideBar RSI': InsidebarRsi,
        'SuperTrend EMA': SupertrendEma,
        'SuperTrend MACD RSI EMA': SupertrendMacdRsiEma
    }
    
    results = []
    
    for symbol, timeframes in available_files.items():
        for timeframe, file_path in timeframes.items():
            print(f"\n📈 Loading data: {symbol} - {timeframe}")
            
            try:
                # Load data
                data = pd.read_parquet(file_path)
                if 'time' in data.columns:
                    data.set_index('time', inplace=True)
                
                print(f"Data loaded successfully: {data.shape} rows")
                
                for strategy_name, strategy_class in strategies.items():
                    try:
                        result = simple_backtest(strategy_class, symbol, timeframe, file_path)
                        
                        results.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'strategy': strategy_name,
                            'return': result[0],
                            'trades': result[1],
                            'win_rate': result[2]
                        })
                        
                        print(f"    🎯 {strategy_name}:")
                        print(f"      Return: {result[0]:.2f}%")
                        print(f"      Trades: {result[1]}")
                        print(f"      Win Rate: {result[2]:.1f}%")
                        
                    except Exception as e:
                        print(f"    ❌ {strategy_name}: Error - {e}")
                        
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    # Summary
    if results:
        df_results = pd.DataFrame(results)
        best_result = df_results.loc[df_results['return'].idxmax()]
        
        print(f"\n🏆 SUMMARY RESULTS:")
        print("=" * 60)
        print(f"🥇 Best Performance:")
        print(f"   Strategy: {best_result['strategy']}")
        print(f"   Symbol: {best_result['symbol']}")
        print(f"   Timeframe: {best_result['timeframe']}")
        print(f"   Return: {best_result['return']:.2f}%")
        print(f"   Win Rate: {best_result['win_rate']:.1f}%")
        
        print(f"\n📊 Average Performance by Strategy:")
        strategy_avg = df_results.groupby('strategy')['return'].mean()
        for strategy, avg_return in strategy_avg.items():
            print(f"   {strategy}: {avg_return:.1f}% (avg)")
        
        print(f"\n✅ Manual backtesting completed!")
        print(f"Total tests run: {len(results)}")
        print("=" * 60)
    else:
        print("\n❌ No successful backtests completed.")

if __name__ == "__main__":
    run_backtests() 