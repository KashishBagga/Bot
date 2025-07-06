#!/usr/bin/env python3
"""
View Backtesting Results
Display consolidated backtesting results from the database
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.backtesting_summary import BacktestingSummary

def view_latest_results():
    """View the latest backtesting results"""
    summary = BacktestingSummary()
    summary.print_latest_summary()

def view_historical_results(limit=10):
    """View historical backtesting results"""
    summary = BacktestingSummary()
    runs = summary.get_historical_runs(limit)
    
    if not runs:
        print("❌ No historical backtest data found")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 HISTORICAL BACKTESTING RESULTS (Last {len(runs)} runs)")
    print(f"{'='*80}")
    
    for i, run in enumerate(runs, 1):
        print(f"\n🔢 Run #{i}:")
        print(f"  🕒 Time: {run['run_timestamp']}")
        print(f"  📅 Period: {run['period_days']} days")
        print(f"  ⏰ Timeframe: {run['timeframe']}")
        print(f"  📈 Symbols: {', '.join(run['symbols'])}")
        print(f"  🧠 Strategies: {', '.join(run['strategies'])}")
        print(f"  🎯 Signals: {run['total_signals']}")
        print(f"  💰 P&L: ₹{run['total_pnl']:.2f}")
        print(f"  ⚡ Performance: {run['performance_rate']:.1f} signals/second")
        print(f"  ⏱️ Duration: {run['duration_seconds']:.2f} seconds")

def view_strategy_performance():
    """View strategy performance comparison"""
    summary = BacktestingSummary()
    performance = summary.get_strategy_performance()
    
    if not performance:
        print("❌ No strategy performance data found")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 STRATEGY PERFORMANCE COMPARISON (Latest Run)")
    print(f"{'='*80}")
    
    # Group by strategy
    strategy_totals = {}
    for data in performance:
        strategy = data['strategy_name']
        if strategy not in strategy_totals:
            strategy_totals[strategy] = {
                'total_signals': 0,
                'total_pnl': 0.0,
                'total_trades': 0,
                'symbols': []
            }
        
        strategy_totals[strategy]['total_signals'] += data['signals_count']
        strategy_totals[strategy]['total_pnl'] += data['pnl']
        strategy_totals[strategy]['total_trades'] += data['total_trades']
        strategy_totals[strategy]['symbols'].append({
            'symbol': data['symbol'],
            'signals': data['signals_count'],
            'pnl': data['pnl'],
            'win_rate': data['win_rate']
        })
    
    # Sort by total P&L
    sorted_strategies = sorted(strategy_totals.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
    
    for strategy, totals in sorted_strategies:
        avg_win_rate = sum(s['win_rate'] for s in totals['symbols']) / len(totals['symbols']) if totals['symbols'] else 0
        
        print(f"\n🎯 {strategy.upper()}:")
        print(f"  📊 Total: {totals['total_signals']} signals, ₹{totals['total_pnl']:.2f} P&L, {avg_win_rate:.1f}% avg win rate")
        
        for symbol_data in totals['symbols']:
            if symbol_data['signals'] > 0:
                print(f"    📈 {symbol_data['symbol']}: {symbol_data['signals']} signals, "
                      f"₹{symbol_data['pnl']:.2f} P&L, {symbol_data['win_rate']:.1f}% win rate")

def clear_old_results(keep_last_n=50):
    """Clear old backtest results"""
    summary = BacktestingSummary()
    summary.clear_old_runs(keep_last_n)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='View consolidated backtesting results')
    parser.add_argument('--latest', action='store_true', help='View latest backtesting results')
    parser.add_argument('--history', type=int, default=0, help='View historical results (specify number of runs)')
    parser.add_argument('--performance', action='store_true', help='View strategy performance comparison')
    parser.add_argument('--clear', type=int, default=0, help='Clear old results, keep last N runs')
    parser.add_argument('--all', action='store_true', help='View all available information')
    
    args = parser.parse_args()
    
    if args.clear > 0:
        clear_old_results(args.clear)
        return
    
    if args.all:
        view_latest_results()
        view_strategy_performance()
        view_historical_results(10)
    elif args.latest:
        view_latest_results()
    elif args.history > 0:
        view_historical_results(args.history)
    elif args.performance:
        view_strategy_performance()
    else:
        # Default: show latest results
        view_latest_results()

if __name__ == "__main__":
    main() 