# Consolidated Backtesting Results System

This system provides comprehensive tracking and analysis of backtesting results with automatic logging and easy viewing capabilities.

## Features

- **Automatic Logging**: Every backtest run is automatically logged to the database
- **Consolidated Views**: View results across multiple strategies and symbols
- **Historical Tracking**: Keep track of multiple backtest runs over time
- **Performance Comparison**: Compare strategy performance side-by-side
- **Database Views**: Optimized database views for fast querying
- **Cleanup Functionality**: Automatic cleanup of old results

## Database Tables

### `backtesting_runs`
Tracks each backtest execution with metadata:
- Run timestamp, period, timeframe
- Symbols and strategies tested
- Overall performance metrics
- Total signals, P&L, duration

### `backtesting_strategy_results`
Detailed results for each strategy-symbol combination:
- Strategy name and symbol
- Signal count, P&L, win rate
- Number of trades executed
- Links to parent backtest run

### Database Views

#### `latest_backtesting_summary`
Shows summary of the most recent backtest run with aggregated metrics.

#### `strategy_performance_comparison`
Compares strategy performance from the latest run, sorted by P&L.

## Usage

### Running Backtests
All backtest runs are automatically logged when using:
```bash
python3 all_strategies_parquet.py --days 30 --timeframe 15min --no-save
```

### Viewing Results

#### Latest Results (Default)
```bash
python3 view_backtest_results.py
# or
python3 view_backtest_results.py --latest
```

#### Strategy Performance Comparison
```bash
python3 view_backtest_results.py --performance
```

#### Historical Results
```bash
python3 view_backtest_results.py --history 10  # Show last 10 runs
```

#### Comprehensive View
```bash
python3 view_backtest_results.py --all
```

#### Cleanup Old Results
```bash
python3 view_backtest_results.py --clear 50  # Keep last 50 runs
```

## Example Output

### Latest Results
```
📊 LATEST BACKTESTING SUMMARY
🕒 Run Time: 2025-07-06 13:54:31
📅 Period: 7 days
⏰ Timeframe: 15min
📈 Symbols: BANKNIFTY, NIFTY50
🧠 Strategies: insidebar_rsi
🎯 Total Signals: 7
💰 Total P&L: ₹1475.05
📊 Average Win Rate: 70.8%
⚡ Performance: 26.1 signals/second
```

### Strategy Performance
```
🎯 INSIDEBAR_RSI:
  📊 Total: 7 signals, ₹1475.05 P&L, 70.8% avg win rate
    📈 BANKNIFTY: 4 signals, ₹1274.75 P&L, 75.0% win rate
    📈 NIFTY50: 3 signals, ₹200.30 P&L, 66.7% win rate
```

## Benefits

1. **Consistent Tracking**: Never lose backtest results again
2. **Easy Comparison**: Compare different strategies and timeframes
3. **Historical Analysis**: Track performance improvements over time
4. **Quick Access**: Get latest results without re-running backtests
5. **Data Integrity**: Structured database storage with relationships
6. **Performance Monitoring**: Track system performance and execution speed

## Integration

The system is fully integrated into the existing backtesting workflow:
- No changes needed to existing strategy code
- Automatic logging happens transparently
- Results are available immediately after each backtest
- Database tables are created automatically on first use

## Database Location

Results are stored in `trading_signals.db` in the project root directory.

## Future Enhancements

- Web dashboard for viewing results
- Export functionality (CSV, JSON)
- Advanced filtering and search
- Performance trend analysis
- Strategy optimization recommendations 