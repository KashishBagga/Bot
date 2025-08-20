# Live Trading System Documentation

## Overview

This live trading system provides automated trading capabilities with the same strategies used in backtesting, ensuring consistency between historical analysis and live execution. The system includes automated scheduling, risk management, performance tracking, and comprehensive reporting.

## Features

### 🤖 Automated Trading Bot
- **Consistent Strategy Implementation**: Uses the same strategies as backtesting system
- **Real-time Signal Generation**: Processes market data and generates trading signals
- **Risk Management**: Built-in position sizing and loss limits
- **Performance Tracking**: Comprehensive metrics and statistics
- **Database Integration**: Automatic logging of all signals and trades

### ⏰ Automated Scheduling
- **Market Hours Operation**: Automatically starts at 9:00 AM and stops at 3:30 PM
- **Weekday Only**: Operates Monday through Friday only
- **Health Monitoring**: Checks bot status every 30 minutes
- **Auto-restart**: Automatically restarts bot if it crashes
- **Daily Reports**: Generates summary reports at 4:00 PM

### 📊 Comprehensive Reporting
- **Daily Summaries**: Complete trading performance for each day
- **Weekly Analysis**: Multi-day performance tracking
- **Strategy Performance**: Individual strategy metrics and comparison
- **Signal Analysis**: Detailed signal generation and execution data

## System Architecture

### Database Tables

#### `live_signals`
Stores all generated trading signals:
- `id`: Unique identifier
- `timestamp`: Signal generation time
- `strategy`: Strategy name
- `symbol`: Trading symbol (BANKNIFTY/NIFTY50)
- `signal`: Signal type (BUY CALL/BUY PUT)
- `confidence_score`: Signal confidence (0-100)
- `price`: Current market price
- `target`: Target price
- `stop_loss`: Stop loss price
- `status`: Signal status (GENERATED/EXECUTED/REJECTED)
- `created_at`: Database timestamp

#### `live_trade_executions`
Tracks actual trade executions:
- `id`: Unique identifier
- `signal_id`: Reference to live_signals
- `entry_price`: Trade entry price
- `exit_price`: Trade exit price
- `quantity`: Trade quantity
- `pnl`: Profit/Loss amount
- `status`: Trade status (OPEN/CLOSED)
- `exit_reason`: Reason for exit (TARGET/STOP_LOSS/TIME)
- `created_at`: Entry timestamp
- `updated_at`: Exit timestamp

#### `daily_trading_summary`
Daily performance summaries:
- `date`: Trading date
- `market_start_time`: Session start time
- `market_end_time`: Session end time
- `session_duration_minutes`: Trading session duration
- `signals_generated`: Total signals generated
- `trades_taken`: Total trades executed
- `profitable_trades`: Number of profitable trades
- `total_pnl`: Total profit/loss
- `win_rate`: Win rate percentage
- `strategies_active`: List of active strategies (JSON)
- `created_at`: Summary creation time

## Usage

### Starting the System

#### Manual Start
```bash
# Start the live trading bot directly
python3 live_trading_bot.py

# Start with automated scheduling
python3 start_trading_bot.py
```

#### Automated Start (Recommended)
```bash
# Run the scheduler (handles all automation)
python3 start_trading_bot.py
```

### Viewing Results

#### Today's Summary
```bash
# View today's trading summary
python3 view_daily_trading_summary.py --today

# View specific date
python3 view_daily_trading_summary.py --date 2024-01-15
```

#### Weekly Analysis
```bash
# View last week's performance
python3 view_daily_trading_summary.py --weekly 1

# View last 2 weeks
python3 view_daily_trading_summary.py --weekly 2
```

#### Strategy Performance
```bash
# View strategy performance for last 7 days
python3 view_daily_trading_summary.py --strategy 7

# View strategy performance for last 30 days
python3 view_daily_trading_summary.py --strategy 30
```

#### Recent Signals
```bash
# View last 20 signals
python3 view_daily_trading_summary.py --signals 20

# View last 50 signals
python3 view_daily_trading_summary.py --signals 50
```

#### Complete Overview
```bash
# View all information at once
python3 view_daily_trading_summary.py --all
```

## Example Outputs

### Daily Summary
```
================================================================================
📊 DAILY TRADING SUMMARY - 2024-01-15
================================================================================
🕘 Session: 09:00:00 - 15:30:00
⏱️ Duration: 390 minutes
🎯 Signals Generated: 23
💼 Trades Taken: 15
✅ Profitable Trades: 9
💰 Total P&L: ₹2,450.75
📈 Win Rate: 60.0%
🧠 Active Strategies: insidebar_rsi, supertrend_ema, ema_crossover, supertrend_macd_rsi_ema

📊 Signal Breakdown:
  🎯 insidebar_rsi: BUY CALL (Confidence: 75, Price: ₹45,234.50, Count: 3)
  🎯 supertrend_ema: BUY PUT (Confidence: 68, Price: ₹19,876.25, Count: 2)
  🎯 ema_crossover: BUY CALL (Confidence: 82, Price: ₹45,456.75, Count: 1)

💼 Trade Executions:
  ✅ insidebar_rsi - BANKNIFTY: BUY CALL | Entry: ₹45,234.50 | P&L: ₹340.25 | Status: CLOSED
  ❌ supertrend_ema - NIFTY50: BUY PUT | Entry: ₹19,876.25 | P&L: ₹-125.50 | Status: CLOSED
================================================================================
```

### Weekly Summary
```
================================================================================
📊 WEEKLY TRADING SUMMARY (Last 1 week(s))
================================================================================
📊 OVERALL PERFORMANCE:
  🎯 Total Signals: 127
  💼 Total Trades: 89
  ✅ Profitable Trades: 52
  💰 Total P&L: ₹8,945.30
  📈 Average Win Rate: 58.4%
  ⏱️ Total Trading Time: 1950 minutes

📅 DAILY BREAKDOWN:
  🟢 2024-01-15: 23 signals, 15 trades, ₹2,450.75 P&L, 60.0% win rate
  🟢 2024-01-14: 19 signals, 12 trades, ₹1,875.25 P&L, 58.3% win rate
  🔴 2024-01-13: 25 signals, 18 trades, ₹-345.80 P&L, 44.4% win rate
  🟢 2024-01-12: 31 signals, 22 trades, ₹3,125.45 P&L, 63.6% win rate
  🟢 2024-01-11: 29 signals, 22 trades, ₹1,839.65 P&L, 59.1% win rate
================================================================================
```

### Strategy Performance
```
================================================================================
📊 STRATEGY PERFORMANCE ANALYSIS (Last 7 days)
================================================================================
🟢 INSIDEBAR_RSI:
  📊 Signals: 45
  💼 Executed: 32
  ✅ Profitable: 19
  💰 P&L: ₹3,245.75
  📈 Win Rate: 59.4%
  🎯 Avg Confidence: 72.3

🟢 SUPERTREND_EMA:
  📊 Signals: 38
  💼 Executed: 28
  ✅ Profitable: 16
  💰 P&L: ₹2,134.50
  📈 Win Rate: 57.1%
  🎯 Avg Confidence: 68.9

🟢 EMA_CROSSOVER:
  📊 Signals: 25
  💼 Executed: 18
  ✅ Profitable: 11
  💰 P&L: ₹1,876.25
  📈 Win Rate: 61.1%
  🎯 Avg Confidence: 75.2

🟢 SUPERTREND_MACD_RSI_EMA:
  📊 Signals: 19
  💼 Executed: 11
  ✅ Profitable: 6
  💰 P&L: ₹1,688.80
  📈 Win Rate: 54.5%
  🎯 Avg Confidence: 71.8
================================================================================
```

## Configuration

### Risk Management Settings
```python
# In live_trading_bot.py
self.min_confidence_score = 60  # Minimum confidence for trade execution
self.max_daily_loss = 5000      # Maximum daily loss limit (₹)
self.max_position_size = 1      # Maximum position size per trade
```

### Strategy Configuration
```python
# Active strategies
self.strategies = {
    'insidebar_rsi': InsidebarRsi(),
    'ema_crossover': EmaCrossover(),
    'supertrend_ema': SupertrendEma(),
    'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
}
```

### Market Hours
```python
# In start_trading_bot.py
market_start = 9:00 AM
market_end = 3:30 PM
trading_days = Monday to Friday
```

## Monitoring and Alerts

### Log Files
- `logs/live_trading_bot.log`: Main bot operations
- `logs/scheduler.log`: Scheduler operations
- `logs/signals.log`: Signal generation details
- `logs/trades.log`: Trade execution details

### Health Checks
- Bot status monitoring every 30 minutes
- Automatic restart on crashes
- Market hours validation
- Database connectivity checks

## Benefits

### ✅ Consistency
- Same strategies as backtesting
- Identical signal generation logic
- Consistent risk management rules

### ✅ Automation
- No manual intervention required
- Automatic start/stop based on market hours
- Self-healing system with auto-restart

### ✅ Comprehensive Tracking
- Every signal and trade logged
- Performance metrics calculated automatically
- Historical data for analysis

### ✅ Risk Management
- Built-in position sizing
- Daily loss limits
- Confidence-based filtering

### ✅ Reporting
- Daily performance summaries
- Strategy comparison analysis
- Historical trend tracking

## Installation Requirements

```bash
# Install required packages
pip install pandas numpy sqlite3 schedule logging pathlib datetime
```

## Database Location

- **File**: `trading_signals.db`
- **Location**: Project root directory
- **Backup**: Automatic daily backups recommended

## Future Enhancements

1. **Real-time Alerts**: SMS/Email notifications for important events
2. **Web Dashboard**: Real-time monitoring interface
3. **Advanced Analytics**: Machine learning insights
4. **Multi-timeframe**: Support for different timeframes
5. **Portfolio Management**: Multi-strategy portfolio optimization
6. **Paper Trading**: Simulation mode for testing
7. **API Integration**: Real broker integration
8. **Cloud Deployment**: AWS/GCP deployment options

## Support

For issues or questions:
1. Check log files for error details
2. Verify database connectivity
3. Ensure all required files are present
4. Check market hours and trading days
5. Review configuration settings

## Security Notes

- Database contains sensitive trading data
- Implement proper access controls
- Regular backups recommended
- Monitor for unauthorized access
- Use secure network connections 