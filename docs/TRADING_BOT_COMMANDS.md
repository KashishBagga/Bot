# 🤖 Trading Bot Commands Reference

## 🚀 **Quick Start Commands**

### **Production Trading (Recommended)**
```bash
# Start automated trading bot scheduler
python3 start_trading_bot.py
```
**Features:**
- ⏰ Auto-start at 9:00 AM (Mon-Fri)
- 🛑 Auto-stop at 3:30 PM (Mon-Fri)
- 💚 Health monitoring every 30 minutes
- 📊 Daily reports at 4:00 PM
- 🔄 Automatic restart on crashes

### **Manual Trading (Testing)**
```bash
# Run live trading bot manually
python3 live_trading_bot.py
```

---

## 📊 **Backtesting Commands**

### **Primary Backtesting**
```bash
# Run comprehensive backtesting with parquet data (RECOMMENDED)
python3 backtesting_parquet.py

# Alternative backtesting with all strategies
python3 all_strategies_parquet.py
```

### **Alternative Backtesting Options**
```bash
# Run backtesting through trading system
python3 run_trading_system.py backtest

# Legacy backtesting (if needed)
python3 backtest.py
```

---

## 🧪 **Testing & Validation Commands**

### **System Testing**
```bash
# Test unified system (all strategies + database)
python3 test_unified_system.py

# Test live trading bot functionality
python3 test_live_trading_bot.py

# Test Fyers API connection
python3 test_fyers.py
```

### **Demo & Simulation**
```bash
# Run demo live trading system
python3 demo_live_trading_system.py
```

---

## 📈 **Analysis & Monitoring Commands**

### **Performance Analysis**
```bash
# View daily trading summaries
python3 view_daily_trading_summary.py

# View backtesting results
python3 view_backtest_results.py

# Show strategy performance summary
python3 show_strategy_summary.py

# Analyze trading data comprehensively
python3 trading_data_analyzer.py
```

### **Performance Monitoring**
```bash
# Monitor system performance
python3 performance_monitor.py

# Update performance metrics
python3 update_performance_metrics.py

# Backfill performance metrics
python3 backfill_performance_metrics.py
```

---

## 🛠️ **Database Management Commands**

### **Database Operations**
```bash
# Drop/reset database (⚠️ DESTRUCTIVE)
python3 drop_database.py

# Optimize database performance
python3 optimize_db.py

# Analyze strategy tables
python3 analyze_strategy_tables.py
```

### **Database Maintenance**
```bash
# Auto-ensure performance metrics
python3 auto_ensure_performance_metrics.py

# Enhanced signal tracking
python3 enhanced_signal_tracker.py
```

---

## 🗂️ **Data Setup Commands**

### **Historical Data Setup**
```bash
# Setup 20-year historical parquet data
python3 setup_20_year_parquet_data.py

# Setup basic system
python3 setup.py
```

### **Configuration**
```bash
# Configure backtesting parameters
python3 backtest_config.py

# Run comprehensive trading system
python3 run_trading_system.py
```

---

## 🔧 **System Management Commands**

### **Auto-Start Setup**
```bash
# Make auto-start script executable
chmod +x auto_start_trading.sh

# Run auto-start setup
./auto_start_trading.sh
```

### **Process Management**
```bash
# Check if trading bot is running
ps aux | grep python3 | grep live_trading_bot

# Kill trading bot process (if needed)
pkill -f "python3 live_trading_bot.py"

# Check scheduler process
ps aux | grep python3 | grep start_trading_bot
```

---

## 📱 **Communication & Alerts**

### **Telegram Integration**
```bash
# Test Telegram notifications
python3 telegram.py
```

---

## 🎯 **Strategy-Specific Commands**

### **All Strategies**
```bash
# Run all strategies with legacy system
python3 all_strategies.py

# Run all strategies with parquet data
python3 all_strategies_parquet.py
```

### **20-Year Backtesting**
```bash
# Run 20-year backtesting engine
python3 backtest_20_year_engine.py
```

---

## 📋 **Command Usage Examples**

### **Daily Workflow**
```bash
# 1. Start trading bot (morning)
python3 start_trading_bot.py

# 2. Monitor performance (during day)
python3 view_daily_trading_summary.py

# 3. Check logs (evening)
tail -f logs/live_trading_$(date +%Y-%m-%d).log
```

### **Weekly Analysis**
```bash
# 1. Run comprehensive backtesting
python3 backtesting_parquet.py

# 2. Analyze results
python3 view_backtest_results.py

# 3. Check strategy performance
python3 show_strategy_summary.py
```

### **System Maintenance**
```bash
# 1. Test system integrity
python3 test_unified_system.py

# 2. Optimize database
python3 optimize_db.py

# 3. Update performance metrics
python3 update_performance_metrics.py
```

---

## 🚨 **Emergency Commands**

### **Stop Trading**
```bash
# Stop scheduler (graceful)
pkill -TERM -f "python3 start_trading_bot.py"

# Force stop all trading processes
pkill -KILL -f "python3.*trading"
```

### **Reset System**
```bash
# Reset database (⚠️ DESTRUCTIVE)
python3 drop_database.py

# Reinitialize system
python3 setup.py
```

---

## 📊 **Log Monitoring**

### **View Logs**
```bash
# View today's trading logs
tail -f logs/live_trading_$(date +%Y-%m-%d).log

# View scheduler logs
tail -f logs/scheduler_$(date +%Y-%m-%d).log

# View all logs
ls -la logs/
```

### **Log Analysis**
```bash
# Check for errors in logs
grep -i "error\|failed\|exception" logs/live_trading_*.log

# Check signal generation
grep -i "signal\|trade" logs/live_trading_*.log

# Check health status
grep -i "healthy\|health" logs/scheduler_*.log
```

---

## 🔍 **System Status Commands**

### **Check System Health**
```bash
# Check if scheduler is running
pgrep -f "python3 start_trading_bot.py"

# Check if trading bot is running
pgrep -f "python3 live_trading_bot.py"

# Check database size
ls -lh trading_signals.db

# Check available disk space
df -h .
```

### **Performance Monitoring**
```bash
# Check system resources
top -p $(pgrep -f "python3.*trading")

# Monitor memory usage
ps aux | grep python3 | grep trading
```

---

## 🎯 **Command Categories Summary**

| Category | Primary Commands | Purpose |
|----------|------------------|---------|
| **Production** | `python3 start_trading_bot.py` | Live trading with automation |
| **Backtesting** | `python3 backtesting_parquet.py` | Historical strategy testing |
| **Testing** | `python3 test_unified_system.py` | System validation |
| **Analysis** | `python3 view_daily_trading_summary.py` | Performance monitoring |
| **Database** | `python3 optimize_db.py` | Database maintenance |
| **Setup** | `python3 setup_20_year_parquet_data.py` | Data preparation |

---

## 🚀 **Recommended Daily Workflow**

### **Morning (Before Market Open)**
```bash
# 1. Check system status
python3 test_unified_system.py

# 2. Start automated trading
python3 start_trading_bot.py
```

### **During Market Hours**
```bash
# Monitor in real-time
tail -f logs/live_trading_$(date +%Y-%m-%d).log
```

### **Evening (After Market Close)**
```bash
# 1. View daily summary
python3 view_daily_trading_summary.py

# 2. Analyze performance
python3 trading_data_analyzer.py
```

### **Weekly Maintenance**
```bash
# 1. Run backtesting
python3 backtesting_parquet.py

# 2. Optimize database
python3 optimize_db.py

# 3. Update metrics
python3 update_performance_metrics.py
```

---

## 📞 **Support & Troubleshooting**

### **Common Issues**
```bash
# Database locked
python3 drop_database.py  # ⚠️ DESTRUCTIVE

# Process stuck
pkill -f "python3.*trading"

# Memory issues
python3 optimize_db.py
```

### **Health Checks**
```bash
# System test
python3 test_unified_system.py

# Database integrity
python3 analyze_strategy_tables.py

# Performance check
python3 performance_monitor.py
```

---

## ⚡ **Quick Reference**

**Start Trading:** `python3 start_trading_bot.py`  
**Run Backtest:** `python3 backtesting_parquet.py`  
**View Results:** `python3 view_daily_trading_summary.py`  
**Test System:** `python3 test_unified_system.py`  
**Stop Trading:** `pkill -TERM -f "python3 start_trading_bot.py"`

---

*Last Updated: $(date)*  
*System Status: ✅ Operational* 