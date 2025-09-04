# 📊 Historical Data Management System

This system automatically keeps your historical market data up to date by fetching missing data from Fyers API daily.

## 🎯 **What This System Does**

1. **✅ Completes Missing Data**: Fetches data from August 25, 2025 to today
2. **🔄 Daily Updates**: Automatically updates data every weekday at 9:00 AM IST
3. **📁 Local Storage**: Stores data in efficient parquet format
4. **🚫 No Data Gaps**: Ensures continuous historical data for strategies

## 📁 **Files Created**

- `update_historical_data.py` - One-time script to complete missing data
- `daily_data_updater.py` - Automated daily updater script
- `setup_daily_updater.sh` - Setup script for automation
- `HISTORICAL_DATA_README.md` - This documentation

## 🚀 **Quick Start**

### **Step 1: Complete Missing Data (One-time)**

```bash
# Update all symbols and timeframes
python3 update_historical_data.py --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --timeframes 5min 15min 1D --complete

# Update specific symbol/timeframe
python3 update_historical_data.py --symbols NSE:NIFTY50-INDEX --timeframes 5min --complete
```

### **Step 2: Setup Daily Automation**

```bash
# Option A: Cron job (recommended)
bash setup_daily_updater.sh cron

# Option B: Systemd service (requires sudo)
sudo bash setup_daily_updater.sh systemd

# Test the updater
bash setup_daily_updater.sh test

# Check status
bash setup_daily_updater.sh status
```

## 📊 **Data Coverage**

### **✅ NIFTY50-INDEX**
- **5min**: 150,692 candles (2017-07-17 to 2025-09-02)
- **15min**: 50,285 candles (2017-07-17 to 2025-09-02)  
- **1D**: 6,364 candles (2000-01-03 to 2025-09-02)

### **✅ NIFTYBANK-INDEX**
- **5min**: 150,683 candles (2017-07-17 to 2025-09-02)
- **15min**: 50,284 candles (2017-07-17 to 2025-09-02)
- **1D**: 2,069 candles (2020-01-01 to 2025-09-02)

## ⏰ **Automation Schedule**

- **Frequency**: Every weekday (Monday-Friday)
- **Time**: 9:00 AM IST (India Standard Time)
- **Duration**: Automatic, runs until completion
- **Logs**: Stored in `logs/daily_data_updater.log`

## 🔧 **Manual Usage**

### **Complete Missing Data**
```bash
python3 update_historical_data.py --complete
```

### **Update Specific Data**
```bash
python3 update_historical_data.py --symbols NSE:NIFTY50-INDEX --timeframes 5min
```

### **Daily Update (Manual)**
```bash
python3 daily_data_updater.py --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --timeframes 5min 15min 1D
```

## 📝 **Logging and Monitoring**

### **Log Files**
- `historical_data_update.log` - One-time updates
- `logs/daily_data_updater.log` - Daily automated updates
- `logs/cron.log` - Cron job execution logs

### **Check Status**
```bash
# View cron jobs
crontab -l

# View systemd timers
systemctl list-timers

# Check updater status
bash setup_daily_updater.sh status

# View recent logs
tail -f logs/daily_data_updater.log
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **"Fyers client not initialized"**
   - Ensure your `.env` file has correct Fyers credentials
   - Check if Fyers authentication is working

2. **"Market closed" messages**
   - This is normal outside market hours (9:15 AM - 3:30 PM IST)
   - The system will wait for market hours automatically

3. **"No data received" warnings**
   - Some dates may have no data (holidays, weekends)
   - This is normal and handled automatically

4. **Timezone issues**
   - All timestamps are stored in Asia/Kolkata (IST)
   - The system automatically handles timezone conversion

### **Manual Recovery**
```bash
# Stop any running processes
pkill -f daily_data_updater.py

# Check logs for errors
tail -50 logs/daily_data_updater.log

# Re-run the updater
python3 daily_data_updater.py --symbols NSE:NIFTY50-INDEX --timeframes 5min
```

## 🔒 **Security and Best Practices**

1. **Environment Variables**: Store Fyers credentials in `.env` file
2. **Rate Limiting**: Built-in delays prevent API rate limit issues
3. **Error Handling**: Comprehensive error handling and logging
4. **Data Validation**: Automatic validation of fetched data
5. **Backup**: Original data is preserved during updates

## 📈 **Performance Benefits**

1. **Faster Strategy Testing**: Local data vs API calls
2. **No API Rate Limits**: Unlimited strategy backtesting
3. **Consistent Data**: Same data across all testing sessions
4. **Offline Capability**: Work without internet connection
5. **Cost Savings**: Reduce Fyers API usage

## 🔄 **Maintenance**

### **Daily**
- Automatic updates run at 9:00 AM IST
- Check logs for any errors
- Monitor data freshness

### **Weekly**
- Review log files for patterns
- Check data completeness
- Verify automation is working

### **Monthly**
- Review data quality
- Check storage usage
- Update Fyers credentials if needed

## 📞 **Support**

If you encounter issues:

1. **Check logs first**: `tail -f logs/daily_data_updater.log`
2. **Verify credentials**: Ensure `.env` file is correct
3. **Test manually**: Run `python3 daily_data_updater.py --symbols NSE:NIFTY50-INDEX --timeframes 5min`
4. **Check status**: `bash setup_daily_updater.sh status`

## 🎉 **Success Indicators**

- ✅ **Data is up to date**: Latest data from today
- ✅ **No missing dates**: Continuous data from 2017 to present
- ✅ **Automation working**: Daily updates running automatically
- ✅ **Logs clean**: No errors in daily update logs
- ✅ **Strategies working**: All indicators calculating correctly

---

**Your historical data is now production-ready and will stay automatically updated! 🚀** 