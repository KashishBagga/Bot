# ⏰ Cron Job Setup Instructions

## 🔧 **Manual Cron Setup**

Since the automated setup requires system permissions, here's how to set it up manually:

### **Step 1: Open Crontab Editor**
```bash
crontab -e
```

### **Step 2: Add This Line**
```bash
# Daily Historical Data Update - Every weekday at 9:00 AM IST
0 9 * * 1-5 cd /Users/kashishbaggafeast/Desktop/Bot && /usr/bin/python3 daily_data_updater.py --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --timeframes 5min 15min 1D >> logs/cron.log 2>&1
```

### **Step 3: Save and Exit**
- Press `Ctrl + X` to exit
- Press `Y` to confirm
- Press `Enter` to save

### **Step 4: Verify Setup**
```bash
crontab -l
```

You should see the cron job listed.

## 📅 **Cron Schedule Explanation**

```
0 9 * * 1-5
│ │ │ │ │ └── Day of week (1-5 = Monday to Friday)
│ │ │ │ └──── Month (any)
│ │ │ └────── Day of month (any)
│ │ └──────── Hour (9 = 9:00 AM)
│ └────────── Minute (0 = at minute 0)
└──────────── Second (0 = at second 0)
```

**Translation**: Run every weekday (Monday-Friday) at 9:00 AM IST

## 🔍 **Monitor the Cron Job**

### **Check if it's running:**
```bash
ps aux | grep daily_data_updater.py
```

### **View cron logs:**
```bash
tail -f logs/cron.log
```

### **Check cron job status:**
```bash
crontab -l
```

## 🚨 **Troubleshooting**

### **If cron job doesn't run:**
1. Check if cron service is running: `sudo launchctl list | grep cron`
2. Verify the path is correct: `/Users/kashishbaggafeast/Desktop/Bot`
3. Check permissions: `ls -la daily_data_updater.py`
4. Test manually: `python3 daily_data_updater.py --symbols NSE:NIFTY50-INDEX --timeframes 5min`

### **If you get permission errors:**
```bash
# Make sure the script is executable
chmod +x daily_data_updater.py

# Check file ownership
ls -la daily_data_updater.py
```

## 📝 **Alternative: Manual Daily Updates**

If you prefer not to use cron, you can run updates manually:

```bash
# Every morning at 9:00 AM IST, run:
cd /Users/kashishbaggafeast/Desktop/Bot
python3 daily_data_updater.py --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --timeframes 5min 15min 1D
```

## 🎯 **What Happens Daily**

1. **9:00 AM IST**: Cron job triggers
2. **Market Check**: Waits for market to open (9:15 AM IST)
3. **Data Fetch**: Downloads missing data from Fyers
4. **Data Update**: Updates local parquet files
5. **Logging**: Records all activities in `logs/cron.log`

## ✅ **Success Indicators**

- ✅ **Cron job listed**: `crontab -l` shows your job
- ✅ **Logs created**: `logs/cron.log` file exists
- ✅ **Data updated**: Parquet files show today's date
- ✅ **No errors**: Clean logs without error messages

---

**Your historical data will now update automatically every weekday! 🚀** 