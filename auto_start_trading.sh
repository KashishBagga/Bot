#!/bin/bash

# Auto-start Live Trading Bot
# This script can be run at system startup or manually

# Set the trading bot directory
TRADING_BOT_DIR="/Users/kashishbaggafeast/Desktop/Bot"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$TRADING_BOT_DIR/logs/auto_start.log"
}

# Create logs directory if it doesn't exist
mkdir -p "$TRADING_BOT_DIR/logs"

log_message "🚀 Auto-start script initiated"

# Change to the trading bot directory
cd "$TRADING_BOT_DIR" || {
    log_message "❌ Failed to change to trading bot directory"
    exit 1
}

# Check if it's a weekday
current_day=$(date +%u)  # 1=Monday, 7=Sunday
if [ "$current_day" -gt 5 ]; then
    log_message "📅 Weekend detected - no trading today"
    exit 0
fi

# Check if it's a reasonable time (between 8:30 AM and 4:00 PM)
current_hour=$(date +%H)
current_minute=$(date +%M)
current_time_minutes=$((current_hour * 60 + current_minute))
start_time_minutes=$((8 * 60 + 30))  # 8:30 AM
end_time_minutes=$((16 * 60))         # 4:00 PM

if [ "$current_time_minutes" -lt "$start_time_minutes" ] || [ "$current_time_minutes" -gt "$end_time_minutes" ]; then
    log_message "⏰ Outside trading hours - not starting bot"
    exit 0
fi

# Check if the bot is already running
if pgrep -f "start_trading_bot.py" > /dev/null; then
    log_message "✅ Trading bot scheduler already running"
    exit 0
fi

# Start the trading bot scheduler
log_message "🤖 Starting trading bot scheduler..."
nohup python3 start_trading_bot.py > "$TRADING_BOT_DIR/logs/nohup_scheduler.log" 2>&1 &

# Wait a moment and check if it started successfully
sleep 5
if pgrep -f "start_trading_bot.py" > /dev/null; then
    log_message "✅ Trading bot scheduler started successfully"
else
    log_message "❌ Failed to start trading bot scheduler"
    exit 1
fi

log_message "🎉 Auto-start completed successfully" 