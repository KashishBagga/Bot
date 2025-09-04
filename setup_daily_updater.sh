#!/bin/bash
"""
Daily Data Updater Setup Script
===============================

This script sets up automated daily historical data updates using either:
1. Cron job (recommended for most users)
2. Systemd service (for systemd-based systems)

Usage:
    bash setup_daily_updater.sh [cron|systemd]
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATER_SCRIPT="$SCRIPT_DIR/daily_data_updater.py"
LOG_DIR="$SCRIPT_DIR/logs"
CRON_LOG="$LOG_DIR/cron.log"

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}🚀 Setting up Daily Data Updater...${NC}"
echo -e "${BLUE}📁 Script directory: $SCRIPT_DIR${NC}"
echo -e "${BLUE}📝 Updater script: $UPDATER_SCRIPT${NC}"

# Check if updater script exists
if [ ! -f "$UPDATER_SCRIPT" ]; then
    echo -e "${RED}❌ Error: daily_data_updater.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Make script executable
chmod +x "$UPDATER_SCRIPT"

# Function to setup cron job
setup_cron() {
    echo -e "${BLUE}⏰ Setting up cron job...${NC}"
    
    # Create cron job entry (runs at 9:00 AM IST every weekday)
    CRON_JOB="0 9 * * 1-5 cd $SCRIPT_DIR && /usr/bin/python3 $UPDATER_SCRIPT --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --timeframes 5min 15min 1D >> $CRON_LOG 2>&1"
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "daily_data_updater.py"; then
        echo -e "${YELLOW}⚠️ Cron job already exists. Updating...${NC}"
        # Remove existing cron job
        crontab -l 2>/dev/null | grep -v "daily_data_updater.py" | crontab -
    fi
    
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    
    echo -e "${GREEN}✅ Cron job added successfully!${NC}"
    echo -e "${BLUE}📅 Schedule: Every weekday (Mon-Fri) at 9:00 AM IST${NC}"
    echo -e "${BLUE}📝 Logs: $CRON_LOG${NC}"
    
    # Show current cron jobs
    echo -e "${BLUE}📋 Current cron jobs:${NC}"
    crontab -l
}

# Function to setup systemd service
setup_systemd() {
    echo -e "${BLUE}🔧 Setting up systemd service...${NC}"
    
    # Create systemd service file
    SERVICE_FILE="/etc/systemd/system/daily-data-updater.service"
    TIMER_FILE="/etc/systemd/system/daily-data-updater.timer"
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}❌ Error: systemd setup requires root privileges${NC}"
        echo -e "${YELLOW}💡 Run with: sudo bash setup_daily_updater.sh systemd${NC}"
        exit 1
    fi
    
    # Create service file
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Daily Historical Data Updater
After=network.target

[Service]
Type=oneshot
User=$SUDO_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $UPDATER_SCRIPT --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX --timeframes 5min 15min 1D
StandardOutput=append:$CRON_LOG
StandardError=append:$CRON_LOG

[Install]
WantedBy=multi-user.target
EOF
    
    # Create timer file
    cat > "$TIMER_FILE" << EOF
[Unit]
Description=Run Daily Data Updater every weekday at 9:00 AM IST
Requires=daily-data-updater.service

[Timer]
OnCalendar=Mon..Fri 09:00:00
Persistent=true
Unit=daily-data-updater.service

[Install]
WantedBy=timers.target
EOF
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable daily-data-updater.timer
    systemctl start daily-data-updater.timer
    
    echo -e "${GREEN}✅ Systemd service and timer created successfully!${NC}"
    echo -e "${BLUE}📅 Schedule: Every weekday (Mon-Fri) at 9:00 AM IST${NC}"
    echo -e "${BLUE}📝 Logs: $CRON_LOG${NC}"
    
    # Show service status
    echo -e "${BLUE}📋 Service status:${NC}"
    systemctl status daily-data-updater.timer --no-pager -l
}

# Function to test the updater
test_updater() {
    echo -e "${BLUE}🧪 Testing the daily data updater...${NC}"
    
    if [ -f "$UPDATER_SCRIPT" ]; then
        echo -e "${BLUE}📝 Running test update...${NC}"
        cd "$SCRIPT_DIR"
        python3 "$UPDATER_SCRIPT" --symbols NSE:NIFTY50-INDEX --timeframes 5min
        echo -e "${GREEN}✅ Test completed successfully!${NC}"
    else
        echo -e "${RED}❌ Error: Cannot find updater script${NC}"
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Daily Data Updater Status${NC}"
    echo -e "${BLUE}========================${NC}"
    
    # Check cron jobs
    if crontab -l 2>/dev/null | grep -q "daily_data_updater.py"; then
        echo -e "${GREEN}✅ Cron job: ACTIVE${NC}"
        crontab -l | grep "daily_data_updater.py"
    else
        echo -e "${RED}❌ Cron job: NOT FOUND${NC}"
    fi
    
    # Check systemd service
    if systemctl list-timers 2>/dev/null | grep -q "daily-data-updater"; then
        echo -e "${GREEN}✅ Systemd timer: ACTIVE${NC}"
        systemctl list-timers | grep "daily-data-updater"
    else
        echo -e "${RED}❌ Systemd timer: NOT FOUND${NC}"
    fi
    
    # Check log file
    if [ -f "$CRON_LOG" ]; then
        echo -e "${GREEN}✅ Log file: EXISTS${NC}"
        echo -e "${BLUE}📝 Log file: $CRON_LOG${NC}"
        echo -e "${BLUE}📊 Log size: $(du -h "$CRON_LOG" | cut -f1)${NC}"
    else
        echo -e "${YELLOW}⚠️ Log file: NOT FOUND${NC}"
    fi
}

# Main script logic
case "${1:-cron}" in
    "cron")
        setup_cron
        ;;
    "systemd")
        setup_systemd
        ;;
    "test")
        test_updater
        ;;
    "status")
        show_status
        ;;
    "help"|"-h"|"--help")
        echo -e "${BLUE}Daily Data Updater Setup Script${NC}"
        echo -e "${BLUE}===============================${NC}"
        echo ""
        echo "Usage: bash setup_daily_updater.sh [OPTION]"
        echo ""
        echo "Options:"
        echo "  cron     Setup cron job (default)"
        echo "  systemd  Setup systemd service and timer"
        echo "  test     Test the updater script"
        echo "  status   Show current status"
        echo "  help     Show this help message"
        echo ""
        echo "Examples:"
        echo "  bash setup_daily_updater.sh cron"
        echo "  sudo bash setup_daily_updater.sh systemd"
        echo "  bash setup_daily_updater.sh test"
        echo "  bash setup_daily_updater.sh status"
        ;;
    *)
        echo -e "${RED}❌ Error: Unknown option '$1'${NC}"
        echo -e "${YELLOW}💡 Use 'bash setup_daily_updater.sh help' for usage information${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo -e "${BLUE}📝 Check logs at: $CRON_LOG${NC}"
echo -e "${BLUE}📋 View status with: bash setup_daily_updater.sh status${NC}" 