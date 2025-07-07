#!/usr/bin/env python3
"""
Start Trading Bot
Automated startup script for live trading bot with scheduling
"""

import os
import sys
import time
import signal
import logging
import schedule
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

class TradingBotScheduler:
    """Schedule and manage the live trading bot"""
    
    def __init__(self):
        self.bot_process = None
        self.is_running = False
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the scheduler"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_market_day(self):
        """Check if today is a market day (Monday to Friday)"""
        today = datetime.now()
        return today.weekday() < 5  # 0-4 are Monday to Friday
    
    def is_market_hours(self):
        """Check if current time is within market hours (9:00 AM to 3:30 PM)"""
        now = datetime.now()
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end
    
    def start_trading_bot(self):
        """Start the live trading bot"""
        if not self.is_market_day():
            self.logger.info("🚫 Not a market day - skipping bot start")
            return
        
        if self.bot_process and self.bot_process.poll() is None:
            self.logger.info("🤖 Trading bot is already running")
            return
        
        try:
            self.logger.info("🚀 Starting live trading bot...")
            
            # Start the bot as a subprocess
            self.bot_process = subprocess.Popen([
                sys.executable, 'live_trading_bot.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.is_running = True
            self.logger.info(f"✅ Trading bot started with PID: {self.bot_process.pid}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start trading bot: {e}")
    
    def stop_trading_bot(self):
        """Stop the live trading bot"""
        if self.bot_process and self.bot_process.poll() is None:
            try:
                self.logger.info("🛑 Stopping live trading bot...")
                self.bot_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.bot_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.logger.warning("⚠️ Bot didn't stop gracefully, forcing shutdown...")
                    self.bot_process.kill()
                    self.bot_process.wait()
                
                self.is_running = False
                self.logger.info("✅ Trading bot stopped successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Error stopping trading bot: {e}")
        else:
            self.logger.info("ℹ️ Trading bot is not running")
    
    def check_bot_health(self):
        """Check if the bot is still running and healthy"""
        if not self.is_market_day():
            return
        
        if not self.is_market_hours():
            if self.is_running:
                self.logger.info("🕐 Market closed - stopping bot")
                self.stop_trading_bot()
            return
        
        if self.bot_process:
            if self.bot_process.poll() is None:
                self.logger.info("💚 Trading bot is healthy and running")
            else:
                self.logger.warning("⚠️ Trading bot has stopped unexpectedly - restarting...")
                self.start_trading_bot()
        else:
            self.logger.warning("⚠️ Trading bot not found - starting...")
            self.start_trading_bot()
    
    def generate_daily_report(self):
        """Generate daily trading report"""
        if not self.is_market_day():
            return
        
        try:
            self.logger.info("📊 Generating daily trading report...")
            
            # Run the daily summary viewer
            result = subprocess.run([
                sys.executable, 'view_daily_trading_summary.py', '--today'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Daily report generated successfully")
            else:
                self.logger.error(f"❌ Failed to generate daily report: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ Error generating daily report: {e}")
    
    def setup_schedule(self):
        """Setup the trading schedule"""
        # Start bot at 9:00 AM on weekdays
        schedule.every().monday.at("09:00").do(self.start_trading_bot)
        schedule.every().tuesday.at("09:00").do(self.start_trading_bot)
        schedule.every().wednesday.at("09:00").do(self.start_trading_bot)
        schedule.every().thursday.at("09:00").do(self.start_trading_bot)
        schedule.every().friday.at("09:00").do(self.start_trading_bot)
        
        # Stop bot at 3:30 PM on weekdays
        schedule.every().monday.at("15:30").do(self.stop_trading_bot)
        schedule.every().tuesday.at("15:30").do(self.stop_trading_bot)
        schedule.every().wednesday.at("15:30").do(self.stop_trading_bot)
        schedule.every().thursday.at("15:30").do(self.stop_trading_bot)
        schedule.every().friday.at("15:30").do(self.stop_trading_bot)
        
        # Health check every 30 minutes during market hours
        schedule.every(30).minutes.do(self.check_bot_health)
        
        # Daily report at 4:00 PM on weekdays
        schedule.every().monday.at("16:00").do(self.generate_daily_report)
        schedule.every().tuesday.at("16:00").do(self.generate_daily_report)
        schedule.every().wednesday.at("16:00").do(self.generate_daily_report)
        schedule.every().thursday.at("16:00").do(self.generate_daily_report)
        schedule.every().friday.at("16:00").do(self.generate_daily_report)
        
        self.logger.info("📅 Trading schedule configured successfully")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum} - shutting down...")
        self.stop_trading_bot()
        sys.exit(0)
    
    def run(self):
        """Run the scheduler"""
        self.logger.info("🚀 Starting Trading Bot Scheduler...")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Setup schedule
        self.setup_schedule()
        
        # If it's currently market hours and a market day, start immediately
        if self.is_market_day() and self.is_market_hours():
            self.logger.info("🏁 Market is open - starting bot immediately")
            self.start_trading_bot()
        
        self.logger.info("⏰ Scheduler is running - waiting for scheduled events...")
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Keyboard interrupt received - shutting down...")
                break
            except Exception as e:
                self.logger.error(f"❌ Scheduler error: {e}")
                time.sleep(60)
        
        self.stop_trading_bot()
        self.logger.info("👋 Trading Bot Scheduler stopped")


def main():
    """Main execution function"""
    print("🤖 Live Trading Bot Scheduler")
    print("=" * 50)
    print("Features:")
    print("• ⏰ Automatic start at 9:00 AM (Mon-Fri)")
    print("• 🛑 Automatic stop at 3:30 PM (Mon-Fri)")
    print("• 💚 Health monitoring every 30 minutes")
    print("• 📊 Daily reports at 4:00 PM")
    print("• 🔄 Automatic restart on crashes")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ['live_trading_bot.py', 'view_daily_trading_summary.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            sys.exit(1)
    
    scheduler = TradingBotScheduler()
    
    try:
        scheduler.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 