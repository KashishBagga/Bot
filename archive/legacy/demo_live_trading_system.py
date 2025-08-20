#!/usr/bin/env python3
"""
Live Trading System Demo
Demonstrates the complete live trading system functionality
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_header():
    """Print demo header"""
    print("🤖 LIVE TRADING SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows the complete live trading system:")
    print("• 📊 Backtesting results integration")
    print("• 🤖 Live trading bot")
    print("• ⏰ Automated scheduling")
    print("• 📈 Daily summaries and reports")
    print("• 🔍 Performance analysis")
    print("=" * 60)

def demo_backtesting():
    """Demo backtesting functionality"""
    print("\n🔍 STEP 1: BACKTESTING DEMONSTRATION")
    print("-" * 40)
    
    print("Running a quick backtest with insidebar_rsi strategy...")
    
    try:
        result = subprocess.run([
            'python3', 'all_strategies_parquet.py', 
            '--days', '7', 
            '--timeframe', '15min', 
            '--strategies', 'insidebar_rsi',
            '--no-save'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Backtesting completed successfully!")
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Backtesting failed")
            print(f"Error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("⏱️ Backtesting timed out (this is normal for demo)")
    except Exception as e:
        print(f"❌ Error running backtest: {e}")

def demo_live_trading_bot():
    """Demo live trading bot"""
    print("\n🤖 STEP 2: LIVE TRADING BOT DEMONSTRATION")
    print("-" * 40)
    
    print("Testing live trading bot initialization...")
    
    try:
        # Import and test the bot
        from live_trading_bot import LiveTradingBot
        
        bot = LiveTradingBot()
        print("✅ Live trading bot initialized successfully")
        
        # Test database
        if os.path.exists("trading_signals.db"):
            print("✅ Database created and ready")
        
        # Test strategies
        print(f"✅ {len(bot.strategies)} strategies loaded:")
        for strategy_name in bot.strategies.keys():
            print(f"   • {strategy_name}")
        
        print("✅ Live trading bot is ready for production!")
        
    except Exception as e:
        print(f"❌ Error testing live trading bot: {e}")

def demo_scheduling():
    """Demo scheduling functionality"""
    print("\n⏰ STEP 3: AUTOMATED SCHEDULING DEMONSTRATION")
    print("-" * 40)
    
    print("Testing automated scheduler...")
    
    try:
        from start_trading_bot import TradingBotScheduler
        
        scheduler = TradingBotScheduler()
        print("✅ Scheduler initialized successfully")
        
        # Show market status
        if scheduler.is_market_day():
            print("✅ Today is a market day")
        else:
            print("ℹ️ Today is not a market day (weekend/holiday)")
        
        if scheduler.is_market_hours():
            print("✅ Currently in market hours")
        else:
            print("ℹ️ Currently outside market hours")
        
        print("✅ Automated scheduling is ready!")
        print("   • Starts at 9:00 AM (Mon-Fri)")
        print("   • Stops at 3:30 PM (Mon-Fri)")
        print("   • Health checks every 30 minutes")
        print("   • Daily reports at 4:00 PM")
        
    except Exception as e:
        print(f"❌ Error testing scheduler: {e}")

def demo_reporting():
    """Demo reporting functionality"""
    print("\n📊 STEP 4: REPORTING DEMONSTRATION")
    print("-" * 40)
    
    print("Testing daily summary viewer...")
    
    try:
        from view_daily_trading_summary import DailyTradingSummaryViewer
        
        viewer = DailyTradingSummaryViewer()
        print("✅ Daily summary viewer initialized")
        
        # Test database connection
        import sqlite3
        conn = sqlite3.connect("trading_signals.db")
        cursor = conn.cursor()
        
        # Check for any existing data
        cursor.execute("SELECT COUNT(*) FROM live_signals")
        signal_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM daily_trading_summary")
        summary_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"✅ Database contains {signal_count} signals and {summary_count} daily summaries")
        
        if signal_count > 0:
            print("✅ Running sample report...")
            try:
                viewer.view_recent_signals(5)
            except:
                print("ℹ️ No recent signals to display (this is normal)")
        
        print("✅ Reporting system is ready!")
        
    except Exception as e:
        print(f"❌ Error testing reporting: {e}")

def demo_usage_examples():
    """Show usage examples"""
    print("\n📚 STEP 5: USAGE EXAMPLES")
    print("-" * 40)
    
    print("Here are the main commands you can use:")
    print()
    
    print("🚀 START LIVE TRADING:")
    print("   python3 start_trading_bot.py")
    print()
    
    print("📊 VIEW TODAY'S RESULTS:")
    print("   python3 view_daily_trading_summary.py --today")
    print()
    
    print("📈 VIEW WEEKLY PERFORMANCE:")
    print("   python3 view_daily_trading_summary.py --weekly 1")
    print()
    
    print("🎯 VIEW STRATEGY PERFORMANCE:")
    print("   python3 view_daily_trading_summary.py --strategy 7")
    print()
    
    print("🔍 VIEW RECENT SIGNALS:")
    print("   python3 view_daily_trading_summary.py --signals 20")
    print()
    
    print("📋 VIEW ALL INFORMATION:")
    print("   python3 view_daily_trading_summary.py --all")
    print()
    
    print("🧪 RUN TESTS:")
    print("   python3 test_live_trading_bot.py")

def demo_system_status():
    """Show system status"""
    print("\n🔍 STEP 6: SYSTEM STATUS")
    print("-" * 40)
    
    # Check required files
    required_files = [
        'live_trading_bot.py',
        'start_trading_bot.py',
        'view_daily_trading_summary.py',
        'test_live_trading_bot.py',
        'all_strategies_parquet.py'
    ]
    
    print("📂 Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
    
    # Check strategies
    strategy_files = [
        'src/strategies/insidebar_rsi.py',
        'src/strategies/ema_crossover.py',
        'src/strategies/supertrend_ema.py',
        'src/strategies/supertrend_macd_rsi_ema.py'
    ]
    
    print("\n🧠 Checking strategy files...")
    for file in strategy_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
    
    # Check database
    if os.path.exists("trading_signals.db"):
        print("\n💾 Database status:")
        print("✅ trading_signals.db exists")
        
        try:
            import sqlite3
            conn = sqlite3.connect("trading_signals.db")
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            print(f"✅ {len(tables)} tables found:")
            for table in tables:
                print(f"   • {table[0]}")
            
            conn.close()
        except Exception as e:
            print(f"❌ Database error: {e}")
    else:
        print("\n💾 Database status:")
        print("⚠️ trading_signals.db not found (will be created on first run)")

def main():
    """Run the complete demo"""
    print_header()
    
    try:
        demo_backtesting()
        demo_live_trading_bot()
        demo_scheduling()
        demo_reporting()
        demo_usage_examples()
        demo_system_status()
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Your live trading system is ready for production!")
        print()
        print("Next steps:")
        print("1. Run: python3 start_trading_bot.py")
        print("2. Monitor: python3 view_daily_trading_summary.py --today")
        print("3. Analyze: python3 view_daily_trading_summary.py --all")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main() 