#!/usr/bin/env python3
"""
Live Bot Monitor
Real-time monitoring of the live trading bot's signals and activity
"""

import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
import subprocess

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_bot_status():
    """Check if the live bot is running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        
        # Look for python processes running live trading
        live_processes = []
        for line in processes.split('\n'):
            if 'python' in line and any(keyword in line for keyword in ['live', 'trading', 'optimized_live']):
                live_processes.append(line.strip())
        
        return live_processes
    except:
        return []

def get_recent_signals(hours=24):
    """Get recent signals from database"""
    try:
        conn = sqlite3.connect('trading_signals.db')
        
        # Get signals from last N hours
        since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        query = '''
            SELECT timestamp, strategy, symbol, signal, confidence_score, price, reasoning
            FROM live_signals_realtime 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 20
        '''
        
        signals = conn.execute(query, (since_time,)).fetchall()
        conn.close()
        
        return signals
    except Exception as e:
        print(f"❌ Error getting signals: {e}")
        return []

def get_signal_stats():
    """Get signal statistics"""
    try:
        conn = sqlite3.connect('trading_signals.db')
        
        # Today's stats
        today = datetime.now().strftime('%Y-%m-%d')
        
        today_query = '''
            SELECT 
                COUNT(*) as total,
                AVG(confidence_score) as avg_confidence,
                COUNT(CASE WHEN confidence_score >= 70 THEN 1 END) as high_confidence,
                COUNT(CASE WHEN signal = 'BUY' THEN 1 END) as buy_signals,
                COUNT(CASE WHEN signal = 'SELL' THEN 1 END) as sell_signals
            FROM live_signals_realtime 
            WHERE date(timestamp) = ?
        '''
        
        today_stats = conn.execute(today_query, (today,)).fetchone()
        
        # Strategy breakdown
        strategy_query = '''
            SELECT strategy, COUNT(*) as count, AVG(confidence_score) as avg_conf
            FROM live_signals_realtime 
            WHERE date(timestamp) = ?
            GROUP BY strategy
            ORDER BY count DESC
        '''
        
        strategy_stats = conn.execute(strategy_query, (today,)).fetchall()
        
        conn.close()
        
        return today_stats, strategy_stats
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return None, []

def get_recent_logs(lines=10):
    """Get recent log entries"""
    try:
        with open('logs/optimized_live_bot.log', 'r') as f:
            log_lines = f.readlines()
            return log_lines[-lines:]
    except Exception as e:
        print(f"❌ Error reading logs: {e}")
        return []

def monitor_live_bot():
    """Main monitoring function"""
    print("🤖 LIVE TRADING BOT MONITOR")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now()}")
    print()
    
    while True:
        # Clear screen (works on Unix systems)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🤖 LIVE TRADING BOT MONITOR")
        print("=" * 80)
        print(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # 1. Bot Status
        print("🔍 BOT STATUS:")
        print("-" * 40)
        
        processes = check_bot_status()
        if processes:
            print("✅ Live bot is running:")
            for proc in processes[:3]:  # Show first 3 matches
                print(f"   {proc}")
        else:
            print("❌ Live bot is not running")
            print("   Start with: python3 run_trading_system.py live")
        print()
        
        # 2. Signal Statistics
        print("📊 SIGNAL STATISTICS:")
        print("-" * 40)
        
        today_stats, strategy_stats = get_signal_stats()
        
        if today_stats:
            total, avg_conf, high_conf, buy_count, sell_count = today_stats
            print(f"📈 Today's Signals: {total}")
            if total > 0:
                print(f"🎯 Average Confidence: {avg_conf:.1f}")
                print(f"⭐ High Confidence (70+): {high_conf}")
                print(f"📈 Buy Signals: {buy_count}")
                print(f"📉 Sell Signals: {sell_count}")
            else:
                print("📊 No signals generated today")
        else:
            print("❌ Unable to get signal statistics")
        print()
        
        # 3. Strategy Breakdown
        if strategy_stats:
            print("🎯 STRATEGY PERFORMANCE:")
            print("-" * 40)
            for strategy, count, avg_conf in strategy_stats:
                print(f"  {strategy}: {count} signals (Avg: {avg_conf:.1f})")
            print()
        
        # 4. Recent Signals
        print("📋 RECENT SIGNALS (Last 24 hours):")
        print("-" * 40)
        
        recent_signals = get_recent_signals(24)
        if recent_signals:
            for signal in recent_signals[:5]:  # Show last 5
                timestamp, strategy, symbol, signal_type, confidence, price, reasoning = signal
                time_str = datetime.fromisoformat(timestamp).strftime('%H:%M:%S')
                print(f"🎯 {time_str} | {strategy} | {symbol} | {signal_type} | Conf: {confidence}")
        else:
            print("📊 No recent signals")
        print()
        
        # 5. Recent Bot Activity
        print("📝 RECENT BOT ACTIVITY:")
        print("-" * 40)
        
        recent_logs = get_recent_logs(5)
        for log_line in recent_logs:
            # Extract timestamp and message
            if ' - ' in log_line:
                parts = log_line.split(' - ', 3)
                if len(parts) >= 4:
                    timestamp = parts[0]
                    level = parts[2]
                    message = parts[3].strip()
                    
                    # Format timestamp
                    try:
                        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                        time_str = dt.strftime('%H:%M:%S')
                        
                        # Color code by level
                        icon = "🔍" if "INFO" in level else "⚠️" if "WARNING" in level else "❌"
                        print(f"{icon} {time_str} | {message}")
                    except:
                        print(f"📝 {log_line.strip()}")
        print()
        
        # 6. Control Instructions
        print("🎮 CONTROLS:")
        print("-" * 40)
        print("💡 Ctrl+C to exit monitor")
        print("🚀 To start bot: python3 run_trading_system.py live")
        print("🔍 To test: python3 test_complete_trading_flow.py")
        print("📊 Database: trading_signals.db")
        print("📝 Logs: logs/optimized_live_bot.log")
        print()
        print("⏰ Refreshing in 30 seconds...")
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break

def show_signal_details():
    """Show detailed signal information"""
    print("📋 DETAILED SIGNAL HISTORY")
    print("=" * 80)
    
    recent_signals = get_recent_signals(168)  # Last week
    
    if not recent_signals:
        print("📊 No signals found in database")
        return
    
    print(f"📊 Found {len(recent_signals)} signals in the last week:")
    print()
    
    for i, signal in enumerate(recent_signals, 1):
        timestamp, strategy, symbol, signal_type, confidence, price, reasoning = signal
        
        print(f"🎯 Signal #{i}")
        print(f"   📅 Time: {timestamp}")
        print(f"   🎯 Strategy: {strategy}")
        print(f"   📈 Symbol: {symbol}")
        print(f"   📊 Signal: {signal_type}")
        print(f"   ⭐ Confidence: {confidence}")
        print(f"   💰 Price: ₹{price:.2f}" if price else "   💰 Price: N/A")
        if reasoning and reasoning != '{}':
            print(f"   💭 Reasoning: {reasoning[:100]}...")
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "signals":
        show_signal_details()
    else:
        try:
            monitor_live_bot()
        except KeyboardInterrupt:
            print("\n👋 Monitor stopped by user")
        except Exception as e:
            print(f"\n❌ Monitor error: {e}") 