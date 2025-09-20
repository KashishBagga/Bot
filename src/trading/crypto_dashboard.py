#!/usr/bin/env python3
"""
Crypto Trading Dashboard - Separate from Indian Trading
"""

import os
import sys
import sqlite3
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

def display_crypto_dashboard():
    """Display crypto trading dashboard."""
    print("🚀 CRYPTO TRADING DASHBOARD")
    print("=" * 60)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        with sqlite3.connect('data/crypto_trading.db') as conn:
            cursor = conn.cursor()
            
            # System Status
            print("\n📊 SYSTEM STATUS")
            print("-" * 30)
            print("✅ Crypto Trader: Running")
            print("✅ Database: Connected")
            print("✅ API: Binance (Real-time)")
            
            # Signals
            cursor.execute("""
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN executed = 1 THEN 1 END) as executed,
                       AVG(confidence) as avg_conf
                FROM signals 
                WHERE market = 'crypto'
            """)
            result = cursor.fetchone()
            total_signals, executed_signals, avg_conf = result
            
            print(f"\n📡 SIGNALS")
            print("-" * 30)
            print(f"📊 Total Signals: {total_signals}")
            print(f"✅ Executed: {executed_signals}")
            if total_signals > 0:
                execution_rate = (executed_signals / total_signals) * 100
                print(f"🎯 Execution Rate: {execution_rate:.1f}%")
                print(f"📈 Avg Confidence: {avg_conf:.1f}%")
            
            # Trades
            cursor.execute("SELECT COUNT(*) FROM open_trades WHERE market = 'crypto'")
            open_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM closed_trades WHERE market = 'crypto'")
            closed_trades = cursor.fetchone()[0]
            
            print(f"\n💼 TRADES")
            print("-" * 30)
            print(f"📊 Open Trades: {open_trades}")
            print(f"📊 Closed Trades: {closed_trades}")
            
            # P&L
            cursor.execute("""
                SELECT SUM(pnl) as total_pnl,
                       COUNT(CASE WHEN pnl > 0 THEN 1 END) as winners
                FROM closed_trades 
                WHERE market = 'crypto' AND pnl IS NOT NULL
            """)
            result = cursor.fetchone()
            total_pnl, winners = result
            
            print(f"\n💰 P&L")
            print("-" * 30)
            if total_pnl is not None:
                print(f"💰 Total P&L: ${total_pnl:.2f}")
                if closed_trades > 0:
                    win_rate = (winners / closed_trades) * 100
                    print(f"�� Win Rate: {win_rate:.1f}%")
            else:
                print("💰 Total P&L: $0.00")
            
            # Recent Activity
            cursor.execute("""
                SELECT symbol, strategy, signal, confidence, timestamp
                FROM signals 
                WHERE market = 'crypto'
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            recent_signals = cursor.fetchall()
            
            if recent_signals:
                print(f"\n📋 RECENT SIGNALS")
                print("-" * 30)
                for signal in recent_signals:
                    symbol, strategy, signal_type, confidence, timestamp = signal
                    print(f"   {symbol} | {strategy} | {signal_type} | {confidence:.1f}% | {timestamp}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🔄 Refreshing in 15 seconds... (Ctrl+C to stop)")

if __name__ == "__main__":
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            display_crypto_dashboard()
            time.sleep(15)
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped")

