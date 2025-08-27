#!/usr/bin/env python3
"""
Database Status Checker
Shows the current status of all database tables
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unified_database import UnifiedDatabase

def check_database_status():
    """Check the status of all database tables."""
    print("🗄️ Database Status Check")
    print("=" * 50)
    
    try:
        # Initialize database
        db = UnifiedDatabase()
        
        # Check if database file exists
        if not os.path.exists("trading_signals.db"):
            print("❌ Database file not found. Run paper trading to create it.")
            return
        
        print("✅ Database file found")
        
        # Get performance summary
        summary = db.get_performance_summary()
        
        print(f"\n📊 Performance Summary:")
        print(f"   Total Signals: {summary.get('total_signals', 0)}")
        print(f"   Rejected Signals: {summary.get('total_rejected', 0)}")
        print(f"   Total Trades: {summary.get('total_trades', 0)}")
        print(f"   Winning Trades: {summary.get('winning_trades', 0)}")
        print(f"   Win Rate: {summary.get('win_rate', 0):.1f}%")
        print(f"   Total P&L: ₹{summary.get('total_pnl', 0):+.2f}")
        print(f"   Avg P&L: ₹{summary.get('avg_pnl', 0):+.2f}")
        
        # Get recent signals
        recent_signals = db.get_trading_signals()
        if recent_signals:
            print(f"\n📈 Recent Signals (Last 5):")
            for signal in recent_signals[:5]:
                print(f"   {signal['timestamp']} - {signal['strategy']} - {signal['signal']} - ₹{signal['price']:,.2f}")
        
        # Get recent rejected signals
        recent_rejected = db.get_rejected_signals()
        if recent_rejected:
            print(f"\n🚫 Recent Rejected Signals (Last 5):")
            for signal in recent_rejected[:5]:
                print(f"   {signal['timestamp']} - {signal['strategy']} - {signal['rejection_reason']}")
        
        # Get open positions
        open_positions = db.get_open_option_positions()
        if open_positions:
            print(f"\n📈 Open Positions ({len(open_positions)}):")
            for pos in open_positions:
                print(f"   {pos['contract_symbol']} - {pos['strategy']} - Entry: ₹{pos['entry_price']:.2f}")
        
        print(f"\n✅ Database check completed successfully!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database_status() 