#!/usr/bin/env python3
"""
Backtest Database Checker
Shows the current status of backtest database tables
"""

import os
import sys
import sqlite3
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_backtest_database():
    """Check the status of backtest database tables."""
    print("🗄️ Backtest Database Status Check")
    print("=" * 60)
    
    try:
        # Check if database file exists
        if not os.path.exists("backtest_results.db"):
            print("❌ Backtest database file not found.")
            return
        
        print("✅ Backtest database file found")
        
        # Connect to database
        conn = sqlite3.connect("backtest_results.db")
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\n📋 Database Tables: {[table[0] for table in tables]}")
        
        # Check backtest sessions
        cursor.execute("SELECT COUNT(*) FROM backtest_sessions")
        session_count = cursor.fetchone()[0]
        print(f"\n📊 Backtest Sessions: {session_count}")
        
        if session_count > 0:
            cursor.execute("""
                SELECT session_id, session_name, symbol, timeframe, total_trades, win_rate, total_return
                FROM backtest_sessions 
                ORDER BY created_at DESC
            """)
            sessions = cursor.fetchall()
            
            print(f"\n📈 Recent Sessions:")
            for session in sessions:
                print(f"   {session[0]}")
                print(f"   Name: {session[1]}")
                print(f"   Symbol: {session[2]} | Timeframe: {session[3]}")
                print(f"   Trades: {session[4]} | Win Rate: {session[5]:.1f}% | Return: {session[6]:.2f}%")
                print()
        
        # Check signals
        cursor.execute("SELECT COUNT(*) FROM backtest_signals")
        signal_count = cursor.fetchone()[0]
        print(f"📈 Total Signals Generated: {signal_count}")
        
        if signal_count > 0:
            cursor.execute("""
                SELECT strategy_name, COUNT(*) as count
                FROM backtest_signals 
                GROUP BY strategy_name
            """)
            strategy_signals = cursor.fetchall()
            
            print(f"\n📊 Signals by Strategy:")
            for strategy, count in strategy_signals:
                print(f"   {strategy}: {count} signals")
        
        # Check rejected signals
        cursor.execute("SELECT COUNT(*) FROM backtest_rejected_signals")
        rejected_count = cursor.fetchone()[0]
        print(f"\n🚫 Rejected Signals: {rejected_count}")
        
        if rejected_count > 0:
            cursor.execute("""
                SELECT rejection_category, COUNT(*) as count
                FROM backtest_rejected_signals 
                GROUP BY rejection_category
            """)
            rejection_categories = cursor.fetchall()
            
            print(f"\n📊 Rejection Categories:")
            for category, count in rejection_categories:
                print(f"   {category}: {count} signals")
        
        # Check trades
        cursor.execute("SELECT COUNT(*) FROM backtest_trades")
        trade_count = cursor.fetchone()[0]
        print(f"\n📈 Total Trades: {trade_count}")
        
        if trade_count > 0:
            cursor.execute("""
                SELECT strategy_name, COUNT(*) as count, 
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as total_pnl
                FROM backtest_trades 
                GROUP BY strategy_name
            """)
            strategy_trades = cursor.fetchall()
            
            print(f"\n📊 Trades by Strategy:")
            for strategy, count, wins, total_pnl in strategy_trades:
                win_rate = (wins / count * 100) if count > 0 else 0
                print(f"   {strategy}: {count} trades, {wins} wins ({win_rate:.1f}%), P&L: ₹{total_pnl:.2f}")
        
        # Check strategy results
        cursor.execute("SELECT COUNT(*) FROM strategy_results")
        result_count = cursor.fetchone()[0]
        print(f"\n📊 Strategy Results: {result_count}")
        
        if result_count > 0:
            cursor.execute("""
                SELECT strategy_name, total_signals, executed_signals, rejected_signals,
                       total_trades, win_rate, total_pnl
                FROM strategy_results 
                ORDER BY total_pnl DESC
            """)
            results = cursor.fetchall()
            
            print(f"\n📈 Strategy Performance Summary:")
            for result in results:
                print(f"   {result[0]}:")
                print(f"     Signals: {result[1]} generated, {result[2]} executed, {result[3]} rejected")
                print(f"     Trades: {result[4]}, Win Rate: {result[5]:.1f}%, P&L: ₹{result[6]:.2f}")
                print()
        
        conn.close()
        print(f"\n✅ Backtest database check completed successfully!")
        
    except Exception as e:
        print(f"❌ Error checking backtest database: {e}")

if __name__ == "__main__":
    check_backtest_database() 