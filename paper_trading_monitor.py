#!/usr/bin/env python3
"""
Paper Trading Monitor
Real-time monitoring dashboard for paper trading bot
"""

import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

def get_latest_trades(db_path: str = "trading_signals.db", limit: int = 20) -> pd.DataFrame:
    """Get latest trades from database."""
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT 
            timestamp,
            strategy,
            signal,
            price,
            confidence,
            reasoning,
            symbol,
            created_at
        FROM trading_signals 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return pd.DataFrame()

def get_performance_stats(db_path: str = "trading_signals.db") -> dict:
    """Get performance statistics."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Get total trades
        total_trades = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM trading_signals", conn
        ).iloc[0]['count']
        
        # Get recent trades (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_trades = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM trading_signals WHERE timestamp > ?", 
            conn, params=(week_ago,)
        ).iloc[0]['count']
        
        # Get strategy performance
        strategy_stats = pd.read_sql_query("""
            SELECT 
                strategy,
                COUNT(*) as trades,
                AVG(confidence) as avg_confidence
            FROM trading_signals 
            GROUP BY strategy
            ORDER BY trades DESC
        """, conn)
        
        # Get daily performance
        daily_stats = pd.read_sql_query("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as trades,
                SUM(CASE WHEN signal = 'BUY CALL' THEN 1 ELSE 0 END) as buy_signals,
                SUM(CASE WHEN signal = 'BUY PUT' THEN 1 ELSE 0 END) as sell_signals
            FROM trading_signals 
            WHERE timestamp > date('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """, conn)
        
        conn.close()
        
        return {
            'total_trades': total_trades,
            'recent_trades': recent_trades,
            'strategy_stats': strategy_stats,
            'daily_stats': daily_stats
        }
    except Exception as e:
        print(f"❌ Error getting performance stats: {e}")
        return {}

def print_dashboard():
    """Print monitoring dashboard."""
    print("\n" + "="*80)
    print("📊 PAPER TRADING MONITORING DASHBOARD")
    print("="*80)
    print(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get performance stats
    stats = get_performance_stats()
    
    if stats:
        print(f"\n📈 PERFORMANCE OVERVIEW")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Recent Trades (7 days): {stats['recent_trades']}")
        
        if not stats['strategy_stats'].empty:
            print(f"\n🎯 STRATEGY PERFORMANCE")
            for _, row in stats['strategy_stats'].iterrows():
                print(f"   {row['strategy']}: {row['trades']} trades, {row['avg_confidence']:.1f}% avg confidence")
        
        if not stats['daily_stats'].empty:
            print(f"\n📅 DAILY PERFORMANCE (Last 7 days)")
            for _, row in stats['daily_stats'].head(5).iterrows():
                print(f"   {row['date']}: {row['trades']} trades ({row['buy_signals']} buy, {row['sell_signals']} sell)")
    
    # Get latest trades
    latest_trades = get_latest_trades(limit=10)
    
    if not latest_trades.empty:
        print(f"\n🔄 LATEST SIGNALS (Last 10)")
        print("-" * 80)
        print(f"{'Time':<20} {'Strategy':<20} {'Signal':<10} {'Price':<10} {'Confidence':<10}")
        print("-" * 80)
        
        for _, trade in latest_trades.iterrows():
            timestamp = pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M')
            strategy = trade['strategy'][:18] + "..." if len(trade['strategy']) > 18 else trade['strategy']
            signal = trade['signal']
            price = f"₹{trade['price']:.0f}"
            confidence = f"{trade['confidence_score']:.0f}%"
            
            print(f"{timestamp:<20} {strategy:<20} {signal:<10} {price:<10} {confidence:<10}")
    
    print("\n" + "="*80)
    print("💡 Commands:")
    print("   Press 'q' to quit")
    print("   Press 'r' to refresh")
    print("   Press 'Enter' to auto-refresh in 30 seconds")
    print("="*80)

def main():
    """Main monitoring loop."""
    print("🚀 Starting Paper Trading Monitor...")
    print("📊 Press 'q' to quit, 'r' to refresh, or Enter for auto-refresh")
    
    while True:
        print_dashboard()
        
        try:
            user_input = input("\nEnter command: ").strip().lower()
            
            if user_input == 'q':
                print("👋 Goodbye!")
                break
            elif user_input == 'r':
                continue
            elif user_input == '':
                print("⏳ Auto-refreshing in 30 seconds...")
                time.sleep(30)
            else:
                print("❓ Invalid command. Press 'q' to quit, 'r' to refresh, or Enter for auto-refresh")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main() 