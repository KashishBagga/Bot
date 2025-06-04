#!/usr/bin/env python3
"""
Performance Monitor
Tracks strategy performance and alerts for degradation.
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self, db_path="trading_signals.db"):
        self.db_path = db_path
    
    def check_recent_performance(self, days=7):
        """Check performance over recent days."""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent trades
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        query = f"""
        SELECT strategy, COUNT(*) as trades, AVG(pnl) as avg_pnl, SUM(pnl) as total_pnl
        FROM (
            SELECT 'insidebar_bollinger' as strategy, pnl FROM insidebar_bollinger WHERE signal_time >= '{cutoff_date}'
            UNION ALL
            SELECT 'donchian_breakout' as strategy, pnl FROM donchian_breakout WHERE signal_time >= '{cutoff_date}'
            UNION ALL
            SELECT 'range_breakout_volatility' as strategy, pnl FROM range_breakout_volatility WHERE signal_time >= '{cutoff_date}'
        ) combined
        WHERE pnl IS NOT NULL
        GROUP BY strategy
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"📊 Performance Summary (Last {days} days):")
        print("-" * 60)
        
        for _, row in df.iterrows():
            print(f"{row['strategy']:<25} | Trades: {int(row['trades']):>4} | Avg P&L: ₹{row['avg_pnl']:>8.2f} | Total: ₹{row['total_pnl']:>10.2f}")
        
        # Alert for poor performance
        poor_performers = df[df['avg_pnl'] < -10]
        if not poor_performers.empty:
            print("\n⚠️ PERFORMANCE ALERTS:")
            for _, row in poor_performers.iterrows():
                print(f"  • {row['strategy']} showing poor performance: ₹{row['avg_pnl']:.2f} avg P&L")
    
    def daily_summary(self):
        """Generate daily performance summary."""
        conn = sqlite3.connect(self.db_path)
        
        today = datetime.now().strftime('%Y-%m-%d')
        query = f"""
        SELECT 
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            AVG(pnl) as avg_pnl,
            SUM(pnl) as total_pnl
        FROM (
            SELECT pnl FROM insidebar_bollinger WHERE DATE(signal_time) = '{today}'
            UNION ALL
            SELECT pnl FROM donchian_breakout WHERE DATE(signal_time) = '{today}'
            UNION ALL  
            SELECT pnl FROM range_breakout_volatility WHERE DATE(signal_time) = '{today}'
        ) combined
        WHERE pnl IS NOT NULL
        """
        
        result = pd.read_sql_query(query, conn).iloc[0]
        conn.close()
        
        if result['trades'] > 0:
            win_rate = (result['wins'] / result['trades']) * 100
            print(f"📈 Today's Performance ({today}):")
            print(f"  Trades: {int(result['trades'])}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total P&L: ₹{result['total_pnl']:.2f}")
            print(f"  Avg P&L: ₹{result['avg_pnl']:.2f}")
        else:
            print(f"📈 No trades executed today ({today})")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.daily_summary()
    monitor.check_recent_performance(7)
