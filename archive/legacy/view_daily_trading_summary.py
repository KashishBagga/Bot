#!/usr/bin/env python3
"""
View Daily Trading Summary
Display live trading results and daily summaries
"""

import sqlite3
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

class DailyTradingSummaryViewer:
    """View and analyze daily trading summaries"""
    
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
    
    def view_today_summary(self):
        """View today's trading summary"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.view_date_summary(today)
    
    def view_date_summary(self, date: str):
        """View trading summary for a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get daily summary
            query = '''
                SELECT * FROM daily_trading_summary 
                WHERE date = ?
            '''
            summary = pd.read_sql_query(query, conn, params=(date,))
            
            if summary.empty:
                print(f"❌ No trading summary found for {date}")
                conn.close()
                return
            
            row = summary.iloc[0]
            
            print(f"\n{'='*80}")
            print(f"📊 DAILY TRADING SUMMARY - {date}")
            print(f"{'='*80}")
            print(f"🕘 Session: {row['market_start_time']} - {row['market_end_time']}")
            print(f"⏱️ Duration: {row['session_duration_minutes']} minutes")
            print(f"🎯 Signals Generated: {row['signals_generated']}")
            print(f"💼 Trades Taken: {row['trades_taken']}")
            print(f"✅ Profitable Trades: {row['profitable_trades']}")
            print(f"💰 Total P&L: ₹{row['total_pnl']:.2f}")
            print(f"📈 Win Rate: {row['win_rate']:.1f}%")
            
            # Parse strategies
            strategies = json.loads(row['strategies_active'])
            print(f"🧠 Active Strategies: {', '.join(strategies)}")
            
            # Get detailed signals for the day
            query = '''
                SELECT strategy, signal, confidence_score, price, COUNT(*) as count
                FROM live_signals
                WHERE date(created_at) = ?
                GROUP BY strategy, signal, confidence_score, price
                ORDER BY strategy, confidence_score DESC
            '''
            signals = pd.read_sql_query(query, conn, params=(date,))
            
            if not signals.empty:
                print(f"\n📊 Signal Breakdown:")
                for _, signal in signals.iterrows():
                    print(f"  🎯 {signal['strategy']}: {signal['signal']} "
                          f"(Confidence: {signal['confidence_score']}, "
                          f"Price: ₹{signal['price']:.2f}, Count: {signal['count']})")
            
            # Get trade executions
            query = '''
                SELECT ls.strategy, ls.symbol, ls.signal, lte.entry_price, lte.exit_price, 
                       lte.pnl, lte.status, lte.exit_reason
                FROM live_trade_executions lte
                JOIN live_signals ls ON lte.signal_id = ls.id
                WHERE date(lte.created_at) = ?
                ORDER BY lte.created_at
            '''
            trades = pd.read_sql_query(query, conn, params=(date,))
            
            if not trades.empty:
                print(f"\n💼 Trade Executions:")
                for _, trade in trades.iterrows():
                    status_icon = "✅" if trade['pnl'] > 0 else "❌" if trade['pnl'] < 0 else "⏸️"
                    print(f"  {status_icon} {trade['strategy']} - {trade['symbol']}: "
                          f"{trade['signal']} | Entry: ₹{trade['entry_price']:.2f} | "
                          f"P&L: ₹{trade['pnl']:.2f} | Status: {trade['status']}")
            
            print(f"{'='*80}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error viewing daily summary: {e}")
    
    def view_weekly_summary(self, weeks_back: int = 1):
        """View weekly trading summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks_back)
            
            query = '''
                SELECT date, signals_generated, trades_taken, profitable_trades, 
                       total_pnl, win_rate, session_duration_minutes
                FROM daily_trading_summary
                WHERE date >= ? AND date <= ?
                ORDER BY date DESC
            '''
            
            summaries = pd.read_sql_query(query, conn, params=(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ))
            
            if summaries.empty:
                print(f"❌ No trading data found for the last {weeks_back} week(s)")
                conn.close()
                return
            
            print(f"\n{'='*80}")
            print(f"📊 WEEKLY TRADING SUMMARY (Last {weeks_back} week(s))")
            print(f"{'='*80}")
            
            # Calculate totals
            total_signals = summaries['signals_generated'].sum()
            total_trades = summaries['trades_taken'].sum()
            total_profitable = summaries['profitable_trades'].sum()
            total_pnl = summaries['total_pnl'].sum()
            avg_win_rate = summaries['win_rate'].mean()
            total_session_time = summaries['session_duration_minutes'].sum()
            
            print(f"📊 OVERALL PERFORMANCE:")
            print(f"  🎯 Total Signals: {total_signals}")
            print(f"  💼 Total Trades: {total_trades}")
            print(f"  ✅ Profitable Trades: {total_profitable}")
            print(f"  💰 Total P&L: ₹{total_pnl:.2f}")
            print(f"  📈 Average Win Rate: {avg_win_rate:.1f}%")
            print(f"  ⏱️ Total Trading Time: {total_session_time} minutes")
            
            print(f"\n📅 DAILY BREAKDOWN:")
            for _, row in summaries.iterrows():
                pnl_icon = "🟢" if row['total_pnl'] > 0 else "🔴" if row['total_pnl'] < 0 else "⚪"
                print(f"  {pnl_icon} {row['date']}: {row['signals_generated']} signals, "
                      f"{row['trades_taken']} trades, ₹{row['total_pnl']:.2f} P&L, "
                      f"{row['win_rate']:.1f}% win rate")
            
            print(f"{'='*80}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error viewing weekly summary: {e}")
    
    def view_strategy_performance(self, days_back: int = 7):
        """View strategy performance analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            query = '''
                SELECT ls.strategy, COUNT(*) as total_signals, 
                       AVG(ls.confidence_score) as avg_confidence,
                       COUNT(lte.id) as executed_trades,
                       SUM(CASE WHEN lte.pnl > 0 THEN 1 ELSE 0 END) as profitable_trades,
                       SUM(lte.pnl) as total_pnl
                FROM live_signals ls
                LEFT JOIN live_trade_executions lte ON ls.id = lte.signal_id
                WHERE date(ls.created_at) >= ? AND date(ls.created_at) <= ?
                GROUP BY ls.strategy
                ORDER BY total_pnl DESC
            '''
            
            performance = pd.read_sql_query(query, conn, params=(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ))
            
            if performance.empty:
                print(f"❌ No strategy performance data found for the last {days_back} days")
                conn.close()
                return
            
            print(f"\n{'='*80}")
            print(f"📊 STRATEGY PERFORMANCE ANALYSIS (Last {days_back} days)")
            print(f"{'='*80}")
            
            for _, row in performance.iterrows():
                win_rate = 0.0
                if row['executed_trades'] > 0:
                    win_rate = (row['profitable_trades'] / row['executed_trades']) * 100
                
                pnl_icon = "🟢" if row['total_pnl'] > 0 else "🔴" if row['total_pnl'] < 0 else "⚪"
                
                print(f"{pnl_icon} {row['strategy'].upper()}:")
                print(f"  📊 Signals: {row['total_signals']}")
                print(f"  💼 Executed: {row['executed_trades']}")
                print(f"  ✅ Profitable: {row['profitable_trades']}")
                print(f"  💰 P&L: ₹{row['total_pnl']:.2f}")
                print(f"  📈 Win Rate: {win_rate:.1f}%")
                print(f"  🎯 Avg Confidence: {row['avg_confidence']:.1f}")
                print()
            
            print(f"{'='*80}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error viewing strategy performance: {e}")
    
    def view_recent_signals(self, limit: int = 20):
        """View recent trading signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, strategy, symbol, signal, confidence_score, 
                       price, target, stop_loss, status
                FROM live_signals
                ORDER BY created_at DESC
                LIMIT ?
            '''
            
            signals = pd.read_sql_query(query, conn, params=(limit,))
            
            if signals.empty:
                print("❌ No recent signals found")
                conn.close()
                return
            
            print(f"\n{'='*80}")
            print(f"📊 RECENT TRADING SIGNALS (Last {len(signals)} signals)")
            print(f"{'='*80}")
            
            for _, signal in signals.iterrows():
                confidence_icon = "🟢" if signal['confidence_score'] >= 70 else "🟡" if signal['confidence_score'] >= 50 else "🔴"
                status_icon = "✅" if signal['status'] == 'EXECUTED' else "⏸️"
                
                print(f"{status_icon} {signal['timestamp'][:19]} | {confidence_icon} {signal['strategy']} - {signal['symbol']}")
                print(f"    Signal: {signal['signal']} | Confidence: {signal['confidence_score']}")
                print(f"    Price: ₹{signal['price']:.2f} | Target: ₹{signal['target']:.2f} | SL: ₹{signal['stop_loss']:.2f}")
                print()
            
            print(f"{'='*80}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error viewing recent signals: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='View daily trading summaries')
    parser.add_argument('--today', action='store_true', help='View today\'s summary')
    parser.add_argument('--date', type=str, help='View summary for specific date (YYYY-MM-DD)')
    parser.add_argument('--weekly', type=int, default=0, help='View weekly summary (specify weeks back)')
    parser.add_argument('--strategy', type=int, default=0, help='View strategy performance (specify days back)')
    parser.add_argument('--signals', type=int, default=0, help='View recent signals (specify limit)')
    parser.add_argument('--all', action='store_true', help='View all available information')
    
    args = parser.parse_args()
    
    viewer = DailyTradingSummaryViewer()
    
    if args.all:
        viewer.view_today_summary()
        viewer.view_weekly_summary(1)
        viewer.view_strategy_performance(7)
        viewer.view_recent_signals(10)
    elif args.today:
        viewer.view_today_summary()
    elif args.date:
        viewer.view_date_summary(args.date)
    elif args.weekly > 0:
        viewer.view_weekly_summary(args.weekly)
    elif args.strategy > 0:
        viewer.view_strategy_performance(args.strategy)
    elif args.signals > 0:
        viewer.view_recent_signals(args.signals)
    else:
        # Default: show today's summary
        viewer.view_today_summary()


if __name__ == "__main__":
    main() 