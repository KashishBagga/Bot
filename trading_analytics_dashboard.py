#!/usr/bin/env python3
"""
Trading Analytics Dashboard
Comprehensive analysis of trades, P&L, strategy signals, and performance metrics
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional

class TradingAnalyticsDashboard:
    """Comprehensive trading analytics and performance analysis"""
    
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
        
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_recent_trades(self, days: int = 7) -> pd.DataFrame:
        """Get recent trades with full details"""
        conn = self.get_connection()
        
        query = f"""
        SELECT 
            timestamp,
            strategy,
            symbol,
            signal,
            price,
            stop_loss,
            target,
            target2,
            target3,
            reasoning,
            confidence,
            confidence_score,
            outcome,
            pnl,
            targets_hit,
            stoploss_count,
            exit_time,
            market_condition
        FROM trades_backtest 
        WHERE timestamp >= datetime('now', '-{days} days')
        ORDER BY timestamp DESC
        """
        
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_rejected_signals(self, days: int = 7) -> pd.DataFrame:
        """Get rejected signals with reasons"""
        conn = self.get_connection()
        
        query = f"""
        SELECT 
            timestamp,
            strategy,
            symbol,
            signal_attempted,
            rejection_reason,
            rejection_category,
            price,
            confidence,
            confidence_score,
            rsi,
            macd,
            ema_21,
            atr,
            supertrend_direction,
            reasoning,
            market_condition
        FROM rejected_signals_backtest 
        WHERE timestamp >= datetime('now', '-{days} days')
        ORDER BY timestamp DESC
        """
        
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching rejected signals: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_strategy_performance_summary(self, days: int = 30) -> pd.DataFrame:
        """Get strategy performance summary"""
        conn = self.get_connection()
        
        query = f"""
        SELECT 
            strategy,
            symbol,
            COUNT(*) as total_trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl,
            ROUND(AVG(confidence_score), 1) as avg_confidence,
            ROUND(MAX(pnl), 2) as max_profit,
            ROUND(MIN(pnl), 2) as max_loss,
            ROUND(AVG(targets_hit), 1) as avg_targets_hit,
            ROUND(AVG(stoploss_count), 1) as avg_stoploss_count
        FROM trades_backtest 
        WHERE timestamp >= datetime('now', '-{days} days')
        GROUP BY strategy, symbol
        ORDER BY total_pnl DESC
        """
        
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching performance summary: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_daily_pnl_breakdown(self, days: int = 30) -> pd.DataFrame:
        """Get daily P&L breakdown"""
        conn = self.get_connection()
        
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            strategy,
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as daily_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl
        FROM trades_backtest 
        WHERE timestamp >= datetime('now', '-{days} days')
        GROUP BY DATE(timestamp), strategy, symbol
        ORDER BY date DESC, daily_pnl DESC
        """
        
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching daily P&L: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_confidence_analysis(self, days: int = 30) -> pd.DataFrame:
        """Analyze confidence scores vs performance"""
        conn = self.get_connection()
        
        query = f"""
        SELECT 
            strategy,
            symbol,
            CASE 
                WHEN confidence_score >= 80 THEN 'Very High (80+)'
                WHEN confidence_score >= 70 THEN 'High (70-79)'
                WHEN confidence_score >= 60 THEN 'Medium (60-69)'
                WHEN confidence_score >= 50 THEN 'Low (50-59)'
                ELSE 'Very Low (<50)'
            END as confidence_level,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl
        FROM trades_backtest 
        WHERE timestamp >= datetime('now', '-{days} days')
        GROUP BY strategy, symbol, confidence_level
        ORDER BY strategy, symbol, total_pnl DESC
        """
        
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching confidence analysis: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_rejection_analysis(self, days: int = 7) -> pd.DataFrame:
        """Analyze rejection reasons and patterns"""
        conn = self.get_connection()
        
        query = f"""
        SELECT 
            strategy,
            symbol,
            rejection_reason,
            rejection_category,
            COUNT(*) as rejection_count,
            ROUND(AVG(confidence_score), 1) as avg_confidence,
            ROUND(AVG(rsi), 1) as avg_rsi,
            ROUND(AVG(atr), 2) as avg_atr
        FROM rejected_signals_backtest 
        WHERE timestamp >= datetime('now', '-{days} days')
        GROUP BY strategy, symbol, rejection_reason, rejection_category
        ORDER BY rejection_count DESC
        """
        
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching rejection analysis: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def print_comprehensive_report(self, days: int = 7):
        """Print comprehensive trading analytics report"""
        print("🚀 TRADING ANALYTICS DASHBOARD")
        print("=" * 60)
        print(f"📅 Analysis Period: Last {days} days")
        print(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Strategy Performance Summary
        print("📊 STRATEGY PERFORMANCE SUMMARY")
        print("-" * 40)
        perf_df = self.get_strategy_performance_summary(days)
        if not perf_df.empty:
            for _, row in perf_df.iterrows():
                status = "✅ PROFITABLE" if row['total_pnl'] > 0 else "❌ LOSS"
                print(f"🎯 {row['strategy']} ({row['symbol']}):")
                print(f"   Trades: {row['total_trades']} | Win Rate: {row['win_rate']}% | P&L: ₹{row['total_pnl']} | Avg: ₹{row['avg_pnl']}")
                print(f"   Confidence: {row['avg_confidence']} | Max Profit: ₹{row['max_profit']} | Max Loss: ₹{row['max_loss']}")
                print(f"   Status: {status}")
                print()
        else:
            print("❌ No performance data found")
        print()
        
        # 2. Recent Trades
        print("📈 RECENT TRADES")
        print("-" * 40)
        trades_df = self.get_recent_trades(days)
        if not trades_df.empty:
            print(f"Found {len(trades_df)} recent trades:")
            for _, row in trades_df.head(10).iterrows():
                outcome_icon = "✅" if row['outcome'] == 'Win' else "❌"
                print(f"{outcome_icon} {row['strategy']} ({row['symbol']}): {row['signal']} @ ₹{row['price']:.2f}")
                print(f"   Outcome: {row['outcome']} | P&L: ₹{row['pnl']:.2f} | Confidence: {row['confidence_score']}")
                reasoning = row.get('reasoning', 'No reasoning provided')
                print(f"   Reasoning: {reasoning[:100]}...")
                print()
        else:
            print("❌ No recent trades found")
        print()
        
        # 3. Daily P&L Breakdown
        print("📅 DAILY P&L BREAKDOWN")
        print("-" * 40)
        daily_df = self.get_daily_pnl_breakdown(days)
        if not daily_df.empty:
            for _, row in daily_df.head(10).iterrows():
                pnl_icon = "📈" if row['daily_pnl'] > 0 else "📉"
                print(f"{pnl_icon} {row['date']}: {row['strategy']} ({row['symbol']})")
                print(f"   Trades: {row['trades']} | Win Rate: {row['win_rate']}% | P&L: ₹{row['daily_pnl']:.2f}")
                print()
        else:
            print("❌ No daily P&L data found")
        print()
        
        # 4. Confidence Analysis
        print("🎯 CONFIDENCE ANALYSIS")
        print("-" * 40)
        conf_df = self.get_confidence_analysis(days)
        if not conf_df.empty:
            for _, row in conf_df.head(10).iterrows():
                print(f"📊 {row['strategy']} ({row['symbol']}) - {row['confidence_level']}:")
                print(f"   Trades: {row['trades']} | Win Rate: {row['win_rate']}% | P&L: ₹{row['total_pnl']:.2f}")
                print()
        else:
            print("❌ No confidence analysis data found")
        print()
        
        # 5. Rejection Analysis
        print("🚫 REJECTION ANALYSIS")
        print("-" * 40)
        reject_df = self.get_rejection_analysis(days)
        if not reject_df.empty:
            print(f"Found {len(reject_df)} rejection patterns:")
            for _, row in reject_df.head(10).iterrows():
                print(f"❌ {row['strategy']} ({row['symbol']}): {row['rejection_reason']}")
                print(f"   Count: {row['rejection_count']} | Avg Confidence: {row['avg_confidence']}")
                print(f"   Category: {row['rejection_category']}")
                print()
        else:
            print("❌ No rejection data found")
        print()
        
        # 6. Key Metrics Summary
        print("📊 KEY METRICS SUMMARY")
        print("-" * 40)
        if not perf_df.empty:
            total_trades = perf_df['total_trades'].sum()
            total_pnl = perf_df['total_pnl'].sum()
            avg_win_rate = (perf_df['wins'].sum() / perf_df['total_trades'].sum()) * 100
            profitable_strategies = len(perf_df[perf_df['total_pnl'] > 0])
            
            print(f"📈 Total Trades: {total_trades}")
            print(f"💰 Total P&L: ₹{total_pnl:.2f}")
            print(f"🎯 Overall Win Rate: {avg_win_rate:.1f}%")
            print(f"✅ Profitable Strategies: {profitable_strategies}/{len(perf_df)}")
            print(f"📊 Average Confidence: {perf_df['avg_confidence'].mean():.1f}")
        else:
            print("❌ No data available for summary")
        print()
        
        print("🎉 ANALYTICS REPORT COMPLETE")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Trading Analytics Dashboard")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze (default: 7)")
    parser.add_argument("--output", type=str, choices=["console", "csv"], default="console", 
                       help="Output format (default: console)")
    args = parser.parse_args()
    
    dashboard = TradingAnalyticsDashboard()
    
    if args.output == "console":
        dashboard.print_comprehensive_report(args.days)
    elif args.output == "csv":
        # Export data to CSV files
        print("📊 Exporting analytics data to CSV files...")
        
        # Export recent trades
        trades_df = dashboard.get_recent_trades(args.days)
        if not trades_df.empty:
            trades_df.to_csv(f"recent_trades_{args.days}d.csv", index=False)
            print(f"✅ Exported {len(trades_df)} trades to recent_trades_{args.days}d.csv")
        
        # Export performance summary
        perf_df = dashboard.get_strategy_performance_summary(args.days)
        if not perf_df.empty:
            perf_df.to_csv(f"performance_summary_{args.days}d.csv", index=False)
            print(f"✅ Exported performance summary to performance_summary_{args.days}d.csv")
        
        # Export daily P&L
        daily_df = dashboard.get_daily_pnl_breakdown(args.days)
        if not daily_df.empty:
            daily_df.to_csv(f"daily_pnl_{args.days}d.csv", index=False)
            print(f"✅ Exported daily P&L to daily_pnl_{args.days}d.csv")
        
        # Export rejection analysis
        reject_df = dashboard.get_rejection_analysis(args.days)
        if not reject_df.empty:
            reject_df.to_csv(f"rejection_analysis_{args.days}d.csv", index=False)
            print(f"✅ Exported rejection analysis to rejection_analysis_{args.days}d.csv")
        
        print("🎉 CSV export complete!")

if __name__ == "__main__":
    main() 