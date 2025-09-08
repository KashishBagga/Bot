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
            return pd.DataFrame()
        finally:
            conn.close()
    
    def print_comprehensive_report(self, days: int = 7):
        """Print comprehensive trading analytics report"""
        
        # 1. Strategy Performance Summary
        perf_df = self.get_strategy_performance_summary(days)
        if not perf_df.empty:
            for _, row in perf_df.iterrows():
                status = "✅ PROFITABLE" if row['total_pnl'] > 0 else "❌ LOSS"
        else:
        
        # 2. Recent Trades
        trades_df = self.get_recent_trades(days)
        if not trades_df.empty:
            for _, row in trades_df.head(10).iterrows():
                outcome_icon = "✅" if row['outcome'] == 'Win' else "❌"
                reasoning = row.get('reasoning', 'No reasoning provided')
                if reasoning and reasoning != 'No reasoning provided':
        else:
        
        # 3. Daily P&L Breakdown
        daily_df = self.get_daily_pnl_breakdown(days)
        if not daily_df.empty:
            for _, row in daily_df.head(10).iterrows():
                pnl_icon = "📈" if row['daily_pnl'] > 0 else "📉"
        else:
        
        # 4. Confidence Analysis
        conf_df = self.get_confidence_analysis(days)
        if not conf_df.empty:
            for _, row in conf_df.head(10).iterrows():
        else:
        
        # 5. Rejection Analysis
        reject_df = self.get_rejection_analysis(days)
        if not reject_df.empty:
            for _, row in reject_df.head(10).iterrows():
        else:
        
        # 6. Key Metrics Summary
        if not perf_df.empty:
            total_trades = perf_df['total_trades'].sum()
            total_pnl = perf_df['total_pnl'].sum()
            avg_win_rate = (perf_df['wins'].sum() / perf_df['total_trades'].sum()) * 100
            profitable_strategies = len(perf_df[perf_df['total_pnl'] > 0])
            
        else:
        

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
        
        # Export recent trades
        trades_df = dashboard.get_recent_trades(args.days)
        if not trades_df.empty:
            trades_df.to_csv(f"recent_trades_{args.days}d.csv", index=False)
        
        # Export performance summary
        perf_df = dashboard.get_strategy_performance_summary(args.days)
        if not perf_df.empty:
            perf_df.to_csv(f"performance_summary_{args.days}d.csv", index=False)
        
        # Export daily P&L
        daily_df = dashboard.get_daily_pnl_breakdown(args.days)
        if not daily_df.empty:
            daily_df.to_csv(f"daily_pnl_{args.days}d.csv", index=False)
        
        # Export rejection analysis
        reject_df = dashboard.get_rejection_analysis(args.days)
        if not reject_df.empty:
            reject_df.to_csv(f"rejection_analysis_{args.days}d.csv", index=False)
        

if __name__ == "__main__":
    main() 