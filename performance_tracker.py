#!/usr/bin/env python3
"""
Performance Tracker Dashboard
============================

Comprehensive performance analysis and tracking for the trading system.
Provides daily/weekly/monthly P&L, strategy performance, and risk metrics.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from zoneinfo import ZoneInfo
import logging

from src.config.settings import setup_logging

logger = setup_logging('performance_tracker')

class PerformanceTracker:
    """Comprehensive performance tracking and analysis"""
    
    def __init__(self, db_path: str = "unified_trading.db"):
        """Initialize performance tracker"""
        self.db_path = db_path
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("📊 Performance Tracker initialized")
    
    def get_daily_pnl(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get daily P&L summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if closed_option_positions table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='closed_option_positions'")
            if not cursor.fetchone():
                logger.warning("⚠️ closed_option_positions table not found, no trading data available")
                conn.close()
                return pd.DataFrame()
            
            query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as max_profit,
                MIN(pnl) as max_loss,
                SUM(commission) as total_commission,
                SUM(slippage) as total_slippage
            FROM closed_option_positions
            WHERE timestamp IS NOT NULL
            """
            
            if start_date:
                query += f" AND DATE(timestamp) >= '{start_date}'"
            if end_date:
                query += f" AND DATE(timestamp) <= '{end_date}'"
            
            query += " GROUP BY DATE(timestamp) ORDER BY date"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['win_rate'] = (df['winning_trades'] / df['total_trades'] * 100).round(2)
                df['cumulative_pnl'] = df['total_pnl'].cumsum()
                df['net_pnl'] = df['total_pnl'] - df['total_commission'] - df['total_slippage']
                df['cumulative_net_pnl'] = df['net_pnl'].cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting daily P&L: {e}")
            return pd.DataFrame()
    
    def get_strategy_performance(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get strategy-level performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if closed_option_positions table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='closed_option_positions'")
            if not cursor.fetchone():
                logger.warning("⚠️ closed_option_positions table not found, no trading data available")
                conn.close()
                return pd.DataFrame()
            
            query = """
            SELECT 
                strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as max_profit,
                MIN(pnl) as max_loss,
                AVG(ABS(pnl)) as avg_abs_pnl,
                SUM(commission) as total_commission,
                SUM(slippage) as total_slippage,
                AVG(confidence_score) as avg_confidence
            FROM closed_option_positions
            WHERE strategy IS NOT NULL
            """
            
            if start_date:
                query += f" AND DATE(timestamp) >= '{start_date}'"
            if end_date:
                query += f" AND DATE(timestamp) <= '{end_date}'"
            
            query += " GROUP BY strategy ORDER BY total_pnl DESC"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['win_rate'] = (df['winning_trades'] / df['total_trades'] * 100).round(2)
                df['profit_factor'] = (df['winning_trades'] * df['avg_pnl'] / 
                                     (df['losing_trades'] * df['avg_pnl'].abs())).round(2)
                df['net_pnl'] = df['total_pnl'] - df['total_commission'] - df['total_slippage']
                df['avg_commission'] = (df['total_commission'] / df['total_trades']).round(2)
                df['avg_slippage'] = (df['total_slippage'] / df['total_trades']).round(2)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy performance: {e}")
            return pd.DataFrame()
    
    def get_equity_curve(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get equity curve data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if equity_curve table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='equity_curve'")
            if not cursor.fetchone():
                logger.warning("⚠️ equity_curve table not found, no equity data available")
                conn.close()
                return pd.DataFrame()
            
            query = """
            SELECT 
                timestamp,
                equity,
                cash,
                unrealized_pnl,
                realized_pnl,
                total_exposure,
                max_drawdown
            FROM equity_curve
            WHERE timestamp IS NOT NULL
            """
            
            if start_date:
                query += f" AND DATE(timestamp) >= '{start_date}'"
            if end_date:
                query += f" AND DATE(timestamp) <= '{end_date}'"
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting equity curve: {e}")
            return pd.DataFrame()
    
    def get_risk_metrics(self, start_date: str = None, end_date: str = None) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            # Get daily P&L
            daily_pnl = self.get_daily_pnl(start_date, end_date)
            
            if daily_pnl.empty:
                return {}
            
            # Calculate risk metrics
            returns = daily_pnl['net_pnl'] / 20000  # Assuming 20k initial capital
            
            metrics = {
                'total_trades': daily_pnl['total_trades'].sum(),
                'winning_trades': daily_pnl['winning_trades'].sum(),
                'losing_trades': daily_pnl['losing_trades'].sum(),
                'total_pnl': daily_pnl['total_pnl'].sum(),
                'net_pnl': daily_pnl['net_pnl'].sum(),
                'total_commission': daily_pnl['total_commission'].sum(),
                'total_slippage': daily_pnl['total_slippage'].sum(),
                'win_rate': (daily_pnl['winning_trades'].sum() / daily_pnl['total_trades'].sum() * 100).round(2),
                'avg_daily_pnl': daily_pnl['net_pnl'].mean(),
                'std_daily_pnl': daily_pnl['net_pnl'].std(),
                'max_daily_profit': daily_pnl['total_pnl'].max(),
                'max_daily_loss': daily_pnl['total_pnl'].min(),
                'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)).round(3) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(daily_pnl['cumulative_net_pnl']),
                'profit_factor': self._calculate_profit_factor(daily_pnl),
                'avg_trade_duration': self._get_avg_trade_duration(start_date, end_date)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100
            return drawdown.min()
        except:
            return 0.0
    
    def _calculate_profit_factor(self, daily_pnl: pd.DataFrame) -> float:
        """Calculate profit factor"""
        try:
            gross_profit = daily_pnl[daily_pnl['total_pnl'] > 0]['total_pnl'].sum()
            gross_loss = abs(daily_pnl[daily_pnl['total_pnl'] < 0]['total_pnl'].sum())
            return (gross_profit / gross_loss).round(2) if gross_loss > 0 else float('inf')
        except:
            return 0.0
    
    def _get_avg_trade_duration(self, start_date: str = None, end_date: str = None) -> float:
        """Get average trade duration in minutes"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if closed_option_positions table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='closed_option_positions'")
            if not cursor.fetchone():
                logger.warning("⚠️ closed_option_positions table not found")
                conn.close()
                return 0.0
            
            query = """
            SELECT AVG(
                (julianday(exit_timestamp) - julianday(entry_timestamp)) * 24 * 60
            ) as avg_duration_minutes
            FROM closed_option_positions
            WHERE entry_timestamp IS NOT NULL AND exit_timestamp IS NOT NULL
            """
            
            if start_date:
                query += f" AND DATE(entry_timestamp) >= '{start_date}'"
            if end_date:
                query += f" AND DATE(entry_timestamp) <= '{end_date}'"
            
            result = conn.execute(query).fetchone()
            conn.close()
            
            return result[0] if result and result[0] else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error getting average trade duration: {e}")
            return 0.0
    
    def generate_performance_report(self, start_date: str = None, end_date: str = None) -> str:
        """Generate comprehensive performance report"""
        try:
            # Get data
            daily_pnl = self.get_daily_pnl(start_date, end_date)
            strategy_perf = self.get_strategy_performance(start_date, end_date)
            risk_metrics = self.get_risk_metrics(start_date, end_date)
            
            if daily_pnl.empty:
                return "No trading data available for the specified period."
            
            # Generate report
            report = []
            report.append("=" * 60)
            report.append("📊 PERFORMANCE REPORT")
            report.append("=" * 60)
            
            if start_date and end_date:
                report.append(f"📅 Period: {start_date} to {end_date}")
            else:
                report.append("📅 Period: All Time")
            
            report.append("")
            
            # Overall Performance
            report.append("🎯 OVERALL PERFORMANCE")
            report.append("-" * 30)
            report.append(f"💰 Total P&L: ₹{risk_metrics.get('total_pnl', 0):,.2f}")
            report.append(f"💵 Net P&L: ₹{risk_metrics.get('net_pnl', 0):,.2f}")
            report.append(f"📈 Win Rate: {risk_metrics.get('win_rate', 0):.1f}%")
            report.append(f"🎲 Total Trades: {risk_metrics.get('total_trades', 0)}")
            report.append(f"✅ Winning Trades: {risk_metrics.get('winning_trades', 0)}")
            report.append(f"❌ Losing Trades: {risk_metrics.get('losing_trades', 0)}")
            report.append("")
            
            # Risk Metrics
            report.append("⚠️ RISK METRICS")
            report.append("-" * 30)
            report.append(f"📊 Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}")
            report.append(f"📉 Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2f}%")
            report.append(f"🎯 Profit Factor: {risk_metrics.get('profit_factor', 0):.2f}")
            report.append(f"📊 Avg Daily P&L: ₹{risk_metrics.get('avg_daily_pnl', 0):,.2f}")
            report.append(f"📈 Daily P&L Std: ₹{risk_metrics.get('std_daily_pnl', 0):,.2f}")
            report.append(f"⏱️ Avg Trade Duration: {risk_metrics.get('avg_trade_duration', 0):.1f} minutes")
            report.append("")
            
            # Costs
            report.append("💸 TRADING COSTS")
            report.append("-" * 30)
            report.append(f"💳 Total Commission: ₹{risk_metrics.get('total_commission', 0):,.2f}")
            report.append(f"📊 Total Slippage: ₹{risk_metrics.get('total_slippage', 0):,.2f}")
            report.append("")
            
            # Strategy Performance
            if not strategy_perf.empty:
                report.append("🎯 STRATEGY PERFORMANCE")
                report.append("-" * 30)
                for _, row in strategy_perf.iterrows():
                    report.append(f"📊 {row['strategy']}:")
                    report.append(f"   P&L: ₹{row['net_pnl']:,.2f} | Win Rate: {row['win_rate']:.1f}% | Trades: {row['total_trades']}")
                    report.append(f"   Avg P&L: ₹{row['avg_pnl']:,.2f} | Profit Factor: {row['profit_factor']:.2f}")
                    report.append("")
            
            # Daily Summary
            if len(daily_pnl) <= 10:
                report.append("📅 DAILY SUMMARY")
                report.append("-" * 30)
                for _, row in daily_pnl.iterrows():
                    report.append(f"📅 {row['date']}: ₹{row['net_pnl']:,.2f} ({row['total_trades']} trades, {row['win_rate']:.1f}% win rate)")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            return f"Error generating report: {e}"
    
    def plot_equity_curve(self, start_date: str = None, end_date: str = None, save_path: str = None):
        """Plot equity curve"""
        try:
            equity_data = self.get_equity_curve(start_date, end_date)
            
            if equity_data.empty:
                logger.warning("No equity curve data available")
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            ax1.plot(equity_data.index, equity_data['equity'], label='Total Equity', linewidth=2)
            ax1.plot(equity_data.index, equity_data['cash'], label='Cash', alpha=0.7)
            ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Equity (₹)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            peak = equity_data['equity'].expanding().max()
            drawdown = (equity_data['equity'] - peak) / peak * 100
            ax2.fill_between(equity_data.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
            ax2.plot(equity_data.index, drawdown, color='red', linewidth=1)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Equity curve saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Error plotting equity curve: {e}")
    
    def plot_daily_pnl(self, start_date: str = None, end_date: str = None, save_path: str = None):
        """Plot daily P&L"""
        try:
            daily_pnl = self.get_daily_pnl(start_date, end_date)
            
            if daily_pnl.empty:
                logger.warning("No daily P&L data available")
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Daily P&L bars
            colors = ['green' if x > 0 else 'red' for x in daily_pnl['net_pnl']]
            ax1.bar(daily_pnl['date'], daily_pnl['net_pnl'], color=colors, alpha=0.7)
            ax1.set_title('Daily P&L', fontsize=14, fontweight='bold')
            ax1.set_ylabel('P&L (₹)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Cumulative P&L
            ax2.plot(daily_pnl['date'], daily_pnl['cumulative_net_pnl'], linewidth=2, color='blue')
            ax2.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Cumulative P&L (₹)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Daily P&L chart saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Error plotting daily P&L: {e}")


def main():
    """Main function to run performance analysis"""
    tracker = PerformanceTracker()
    
    # Get date range (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print("📊 PERFORMANCE TRACKER")
    print("=" * 50)
    
    # Generate report
    report = tracker.generate_performance_report(start_date, end_date)
    print(report)
    
    # Plot charts
    tracker.plot_equity_curve(start_date, end_date, "equity_curve.png")
    tracker.plot_daily_pnl(start_date, end_date, "daily_pnl.png")
    
    print("\n📊 Charts saved as:")
    print("   - equity_curve.png")
    print("   - daily_pnl.png")


if __name__ == "__main__":
    main() 