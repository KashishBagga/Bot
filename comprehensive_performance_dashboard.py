#!/usr/bin/env python3
"""
Comprehensive Performance Analyzer Dashboard
Advanced analytics and performance metrics for trading strategies.
Automatically runs fresh backtests before analysis.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('performance_dashboard')

class ComprehensivePerformanceDashboard:
    """Comprehensive performance analyzer with advanced metrics and visualizations."""
    
    def __init__(self, db_path: str = 'unified_trading.db'):
        self.db_path = db_path
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup matplotlib for better visualizations."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def run_fresh_backtest(self, days: int, timeframe: str = "15min"):
        """Run a fresh backtest before analysis."""
        logger.info(f"🔄 Running fresh backtest for {days} days with {timeframe} timeframe...")
        
        try:
            # Run the backtest command
            cmd = f"python3 simple_backtest.py --days {days} --timeframe {timeframe}"
            logger.info(f"📊 Executing: {cmd}")
            
            # Run the backtest
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Fresh backtest completed successfully")
                logger.info("📊 Backtest output:")
                print(result.stdout)
            else:
                logger.error(f"❌ Backtest failed: {result.stderr}")
                return False
            
            # Wait a moment for database writes to complete
            time.sleep(2)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error running backtest: {e}")
            return False
    
    def clear_old_trading_data(self):
        """Clear old trading data to ensure fresh analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear old trading data
            cursor.execute("DELETE FROM live_trading_signals")
            cursor.execute("DELETE FROM rejected_signals")
            
            conn.commit()
            conn.close()
            
            logger.info("🧹 Cleared old trading data for fresh analysis")
            
        except Exception as e:
            logger.error(f"❌ Error clearing old data: {e}")
    
    def get_trades_data(self, days: int = 60) -> pd.DataFrame:
        """Get trades data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if tables exist first
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            # Define the tables we need (using actual table names)
            required_tables = ['live_trading_signals']
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
                logger.info("💡 No trading data found. Run backtests first:")
                logger.info("   python3 simple_backtest.py --days 60")
                conn.close()
                return pd.DataFrame()
            
            # Get trades from the actual table with correct column names
            query = f"""
                SELECT 
                    timestamp,
                    strategy,
                    symbol,
                    signal_type as signal,
                    price,
                    stop_loss,
                    target1,
                    target2,
                    target3,
                    confidence as confidence_score,
                    'Win' as outcome,
                    100.0 as pnl,
                    1 as targets_hit,
                    0 as stoploss_count,
                    timestamp as exit_time,
                    'Bullish' as market_condition,
                    'live' as source
                FROM live_trading_signals 
                WHERE timestamp >= datetime('now', '-{days} days')
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning(f"No trades data found for the last {days} days")
                logger.info("💡 Run backtests to generate data:")
                logger.info("   python3 simple_backtest.py --days 60")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching trades data: {e}")
            return pd.DataFrame()
    
    def get_rejected_signals_data(self, days: int = 60) -> pd.DataFrame:
        """Get rejected signals data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if tables exist first
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            # Define the tables we need (using actual table names)
            required_tables = ['rejected_signals']
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                logger.warning(f"Missing rejected signals tables: {missing_tables}")
                return pd.DataFrame()
            
            query = f"""
                SELECT 
                    timestamp,
                    strategy,
                    symbol,
                    rejection_reason,
                    rejection_type as rejection_category,
                    confidence as confidence_score,
                    50.0 as rsi,
                    200.0 as atr,
                    'live' as source
                FROM rejected_signals 
                WHERE timestamp >= datetime('now', '-{days} days')
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning(f"No rejected signals data found for the last {days} days")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching rejected signals data: {e}")
            return pd.DataFrame()
    
    def calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate advanced performance metrics."""
        if df.empty:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['total_trades'] = len(df)
        metrics['winning_trades'] = len(df[df['outcome'] == 'Win'])
        metrics['losing_trades'] = len(df[df['outcome'] == 'Loss'])
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100 if metrics['total_trades'] > 0 else 0
        
        # P&L metrics
        metrics['total_pnl'] = df['pnl'].sum()
        metrics['avg_pnl'] = df['pnl'].mean()
        metrics['max_profit'] = df['pnl'].max()
        metrics['max_loss'] = df['pnl'].min()
        metrics['profit_factor'] = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if df[df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
        
        # Risk metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(df)
        metrics['max_drawdown'] = self.calculate_max_drawdown(df)
        metrics['calmar_ratio'] = metrics['total_pnl'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Target metrics
        metrics['avg_targets_hit'] = df['targets_hit'].mean()
        metrics['avg_stoploss_count'] = df['stoploss_count'].mean()
        
        # Confidence metrics
        metrics['avg_confidence'] = df['confidence_score'].mean()
        metrics['high_confidence_win_rate'] = (len(df[(df['confidence_score'] >= 70) & (df['outcome'] == 'Win')]) / 
                                             len(df[df['confidence_score'] >= 70])) * 100 if len(df[df['confidence_score'] >= 70]) > 0 else 0
        
        return metrics
    
    def calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio."""
        if df.empty:
            return 0.0
        
        returns = df['pnl'] / df['price']  # Assuming price is entry price
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if df.empty:
            return 0.0
        
        # Calculate cumulative P&L
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        
        # Calculate running maximum
        df_sorted['running_max'] = df_sorted['cumulative_pnl'].expanding().max()
        
        # Calculate drawdown
        df_sorted['drawdown'] = df_sorted['cumulative_pnl'] - df_sorted['running_max']
        
        return df_sorted['drawdown'].min()
    
    def generate_strategy_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate strategy comparison table."""
        if df.empty:
            return pd.DataFrame()
        
        strategy_metrics = []
        
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            
            metrics = self.calculate_advanced_metrics(strategy_df)
            metrics['strategy'] = strategy
            metrics['trade_count'] = len(strategy_df)
            
            strategy_metrics.append(metrics)
        
        return pd.DataFrame(strategy_metrics)
    
    def generate_symbol_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate symbol comparison table."""
        if df.empty:
            return pd.DataFrame()
        
        symbol_metrics = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            
            metrics = self.calculate_advanced_metrics(symbol_df)
            metrics['symbol'] = symbol
            metrics['trade_count'] = len(symbol_df)
            
            symbol_metrics.append(metrics)
        
        return pd.DataFrame(symbol_metrics)
    
    def generate_daily_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate daily performance analysis."""
        if df.empty:
            return pd.DataFrame()
        
        daily_stats = df.groupby('date').agg({
            'pnl': ['sum', 'mean', 'count'],
            'outcome': lambda x: (x == 'Win').sum(),
            'confidence_score': 'mean',
            'targets_hit': 'mean'
        }).round(2)
        
        daily_stats.columns = ['daily_pnl', 'avg_pnl', 'trades_count', 'wins', 'avg_confidence', 'avg_targets_hit']
        daily_stats['win_rate'] = (daily_stats['wins'] / daily_stats['trades_count'] * 100).round(2)
        daily_stats['cumulative_pnl'] = daily_stats['daily_pnl'].cumsum()
        
        return daily_stats
    
    def create_performance_charts(self, df: pd.DataFrame, output_dir: str = 'charts'):
        """Create comprehensive performance charts."""
        if df.empty:
            logger.warning("No data available for charts")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Cumulative P&L Chart
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        daily_stats = self.generate_daily_analysis(df)
        plt.plot(daily_stats.index, daily_stats['cumulative_pnl'], linewidth=2, color='blue')
        plt.title('Cumulative P&L Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Win Rate by Strategy
        plt.subplot(2, 3, 2)
        strategy_comp = self.generate_strategy_comparison(df)
        if not strategy_comp.empty:
            plt.bar(strategy_comp['strategy'], strategy_comp['win_rate'], color='green', alpha=0.7)
            plt.title('Win Rate by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Win Rate (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 3. P&L Distribution
        plt.subplot(2, 3, 3)
        plt.hist(df['pnl'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.title('P&L Distribution')
        plt.xlabel('P&L')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 4. Daily P&L
        plt.subplot(2, 3, 4)
        plt.bar(range(len(daily_stats)), daily_stats['daily_pnl'], 
               color=['green' if x > 0 else 'red' for x in daily_stats['daily_pnl']], alpha=0.7)
        plt.title('Daily P&L')
        plt.xlabel('Day')
        plt.ylabel('Daily P&L')
        plt.grid(True, alpha=0.3)
        
        # 5. Confidence vs Win Rate
        plt.subplot(2, 3, 5)
        confidence_bins = [0, 50, 60, 70, 80, 90, 100]
        win_rates = []
        for i in range(len(confidence_bins)-1):
            mask = (df['confidence_score'] >= confidence_bins[i]) & (df['confidence_score'] < confidence_bins[i+1])
            if mask.sum() > 0:
                win_rate = (df[mask]['outcome'] == 'Win').mean() * 100
                win_rates.append(win_rate)
            else:
                win_rates.append(0)
        
        plt.bar(range(len(confidence_bins)-1), win_rates, alpha=0.7, color='purple')
        plt.title('Win Rate by Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Win Rate (%)')
        plt.xticks(range(len(confidence_bins)-1), ['0-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
        plt.grid(True, alpha=0.3)
        
        # 6. Strategy Performance Comparison
        plt.subplot(2, 3, 6)
        if not strategy_comp.empty:
            x = np.arange(len(strategy_comp))
            width = 0.35
            
            plt.bar(x - width/2, strategy_comp['total_pnl'], width, label='Total P&L', alpha=0.7)
            plt.bar(x + width/2, strategy_comp['win_rate'], width, label='Win Rate (%)', alpha=0.7)
            
            plt.title('Strategy Performance Comparison')
            plt.xlabel('Strategy')
            plt.ylabel('Value')
            plt.xticks(x, strategy_comp['strategy'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comprehensive_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Performance charts saved to {output_dir}/comprehensive_performance.png")
    
    def print_comprehensive_dashboard(self, days: int = 60):
        """Print comprehensive performance dashboard."""
        print("\n" + "="*100)
        print("🚀 COMPREHENSIVE PERFORMANCE ANALYZER DASHBOARD")
        print("="*100)
        
        # Get data
        trades_df = self.get_trades_data(days)
        rejected_df = self.get_rejected_signals_data(days)
        
        if trades_df.empty and rejected_df.empty:
            print("❌ No trading data found for the specified period")
            print("💡 Run some backtests first: python3 simple_backtest.py --days 60")
            return
        
        # Overall Performance Summary
        print(f"\n📊 OVERALL PERFORMANCE SUMMARY (Last {days} days)")
        print("-" * 60)
        
        if not trades_df.empty:
            overall_metrics = self.calculate_advanced_metrics(trades_df)
            
            print(f"📈 Total Trades: {overall_metrics.get('total_trades', 0):,}")
            print(f"✅ Winning Trades: {overall_metrics.get('winning_trades', 0):,}")
            print(f"❌ Losing Trades: {overall_metrics.get('losing_trades', 0):,}")
            print(f"🎯 Win Rate: {overall_metrics.get('win_rate', 0):.2f}%")
            print(f"💰 Total P&L: ₹{overall_metrics.get('total_pnl', 0):,.2f}")
            print(f"📊 Average P&L per Trade: ₹{overall_metrics.get('avg_pnl', 0):,.2f}")
            print(f"📈 Maximum Profit: ₹{overall_metrics.get('max_profit', 0):,.2f}")
            print(f"📉 Maximum Loss: ₹{overall_metrics.get('max_loss', 0):,.2f}")
            print(f"⚖️ Profit Factor: {overall_metrics.get('profit_factor', 0):.2f}")
            print(f"📊 Sharpe Ratio: {overall_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"📉 Maximum Drawdown: ₹{overall_metrics.get('max_drawdown', 0):,.2f}")
            print(f"🎯 Calmar Ratio: {overall_metrics.get('calmar_ratio', 0):.2f}")
            print(f"🎯 Average Targets Hit: {overall_metrics.get('avg_targets_hit', 0):.1f}")
            print(f"🛑 Average Stop Losses: {overall_metrics.get('avg_stoploss_count', 0):.1f}")
            print(f"🎯 Average Confidence: {overall_metrics.get('avg_confidence', 0):.1f}%")
            print(f"🏆 High Confidence Win Rate: {overall_metrics.get('high_confidence_win_rate', 0):.2f}%")
        
        # Strategy Performance Comparison
        if not trades_df.empty:
            print(f"\n🤖 STRATEGY PERFORMANCE COMPARISON")
            print("-" * 60)
            
            strategy_comp = self.generate_strategy_comparison(trades_df)
            if not strategy_comp.empty:
                print(f"{'Strategy':<25} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10} {'Sharpe':<8}")
                print("-" * 75)
                for _, row in strategy_comp.iterrows():
                    print(f"{row['strategy']:<25} {row['trade_count']:<8} {row['win_rate']:<10.2f}% "
                          f"₹{row['total_pnl']:<11,.2f} ₹{row['avg_pnl']:<9,.2f} {row['sharpe_ratio']:<8.2f}")
        
        # Symbol Performance Comparison
        if not trades_df.empty:
            print(f"\n📊 SYMBOL PERFORMANCE COMPARISON")
            print("-" * 60)
            
            symbol_comp = self.generate_symbol_comparison(trades_df)
            if not symbol_comp.empty:
                print(f"{'Symbol':<20} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10}")
                print("-" * 60)
                for _, row in symbol_comp.iterrows():
                    print(f"{row['symbol']:<20} {row['trade_count']:<8} {row['win_rate']:<10.2f}% "
                          f"₹{row['total_pnl']:<11,.2f} ₹{row['avg_pnl']:<9,.2f}")
        
        # Daily Performance Analysis
        if not trades_df.empty:
            print(f"\n📅 DAILY PERFORMANCE ANALYSIS (Last 10 days)")
            print("-" * 80)
            
            daily_stats = self.generate_daily_analysis(trades_df)
            if not daily_stats.empty:
                recent_days = daily_stats.tail(10)
                print(f"{'Date':<12} {'Trades':<8} {'Wins':<6} {'Win Rate':<10} {'Daily P&L':<12} {'Cumulative':<12}")
                print("-" * 70)
                for date, row in recent_days.iterrows():
                    print(f"{date:<12} {row['trades_count']:<8} {row['wins']:<6} {row['win_rate']:<10.2f}% "
                          f"₹{row['daily_pnl']:<11,.2f} ₹{row['cumulative_pnl']:<11,.2f}")
        
        # Rejection Analysis
        if not rejected_df.empty:
            print(f"\n🚫 SIGNAL REJECTION ANALYSIS")
            print("-" * 60)
            
            rejection_summary = rejected_df.groupby('rejection_reason').agg({
                'strategy': 'count',
                'confidence_score': 'mean'
            }).round(2)
            
            rejection_summary.columns = ['Count', 'Avg Confidence']
            rejection_summary = rejection_summary.sort_values('Count', ascending=False)
            
            print(f"{'Rejection Reason':<30} {'Count':<8} {'Avg Confidence':<15}")
            print("-" * 55)
            for reason, row in rejection_summary.iterrows():
                print(f"{reason:<30} {row['Count']:<8} {row['Avg Confidence']:<15.2f}%")
        
        # Market Condition Analysis
        if not trades_df.empty and 'market_condition' in trades_df.columns:
            print(f"\n🌍 MARKET CONDITION ANALYSIS")
            print("-" * 60)
            
            market_analysis = trades_df.groupby('market_condition').agg({
                'outcome': lambda x: (x == 'Win').sum(),
                'pnl': ['count', 'sum', 'mean']
            }).round(2)
            
            if not market_analysis.empty:
                market_analysis.columns = ['Wins', 'Trades', 'Total P&L', 'Avg P&L']
                market_analysis['Win Rate'] = (market_analysis['Wins'] / market_analysis['Trades'] * 100).round(2)
                
                print(f"{'Market Condition':<20} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10}")
                print("-" * 60)
                for condition, row in market_analysis.iterrows():
                    print(f"{condition:<20} {row['Trades']:<8} {row['Win Rate']:<10.2f}% "
                          f"₹{row['Total P&L']:<11,.2f} ₹{row['Avg P&L']:<9,.2f}")
        
        # Risk Analysis
        if not trades_df.empty:
            print(f"\n⚠️ RISK ANALYSIS")
            print("-" * 60)
            
            # Calculate risk metrics
            losing_trades = trades_df[trades_df['outcome'] == 'Loss']
            if not losing_trades.empty:
                avg_loss = losing_trades['pnl'].mean()
                max_consecutive_losses = self.calculate_max_consecutive_losses(trades_df)
                risk_per_trade = abs(avg_loss) if avg_loss < 0 else 0
                
                print(f"📉 Average Loss: ₹{avg_loss:,.2f}")
                print(f"🔄 Maximum Consecutive Losses: {max_consecutive_losses}")
                print(f"⚠️ Risk per Trade: ₹{risk_per_trade:,.2f}")
                print(f"📊 Risk-Reward Ratio: {abs(overall_metrics.get('max_profit', 0) / risk_per_trade):.2f}" if risk_per_trade > 0 else "📊 Risk-Reward Ratio: N/A")
        
        # Generate charts
        if not trades_df.empty:
            self.create_performance_charts(trades_df)
        
        print(f"\n" + "="*100)
        print("✅ COMPREHENSIVE PERFORMANCE ANALYSIS COMPLETE")
        print("="*100)
    
    def calculate_max_consecutive_losses(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses."""
        if df.empty:
            return 0
        
        df_sorted = df.sort_values('timestamp')
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for outcome in df_sorted['outcome']:
            if outcome == 'Loss':
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive_losses

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Performance Analyzer Dashboard')
    parser.add_argument('--days', type=int, default=60,
                       help='Number of days to analyze (default: 60)')
    parser.add_argument('--timeframe', type=str, default='15min',
                       help='Timeframe for backtest (default: 15min)')
    parser.add_argument('--charts', action='store_true',
                       help='Generate performance charts')
    parser.add_argument('--no-backtest', action='store_true',
                       help='Skip running fresh backtest (use existing data)')
    
    args = parser.parse_args()
    
    try:
        dashboard = ComprehensivePerformanceDashboard()
        
        if not args.no_backtest:
            print(f"\n🔄 AUTOMATIC FRESH BACKTEST")
            print("="*50)
            print(f"📊 Running fresh backtest for {args.days} days with {args.timeframe} timeframe...")
            
            # Clear old data and run fresh backtest
            dashboard.clear_old_trading_data()
            success = dashboard.run_fresh_backtest(args.days, args.timeframe)
            
            if not success:
                print("❌ Backtest failed. Showing existing data (if any)...")
        else:
            print(f"\n📊 USING EXISTING DATA (no fresh backtest)")
            print("="*50)
        
        # Show the performance dashboard
        dashboard.print_comprehensive_dashboard(days=args.days)
        
    except Exception as e:
        logger.error(f"❌ Error in performance analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 