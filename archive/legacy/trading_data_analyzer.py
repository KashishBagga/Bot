#!/usr/bin/env python3
"""
Advanced Trading Data Analyzer
Analyzes trading signals database to identify profit/loss patterns and optimization opportunities.
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingDataAnalyzer:
    def __init__(self, db_path="trading_signals.db"):
        """Initialize the analyzer with database connection."""
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def get_table_names(self):
        """Get all table names in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql_query(query, self.connection)
        return tables['name'].tolist()
    
    def load_all_trading_data(self):
        """Load all trading data from all strategy tables."""
        tables = self.get_table_names()
        all_data = []
        
        print("📊 Loading trading data from all strategies...")
        
        for table in tables:
            try:
                # Check if table has the basic required columns
                query = f"PRAGMA table_info({table})"
                columns_info = pd.read_sql_query(query, self.connection)
                columns = columns_info['name'].tolist()
                
                # Skip if doesn't look like a trading table
                required_cols = ['signal_time', 'index_name', 'signal', 'price']
                if not all(col in columns for col in required_cols):
                    continue
                
                # Load data from this table
                query = f"SELECT * FROM {table} WHERE pnl IS NOT NULL"
                df = pd.read_sql_query(query, self.connection)
                
                if not df.empty:
                    df['strategy'] = table
                    df['signal_time'] = pd.to_datetime(df['signal_time'])
                    all_data.append(df)
                    print(f"  ✅ {table}: {len(df)} trades loaded")
                
            except Exception as e:
                print(f"  ⚠️ Error loading {table}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            print(f"\n📈 Total trades loaded: {len(combined_df)} from {len(all_data)} strategies")
            return combined_df
        else:
            print("❌ No trading data found!")
            return pd.DataFrame()
    
    def basic_statistics(self, df):
        """Generate basic trading statistics."""
        print("="*80)
        print("📊 BASIC TRADING STATISTICS")
        print("="*80)
        
        # Overall stats
        total_trades = len(df)
        profitable_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        breakeven_trades = len(df[df['pnl'] == 0])
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_profit = df[df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        print(f"📈 Total Trades: {total_trades:,}")
        print(f"✅ Profitable: {profitable_trades:,} ({win_rate:.1f}%)")
        print(f"❌ Losing: {losing_trades:,} ({losing_trades/total_trades*100:.1f}%)")
        print(f"🔄 Breakeven: {breakeven_trades:,} ({breakeven_trades/total_trades*100:.1f}%)")
        print()
        print(f"💰 Total P&L: ₹{total_pnl:,.2f}")
        print(f"📊 Average Profit: ₹{avg_profit:.2f}")
        print(f"📊 Average Loss: ₹{avg_loss:.2f}")
        print(f"📊 Profit Factor: {abs(avg_profit/avg_loss):.2f}" if avg_loss != 0 else "N/A")
        
        # Best and worst trades
        best_trade = df.loc[df['pnl'].idxmax()] if not df.empty else None
        worst_trade = df.loc[df['pnl'].idxmin()] if not df.empty else None
        
        if best_trade is not None:
            print(f"\n🏆 Best Trade: ₹{best_trade['pnl']:.2f} ({best_trade['strategy']} - {best_trade['index_name']})")
        if worst_trade is not None:
            print(f"💸 Worst Trade: ₹{worst_trade['pnl']:.2f} ({worst_trade['strategy']} - {worst_trade['index_name']})")
    
    def strategy_performance_analysis(self, df):
        """Analyze performance by strategy."""
        print("\n" + "="*80)
        print("🧠 STRATEGY PERFORMANCE ANALYSIS")
        print("="*80)
        
        strategy_stats = df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean', 'std'],
            'signal': lambda x: (x != 'NO TRADE').sum() if 'NO TRADE' in x.values else len(x)
        }).round(2)
        
        strategy_stats.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Std_PnL', 'Signals']
        
        # Calculate win rates
        win_rates = df.groupby('strategy').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(1)
        strategy_stats['Win_Rate'] = win_rates
        
        # Sort by total PnL
        strategy_stats = strategy_stats.sort_values('Total_PnL', ascending=False)
        
        print("Strategy Performance Ranking:")
        print("-" * 100)
        print(f"{'Strategy':<25} {'Trades':<8} {'Total P&L':<12} {'Avg P&L':<10} {'Win Rate':<10} {'Std Dev':<10}")
        print("-" * 100)
        
        for strategy, row in strategy_stats.iterrows():
            print(f"{strategy:<25} {int(row['Trades']):<8} ₹{row['Total_PnL']:<11.2f} ₹{row['Avg_PnL']:<9.2f} {row['Win_Rate']:<9.1f}% ₹{row['Std_PnL']:<9.2f}")
        
        return strategy_stats
    
    def symbol_analysis(self, df):
        """Analyze performance by symbol/index."""
        print("\n" + "="*80)
        print("📈 SYMBOL/INDEX ANALYSIS")
        print("="*80)
        
        symbol_stats = df.groupby('index_name').agg({
            'pnl': ['count', 'sum', 'mean'],
            'signal': lambda x: len(x)
        }).round(2)
        
        symbol_stats.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Signals']
        
        # Calculate win rates
        win_rates = df.groupby('index_name').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(1)
        symbol_stats['Win_Rate'] = win_rates
        
        symbol_stats = symbol_stats.sort_values('Total_PnL', ascending=False)
        
        print("Symbol Performance:")
        print("-" * 70)
        print(f"{'Symbol':<15} {'Trades':<8} {'Total P&L':<12} {'Avg P&L':<10} {'Win Rate':<10}")
        print("-" * 70)
        
        for symbol, row in symbol_stats.iterrows():
            print(f"{symbol:<15} {int(row['Trades']):<8} ₹{row['Total_PnL']:<11.2f} ₹{row['Avg_PnL']:<9.2f} {row['Win_Rate']:<9.1f}%")
        
        return symbol_stats
    
    def time_based_analysis(self, df):
        """Analyze performance by time patterns."""
        print("\n" + "="*80)
        print("⏰ TIME-BASED ANALYSIS")
        print("="*80)
        
        df['hour'] = df['signal_time'].dt.hour
        df['day_of_week'] = df['signal_time'].dt.day_name()
        df['month'] = df['signal_time'].dt.month_name()
        
        # Hour analysis
        print("\n🕐 Performance by Hour:")
        hour_stats = df.groupby('hour').agg({
            'pnl': ['count', 'mean', 'sum']
        }).round(2)
        hour_stats.columns = ['Trades', 'Avg_PnL', 'Total_PnL']
        hour_stats['Win_Rate'] = df.groupby('hour').apply(lambda x: (x['pnl'] > 0).sum() / len(x) * 100).round(1)
        
        best_hours = hour_stats.sort_values('Avg_PnL', ascending=False).head(3)
        worst_hours = hour_stats.sort_values('Avg_PnL', ascending=True).head(3)
        
        print("Best Trading Hours:")
        for hour, row in best_hours.iterrows():
            print(f"  {hour:02d}:00 - Avg P&L: ₹{row['Avg_PnL']:.2f}, Win Rate: {row['Win_Rate']:.1f}%, Trades: {int(row['Trades'])}")
        
        print("\nWorst Trading Hours:")
        for hour, row in worst_hours.iterrows():
            print(f"  {hour:02d}:00 - Avg P&L: ₹{row['Avg_PnL']:.2f}, Win Rate: {row['Win_Rate']:.1f}%, Trades: {int(row['Trades'])}")
        
        # Day of week analysis
        print("\n📅 Performance by Day of Week:")
        dow_stats = df.groupby('day_of_week').agg({
            'pnl': ['count', 'mean', 'sum']
        }).round(2)
        dow_stats.columns = ['Trades', 'Avg_PnL', 'Total_PnL']
        dow_stats['Win_Rate'] = df.groupby('day_of_week').apply(lambda x: (x['pnl'] > 0).sum() / len(x) * 100).round(1)
        
        # Reorder by weekday
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats = dow_stats.reindex([day for day in day_order if day in dow_stats.index])
        
        for day, row in dow_stats.iterrows():
            print(f"  {day:<10} - Avg P&L: ₹{row['Avg_PnL']:>8.2f}, Win Rate: {row['Win_Rate']:>5.1f}%, Trades: {int(row['Trades']):>4}")
        
        return hour_stats, dow_stats
    
    def confidence_analysis(self, df):
        """Analyze performance by confidence levels."""
        print("\n" + "="*80)
        print("🎯 CONFIDENCE LEVEL ANALYSIS")
        print("="*80)
        
        if 'confidence' in df.columns:
            conf_stats = df.groupby('confidence').agg({
                'pnl': ['count', 'mean', 'sum']
            }).round(2)
            conf_stats.columns = ['Trades', 'Avg_PnL', 'Total_PnL']
            conf_stats['Win_Rate'] = df.groupby('confidence').apply(lambda x: (x['pnl'] > 0).sum() / len(x) * 100).round(1)
            
            print("Performance by Confidence Level:")
            print("-" * 60)
            print(f"{'Confidence':<12} {'Trades':<8} {'Avg P&L':<10} {'Win Rate':<10}")
            print("-" * 60)
            
            for conf, row in conf_stats.iterrows():
                print(f"{conf:<12} {int(row['Trades']):<8} ₹{row['Avg_PnL']:<9.2f} {row['Win_Rate']:<9.1f}%")
            
            return conf_stats
        else:
            print("No confidence data available")
            return None
    
    def signal_type_analysis(self, df):
        """Analyze performance by signal type (CALL/PUT)."""
        print("\n" + "="*80)
        print("📊 SIGNAL TYPE ANALYSIS")
        print("="*80)
        
        signal_stats = df.groupby('signal').agg({
            'pnl': ['count', 'mean', 'sum']
        }).round(2)
        signal_stats.columns = ['Trades', 'Avg_PnL', 'Total_PnL']
        signal_stats['Win_Rate'] = df.groupby('signal').apply(lambda x: (x['pnl'] > 0).sum() / len(x) * 100).round(1)
        
        signal_stats = signal_stats.sort_values('Avg_PnL', ascending=False)
        
        print("Performance by Signal Type:")
        print("-" * 55)
        print(f"{'Signal':<8} {'Trades':<8} {'Avg P&L':<10} {'Win Rate':<10}")
        print("-" * 55)
        
        for signal, row in signal_stats.iterrows():
            if signal != 'NO TRADE':
                print(f"{signal:<8} {int(row['Trades']):<8} ₹{row['Avg_PnL']:<9.2f} {row['Win_Rate']:<9.1f}%")
        
        return signal_stats
    
    def risk_analysis(self, df):
        """Analyze risk metrics and patterns."""
        print("\n" + "="*80)
        print("⚠️ RISK ANALYSIS")
        print("="*80)
        
        # Calculate risk metrics
        returns = df['pnl']
        
        # Drawdown analysis
        cumulative_pnl = returns.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Consecutive losses
        df['is_loss'] = df['pnl'] < 0
        df['loss_streak'] = df.groupby((~df['is_loss']).cumsum())['is_loss'].cumsum()
        max_consecutive_losses = df['loss_streak'].max()
        
        # Risk metrics
        volatility = returns.std()
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Sharpe-like ratio (using 0 as risk-free rate)
        sharpe_ratio = returns.mean() / volatility if volatility != 0 else 0
        
        print(f"📉 Maximum Drawdown: ₹{max_drawdown:.2f}")
        print(f"🔴 Maximum Consecutive Losses: {max_consecutive_losses}")
        print(f"📊 Volatility (Std Dev): ₹{volatility:.2f}")
        print(f"📊 Downside Volatility: ₹{downside_volatility:.2f}")
        print(f"📊 Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Large loss analysis
        large_losses = df[df['pnl'] < -1000]  # Losses > ₹1000
        if not large_losses.empty:
            print(f"\n💸 Large Losses (>₹1000): {len(large_losses)} trades")
            print("Top 5 largest losses:")
            worst_losses = large_losses.nlargest(5, 'pnl', keep='all')[['signal_time', 'strategy', 'index_name', 'signal', 'pnl']]
            for _, trade in worst_losses.iterrows():
                print(f"  {trade['signal_time'].strftime('%Y-%m-%d %H:%M')} - {trade['strategy']} - {trade['index_name']} - {trade['signal']} - ₹{trade['pnl']:.2f}")
        
        return {
            'max_drawdown': max_drawdown,
            'max_consecutive_losses': max_consecutive_losses,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def indicator_analysis(self, df):
        """Analyze performance based on technical indicators."""
        print("\n" + "="*80)
        print("📈 TECHNICAL INDICATOR ANALYSIS")
        print("="*80)
        
        # RSI analysis
        if 'rsi' in df.columns:
            df_rsi = df[df['rsi'].notna()].copy()
            if not df_rsi.empty:
                print("\n📊 RSI Analysis:")
                
                # Create RSI bins
                df_rsi['rsi_bin'] = pd.cut(df_rsi['rsi'], 
                                         bins=[0, 30, 50, 70, 100], 
                                         labels=['Oversold (<30)', 'Neutral (30-50)', 'Neutral (50-70)', 'Overbought (>70)'])
                
                rsi_stats = df_rsi.groupby('rsi_bin').agg({
                    'pnl': ['count', 'mean']
                }).round(2)
                rsi_stats.columns = ['Trades', 'Avg_PnL']
                rsi_stats['Win_Rate'] = df_rsi.groupby('rsi_bin').apply(lambda x: (x['pnl'] > 0).sum() / len(x) * 100).round(1)
                
                for rsi_level, row in rsi_stats.iterrows():
                    print(f"  {rsi_level:<20} - Trades: {int(row['Trades']):>4}, Avg P&L: ₹{row['Avg_PnL']:>8.2f}, Win Rate: {row['Win_Rate']:>5.1f}%")
        
        # Price analysis (if available)
        if 'price' in df.columns:
            df_price = df[df['price'].notna()].copy()
            if not df_price.empty:
                print("\n💰 Price Level Analysis:")
                
                # Create price bins (quartiles)
                df_price['price_quartile'] = pd.qcut(df_price['price'], 
                                                   q=4, 
                                                   labels=['Q1 (Low)', 'Q2 (Med-Low)', 'Q3 (Med-High)', 'Q4 (High)'])
                
                price_stats = df_price.groupby('price_quartile').agg({
                    'pnl': ['count', 'mean']
                }).round(2)
                price_stats.columns = ['Trades', 'Avg_PnL']
                price_stats['Win_Rate'] = df_price.groupby('price_quartile').apply(lambda x: (x['pnl'] > 0).sum() / len(x) * 100).round(1)
                
                for price_level, row in price_stats.iterrows():
                    print(f"  {price_level:<15} - Trades: {int(row['Trades']):>4}, Avg P&L: ₹{row['Avg_PnL']:>8.2f}, Win Rate: {row['Win_Rate']:>5.1f}%")
    
    def generate_recommendations(self, df, strategy_stats, symbol_stats, hour_stats=None, conf_stats=None):
        """Generate actionable recommendations based on analysis."""
        print("\n" + "="*80)
        print("💡 ACTIONABLE RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        # Strategy recommendations
        best_strategy = strategy_stats.index[0]
        worst_strategy = strategy_stats.index[-1]
        
        if strategy_stats.loc[best_strategy, 'Total_PnL'] > 0:
            recommendations.append(f"🎯 Focus on '{best_strategy}' strategy - it's your most profitable with ₹{strategy_stats.loc[best_strategy, 'Total_PnL']:.2f} total P&L")
        
        if strategy_stats.loc[worst_strategy, 'Total_PnL'] < -500:
            recommendations.append(f"⚠️ Consider disabling '{worst_strategy}' strategy - it's causing significant losses (₹{strategy_stats.loc[worst_strategy, 'Total_PnL']:.2f})")
        
        # Symbol recommendations
        best_symbol = symbol_stats.index[0]
        worst_symbol = symbol_stats.index[-1]
        
        if symbol_stats.loc[best_symbol, 'Total_PnL'] > symbol_stats.loc[worst_symbol, 'Total_PnL'] * 2:
            recommendations.append(f"📈 Prioritize trading {best_symbol} over {worst_symbol} - significantly better performance")
        
        # Confidence recommendations
        if conf_stats is not None:
            high_conf_avg = conf_stats.loc[conf_stats.index.str.contains('HIGH', case=False, na=False), 'Avg_PnL'].mean() if any(conf_stats.index.str.contains('HIGH', case=False, na=False)) else 0
            low_conf_avg = conf_stats.loc[conf_stats.index.str.contains('LOW', case=False, na=False), 'Avg_PnL'].mean() if any(conf_stats.index.str.contains('LOW', case=False, na=False)) else 0
            
            if high_conf_avg > low_conf_avg + 50:  # At least ₹50 difference
                recommendations.append(f"🎯 Trade only HIGH confidence signals - they perform ₹{high_conf_avg - low_conf_avg:.2f} better on average")
        
        # Time-based recommendations
        if hour_stats is not None:
            best_hours = hour_stats.nlargest(3, 'Avg_PnL')
            worst_hours = hour_stats.nsmallest(3, 'Avg_PnL')
            
            if best_hours['Avg_PnL'].mean() > 0 and worst_hours['Avg_PnL'].mean() < -20:
                best_hour_list = [f"{h:02d}:00" for h in best_hours.index]
                worst_hour_list = [f"{h:02d}:00" for h in worst_hours.index]
                recommendations.append(f"⏰ Trade during {', '.join(best_hour_list)} and avoid {', '.join(worst_hour_list)}")
        
        # Risk management recommendations
        large_losses = df[df['pnl'] < -1000]
        if len(large_losses) > len(df) * 0.05:  # More than 5% of trades are large losses
            recommendations.append(f"⚠️ Implement stricter stop-losses - {len(large_losses)} trades lost >₹1000 each")
        
        # Win rate recommendations
        overall_win_rate = (df['pnl'] > 0).sum() / len(df) * 100
        if overall_win_rate < 40:
            recommendations.append(f"📊 Win rate is only {overall_win_rate:.1f}% - consider tightening entry criteria")
        
        # Print recommendations
        print("🚀 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:8], 1):  # Top 8 recommendations
            print(f"  {i}. {rec}")
        
        if not recommendations:
            print("✅ Your trading system appears to be well-balanced. Continue monitoring performance.")
        
        return recommendations
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("🚀 Starting Comprehensive Trading Data Analysis...")
        print("="*80)
        
        # Load data
        df = self.load_all_trading_data()
        if df.empty:
            print("❌ No trading data found. Make sure you have run some backtests first.")
            return
        
        # Run all analyses
        self.basic_statistics(df)
        strategy_stats = self.strategy_performance_analysis(df)
        symbol_stats = self.symbol_analysis(df)
        hour_stats, dow_stats = self.time_based_analysis(df)
        conf_stats = self.confidence_analysis(df)
        self.signal_type_analysis(df)
        self.risk_analysis(df)
        self.indicator_analysis(df)
        
        # Generate recommendations
        self.generate_recommendations(df, strategy_stats, symbol_stats, hour_stats, conf_stats)
        
        print("\n" + "="*80)
        print("✅ Analysis Complete!")
        print("="*80)
        
        self.connection.close()

def main():
    """Main execution function."""
    analyzer = TradingDataAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 