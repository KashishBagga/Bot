#!/usr/bin/env python3
"""
Simple Backtesting System
A reliable and working backtesting solution using local parquet data
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.local_data_loader import LocalDataLoader
from src.models.unified_database import UnifiedDatabase
from src.core.indicators import add_technical_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleBacktester:
    def __init__(self):
        """Initialize the simple backtester"""
        self.base_dir = Path("historical_data_20yr")
        self.db = UnifiedDatabase()
        
        logger.info("🚀 Simple Backtester Initialized")
        logger.info(f"📁 Data Directory: {self.base_dir}")
    
    def load_data(self, symbol, timeframe, days=30):
        """Load data from parquet files with optimization"""
        try:
            symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
            parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
            
            if not parquet_file.exists():
                logger.error(f"❌ Data file not found: {parquet_file}")
                return None
            
            # Load only required columns for better performance
            df = pd.read_parquet(parquet_file, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if df.empty:
                logger.error(f"❌ Empty data file: {parquet_file}")
                return None
            
            # Filter for recent days
            if days:
                end_date = df['timestamp'].max()
                start_date = end_date - timedelta(days=days)
                df = df[df['timestamp'] >= start_date]
            
            # Limit data size for faster processing (max 2000 candles)
            if len(df) > 2000:
                df = df.tail(2000)
                logger.info(f"📊 Limited to last 2000 candles for faster processing")
            
            logger.info(f"✅ Loaded {len(df):,} candles for {symbol} {timeframe}")
            logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def add_indicators(self, df):
        """Add technical indicators to the data"""
        try:
            # Use the unified indicators function
            df = add_technical_indicators(df)
            
            # Add market condition analysis
            from src.core.market_conditions import analyze_market_conditions
            df = analyze_market_conditions(df)
            
            # Debug: Show market condition distribution
            if 'market_condition' in df.columns:
                condition_counts = df['market_condition'].value_counts()
                logger.info(f"📊 Market Condition Distribution: {dict(condition_counts)}")
                
                tradeable_count = df['market_tradeable'].sum() if 'market_tradeable' in df.columns else 0
                logger.info(f"📊 Tradeable Candles: {tradeable_count}/{len(df)} ({tradeable_count/len(df)*100:.1f}%)")
            
            logger.info("✅ Indicators and market conditions added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding indicators: {e}")
            return df
    
    def run_enhanced_strategies(self, df, symbol):
        """Run all enhanced strategies with new implementations"""
        signals = []
        
        # Import enhanced original strategies
        from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
        from src.strategies.supertrend_ema import SupertrendEma
        from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma

        
        # Initialize enhanced original strategies
        strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }
        
        logger.info(f"🧠 Running {len(strategies)} enhanced strategies...")
        
        # Pre-calculate indicators ONCE for the entire dataset with optimization
        logger.info("📊 Pre-calculating indicators...")
        df_with_indicators = add_technical_indicators(df)
        
        # Cache frequently used calculations
        df_with_indicators['timestamp'] = pd.to_datetime(df_with_indicators['timestamp'])
        logger.info(f"📊 Indicators calculated for {len(df_with_indicators)} candles")
        
        # Test each strategy with optimized processing
        for strategy_name, strategy in strategies.items():
            logger.info(f"Testing {strategy_name}...")
            strategy_signals = 0
            
            # Process in chunks for better performance
            chunk_size = 100
            for i in range(200, len(df_with_indicators), chunk_size):
                chunk_end = min(i + chunk_size, len(df_with_indicators))
                
                for j in range(i, chunk_end):
                    try:
                        # Get data slice for analysis
                        data_slice = df_with_indicators.iloc[:j+1]
                        
                        # Analyze with current strategy
                        result = strategy.analyze(data_slice)
                        
                        if result and result.get('signal') != 'NO TRADE':
                            signal = {
                                'timestamp': data_slice.iloc[-1]['timestamp'],
                                'strategy': strategy_name,
                                'signal': result['signal'],
                                'price': result.get('price', data_slice.iloc[-1]['close']),
                                'confidence': result.get('confidence_score', 0),
                                'reasoning': result.get('reasoning', '')[:100],
                                'stop_loss': result.get('stop_loss'),
                                'target1': result.get('target1'),
                                'target2': result.get('target2'),
                                'target3': result.get('target3'),
                                'position_multiplier': result.get('position_multiplier', 1.0)
                            }
                            signals.append(signal)
                            strategy_signals += 1
                            
                    except Exception as e:
                        continue
                
                # Progress update every chunk
                if i % 500 == 0:
                    logger.info(f"📊 {strategy_name}: Processed {i}/{len(df_with_indicators)} candles...")
            
            logger.info(f"📊 {strategy_name}: {strategy_signals} signals")
        
        return signals
    
    def simulate_trades(self, signals, df):
        """Simulate trades and calculate P&L with position sizing and multiple targets"""
        trades = []
        for signal in signals:
            try:
                signal_time = signal['timestamp']
                signal_price = signal['price']
                signal_type = signal['signal']
                
                # Get position multiplier from signal (default to 1.0 if not provided)
                position_multiplier = signal.get('position_multiplier', 1.0)
                
                # Use ATR-based stops and targets from enhanced strategies
                if signal.get('stop_loss') and signal.get('target1'):
                    stop_loss_pct = signal['stop_loss'] / signal_price
                    target_pct = signal['target1'] / signal_price
                    target2_pct = signal.get('target2', signal['target1']) / signal_price
                    target3_pct = signal.get('target3', signal['target2'] if signal.get('target2') else signal['target1']) / signal_price
                else:
                    # Default percentage-based values
                    stop_loss_pct = 0.015  # 1.5% stop loss
                    target_pct = 0.025     # 2.5% first target
                    target2_pct = 0.04     # 4% second target
                    target3_pct = 0.06     # 6% third target
                
                # Find future data after signal
                future_data = df[df['timestamp'] > signal_time].head(50)  # Check next 50 candles
                
                if future_data.empty:
                    continue
                
                # Simulate trade outcome with position sizing
                outcome = "Pending"
                pnl = 0
                exit_price = signal_price
                target_hit = False
                target2_hit = False
                
                for _, candle in future_data.iterrows():
                    if signal_type == "BUY CALL":
                        # Check stop loss
                        if candle['low'] <= signal_price * (1 - stop_loss_pct):
                            outcome = "Loss"
                            pnl = -signal_price * stop_loss_pct * position_multiplier
                            exit_price = signal_price * (1 - stop_loss_pct)
                            break
                        # Check first target
                        elif not target_hit and candle['high'] >= signal_price * (1 + target_pct):
                            target_hit = True
                            # Continue to check for second target
                        # Check second target
                        elif target_hit and not target2_hit and candle['high'] >= signal_price * (1 + target2_pct):
                            target2_hit = True
                            # Continue to check for third target
                        # Check third target
                        elif target2_hit and candle['high'] >= signal_price * (1 + target3_pct):
                            outcome = "Win"
                            pnl = signal_price * target3_pct * position_multiplier
                            exit_price = signal_price * (1 + target3_pct)
                            break
                        # If second target hit but third not hit, close at second target
                        elif target2_hit and candle['close'] <= signal_price * (1 + target2_pct * 0.8):
                            outcome = "Win"
                            pnl = signal_price * target2_pct * position_multiplier
                            exit_price = signal_price * (1 + target2_pct)
                            break
                        # If first target hit but second not hit, close at first target
                        elif target_hit and candle['close'] <= signal_price * (1 + target_pct * 0.8):
                            outcome = "Win"
                            pnl = signal_price * target_pct * position_multiplier
                            exit_price = signal_price * (1 + target_pct)
                            break
                    
                    elif signal_type == "BUY PUT":
                        # Check stop loss
                        if candle['high'] >= signal_price * (1 + stop_loss_pct):
                            outcome = "Loss"
                            pnl = -signal_price * stop_loss_pct * position_multiplier
                            exit_price = signal_price * (1 + stop_loss_pct)
                            break
                        # Check first target
                        elif not target_hit and candle['low'] <= signal_price * (1 - target_pct):
                            target_hit = True
                            # Continue to check for second target
                        # Check second target
                        elif target_hit and not target2_hit and candle['low'] <= signal_price * (1 - target2_pct):
                            target2_hit = True
                            # Continue to check for third target
                        # Check third target
                        elif target2_hit and candle['low'] <= signal_price * (1 - target3_pct):
                            outcome = "Win"
                            pnl = signal_price * target3_pct * position_multiplier
                            exit_price = signal_price * (1 - target3_pct)
                            break
                        # If second target hit but third not hit, close at second target
                        elif target2_hit and candle['close'] >= signal_price * (1 - target2_pct * 0.8):
                            outcome = "Win"
                            pnl = signal_price * target2_pct * position_multiplier
                            exit_price = signal_price * (1 - target2_pct)
                            break
                        # If first target hit but second not hit, close at first target
                        elif target_hit and candle['close'] >= signal_price * (1 - target_pct * 0.8):
                            outcome = "Win"
                            pnl = signal_price * target_pct * position_multiplier
                            exit_price = signal_price * (1 - target_pct)
                            break
                
                # If still pending after 50 candles, close at current price
                if outcome == "Pending":
                    last_candle = future_data.iloc[-1]
                    if signal_type == "BUY CALL":
                        pnl = (last_candle['close'] - signal_price) * position_multiplier
                    else:  # BUY PUT
                        pnl = (signal_price - last_candle['close']) * position_multiplier
                    
                    exit_price = last_candle['close']
                    outcome = "Win" if pnl > 0 else "Loss"
                
                trade = {
                    'timestamp': signal_time,
                    'strategy': signal['strategy'],
                    'signal': signal_type,
                    'entry_price': signal_price,
                    'exit_price': exit_price,
                    'outcome': outcome,
                    'pnl': pnl,
                    'confidence': signal['confidence'],
                    'reasoning': signal['reasoning'],
                    'position_multiplier': position_multiplier
                }
                trades.append(trade)
                
            except Exception as e:
                continue
        
        return trades
    
    def run_backtest(self, symbol, timeframe, days=30):
        """Run complete backtest"""
        logger.info(f"🧠 Running backtest: {symbol} {timeframe} ({days} days)")
        
        # Load data
        df = self.load_data(symbol, timeframe, days)
        if df is None:
            return None
        
        # Add indicators
        df = self.add_indicators(df)
        
        # Generate signals
        signals = self.run_enhanced_strategies(df, symbol)
        logger.info(f"📊 Generated {len(signals)} signals")
        
        # Simulate trades
        trades = self.simulate_trades(signals, df)
        logger.info(f"📈 Simulated {len(trades)} trades")
        
        return trades
    
    def analyze_results(self, trades):
        """Analyze backtest results"""
        if not trades:
            logger.info("❌ No trades to analyze")
            return
        
        df = pd.DataFrame(trades)
        
        # Overall statistics
        total_trades = len(trades)
        wins = len(df[df['outcome'] == 'Win'])
        losses = len(df[df['outcome'] == 'Loss'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        max_profit = df['pnl'].max()
        max_loss = df['pnl'].min()
        
        # Strategy-wise analysis
        strategy_stats = df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean'],
            'outcome': lambda x: (x == 'Win').sum() / len(x) * 100
        }).round(2)
        
        strategy_stats.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Win_Rate']
        
        # Print results
        print(f"\n📊 ENHANCED STRATEGIES BACKTEST RESULTS")
        print("=" * 60)
        print(f"📈 Total Trades: {total_trades}")
        print(f"✅ Wins: {wins}")
        print(f"❌ Losses: {losses}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"💰 Total P&L: ₹{total_pnl:.2f}")
        print(f"📊 Average P&L: ₹{avg_pnl:.2f}")
        print(f"🚀 Max Profit: ₹{max_profit:.2f}")
        print(f"📉 Max Loss: ₹{max_loss:.2f}")
        
        print(f"\n📊 STRATEGY PERFORMANCE")
        print("-" * 60)
        for strategy, stats in strategy_stats.iterrows():
            status = "✅ PROFITABLE" if stats['Total_PnL'] > 0 else "❌ LOSS"
            print(f"🎯 {strategy}:")
            print(f"   Trades: {stats['Trades']} | Win Rate: {stats['Win_Rate']:.1f}%")
            print(f"   P&L: ₹{stats['Total_PnL']:.2f} | Avg: ₹{stats['Avg_PnL']:.2f}")
            print(f"   Status: {status}")
            print()
        
        if total_pnl > 0:
            print(f"🎉 OVERALL RESULT: PROFITABLE")
        else:
            print(f"⚠️ OVERALL RESULT: LOSS MAKING")
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'strategy_stats': strategy_stats
        }
    
    def save_trades_to_db(self, trades_df, symbol, timeframe):
        """Save trades to database"""
        try:
            # Add metadata columns
            trades_df['symbol'] = symbol
            trades_df['timeframe'] = timeframe
            trades_df['backtest_date'] = datetime.now()
            
            # Convert timestamp to string for database storage
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = trades_df['timestamp'].astype(str)
            
            # Save to database using the unified database
            self.db.save_trades(trades_df)
            logger.info(f"✅ Saved {len(trades_df)} trades to database")
            
        except Exception as e:
            logger.error(f"❌ Error saving trades to database: {e}")
            # Continue without failing the backtest

def main():
    parser = argparse.ArgumentParser(description="Simple Backtesting")
    parser.add_argument("--symbol", type=str, default="NSE:NIFTY50-INDEX", help="Symbol to test")
    parser.add_argument("--timeframe", type=str, default="5min", help="Timeframe to test")
    parser.add_argument("--days", type=int, default=180, help="Days to backtest")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample data (NOT RECOMMENDED)")
    parser.add_argument("--quick", action="store_true", help="Run quick test with sample data (NOT RECOMMENDED)")
    args = parser.parse_args()
    
    if args.demo:
        print("⚠️ WARNING: Using sample data - NOT recommended for real trading!")
        print("💡 Use real market data instead: python3 simple_backtest.py --symbol 'NSE:NIFTY50-INDEX' --days 60")
        # Run demo with sample data
        run_demo_backtest()
    elif args.quick:
        print("⚠️ WARNING: Using sample data - NOT recommended for real trading!")
        print("💡 Use real market data instead: python3 simple_backtest.py --symbol 'NSE:NIFTY50-INDEX' --days 60")
        # Run quick test with sample data
        run_quick_test()
    else:
        backtester = SimpleBacktester()
        
        # Run backtest
        trades = backtester.run_backtest(args.symbol, args.timeframe, args.days)
        
        # Analyze results
        if trades:
            results = backtester.analyze_results(trades)
            
                    # Save detailed trades to database only
        if trades:
            trades_df = pd.DataFrame(trades)
            backtester.save_trades_to_db(trades_df, symbol=args.symbol, timeframe=args.timeframe)
            print(f"💾 Trades saved to database")
        else:
            logger.error("❌ Backtest failed")

def run_quick_test():
    """Run a quick test with sample data to show all strategies working"""
    logger.info("🎯 Running QUICK TEST with sample data for all strategies...")
    
    # Generate sample data with trends
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=800, freq='5min')
    
    # Create trending data
    prices = [1000]
    for i in range(1, 800):
        # Add some trend and volatility
        if i < 200:
            change = np.random.normal(0.001, 0.002)  # Uptrend
        elif i < 400:
            change = np.random.normal(-0.0005, 0.002)  # Downtrend
        elif i < 600:
            change = np.random.normal(0.0008, 0.002)  # Strong uptrend
        else:
            change = np.random.normal(0.0002, 0.002)  # Sideways
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.randint(5000, 15000, 800)
    })
    
    # Test enhanced original strategies
    from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
    from src.strategies.supertrend_ema import SupertrendEma
    from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
    
    strategies = {
        'ema_crossover_enhanced': EmaCrossoverEnhanced(),
        'supertrend_ema': SupertrendEma(),
        'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
    }
    
    signals = []
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Testing {strategy_name}...")
        strategy_signals = 0
        
        for i in range(200, len(data)):
            try:
                result = strategy.analyze(data.iloc[:i+1])
                if result and result.get('signal') != 'NO TRADE':
                    signal = {
                        'timestamp': data.iloc[i]['timestamp'],
                        'strategy': strategy_name,
                        'signal': result['signal'],
                        'price': result.get('price', data.iloc[i]['close']),
                        'confidence': result.get('confidence_score', 0),
                        'stop_loss': result.get('stop_loss'),
                        'target1': result.get('target1'),
                        'target2': result.get('target2'),
                        'target3': result.get('target3'),
                        'position_multiplier': result.get('position_multiplier', 1.0)
                    }
                    signals.append(signal)
                    strategy_signals += 1
            except Exception as e:
                continue
        
        logger.info(f"📊 {strategy_name}: {strategy_signals} signals")
    
    # Simulate trades
    backtester = SimpleBacktester()
    trades = backtester.simulate_trades(signals, data)
    
    if trades:
        results = backtester.analyze_results(trades)
        
        # Save to database only
        if trades:
            trades_df = pd.DataFrame(trades)
            backtester.save_trades_to_db(trades_df, symbol="QUICK_TEST", timeframe="5min")
            print(f"💾 Quick test trades saved to database")
    else:
        logger.error("❌ Quick test failed - no trades generated")

def run_demo_backtest():
    """Run a demo backtest with sample data to show P&L"""
    logger.info("🎯 Running DEMO backtest with sample data...")
    
    # Generate sample data with trends
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='5min')
    
    # Create trending data
    prices = [1000]
    for i in range(1, 1000):
        # Add some trend and volatility
        if i < 300:
            change = np.random.normal(0.001, 0.002)  # Uptrend
        elif i < 600:
            change = np.random.normal(-0.0005, 0.002)  # Downtrend
        else:
            change = np.random.normal(0.0005, 0.002)  # Sideways
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Test enhanced original strategies
    from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
    from src.strategies.supertrend_ema import SupertrendEma
    from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
    
    strategies = {
        'ema_crossover_enhanced': EmaCrossoverEnhanced(),
        'supertrend_ema': SupertrendEma(),
        'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
    }
    
    signals = []
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Testing {strategy_name}...")
        strategy_signals = 0
        
        for i in range(200, len(data)):
            try:
                result = strategy.analyze(data.iloc[:i+1])
                if result and result.get('signal') != 'NO TRADE':
                    signal = {
                        'timestamp': data.iloc[i]['timestamp'],
                        'strategy': strategy_name,
                        'signal': result['signal'],
                        'price': result.get('price', data.iloc[i]['close']),
                        'confidence': result.get('confidence_score', 0),
                        'stop_loss': result.get('stop_loss'),
                        'target1': result.get('target1'),
                        'target2': result.get('target2'),
                        'target3': result.get('target3'),
                        'position_multiplier': result.get('position_multiplier', 1.0)
                    }
                    signals.append(signal)
                    strategy_signals += 1
            except Exception as e:
                continue
        
        logger.info(f"📊 {strategy_name}: {strategy_signals} signals")
    
    # Simulate trades
    backtester = SimpleBacktester()
    trades = backtester.simulate_trades(signals, data)
    
    if trades:
        results = backtester.analyze_results(trades)
        
        # Save to database only
        if trades:
            trades_df = pd.DataFrame(trades)
            backtester.save_trades_to_db(trades_df, symbol="DEMO", timeframe="5min")
            print(f"💾 Demo trades saved to database")
    else:
        logger.error("❌ Demo backtest failed - no trades generated")

if __name__ == "__main__":
    main() 