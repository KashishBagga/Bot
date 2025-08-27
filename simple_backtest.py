#!/usr/bin/env python3
"""
Optimized Simple Backtesting System
A high-performance backtesting solution using local parquet data
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
import gc
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

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

class OptimizedBacktester:
    def __init__(self):
        """Initialize the optimized backtester"""
        self.base_dir = Path("historical_data_20yr")
        self.db = UnifiedDatabase()
        self.data_loader = LocalDataLoader()
        
        # Performance tracking
        self.start_time = None
        self.processing_stats = {
            'data_load_time': 0,
            'indicator_time': 0,
            'strategy_time': 0,
            'trade_sim_time': 0,
            'db_save_time': 0
        }
        self.disable_tqdm = False  # Can be set to True for non-interactive environments
        
        logger.info("🚀 Optimized Backtester Initialized")
        logger.info(f"📁 Data Directory: {self.base_dir}")
    
    def load_data_optimized(self, symbol, timeframe, days=30):
        """Load data with optimized performance using pyarrow engine"""
        start_time = time.time()
        
        try:
            # Direct parquet loading with pyarrow engine for maximum speed
            symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
            parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
            
            if not parquet_file.exists():
                logger.error(f"❌ Data file not found: {parquet_file}")
                return None
            
            # Use pyarrow engine for much faster loading
            df = pd.read_parquet(
                parquet_file, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                engine="pyarrow"
            )
            
            if df.empty:
                logger.error(f"❌ Empty data file: {parquet_file}")
                return None
            
            # Convert timestamp to datetime64[ns] once at load
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
            
            # Filter by date range using NumPy masks for speed
            if days:
                end_date = df['timestamp'].max()
                start_date = end_date - timedelta(days=days)
                mask = df['timestamp'] >= start_date
                df = df[mask].reset_index(drop=True)
            
            # Limit data size for faster processing (max 3000 candles)
            if len(df) > 3000:
                df = df.tail(3000).reset_index(drop=True)
                logger.info(f"📊 Limited to last 3000 candles for faster processing")
            
            load_time = time.time() - start_time
            self.processing_stats['data_load_time'] = load_time
            
            logger.info(f"✅ Loaded {len(df):,} candles for {symbol} {timeframe} in {load_time:.2f}s")
            logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    

    
    def add_indicators_optimized(self, df):
        """Add technical indicators with optimized performance - compute once only"""
        start_time = time.time()
        
        try:
            # Check if indicators already exist to avoid recalculation
            required_indicators = {'ema_20', 'ema_50', 'supertrend', 'rsi', 'macd'}
            if not required_indicators.issubset(set(df.columns)):
                # Use the unified indicators function - compute once only
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
            
            indicator_time = time.time() - start_time
            self.processing_stats['indicator_time'] = indicator_time
            
            logger.info(f"✅ Indicators and market conditions added in {indicator_time:.2f}s")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding indicators: {e}")
            return df
    
    def run_enhanced_strategies_optimized(self, df, symbol, disable_tqdm: bool = False):
        """Run all enhanced strategies with optimized processing - row-wise + parallel.
           Assumes indicators are already present in df (do NOT recompute here)."""
        start_time = time.time()
        signals = []

        # Import enhanced original strategies
        from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
        from src.strategies.supertrend_ema import SupertrendEma
        from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma

        strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }

        logger.info(f"🧠 Running {len(strategies)} enhanced strategies with optimized processing...")

        # Ensure timestamp is datetime
        if df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Pre-cache length
        n = len(df)
        min_candles = 200

        # Prepare an index map and maybe column arrays for vectorized access if strategies need them
        # NOTE: we still call analyze_row for compatibility. If strategies support analyze_vectorized,
        # that would be faster and we can call it here once per strategy.
        def process_strategy(strategy_name, strategy):
            strategy_signals = []
            try:
                # prefer an analyze_row API (row-wise), fallback to analyze (slice)
                use_row = hasattr(strategy, 'analyze_row')

                rng = range(min_candles, n)
                it = tqdm(rng, desc=f"Processing {strategy_name}", leave=False, disable=disable_tqdm)

                for j in it:
                    try:
                        if use_row:
                            # lightweight row access
                            row = df.iloc[j]  # acceptable given it's a single Series
                            result = strategy.analyze_row(j, row, df)
                        else:
                            # fallback - still slower
                            result = strategy.analyze(df.iloc[:j+1])

                        if result and result.get('signal') and result.get('signal') != 'NO TRADE':
                            signal = {
                                'timestamp': df.iat[j, df.columns.get_loc('timestamp')],
                                'strategy': strategy_name,
                                'signal': result['signal'],
                                'price': result.get('price', df.iat[j, df.columns.get_loc('close')]),
                                'confidence': result.get('confidence_score', 0),
                                'reasoning': str(result.get('reasoning', ''))[:100],
                                'stop_loss': result.get('stop_loss'),
                                'target1': result.get('target1'),
                                'target2': result.get('target2'),
                                'target3': result.get('target3'),
                                'position_multiplier': result.get('position_multiplier', 1.0)
                            }
                            strategy_signals.append(signal)
                    except Exception as e:
                        # log for debugging, but continue processing
                        logger.debug(f"⚠️ {strategy_name} row {j} error: {e}")
                        logger.debug(traceback.format_exc())
                        continue
            except Exception as e:
                logger.error(f"❌ Fatal error processing strategy {strategy_name}: {e}")
                logger.error(traceback.format_exc())

            logger.info(f"📊 {strategy_name}: {len(strategy_signals)} signals")
            return strategy_signals

        # Execute strategies in parallel
        max_workers = min(len(strategies), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {executor.submit(process_strategy, name, strat): name for name, strat in strategies.items()}

            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    res = fut.result()
                    if res:
                        signals.extend(res)
                except Exception as e:
                    logger.error(f"❌ Error in strategy future {name}: {e}")
                    logger.error(traceback.format_exc())

        strategy_time = time.time() - start_time
        self.processing_stats['strategy_time'] = strategy_time
        logger.info(f"✅ Strategy analysis completed in {strategy_time:.2f}s")
        return signals
    
    def simulate_trades_optimized(self, signals, df):
        """Simulate trades with optimized performance using NumPy vectorized operations and correct earliest-hit logic."""
        start_time = time.time()
        trades = []

        # Pre-extract arrays for speed
        timestamps = df['timestamp'].values
        lows_arr = df['low'].values
        highs_arr = df['high'].values
        closes_arr = df['close'].values

        for signal in signals:
            try:
                signal_time = signal['timestamp']
                signal_price = float(signal['price'])
                signal_type = signal['signal']
                position_multiplier = float(signal.get('position_multiplier', 1.0))

                # compute percents
                if signal.get('stop_loss') and signal.get('target1'):
                    stop_loss_pct = float(signal['stop_loss']) / signal_price
                    target_pct = float(signal['target1']) / signal_price
                    target2_pct = float(signal.get('target2', signal['target1'])) / signal_price
                    target3_pct = float(signal.get('target3', signal.get('target2', signal['target1']))) / signal_price
                else:
                    stop_loss_pct = 0.015
                    target_pct = 0.025
                    target2_pct = 0.04
                    target3_pct = 0.06

                # find start index (first index where timestamp > signal_time)
                # use numpy searchsorted for speed (timestamps must be sorted)
                try:
                    start_idx = np.searchsorted(timestamps, np.datetime64(signal_time), side='right')
                except Exception:
                    # fallback to boolean mask
                    mask = df['timestamp'].values > np.datetime64(signal_time)
                    start_idx = np.argmax(mask) if mask.any() else len(df)

                if start_idx >= len(df):
                    continue

                end_idx = min(start_idx + 50, len(df))
                # slice views (no copy if possible)
                lows = lows_arr[start_idx:end_idx]
                highs = highs_arr[start_idx:end_idx]
                closes = closes_arr[start_idx:end_idx]

                # init outcome values
                outcome = "Pending"
                pnl = 0.0
                exit_price = signal_price

                if signal_type == "BUY CALL":
                    stop_price = signal_price * (1 - stop_loss_pct)
                    t1_price = signal_price * (1 + target_pct)
                    t2_price = signal_price * (1 + target2_pct)
                    t3_price = signal_price * (1 + target3_pct)

                    # boolean arrays
                    stop_hit = (lows <= stop_price)
                    t1_hit = (highs >= t1_price)
                    t2_hit = (highs >= t2_price)
                    t3_hit = (highs >= t3_price)

                    # find first indices (or set to large number if not hit)
                    def first_idx(bool_arr):
                        idxs = np.nonzero(bool_arr)[0]
                        return idxs[0] if idxs.size > 0 else np.iinfo(np.int32).max

                    idx_stop = first_idx(stop_hit)
                    idx_t1 = first_idx(t1_hit)
                    idx_t2 = first_idx(t2_hit)
                    idx_t3 = first_idx(t3_hit)

                    # pick earliest event
                    first_event_idx = min(idx_stop, idx_t1, idx_t2, idx_t3)

                    if first_event_idx == np.iinfo(np.int32).max:
                        # no events -> close at last close
                        last_close = float(closes[-1])
                        pnl = (last_close - signal_price) * position_multiplier
                        exit_price = last_close
                        outcome = "Win" if pnl > 0 else "Loss"
                    else:
                        # decide which event it was
                        if idx_stop == first_event_idx:
                            outcome = "Loss"
                            pnl = -signal_price * stop_loss_pct * position_multiplier
                            exit_price = stop_price
                        elif idx_t3 == first_event_idx:
                            outcome = "Win"
                            pnl = signal_price * target3_pct * position_multiplier
                            exit_price = t3_price
                        elif idx_t2 == first_event_idx:
                            outcome = "Win"
                            pnl = signal_price * target2_pct * position_multiplier
                            exit_price = t2_price
                        else:  # idx_t1 == first_event_idx
                            outcome = "Win"
                            pnl = signal_price * target_pct * position_multiplier
                            exit_price = t1_price

                elif signal_type == "BUY PUT":
                    stop_price = signal_price * (1 + stop_loss_pct)
                    t1_price = signal_price * (1 - target_pct)
                    t2_price = signal_price * (1 - target2_pct)
                    t3_price = signal_price * (1 - target3_pct)

                    stop_hit = (highs >= stop_price)
                    t1_hit = (lows <= t1_price)
                    t2_hit = (lows <= t2_price)
                    t3_hit = (lows <= t3_price)

                    def first_idx(bool_arr):
                        idxs = np.nonzero(bool_arr)[0]
                        return idxs[0] if idxs.size > 0 else np.iinfo(np.int32).max

                    idx_stop = first_idx(stop_hit)
                    idx_t1 = first_idx(t1_hit)
                    idx_t2 = first_idx(t2_hit)
                    idx_t3 = first_idx(t3_hit)

                    first_event_idx = min(idx_stop, idx_t1, idx_t2, idx_t3)

                    if first_event_idx == np.iinfo(np.int32).max:
                        last_close = float(closes[-1])
                        pnl = (signal_price - last_close) * position_multiplier
                        exit_price = last_close
                        outcome = "Win" if pnl > 0 else "Loss"
                    else:
                        if idx_stop == first_event_idx:
                            outcome = "Loss"
                            pnl = -signal_price * stop_loss_pct * position_multiplier
                            exit_price = stop_price
                        elif idx_t3 == first_event_idx:
                            outcome = "Win"
                            pnl = signal_price * target3_pct * position_multiplier
                            exit_price = t3_price
                        elif idx_t2 == first_event_idx:
                            outcome = "Win"
                            pnl = signal_price * target2_pct * position_multiplier
                            exit_price = t2_price
                        else:  # idx_t1 == first_event_idx
                            outcome = "Win"
                            pnl = signal_price * target_pct * position_multiplier
                            exit_price = t1_price

                # record trade
                trade = {
                    'timestamp': signal_time,
                    'strategy': signal['strategy'],
                    'signal': signal_type,
                    'entry_price': signal_price,
                    'exit_price': float(exit_price),
                    'outcome': outcome,
                    'pnl': float(pnl),
                    'confidence': signal.get('confidence', 0),
                    'reasoning': signal.get('reasoning', ''),
                    'position_multiplier': position_multiplier
                }
                trades.append(trade)

            except Exception as e:
                logger.debug(f"⚠️ simulate_trades error for signal {signal}: {e}")
                logger.debug(traceback.format_exc())
                continue

        trade_sim_time = time.time() - start_time
        self.processing_stats['trade_sim_time'] = trade_sim_time
        logger.info(f"✅ Trade simulation completed in {trade_sim_time:.2f}s")
        return trades
    
    def run_backtest(self, symbol, timeframe, days=30):
        """Run complete optimized backtest"""
        self.start_time = time.time()
        logger.info(f"🧠 Running optimized backtest: {symbol} {timeframe} ({days} days)")
        
        # Load data
        df = self.load_data_optimized(symbol, timeframe, days)
        if df is None:
            return None
        
        # Add indicators ONCE
        df = self.add_indicators_optimized(df)
        
        # Generate signals (DO NOT recalc indicators inside the function)
        signals = self.run_enhanced_strategies_optimized(df, symbol, disable_tqdm=self.disable_tqdm)
        logger.info(f"📊 Generated {len(signals)} signals")
        
        # Simulate trades
        trades = self.simulate_trades_optimized(signals, df)
        logger.info(f"📈 Simulated {len(trades)} trades")
        
        # Clean up memory
        del df, signals
        gc.collect()
        
        total_time = time.time() - self.start_time
        logger.info(f"✅ Complete backtest finished in {total_time:.2f}s")
        
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
        print(f"\n📊 OPTIMIZED STRATEGIES BACKTEST RESULTS")
        print("=" * 60)
        print(f"📈 Total Trades: {total_trades}")
        print(f"✅ Wins: {wins}")
        print(f"❌ Losses: {losses}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"💰 Total P&L: ₹{total_pnl:.2f}")
        print(f"📊 Average P&L: ₹{avg_pnl:.2f}")
        print(f"🚀 Max Profit: ₹{max_profit:.2f}")
        print(f"📉 Max Loss: ₹{max_loss:.2f}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print("-" * 60)
        for metric, time_taken in self.processing_stats.items():
            print(f"⏱️ {metric.replace('_', ' ').title()}: {time_taken:.2f}s")
        
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
            'strategy_stats': strategy_stats,
            'performance_stats': self.processing_stats
        }
    
    def save_trades_to_db_optimized(self, trades_df, symbol, timeframe):
        """Save trades to database with optimized batch operations"""
        start_time = time.time()
        
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
            
            db_save_time = time.time() - start_time
            self.processing_stats['db_save_time'] = db_save_time
            
            logger.info(f"✅ Saved {len(trades_df)} trades to database in {db_save_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error saving trades to database: {e}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Simple Backtesting")
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
        backtester = OptimizedBacktester()
        
        # Run backtest
        trades = backtester.run_backtest(args.symbol, args.timeframe, args.days)
        
        # Analyze results
        if trades:
            results = backtester.analyze_results(trades)
            
            # Save detailed trades to database only
            if trades:
                trades_df = pd.DataFrame(trades)
                backtester.save_trades_to_db_optimized(trades_df, symbol=args.symbol, timeframe=args.timeframe)
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
    backtester = OptimizedBacktester()
    trades = backtester.simulate_trades_optimized(signals, data)
    
    if trades:
        results = backtester.analyze_results(trades)
        
        # Save to database only
        if trades:
            trades_df = pd.DataFrame(trades)
            backtester.save_trades_to_db_optimized(trades_df, symbol="QUICK_TEST", timeframe="5min")
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
    backtester = OptimizedBacktester()
    trades = backtester.simulate_trades_optimized(signals, data)
    
    if trades:
        results = backtester.analyze_results(trades)
        
        # Save to database only
        if trades:
            trades_df = pd.DataFrame(trades)
            backtester.save_trades_to_db_optimized(trades_df, symbol="DEMO", timeframe="5min")
            print(f"💾 Demo trades saved to database")
    else:
        logger.error("❌ Demo backtest failed - no trades generated")

if __name__ == "__main__":
    main() 