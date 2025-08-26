#!/usr/bin/env python3
"""
Backtesting with Local Parquet Data
Uses local parquet files for backtesting instead of fetching from API
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.local_data_loader import LocalDataLoader
from src.core.strategy import Strategy
from src.models.database import Database
from src.models.enhanced_rejected_signals import EnhancedRejectedSignals
from src.config.settings import SYMBOLS, TIMEFRAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalBacktestingEngine:
    def __init__(self):
        """Initialize the local backtesting engine"""
        self.data_loader = LocalDataLoader()
        self.db = Database()
        self.rejected_signals = EnhancedRejectedSignals()
        
        # Initialize strategies
        self.strategies = self.load_strategies()
        
        logger.info("🚀 Local Backtesting Engine Initialized")
        logger.info(f"📊 Available strategies: {list(self.strategies.keys())}")
    
    def load_strategies(self):
        """Load all available trading strategies"""
        strategies = {}
        
        try:
            from src.strategies.supertrend_ema import SupertrendEma
            from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
            from src.strategies.insidebar_rsi import InsidebarRsi
            from src.strategies.hybrid_momentum_trend import HybridMomentumTrend
            from src.strategies.rsi_mean_reversion_bb import RsiMeanReversionBb
            from src.strategies.ema_crossover import EmaCrossover
            from src.strategies.macd_cross_rsi_filter import MacdCrossRsiFilter
            
            strategies['supertrend_ema'] = SupertrendEma()
            strategies['supertrend_macd_rsi_ema'] = SupertrendMacdRsiEma()
            strategies['insidebar_rsi'] = InsidebarRsi()
            strategies['hybrid_momentum_trend'] = HybridMomentumTrend()
            strategies['rsi_mean_reversion_bb'] = RsiMeanReversionBb()
            strategies['ema_crossover'] = EmaCrossover()
            strategies['macd_cross_rsi_filter'] = MacdCrossRsiFilter()
            
        except ImportError as e:
            logger.error(f"❌ Error importing strategies: {e}")
        
        return strategies
    
    def run_backtest(self, strategy_name: str, symbol: str, timeframe: str, 
                    start_date: str = None, end_date: str = None, days: int = None):
        """Run backtest for a specific strategy and symbol"""
        
        if strategy_name not in self.strategies:
            logger.error(f"❌ Strategy '{strategy_name}' not found")
            return
        
        strategy = self.strategies[strategy_name]
        
        logger.info(f"🧠 Testing Strategy: {strategy_name}")
        logger.info("-" * 40)
        
        # Load data from local parquet files
        logger.info(f"🔄 Loading data for {symbol} {timeframe}")
        data = self.data_loader.load_data(symbol, timeframe, start_date, end_date, days)
        
        if data is None or data.empty:
            logger.error(f"❌ No data available for {symbol} {timeframe}")
            return
        
        logger.info(f"📊 Loaded {len(data):,} candles for {symbol}")
        
        # Run backtest
        total_signals = 0
        rejected_signals = 0
        
        for i in range(len(data) - 1):
            try:
                # Get current candle and previous candles for analysis
                current_candle = data.iloc[i]
                previous_candles = data.iloc[:i+1]
                
                # Analyze with strategy
                signal = strategy.analyze(previous_candles)
                
                if signal:
                    total_signals += 1
                    
                    # Simulate trade outcome
                    trade_result = self.simulate_trade(signal, data.iloc[i+1:], current_candle)
                    
                    if trade_result:
                        # Store trade result
                        self.store_trade_result(strategy_name, symbol, signal, trade_result, current_candle)
                    else:
                        # Store rejected signal
                        self.store_rejected_signal(strategy_name, symbol, signal, current_candle)
                        rejected_signals += 1
                
                # Progress logging
                if (i + 1) % 1000 == 0:
                    logger.info(f"📊 {symbol}: {i + 1:,} candles processed...")
                
            except Exception as e:
                logger.error(f"❌ Error processing candle {i}: {e}")
                continue
        
        logger.info(f"✅ Backtest completed: {total_signals} total signals, {rejected_signals} rejected")
        logger.info(f"📊 {symbol}: {total_signals} signals, {rejected_signals} rejected ({rejected_signals/total_signals*100:.1f}%)")
    
    def simulate_trade(self, signal, future_data, entry_candle):
        """Simulate trade outcome based on signal"""
        if future_data.empty:
            return None
        
        entry_price = entry_candle['close']
        signal_type = signal.get('signal')
        stop_loss = signal.get('stop_loss', 0)
        target1 = signal.get('target1', 0)
        target2 = signal.get('target2', 0)
        target3 = signal.get('target3', 0)
        
        # Check if stop loss or targets are hit
        for _, candle in future_data.iterrows():
            high = candle['high']
            low = candle['low']
            
            if signal_type == "BUY CALL":
                # Check for stop loss
                if low <= entry_price - stop_loss:
                    return {
                        'outcome': 'Loss',
                        'pnl': -stop_loss,
                        'targets_hit': 0,
                        'stoploss_count': 1,
                        'exit_price': entry_price - stop_loss,
                        'exit_time': candle['timestamp']
                    }
                
                # Check for targets
                targets_hit = 0
                if high >= entry_price + target1:
                    targets_hit = 1
                if high >= entry_price + target2:
                    targets_hit = 2
                if high >= entry_price + target3:
                    targets_hit = 3
                
                if targets_hit > 0:
                    return {
                        'outcome': 'Win',
                        'pnl': target1 if targets_hit == 1 else target2 if targets_hit == 2 else target3,
                        'targets_hit': targets_hit,
                        'stoploss_count': 0,
                        'exit_price': entry_price + (target1 if targets_hit == 1 else target2 if targets_hit == 2 else target3),
                        'exit_time': candle['timestamp']
                    }
            
            elif signal_type == "BUY PUT":
                # Check for stop loss
                if high >= entry_price + stop_loss:
                    return {
                        'outcome': 'Loss',
                        'pnl': -stop_loss,
                        'targets_hit': 0,
                        'stoploss_count': 1,
                        'exit_price': entry_price + stop_loss,
                        'exit_time': candle['timestamp']
                    }
                
                # Check for targets
                targets_hit = 0
                if low <= entry_price - target1:
                    targets_hit = 1
                if low <= entry_price - target2:
                    targets_hit = 2
                if low <= entry_price - target3:
                    targets_hit = 3
                
                if targets_hit > 0:
                    return {
                        'outcome': 'Win',
                        'pnl': target1 if targets_hit == 1 else target2 if targets_hit == 2 else target3,
                        'targets_hit': targets_hit,
                        'stoploss_count': 0,
                        'exit_price': entry_price - (target1 if targets_hit == 1 else target2 if targets_hit == 2 else target3),
                        'exit_time': candle['timestamp']
                    }
        
        return None
    
    def store_trade_result(self, strategy_name, symbol, signal, trade_result, entry_candle):
        """Store trade result in database"""
        try:
            trade_data = {
                'strategy': strategy_name,
                'symbol': symbol,
                'signal_type': signal['signal'],
                'entry_price': entry_candle['close'],
                'exit_price': trade_result['exit_price'],
                'pnl': trade_result['pnl'],
                'outcome': trade_result['outcome'],
                'targets_hit': trade_result['targets_hit'],
                'stoploss_count': trade_result['stoploss_count'],
                'entry_time': entry_candle['timestamp'],
                'exit_time': trade_result['exit_time'],
                'confidence_score': signal.get('confidence_score', 0),
                'timestamp': entry_candle['timestamp']
            }
            
            # Insert into database
            self.db.insert_trade_backtest(trade_data)
            
        except Exception as e:
            logger.error(f"❌ Error storing trade result: {e}")
    
    def store_rejected_signal(self, strategy_name, symbol, signal, candle):
        """Store rejected signal in database"""
        try:
            rejected_data = {
                'strategy': strategy_name,
                'symbol': symbol,
                'signal_type': signal['signal'],
                'price': candle['close'],
                'timestamp': candle['timestamp'],
                'confidence_score': signal.get('confidence_score', 0),
                'reason': signal.get('reason', 'Low confidence')
            }
            
            # Insert into database
            self.rejected_signals.insert_rejected_signal_backtest(rejected_data)
            
        except Exception as e:
            logger.error(f"❌ Error storing rejected signal: {e}")
    
    def run_comprehensive_backtest(self, strategies=None, symbols=None, timeframes=None, days=30):
        """Run comprehensive backtest for multiple strategies/symbols/timeframes"""
        
        if strategies is None:
            strategies = list(self.strategies.keys())
        if symbols is None:
            symbols = SYMBOLS
        if timeframes is None:
            timeframes = ['5min']  # Default to 5min for backtesting
        
        logger.info("🚀 BACKTESTING WITH LOCAL PARQUET DATA")
        logger.info("=" * 60)
        
        total_combinations = len(strategies) * len(symbols) * len(timeframes)
        current_combination = 0
        
        for strategy_name in strategies:
            for symbol in symbols:
                for timeframe in timeframes:
                    current_combination += 1
                    logger.info(f"📊 Progress: {current_combination}/{total_combinations}")
                    
                    try:
                        self.run_backtest(strategy_name, symbol, timeframe, days=days)
                    except Exception as e:
                        logger.error(f"❌ Error in backtest {strategy_name} {symbol} {timeframe}: {e}")
                        continue
        
        logger.info("🎉 COMPREHENSIVE BACKTESTING COMPLETE!")
    
    def print_results_summary(self):
        """Print summary of backtesting results"""
        try:
            # Query database for results
            query = """
            SELECT strategy, symbol, 
                   COUNT(*) as total_trades,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl,
                   SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM trades_backtest 
            GROUP BY strategy, symbol 
            ORDER BY total_pnl DESC
            """
            
            results = self.db.execute_query(query)
            
            if results:
                logger.info("📊 BACKTESTING RESULTS SUMMARY:")
                logger.info("=" * 80)
                
                for row in results:
                    strategy, symbol, trades, total_pnl, avg_pnl, win_rate = row
                    logger.info(f"{strategy:<20} {symbol:<20} trades={trades:>5} total={total_pnl:>8.2f} avg={avg_pnl:>6.2f} W/L={win_rate:>5.1f}%")
                
                # Calculate totals
                total_pnl = sum(row[3] for row in results)
                total_trades = sum(row[2] for row in results)
                
                logger.info("=" * 80)
                logger.info(f"📈 TOTAL: {total_trades} trades, P&L: {total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error printing results summary: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run backtesting with local parquet data')
    parser.add_argument('--strategies', nargs='+', help='Strategies to test')
    parser.add_argument('--symbols', nargs='+', help='Symbols to test')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes to test')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        engine = LocalBacktestingEngine()
        
        # Print data availability
        engine.data_loader.print_data_summary()
        
        # Run backtest
        engine.run_comprehensive_backtest(
            strategies=args.strategies,
            symbols=args.symbols,
            timeframes=args.timeframes,
            days=args.days
        )
        
        # Print results
        engine.print_results_summary()
        
    except Exception as e:
        logger.error(f"❌ Fatal error in backtesting: {e}")
        raise

if __name__ == "__main__":
    main() 