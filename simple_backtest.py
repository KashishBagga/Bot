#!/usr/bin/env python3
"""
Simple Backtesting Script
Works with existing data and strategies
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

from src.models.database import db
from src.core.indicators import indicators

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
        self.db = db
        
        logger.info("🚀 Simple Backtester Initialized")
        logger.info(f"📁 Data Directory: {self.base_dir}")
    
    def load_data(self, symbol, timeframe, days=30):
        """Load data from parquet files"""
        try:
            symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
            parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
            
            if not parquet_file.exists():
                logger.error(f"❌ Data file not found: {parquet_file}")
                return None
            
            df = pd.read_parquet(parquet_file)
            
            if df.empty:
                logger.error(f"❌ Empty data file: {parquet_file}")
                return None
            
            # Filter for recent days
            if days:
                end_date = df['timestamp'].max()
                start_date = end_date - timedelta(days=days)
                df = df[df['timestamp'] >= start_date]
            
            logger.info(f"✅ Loaded {len(df):,} candles for {symbol} {timeframe}")
            logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def add_indicators(self, df):
        """Add basic indicators to the data"""
        try:
            # Basic indicators
            df['ema_9'] = indicators.ema(df, period=9)
            df['ema_21'] = indicators.ema(df, period=21)
            df['rsi'] = indicators.rsi(df, period=14)
            df['macd'], df['macd_signal'], df['macd_histogram'] = indicators.macd(df)
            
            # ATR
            df['atr'] = indicators.atr(df, period=14)
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price position
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
            
            logger.info("✅ Indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding indicators: {e}")
            return df
    
    def simple_supertrend_ema_strategy(self, df):
        """Simple SuperTrend + EMA strategy"""
        signals = []
        
        for i in range(50, len(df)):
            try:
                candle = df.iloc[i]
                prev_candle = df.iloc[i-1]
                
                # Basic conditions
                ema_9 = candle['ema_9']
                ema_21 = candle['ema_21']
                rsi = candle['rsi']
                macd = candle['macd']
                macd_signal = candle['macd_signal']
                volume_ratio = candle['volume_ratio']
                
                if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(rsi):
                    continue
                
                # Very simple BUY CALL conditions
                if (ema_9 > ema_21 and  # EMA crossover
                    rsi > 25 and rsi < 85):  # Very wide RSI range
                    
                    signal = {
                        'timestamp': candle['timestamp'],
                        'signal': 'BUY CALL',
                        'price': candle['close'],
                        'confidence': 70,
                        'stop_loss': candle['atr'] * 2,
                        'target': candle['atr'] * 3
                    }
                    signals.append(signal)
                
                # Very simple BUY PUT conditions
                elif (ema_9 < ema_21 and  # EMA crossover
                      rsi > 15 and rsi < 75):  # Very wide RSI range
                    
                    signal = {
                        'timestamp': candle['timestamp'],
                        'signal': 'BUY PUT',
                        'price': candle['close'],
                        'confidence': 70,
                        'stop_loss': candle['atr'] * 2,
                        'target': candle['atr'] * 3
                    }
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def simulate_trades(self, signals, df):
        """Simulate trades and calculate P&L"""
        trades = []
        
        for signal in signals:
            try:
                signal_time = signal['timestamp']
                signal_price = signal['price']
                signal_type = signal['signal']
                stop_loss = signal['stop_loss']
                target = signal['target']
                
                # Find future data after signal
                future_data = df[df['timestamp'] > signal_time].head(50)  # Check next 50 candles
                
                if future_data.empty:
                    continue
                
                # Simulate trade outcome
                outcome = "Pending"
                pnl = 0
                exit_price = signal_price
                
                for _, candle in future_data.iterrows():
                    if signal_type == "BUY CALL":
                        # Check stop loss
                        if candle['low'] <= signal_price - stop_loss:
                            outcome = "Loss"
                            pnl = -stop_loss
                            exit_price = signal_price - stop_loss
                            break
                        # Check target
                        elif candle['high'] >= signal_price + target:
                            outcome = "Win"
                            pnl = target
                            exit_price = signal_price + target
                            break
                    
                    elif signal_type == "BUY PUT":
                        # Check stop loss
                        if candle['high'] >= signal_price + stop_loss:
                            outcome = "Loss"
                            pnl = -stop_loss
                            exit_price = signal_price + stop_loss
                            break
                        # Check target
                        elif candle['low'] <= signal_price - target:
                            outcome = "Win"
                            pnl = target
                            exit_price = signal_price - target
                            break
                
                trade = {
                    'timestamp': signal_time,
                    'signal': signal_type,
                    'entry_price': signal_price,
                    'exit_price': exit_price,
                    'outcome': outcome,
                    'pnl': pnl,
                    'confidence': signal['confidence']
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
        signals = self.simple_supertrend_ema_strategy(df)
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
        
        # Basic statistics
        total_trades = len(trades)
        wins = len(df[df['outcome'] == 'Win'])
        losses = len(df[df['outcome'] == 'Loss'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        max_profit = df['pnl'].max()
        max_loss = df['pnl'].min()
        
        # Print results
        print(f"\n📊 BACKTEST RESULTS")
        print("=" * 50)
        print(f"📈 Total Trades: {total_trades}")
        print(f"✅ Wins: {wins}")
        print(f"❌ Losses: {losses}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"💰 Total P&L: ₹{total_pnl:.2f}")
        print(f"📊 Average P&L: ₹{avg_pnl:.2f}")
        print(f"🚀 Max Profit: ₹{max_profit:.2f}")
        print(f"📉 Max Loss: ₹{max_loss:.2f}")
        
        if total_pnl > 0:
            print(f"🎉 RESULT: PROFITABLE")
        else:
            print(f"⚠️ RESULT: LOSS MAKING")
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_profit': max_profit,
            'max_loss': max_loss
        }

def main():
    parser = argparse.ArgumentParser(description="Simple Backtesting")
    parser.add_argument("--symbol", type=str, default="NSE:NIFTY50-INDEX", help="Symbol to test")
    parser.add_argument("--timeframe", type=str, default="5min", help="Timeframe to test")
    parser.add_argument("--days", type=int, default=180, help="Days to backtest")
    args = parser.parse_args()
    
    backtester = SimpleBacktester()
    
    # Run backtest
    trades = backtester.run_backtest(args.symbol, args.timeframe, args.days)
    
    # Analyze results
    if trades:
        backtester.analyze_results(trades)
    else:
        logger.error("❌ Backtest failed")

if __name__ == "__main__":
    main() 