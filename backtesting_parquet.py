#!/usr/bin/env python3
"""
Parquet-Based Backtesting System
Ultra-fast backtesting using pre-stored parquet data with all timeframes readily available.
"""
import sys
import argparse
import time
import concurrent.futures
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import src.warning_filters
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.parquet_data_store import ParquetDataStore
from all_strategies_parquet import run_strategy
from src.models.enhanced_rejected_signals import log_rejected_signal_backtest
from all_strategies_parquet import add_technical_indicators

def setup_database():
    """Setup database with required tables"""
    conn = sqlite3.connect('trading_signals.db')
    cursor = conn.cursor()
    
    # Create trading_signals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence TEXT,
            confidence_score INTEGER,
            price REAL,
            stop_loss REAL,
            target REAL,
            target2 REAL,
            target3 REAL,
            reasoning TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create rejected_signals table (legacy)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rejected_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            rejection_reason TEXT NOT NULL,
            confidence TEXT,
            confidence_score INTEGER,
            price REAL,
            stop_loss REAL,
            target REAL,
            reasoning TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create trades_backtest table to store valid trade outcomes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades_backtest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            price REAL,
            stop_loss REAL,
            target REAL,
            target2 REAL,
            target3 REAL,
            reasoning TEXT,
            confidence TEXT,
            confidence_score INTEGER,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            exit_time TEXT,
            market_condition TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def ensure_recent_data(symbols: list, timeframe: str, max_staleness_days: int = 1):
    """Ensure parquet data is fresh for given symbols and timeframe. Optionally runs sync."""
    data_store = ParquetDataStore()
    stale = []
        for symbol in symbols:
        df = data_store.load_data(symbol, timeframe, days_back=None)
        if df.empty:
            stale.append(symbol)
            continue
        last_ts = df.index.max()
        if (datetime.now() - last_ts).days > max_staleness_days:
            stale.append(symbol)
    if not stale:
        return
    print(f"⚠️ Stale/missing data detected for {timeframe}: {stale}")
    if os.environ.get('ALLOW_SYNC', '0') == '1':
        print("🔄 Running sync_parquet_data.py to fetch missing data...")
        try:
            cmd = [
                sys.executable, 'sync_parquet_data.py',
                '--symbols', ','.join(stale),
                '--timeframes', timeframe
            ]
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"❌ Sync failed to run: {e}")

def run_backtest_with_enhanced_logging(strategy_name, symbol, timeframe, days):
    """Run backtest with enhanced rejected signals logging"""
    print(f"\n🔄 Running backtest: {strategy_name} on {symbol} ({timeframe}, {days} days)")

    # Generate unique backtest run ID
    backtest_run_id = f"backtest_{strategy_name}_{symbol}_{timeframe}_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backtest_parameters = {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "run_timestamp": datetime.now().isoformat()
    }

    # Load data
    data_store = ParquetDataStore()
    data = data_store.load_data(symbol, timeframe, days_back=days)

    if data.empty:
        print(f"❌ No data available for {symbol}")
        return 0, 0

    print(f"📊 Loaded {len(data)} candles for {symbol}")

    # Precompute indicators once
    df_with_indicators = add_technical_indicators(data.copy())

    # Import and instantiate strategy once
    strategy_module = __import__(f"src.strategies.{strategy_name}", fromlist=[strategy_name])
    class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
    strategy_class = getattr(strategy_module, class_name)
    strategy_instance = strategy_class(timeframe_data={})
    # Allow strategy to add its indicators, if needed
    df_with_indicators = strategy_instance.add_indicators(df_with_indicators)

    # Initialize counters
    total_signals = 0
    rejected_signals = 0

    # Iterate efficiently per candle
    for i in range(50, len(df_with_indicators)):
        candle = df_with_indicators.iloc[i]
        current_time = candle.name if hasattr(candle, 'name') else datetime.now()
        future_data = df_with_indicators.iloc[i+1:i+51] if i+1 < len(df_with_indicators) else None

        try:
            # Call analyze with the correct signature per strategy
            if strategy_name == 'insidebar_rsi':
                current_slice = df_with_indicators.iloc[:i+1]
                result = strategy_instance.analyze(current_slice, symbol, future_data)
            else:
                result = strategy_instance.analyze(candle, i, df_with_indicators, future_data)

            if not result or not isinstance(result, dict):
                continue

            signal_type = result.get('signal', 'NO TRADE')
            confidence_score = result.get('confidence_score', 0)
            total_signals += 1

            if total_signals % 50 == 0:
                print(f"  📊 {symbol}: {total_signals} signals processed...")

            # Build enhanced signal payload
            enhanced_signal_data = {
                'timestamp': current_time.isoformat() if hasattr(current_time, 'isoformat') else str(current_time),
                'strategy': strategy_name,
                'symbol': symbol,
                'signal_attempted': signal_type,
                'price': candle.get('close', 0),
                'confidence': result.get('confidence', 'Unknown'),
                'confidence_score': confidence_score,
                'rsi': candle.get('rsi', 0),
                'macd': candle.get('macd', 0),
                'macd_signal': candle.get('macd_signal', 0),
                'macd_histogram': candle.get('macd_histogram', 0),
                'ema_9': candle.get('ema_9', 0),
                'ema_21': candle.get('ema_21', 0),
                'ema_20': candle.get('ema_20', candle.get('ema', 0)),
                'ema_50': candle.get('ema_50', 0),
                'atr': candle.get('atr', 0),
                'supertrend': candle.get('supertrend', 0),
                'supertrend_direction': candle.get('supertrend_direction', 0),
                'bb_upper': candle.get('bb_upper', 0),
                'bb_lower': candle.get('bb_lower', 0),
                'bb_middle': candle.get('bb_middle', 0),
                'volume': candle.get('volume', 0),
                'stop_loss': result.get('stop_loss', 0),
                'target': result.get('target', 0),
                'target2': result.get('target2', 0),
                'target3': result.get('target3', 0),
                'trade_type': 'Intraday',
                'reasoning': result.get('reasoning', result.get('reason', '')),
                'outcome': result.get('outcome', 'Pending'),
                'pnl': result.get('pnl', 0.0),
                'targets_hit': result.get('targets_hit', 0),
                'stoploss_count': result.get('stoploss_count', 0),
                'failure_reason': result.get('failure_reason', ''),
                'exit_time': result.get('exit_time', ''),
                'market_condition': 'Unknown'
            }

            # Rejection policy
            if signal_type == 'NO TRADE' or (confidence_score and confidence_score > 0 and confidence_score < 60):
                enhanced_signal_data['rejection_reason'] = (
                    "No trade signal generated" if signal_type == 'NO TRADE' else f"Low confidence: {confidence_score} < 60 threshold"
                )
                log_rejected_signal_backtest(enhanced_signal_data, future_data, backtest_run_id, backtest_parameters)
                rejected_signals += 1
            else:
                # Valid signal - log to trading_signals table
                conn = sqlite3.connect('trading_signals.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_signals (timestamp, strategy, symbol, signal, confidence, confidence_score, price, stop_loss, target, target2, target3, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    enhanced_signal_data['timestamp'],
                    strategy_name,
                    symbol,
                    signal_type,
                    enhanced_signal_data['confidence'],
                    confidence_score,
                    enhanced_signal_data['price'],
                    enhanced_signal_data['stop_loss'],
                    enhanced_signal_data['target'],
                    enhanced_signal_data['target2'],
                    enhanced_signal_data['target3'],
                    enhanced_signal_data['reasoning']
                ))

                # Also log to trades_backtest with real performance data if present
                try:
                    trade_payload = {
                        'timestamp': enhanced_signal_data['timestamp'],
                        'strategy': strategy_name,
                        'symbol': symbol,
                        'signal': signal_type,
                        'price': enhanced_signal_data['price'],
                        'stop_loss': enhanced_signal_data.get('stop_loss', 0),
                        'target': enhanced_signal_data.get('target', 0),
                        'target2': enhanced_signal_data.get('target2', 0),
                        'target3': enhanced_signal_data.get('target3', 0),
                        'reasoning': enhanced_signal_data.get('reasoning', ''),
                        'confidence': enhanced_signal_data.get('confidence', 'Unknown'),
                        'confidence_score': enhanced_signal_data.get('confidence_score', 0),
                        'outcome': enhanced_signal_data.get('outcome', 'Pending'),
                        'pnl': enhanced_signal_data.get('pnl', 0.0),
                        'targets_hit': enhanced_signal_data.get('targets_hit', 0),
                        'stoploss_count': enhanced_signal_data.get('stoploss_count', 0),
                        'exit_time': enhanced_signal_data.get('exit_time', ''),
                        'market_condition': enhanced_signal_data.get('market_condition', 'Unknown')
                    }
                    cursor.execute('''
                        INSERT INTO trades_backtest (
                            timestamp, strategy, symbol, signal, price, stop_loss, target, target2, target3, reasoning,
                            confidence, confidence_score, outcome, pnl, targets_hit, stoploss_count, exit_time, market_condition
                        ) VALUES (
                            :timestamp, :strategy, :symbol, :signal, :price, :stop_loss, :target, :target2, :target3, :reasoning,
                            :confidence, :confidence_score, :outcome, :pnl, :targets_hit, :stoploss_count, :exit_time, :market_condition
                        )
                    ''', trade_payload)
                except Exception as _:
                    pass
                conn.commit()
                conn.close()

        except Exception as e:
            print(f"❌ Error processing candle {i}: {e}")
            continue

    print(f"✅ Backtest completed: {total_signals} total signals, {rejected_signals} rejected")
    return total_signals, rejected_signals

def main():
    """Main backtesting function"""
    print("🚀 BACKTESTING WITH ENHANCED REJECTED SIGNALS")
    print("=" * 60)
    
    # Setup database
    setup_database()
    
    # Test strategies
    strategies = [
        'ema_crossover',
        'supertrend_macd_rsi_ema',
        'supertrend_ema',
        'insidebar_rsi'
    ]
    
    symbols = ['NSE:NIFTYBANK-INDEX', 'NSE:NIFTY50-INDEX']
    timeframe = '5min'
    days = 5
    # Allow environment overrides for full multi-strategy runs
    timeframe = os.environ.get('TIMEFRAME', timeframe)
    try:
        days = int(os.environ.get('DAYS', days))
    except Exception:
        pass

    # Ensure data fresh
    ensure_recent_data(symbols, timeframe)
    
    total_signals_all = 0
    total_rejected_all = 0
    
    for strategy in strategies:
        print(f"\n🧠 Testing Strategy: {strategy}")
        print("-" * 40)
        
        for symbol in symbols:
            try:
                signals, rejected = run_backtest_with_enhanced_logging(strategy, symbol, timeframe, days)
                total_signals_all += signals
                total_rejected_all += rejected
                
                if signals > 0:
                    rejection_rate = (rejected / signals) * 100
                    print(f"  📊 {symbol}: {signals} signals, {rejected} rejected ({rejection_rate:.1f}%)")
                else:
                    print(f"  📊 {symbol}: No signals generated")
                    
        except Exception as e:
                print(f"  ❌ Error testing {symbol}: {e}")
    
    print(f"\n🎯 OVERALL RESULTS")
    print("=" * 60)
    print(f"Total Signals Generated: {total_signals_all}")
    print(f"Total Signals Rejected: {total_rejected_all}")
    
    if total_signals_all > 0:
        overall_rejection_rate = (total_rejected_all / total_signals_all) * 100
        print(f"Overall Rejection Rate: {overall_rejection_rate:.1f}%")
    
    # Show database status
    conn = sqlite3.connect('trading_signals.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM trading_signals")
    valid_signals = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM rejected_signals")
    legacy_rejected = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM rejected_signals_backtest")
    enhanced_rejected = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n📊 DATABASE STATUS")
    print("=" * 60)
    print(f"✅ Valid Signals (trading_signals): {valid_signals}")
    print(f"📋 Legacy Rejected Signals (rejected_signals): {legacy_rejected}")
    print(f"🎯 Enhanced Rejected Signals (rejected_signals_backtest): {enhanced_rejected}")
    
    print(f"\n🎉 BACKTESTING COMPLETE!")
    print("✅ All strategies tested with enhanced rejected signals logging")
    print("✅ P&L calculations available for all rejected signals")
    print("✅ Separate tables maintained for backtesting vs live trading")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parquet backtesting runner")
    parser.add_argument("--strategy", type=str, default=None, help="Strategy name to run (optional)")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to run (optional)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe (e.g., 5min)")
    parser.add_argument("--days", type=int, default=None, help="Days back to load")
    args = parser.parse_args()

    # If args provided, override defaults inside main via simple wrapper
    if args.strategy or args.symbol or args.timeframe or args.days:
        def _run_single():
            print("🚀 BACKTESTING WITH ENHANCED REJECTED SIGNALS")
            print("=" * 60)
            setup_database()
            strategy = args.strategy or 'ema_crossover'
            symbol = args.symbol or 'NSE:NIFTYBANK-INDEX'
            timeframe = args.timeframe or '5min'
            days = args.days or 1

            # Ensure data fresh for the requested run
            ensure_recent_data([symbol], timeframe)
            print(f"\n🧠 Testing Strategy: {strategy}")
            print("-" * 40)
            try:
                signals, rejected = run_backtest_with_enhanced_logging(strategy, symbol, timeframe, days)
                if signals > 0:
                    rejection_rate = (rejected / signals) * 100
                    print(f"  📊 {symbol}: {signals} signals, {rejected} rejected ({rejection_rate:.1f}%)")
                else:
                    print(f"  📊 {symbol}: No signals generated")
        except Exception as e:
                print(f"  ❌ Error testing {symbol}: {e}")
            print("\n🎉 BACKTESTING COMPLETE!")
        _run_single()
    else:
    main() 