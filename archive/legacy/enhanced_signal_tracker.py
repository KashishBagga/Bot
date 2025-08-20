#!/usr/bin/env python3
"""
Enhanced Signal Tracker
Comprehensive signal tracking with profit/loss calculation and outcome analysis
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimized_live_trading_bot import OptimizedLiveTradingBot

class EnhancedSignalTracker:
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_enhanced_database()
        
    def setup_enhanced_database(self):
        """Setup enhanced database schema with profit/loss tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create enhanced live_signals table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS live_signals_enhanced (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence_score INTEGER,
                    entry_price REAL,
                    stop_loss REAL,
                    target REAL,
                    target2 REAL,
                    target3 REAL,
                    reasoning TEXT,
                    
                    -- Enhanced tracking fields
                    status TEXT DEFAULT 'ACTIVE',
                    exit_price REAL,
                    exit_timestamp TEXT,
                    exit_reason TEXT,
                    outcome TEXT,  -- WIN, LOSS, BREAKEVEN, PARTIAL_WIN
                    holding_period_minutes INTEGER,
                    
                    -- P&L tracking
                    pnl_points REAL,
                    pnl_percentage REAL,
                    pnl_amount REAL,
                    position_size INTEGER DEFAULT 1,
                    
                    -- Performance metrics
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    target_hit INTEGER DEFAULT 0,  -- 0=none, 1=target1, 2=target2, 3=target3
                    stop_loss_hit INTEGER DEFAULT 0,  -- 0=no, 1=yes
                    
                    -- Market context
                    market_condition TEXT,
                    volatility_at_entry REAL,
                    volume_at_entry REAL,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create signal performance summary table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    profitable_signals INTEGER DEFAULT 0,
                    loss_signals INTEGER DEFAULT 0,
                    active_signals INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    avg_pnl_points REAL DEFAULT 0.0,
                    avg_pnl_percentage REAL DEFAULT 0.0,
                    total_pnl_amount REAL DEFAULT 0.0,
                    max_profit REAL DEFAULT 0.0,
                    max_loss REAL DEFAULT 0.0,
                    avg_holding_period REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, strategy, symbol)
                )
            ''')
            
            # Create daily summary table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_signals INTEGER DEFAULT 0,
                    profitable_signals INTEGER DEFAULT 0,
                    loss_signals INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    best_strategy TEXT,
                    worst_strategy TEXT,
                    market_condition TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Enhanced database schema created successfully")
            
        except Exception as e:
            print(f"❌ Database setup error: {e}")
    
    def log_signal(self, signal_data: Dict[str, Any]) -> int:
        """Log a new trading signal with enhanced tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Extract signal data
            timestamp = signal_data.get('timestamp', datetime.now().isoformat())
            strategy = signal_data.get('strategy', 'unknown')
            symbol = signal_data.get('symbol', 'unknown')
            signal_type = signal_data.get('signal', 'NO TRADE')
            confidence = signal_data.get('confidence_score', 0)
            entry_price = signal_data.get('price', 0.0)
            stop_loss = signal_data.get('stop_loss', 0.0)
            target = signal_data.get('target', 0.0)
            target2 = signal_data.get('target2', 0.0)
            target3 = signal_data.get('target3', 0.0)
            reasoning = json.dumps(signal_data.get('reasoning', {}))
            
            # Market context (if available)
            market_condition = signal_data.get('market_condition', 'unknown')
            volatility = signal_data.get('volatility', 0.0)
            volume = signal_data.get('volume', 0.0)
            position_size = signal_data.get('position_size', 1)
            
            # Insert signal
            cursor = conn.execute('''
                INSERT INTO live_signals_enhanced (
                    timestamp, strategy, symbol, signal, confidence_score,
                    entry_price, stop_loss, target, target2, target3, reasoning,
                    market_condition, volatility_at_entry, volume_at_entry, position_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, strategy, symbol, signal_type, confidence,
                entry_price, stop_loss, target, target2, target3, reasoning,
                market_condition, volatility, volume, position_size
            ))
            
            signal_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"✅ Signal logged: {strategy} - {symbol} - {signal_type} (ID: {signal_id})")
            return signal_id
            
        except Exception as e:
            print(f"❌ Error logging signal: {e}")
            return -1
    
    def update_signal_outcome(self, signal_id: int, exit_price: float, outcome: str, 
                            pnl_points: float, pnl_percentage: float, 
                            max_favorable_excursion: float = None, 
                            max_adverse_excursion: float = None):
        """Update signal with trading outcome and P&L"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Update signal outcome
            conn.execute('''
                UPDATE live_signals_enhanced 
                SET exit_price = ?, exit_timestamp = ?, outcome = ?, 
                    pnl_points = ?, pnl_percentage = ?, status = 'CLOSED',
                    max_favorable_excursion = ?, max_adverse_excursion = ?
                WHERE id = ?
            ''', (
                exit_price, datetime.now().isoformat(), outcome,
                pnl_points, pnl_percentage, 
                max_favorable_excursion, max_adverse_excursion,
                signal_id
            ))
            
            # Update performance metrics
            self._update_performance_metrics(outcome, pnl_points, pnl_percentage)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Signal {signal_id} outcome updated: {outcome} ({pnl_points:+.2f} points)")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating signal outcome: {e}")
    
    def _update_performance_metrics(self, outcome: str, pnl_points: float, pnl_percentage: float):
        """Update overall performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get current metrics
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM signal_performance WHERE id = 1')
            current = cursor.fetchone()
            
            if current:
                # Update existing metrics
                total_signals = current[1] + 1
                profitable_signals = current[2] + (1 if outcome == 'WIN' else 0)
                total_pnl = current[3] + pnl_points
                win_rate = (profitable_signals / total_signals) * 100
                
                conn.execute('''
                    UPDATE signal_performance 
                    SET total_signals = ?, profitable_signals = ?, 
                        total_pnl = ?, win_rate = ?, last_updated = ?
                    WHERE id = 1
                ''', (total_signals, profitable_signals, total_pnl, win_rate, datetime.now().isoformat()))
            else:
                # Create initial metrics
                profitable_signals = 1 if outcome == 'WIN' else 0
                win_rate = (profitable_signals / 1) * 100
                
                conn.execute('''
                    INSERT INTO signal_performance 
                    (total_signals, profitable_signals, total_pnl, win_rate, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                ''', (1, profitable_signals, pnl_points, win_rate, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance metrics: {e}")
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active signals for monitoring"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM live_signals_enhanced 
                WHERE status = 'ACTIVE' 
                ORDER BY created_at DESC
            ''')
            
            columns = [description[0] for description in cursor.description]
            signals = []
            
            for row in cursor.fetchall():
                signal = dict(zip(columns, row))
                signals.append(signal)
            
            conn.close()
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error getting active signals: {e}")
            return []
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall performance from signal_performance table
            cursor.execute('SELECT * FROM signal_performance WHERE id = 1')
            perf_data = cursor.fetchone()
            
            # Get recent signals statistics
            cursor.execute('''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN pnl_points > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN pnl_points IS NOT NULL THEN pnl_points ELSE 0 END) as total_pnl,
                       AVG(CASE WHEN pnl_points IS NOT NULL THEN pnl_points ELSE 0 END) as avg_pnl,
                       COUNT(DISTINCT DATE(created_at)) as active_days
                FROM live_signals_enhanced 
                WHERE status = 'CLOSED'
            ''')
            
            stats = cursor.fetchone()
            conn.close()
            
            if perf_data:
                report = {
                    'total_signals': perf_data[1],
                    'profitable_signals': perf_data[2],
                    'total_pnl': perf_data[3],
                    'win_rate': perf_data[4],
                    'last_updated': perf_data[5]
                }
            else:
                report = {
                    'total_signals': 0,
                    'profitable_signals': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'last_updated': None
                }
            
            # Add recent stats
            if stats and stats[0] > 0:
                report.update({
                    'recent_total': stats[0],
                    'recent_wins': stats[1] or 0,
                    'recent_pnl': stats[2] or 0.0,
                    'avg_pnl_per_trade': stats[3] or 0.0,
                    'active_days': stats[4] or 0
                })
            else:
                report.update({
                    'recent_total': 0,
                    'recent_wins': 0,
                    'recent_pnl': 0.0,
                    'avg_pnl_per_trade': 0.0,
                    'active_days': 0
                })
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance report: {e}")
            return {}
    
    def print_performance_report(self, days: int = 7):
        """Print formatted performance report"""
        report = self.generate_performance_report()
        
        if not report:
            print("❌ Unable to generate performance report")
            return
        
        print(f"\n📊 PERFORMANCE REPORT - Last {days} Days")
        print("=" * 80)
        
        overall = report['overall']
        print(f"📈 Overall Performance:")
        print(f"   Total Signals: {overall['total_signals']}")
        print(f"   Closed Signals: {overall['closed_signals']}")
        print(f"   Active Signals: {overall['active_signals']}")
        print(f"   Win Rate: {overall['win_rate']:.1f}%")
        print(f"   Average P&L: {overall['avg_pnl_points']:.2f} points ({overall['avg_pnl_percentage']:.2f}%)")
        print(f"   Total P&L: ₹{overall['total_pnl']:.2f}")
        print(f"   Best Trade: {overall['max_profit']:.2f} points")
        print(f"   Worst Trade: {overall['max_loss']:.2f} points")
        print(f"   Avg Holding Period: {overall['avg_holding_period_minutes']:.0f} minutes")
        
        if report['strategy_performance']:
            print(f"\n🎯 Strategy Performance:")
            for strategy in report['strategy_performance']:
                print(f"   {strategy['strategy']}: {strategy['total_signals']} signals, "
                      f"{strategy['win_rate']:.1f}% win rate, ₹{strategy['total_pnl']:.2f} P&L")
        
        if report['daily_breakdown']:
            print(f"\n📅 Daily Breakdown:")
            for day in report['daily_breakdown'][:7]:  # Show last 7 days
                print(f"   {day['date']}: {day['signals']} signals, "
                      f"{day['win_rate']:.1f}% win rate, ₹{day['daily_pnl']:.2f} P&L")

def main():
    """Main function for testing the enhanced signal tracker"""
    print("🚀 ENHANCED SIGNAL TRACKER TEST")
    print("=" * 80)
    
    # Initialize tracker
    tracker = EnhancedSignalTracker()
    
    # Initialize bot for market data
    bot = OptimizedLiveTradingBot()
    
    # Update all active signals
    tracker.update_all_active_signals(bot)
    
    # Generate and print performance report
    tracker.print_performance_report(7)
    
    print("\n✅ Enhanced signal tracking system is ready!")
    print("📊 Database: trading_signals.db")
    print("📋 Tables: live_signals_enhanced, signal_performance, daily_summary")

if __name__ == "__main__":
    main() 