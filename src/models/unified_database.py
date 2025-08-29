#!/usr/bin/env python3
"""
Unified Database for Trading System
Handles all database operations for backtesting and paper trading
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)


class UnifiedDatabase:
    def __init__(self, db_path: str = "trading_signals.db"):
        """Initialize database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing tables
                cursor.execute("DROP TABLE IF EXISTS trading_signals")
                cursor.execute("DROP TABLE IF EXISTS rejected_signals")
                cursor.execute("DROP TABLE IF EXISTS open_option_positions")
                cursor.execute("DROP TABLE IF EXISTS closed_option_positions")
                cursor.execute("DROP TABLE IF EXISTS equity_curve")
                cursor.execute("DROP TABLE IF EXISTS performance_metrics")
                
                # Create trading_signals table
                cursor.execute("""
                    CREATE TABLE trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                    strategy TEXT NOT NULL,
                        signal TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL,
                        reasoning TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create rejected_signals table
                cursor.execute("""
                    CREATE TABLE rejected_signals (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        underlying TEXT NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        reasoning TEXT,
                        rejection_reason TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create open_option_positions table
                cursor.execute("""
                    CREATE TABLE open_option_positions (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        contract_symbol TEXT NOT NULL,
                        underlying TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity INTEGER NOT NULL,
                        lot_size INTEGER NOT NULL,
                        strike REAL NOT NULL,
                        expiry DATETIME NOT NULL,
                        option_type TEXT NOT NULL,
                        status TEXT DEFAULT 'OPEN',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
                # Create closed_option_positions table
                cursor.execute("""
                    CREATE TABLE closed_option_positions (
                        id TEXT PRIMARY KEY,
                        entry_timestamp DATETIME NOT NULL,
                        exit_timestamp DATETIME,
                        contract_symbol TEXT NOT NULL,
                        underlying TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                    exit_price REAL,
                        quantity INTEGER NOT NULL,
                        lot_size INTEGER NOT NULL,
                        strike REAL NOT NULL,
                        expiry DATETIME NOT NULL,
                        option_type TEXT NOT NULL,
                        pnl REAL,
                    exit_reason TEXT,
                        status TEXT DEFAULT 'CLOSED',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
            
                # Create equity_curve table
                cursor.execute("""
                    CREATE TABLE equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        capital REAL NOT NULL,
                        open_trades INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
            
                # Create performance_metrics table
                cursor.execute("""
                    CREATE TABLE performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                        avg_pnl REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        profit_factor REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
            
            conn.commit()
            logger.info("✅ Unified database schema created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
            raise
    
    def save_trading_signal(self, signal):
        """Save trading signal to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trading_signals 
                    (timestamp, strategy, signal, symbol, price, confidence, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal['timestamp'],
                    signal['strategy'],
                    signal['signal'],
                    signal.get('symbol', ''),
                    signal['price'],
                    signal.get('confidence', 0),
                    signal.get('reasoning', '')
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving trading signal: {e}")
    
    def save_rejected_signal(self, rejected_signal):
        """Save rejected signal to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO rejected_signals 
                    (id, timestamp, strategy, signal_type, underlying, price, confidence, reasoning, rejection_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rejected_signal.id,
                    rejected_signal.timestamp,
                    rejected_signal.strategy,
                    rejected_signal.signal_type,
                    rejected_signal.underlying,
                    rejected_signal.price,
                    rejected_signal.confidence,
                    rejected_signal.reasoning,
                    rejected_signal.rejection_reason
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving rejected signal: {e}")
    
    def save_open_option_position(self, trade):
        """Save open option position to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO open_option_positions 
                    (id, timestamp, contract_symbol, underlying, strategy, signal_type, 
                     entry_price, quantity, lot_size, strike, expiry, option_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id,
                    trade.timestamp,
                    trade.contract_symbol,
                    trade.underlying,
                    trade.strategy,
                    trade.signal_type,
                    trade.entry_price,
                    trade.quantity,
                    trade.lot_size,
                    trade.strike,
                    trade.expiry,
                    trade.option_type
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving open option position: {e}")
    
    def update_option_position_status(self, trade_id: str, status: str, pnl: float = None, exit_reason: str = None):
        """Update option position status and move to closed table if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                if status == 'CLOSED':
                    # Get the open position
                    cursor.execute("SELECT * FROM open_option_positions WHERE id = ?", (trade_id,))
                    open_position = cursor.fetchone()
                    
                    if open_position:
                        # Insert into closed table
                        cursor.execute("""
                            INSERT INTO closed_option_positions 
                            (id, entry_timestamp, exit_timestamp, contract_symbol, underlying, strategy, signal_type,
                             entry_price, exit_price, quantity, lot_size, strike, expiry, option_type, pnl, exit_reason)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            open_position[0],  # id
                            open_position[1],  # timestamp
                            datetime.now(),   # exit_timestamp
                            open_position[2],  # contract_symbol
                            open_position[3],  # underlying
                            open_position[4],  # strategy
                            open_position[5],  # signal_type
                            open_position[6],  # entry_price
                            None,             # exit_price (will be updated)
                            open_position[7],  # quantity
                            open_position[8],  # lot_size
                            open_position[9],  # strike
                            open_position[10], # expiry
                            open_position[11], # option_type
                            pnl,              # pnl
                            exit_reason       # exit_reason
                        ))
                        
                        # Delete from open table
                        cursor.execute("DELETE FROM open_option_positions WHERE id = ?", (trade_id,))
            
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error updating option position status: {e}")
    
    def save_equity_point(self, timestamp: datetime, capital: float, open_trades: int, daily_pnl: float):
        """Save equity curve point."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO equity_curve (timestamp, capital, open_trades, daily_pnl)
                    VALUES (?, ?, ?, ?)
                """, (timestamp, capital, open_trades, daily_pnl))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving equity point: {e}")
    
    def save_performance_metrics(self, metrics: Dict):
        """Save performance metrics for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_metrics 
                    (session_date, initial_capital, final_capital, total_trades, winning_trades, 
                     losing_trades, win_rate, total_pnl, avg_pnl, max_drawdown, rejected_signals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().date(),
                    metrics['initial_capital'],
                    metrics['current_capital'],
                    metrics['total_trades'],
                    metrics['winning_trades'],
                    metrics['losing_trades'],
                    metrics['win_rate'],
                    metrics['total_pnl'],
                    metrics['avg_pnl'],
                    metrics['max_drawdown'],
                    metrics.get('rejected_signals', 0)
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving performance metrics: {e}")

    def get_open_option_positions(self) -> List[Dict]:
        """Get all open option positions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM open_option_positions WHERE status = 'OPEN'")
                rows = cursor.fetchall()
                
                positions = []
                for row in rows:
                    positions.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'contract_symbol': row[2],
                        'underlying': row[3],
                        'strategy': row[4],
                        'signal_type': row[5],
                        'entry_price': row[6],
                        'quantity': row[7],
                        'lot_size': row[8],
                        'strike': row[9],
                        'expiry': row[10],
                        'option_type': row[11]
                    })
                
                return positions
        except Exception as e:
            logger.error(f"❌ Error getting open positions: {e}")
            return []
    
    def get_trading_signals(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get trading signals with optional date filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                query = "SELECT * FROM trading_signals"
                params = []
                
                if start_date and end_date:
                    query += " WHERE timestamp BETWEEN ? AND ?"
                    params = [start_date, end_date]
                elif start_date:
                    query += " WHERE timestamp >= ?"
                    params = [start_date]
                elif end_date:
                    query += " WHERE timestamp <= ?"
                    params = [end_date]
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                signals = []
                for row in rows:
                    signals.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'strategy': row[2],
                        'signal': row[3],
                        'symbol': row[4],
                        'price': row[5],
                        'confidence': row[6],
                        'reasoning': row[7]
                    })
                
                return signals
        except Exception as e:
            logger.error(f"❌ Error getting trading signals: {e}")
            return []
    
    def get_rejected_signals(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get rejected signals with optional date filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                query = "SELECT * FROM rejected_signals"
                params = []
                
                if start_date and end_date:
                    query += " WHERE timestamp BETWEEN ? AND ?"
                    params = [start_date, end_date]
                elif start_date:
                    query += " WHERE timestamp >= ?"
                    params = [start_date]
                elif end_date:
                    query += " WHERE timestamp <= ?"
                    params = [end_date]
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                signals = []
                for row in rows:
                    signals.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'strategy': row[2],
                        'signal_type': row[3],
                        'underlying': row[4],
                        'price': row[5],
                        'confidence': row[6],
                        'reasoning': row[7],
                        'rejection_reason': row[8]
                    })
                
                return signals
        except Exception as e:
            logger.error(f"❌ Error getting rejected signals: {e}")
            return []

    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                # Get total signals
                cursor.execute("SELECT COUNT(*) FROM trading_signals")
                total_signals = cursor.fetchone()[0]
                
                # Get total rejected signals
                cursor.execute("SELECT COUNT(*) FROM rejected_signals")
                total_rejected = cursor.fetchone()[0]
                
                # Get total trades
                cursor.execute("SELECT COUNT(*) FROM closed_option_positions")
                total_trades = cursor.fetchone()[0]
                
                # Get winning trades
                cursor.execute("SELECT COUNT(*) FROM closed_option_positions WHERE pnl > 0")
                winning_trades = cursor.fetchone()[0]
                
                # Get total P&L
                cursor.execute("SELECT SUM(pnl) FROM closed_option_positions")
                total_pnl = cursor.fetchone()[0] or 0
                
                # Get average P&L
                cursor.execute("SELECT AVG(pnl) FROM closed_option_positions")
                avg_pnl = cursor.fetchone()[0] or 0
                
                return {
                    'total_signals': total_signals,
                    'total_rejected': total_rejected,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl
                }
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {} 