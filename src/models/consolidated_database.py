#!/usr/bin/env python3
"""
Consolidated Trading Database
============================
Single database for all markets with comprehensive metrics tracking
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

class ConsolidatedTradingDatabase:
    """Consolidated database for all trading markets"""
    
    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
        self.tz = ZoneInfo("Asia/Kolkata")
        self.init_database()
    
    def init_database(self):
        """Initialize database with all tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        price REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        strength TEXT,
                        confirmed BOOLEAN DEFAULT FALSE,
                        executed BOOLEAN DEFAULT FALSE,
                        rejection_reason TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create open_trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS open_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE NOT NULL,
                        market TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        entry_time TEXT NOT NULL,
                        stop_loss_price REAL,
                        take_profit_price REAL,
                        status TEXT DEFAULT 'OPEN',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create closed_trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS closed_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT NOT NULL,
                        market TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        entry_time TEXT NOT NULL,
                        exit_time TEXT NOT NULL,
                        exit_reason TEXT,
                        pnl REAL,
                        commission REAL DEFAULT 0.0,
                        duration_minutes INTEGER,
                        status TEXT DEFAULT 'CLOSED',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create system_health table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market TEXT NOT NULL,
                        status TEXT NOT NULL,
                        memory_usage REAL,
                        cpu_usage REAL,
                        active_connections INTEGER,
                        last_heartbeat TEXT NOT NULL,
                        error_count INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
                # Verify tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                logger.info(f"Database initialized successfully with {len(tables)} tables: {[t[0] for t in tables]}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_open_trade(self, trade_id: str, market: str, symbol: str, strategy: str, 
                       signal: str, entry_price: float, quantity: float, 
                       entry_time: datetime, stop_loss_price: float, 
                       take_profit_price: float) -> bool:
        """Save an open trade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO open_trades 
                    (trade_id, market, symbol, strategy, signal, entry_price, quantity, 
                     entry_time, stop_loss_price, take_profit_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (trade_id, market, symbol, strategy, signal, entry_price, quantity,
                      entry_time.isoformat(), stop_loss_price, take_profit_price))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save open trade: {e}")
            return False
    
    def close_trade(self, trade_id: str, market: str, exit_price: float, 
                   exit_time: datetime, exit_reason: str, pnl: float, 
                   commission: float = 0.0) -> bool:
        """Close a trade and move to closed_trades"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the open trade
                cursor.execute('''
                    SELECT * FROM open_trades WHERE trade_id = ? AND market = ?
                ''', (trade_id, market))
                trade = cursor.fetchone()
                
                if not trade:
                    logger.warning(f"Trade {trade_id} not found in open trades")
                    return False
                
                # Calculate duration
                entry_time = datetime.fromisoformat(trade[8])  # entry_time is at index 8
                duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
                
                # Insert into closed_trades
                cursor.execute('''
                    INSERT INTO closed_trades 
                    (trade_id, market, symbol, strategy, signal, entry_price, exit_price, 
                     quantity, entry_time, exit_time, exit_reason, pnl, commission, 
                     duration_minutes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (trade_id, market, trade[3], trade[4], trade[5], trade[6], exit_price,
                      trade[7], trade[8], exit_time.isoformat(), exit_reason, pnl, 
                      commission, duration_minutes))
                
                # Remove from open_trades
                cursor.execute('''
                    DELETE FROM open_trades WHERE trade_id = ? AND market = ?
                ''', (trade_id, market))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to close trade: {e}")
            return False
    
    def save_signal(self, market: str, symbol: str, strategy: str, signal: str, 
                   confidence: float, price: float, timestamp: datetime, 
                   timeframe: str, strength: str = None, confirmed: bool = False, 
                   rejection_reason: str = None) -> int:
        """Save a trading signal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals 
                    (market, symbol, strategy, signal, confidence, price, timestamp, 
                     timeframe, strength, confirmed, executed, rejection_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (market, symbol, strategy, signal, confidence, price, 
                     timestamp.isoformat(), timeframe, strength, confirmed, False, rejection_reason))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return -1
    
    def update_signal_executed_status(self, signal_id: int, executed: bool, rejection_reason: str = None) -> bool:
        """Update the executed status of a signal and optionally add a rejection reason."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE signals
                    SET executed = ?, rejection_reason = ?
                    WHERE id = ?
                ''', (executed, rejection_reason, signal_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update signal executed status for signal_id {signal_id}: {e}")
            return False
    def update_signal_execution_status(self, signal_id: int, executed: bool, rejection_reason: str = None) -> bool:
        """Update the execution status of a signal (alias for update_signal_executed_status)."""
        return self.update_signal_executed_status(signal_id, executed, rejection_reason)
    def get_recent_signals_with_execution(self, market: str, executed: bool = None, limit: int = 5) -> List[Tuple]:
        """Get recent signals with execution status and rejection reasons."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = '''
                    SELECT id, market, symbol, strategy, signal, confidence, price, timestamp, 
                           timeframe, strength, confirmed, executed, rejection_reason
                    FROM signals 
                    WHERE market = ?
                '''
                params = [market]
                if executed is not None:
                    query += ' AND executed = ?'
                    params.append(1 if executed else 0)
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                cursor.execute(query, tuple(params))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get recent signals with execution status: {e}")
            return []
    
    def get_open_trades(self, market: str = None) -> List[Tuple]:
        """Get open trades, optionally filtered by market"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if market:
                    cursor.execute('''
                        SELECT * FROM open_trades WHERE market = ? AND status = 'OPEN'
                        ORDER BY entry_time DESC
                    ''', (market,))
                else:
                    cursor.execute('''
                        SELECT * FROM open_trades WHERE status = 'OPEN'
                        ORDER BY entry_time DESC
                    ''')
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}")
            return []
    
    def get_market_stats(self, market: str) -> Dict[str, Any]:
        """Get comprehensive market statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Open trades count
                cursor.execute('''
                    SELECT COUNT(*) FROM open_trades 
                    WHERE market = ? AND status = 'OPEN'
                ''', (market,))
                open_trades = cursor.fetchone()[0]
                
                # Closed trades stats
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_closed,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MIN(pnl) as worst_trade,
                        MAX(pnl) as best_trade,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(CASE WHEN exit_reason = 'TARGET_HIT' THEN 1 ELSE 0 END) as target_hits,
                        SUM(CASE WHEN exit_reason = 'STOP_LOSS' THEN 1 ELSE 0 END) as stop_losses,
                        SUM(CASE WHEN exit_reason = 'TIME_EXIT' THEN 1 ELSE 0 END) as time_exits
                    FROM closed_trades 
                    WHERE market = ? AND pnl IS NOT NULL
                ''', (market,))
                closed_stats = cursor.fetchone()
                
                total_closed = closed_stats[0] or 0
                total_pnl = closed_stats[1] or 0
                avg_pnl = closed_stats[2] or 0
                worst_trade = closed_stats[3] or 0
                best_trade = closed_stats[4] or 0
                winning_trades = closed_stats[5] or 0
                losing_trades = closed_stats[6] or 0
                target_hits = closed_stats[7] or 0
                stop_losses = closed_stats[8] or 0
                time_exits = closed_stats[9] or 0
                
                win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0
                
                return {
                    'market': market,
                    'open_trades': open_trades,
                    'closed_trades': total_closed,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'worst_trade': worst_trade,
                    'best_trade': best_trade,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'target_hits': target_hits,
                    'stop_losses': stop_losses,
                    'time_exits': time_exits,
                    'win_rate': win_rate
                }
        except Exception as e:
            logger.error(f"Failed to get market stats: {e}")
            return {}

    def get_recent_trades(self, market: str, limit: int = 5) -> List[Tuple]:
        """Get recent trade entries for a market."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT trade_id, market, symbol, strategy, signal, entry_price, quantity, 
                           entry_time, stop_loss_price, take_profit_price
                    FROM open_trades 
                    WHERE market = ? 
                    ORDER BY entry_time DESC 
                    LIMIT ?
                ''', (market, limit))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get recent trades for market {market}: {e}")
            return []

def initialize_connection_pools():
    """Initialize database connection pools."""
    try:
        # This is a placeholder for connection pool initialization
        # In a production system, you would initialize actual connection pools here
        logger.info("Database connection pools initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize connection pools: {e}")
        return False

