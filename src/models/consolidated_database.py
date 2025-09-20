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
                   confidence: float, price: float, timestamp, 
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
                     timestamp if isinstance(timestamp, str) else timestamp.isoformat(), timeframe, strength, confirmed, False, rejection_reason))
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
    def get_strategy_performance(self, market: str) -> List[Tuple]:
        """Get strategy performance statistics for a market"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        strategy,
                        COUNT(*) as total_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        MIN(pnl) as worst_trade,
                        MAX(pnl) as best_trade,
                        SUM(CASE WHEN exit_reason = 'TARGET_HIT' THEN 1 ELSE 0 END) as targets,
                        SUM(CASE WHEN exit_reason = 'STOP_LOSS' THEN 1 ELSE 0 END) as stops
                    FROM closed_trades 
                    WHERE market = ? AND pnl IS NOT NULL
                    GROUP BY strategy
                    ORDER BY total_pnl DESC
                ''', (market,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get strategy performance for market {market}: {e}")
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



    def get_trade_by_id(self, trade_id: str) -> Optional[Tuple]:
        """Get trade data by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT trade_id, market, symbol, strategy, signal, entry_price, quantity, 
                           entry_time, stop_loss_price, take_profit_price, status
                    FROM open_trades 
                    WHERE trade_id = ? AND status = 'OPEN'
                ''', (trade_id,))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Failed to get trade by ID {trade_id}: {e}")
            return None
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str, pnl: float):
        """Close a trade and move it to closed_trades."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get trade data
                trade_data = self.get_trade_by_id(trade_id)
                if not trade_data:
                    logger.error(f"Trade {trade_id} not found")
                    return False
                
                # Insert into closed_trades
                cursor.execute('''
                    INSERT INTO closed_trades (
                        trade_id, market, symbol, strategy, signal, entry_price, exit_price,
                        quantity, entry_time, exit_time, pnl, exit_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data[0], trade_data[1], trade_data[2], trade_data[3], trade_data[4],
                    trade_data[5], exit_price, trade_data[6], trade_data[7], 
                    datetime.now().isoformat(), pnl, exit_reason
                ))
                
                # Remove from open_trades
                cursor.execute('''
                    DELETE FROM open_trades WHERE trade_id = ?
                ''', (trade_id,))
                
                conn.commit()
                logger.info(f"Trade {trade_id} closed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return False

    def calculate_unrealized_pnl(self, market: str, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L for open trades."""
        try:
            total_unrealized = 0.0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT trade_id, symbol, signal, entry_price, quantity
                    FROM open_trades 
                    WHERE market = ? AND status = 'OPEN'
                ''', (market,))
                
                for trade in cursor.fetchall():
                    trade_id, symbol, signal, entry_price, quantity = trade
                    
                    if symbol in current_prices:
                        current_price = current_prices[symbol]
                        
                        if signal == 'BUY CALL':
                            unrealized = (current_price - entry_price) * quantity
                        else:  # BUY PUT
                            unrealized = (entry_price - current_price) * quantity
                        
                        total_unrealized += unrealized
                
                return total_unrealized
                
        except Exception as e:
            logger.error(f"Failed to calculate unrealized P&L for {market}: {e}")
            return 0.0

    def get_market_statistics(self, market: str) -> Dict[str, Any]:
        """Get comprehensive market statistics for dashboard."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get basic counts
                cursor.execute("SELECT COUNT(*) FROM open_trades WHERE market = ?", (market,))
                open_trades_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM closed_trades WHERE market = ?", (market,))
                closed_trades_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM signals WHERE market = ?", (market,))
                total_signals = cursor.fetchone()[0]
                
                # Get P&L statistics
                cursor.execute("SELECT SUM(pnl) FROM closed_trades WHERE market = ?", (market,))
                total_pnl = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT AVG(pnl) FROM closed_trades WHERE market = ? AND pnl > 0", (market,))
                avg_win = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT AVG(pnl) FROM closed_trades WHERE market = ? AND pnl < 0", (market,))
                avg_loss = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT COUNT(*) FROM closed_trades WHERE market = ? AND pnl > 0", (market,))
                winning_trades = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM closed_trades WHERE market = ? AND pnl < 0", (market,))
                losing_trades = cursor.fetchone()[0]
                
                # Calculate win rate
                win_rate = (winning_trades / max(1, closed_trades_count)) * 100
                
                # Get recent performance (last 7 days)
                cursor.execute("""
                    SELECT SUM(pnl) FROM closed_trades 
                    WHERE market = ? AND date(exit_time) >= date('now', '-7 days')
                """, (market,))
                weekly_pnl = cursor.fetchone()[0] or 0.0
                
                # Get strategy performance
                cursor.execute("""
                    SELECT strategy, COUNT(*), SUM(pnl), AVG(pnl)
                    FROM closed_trades 
                    WHERE market = ? 
                    GROUP BY strategy
                """, (market,))
                strategy_stats = cursor.fetchall()
                
                return {
                    'market': market,
                    'open_trades': open_trades_count,
                    'closed_trades': closed_trades_count,
                    'total_signals': total_signals,
                    'total_pnl': round(total_pnl, 2),
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2),
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': round(win_rate, 2),
                    'weekly_pnl': round(weekly_pnl, 2),
                    'strategy_performance': [
                        {
                            'strategy': row[0],
                            'trades': row[1],
                            'total_pnl': round(row[2] or 0, 2),
                            'avg_pnl': round(row[3] or 0, 2)
                        }
                        for row in strategy_stats
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get market statistics for {market}: {e}")
            return {
                'market': market,
                'open_trades': 0,
                'closed_trades': 0,
                'total_signals': 0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'weekly_pnl': 0.0,
                'strategy_performance': []
            }

        """Update signal execution status by symbol and strategy."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE signals 
                    SET executed = ?, rejection_reason = ?
                    WHERE symbol = ? AND strategy = ? AND executed = 0
                    ORDER BY timestamp DESC LIMIT 1
                """, (1 if executed else 0, rejection_reason, symbol, strategy))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update signal execution: {e}")
            return False


    def update_signal_execution(self, symbol: str, strategy: str, executed: bool, rejection_reason: str = None) -> bool:
        """Update signal execution status by symbol and strategy."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE signals 
                    SET executed = ?, rejection_reason = ?
                    WHERE symbol = ? AND strategy = ? AND executed = 0
                    ORDER BY timestamp DESC LIMIT 1
                """, (1 if executed else 0, rejection_reason, symbol, strategy))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update signal execution: {e}")
            return False

