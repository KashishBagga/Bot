#!/usr/bin/env python3
"""
Backtest Database Module
Separate database for backtesting results with strategy-specific tables
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

class BacktestDatabase:
    """Database manager for backtesting results."""
    
    def __init__(self, db_path: str = "backtest_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize backtesting database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # ========================================
                # BACKTEST SESSIONS
                # ========================================
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_sessions (
                        session_id TEXT PRIMARY KEY,
                        session_name TEXT NOT NULL,
                        start_date DATETIME NOT NULL,
                        end_date DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        initial_capital REAL NOT NULL,
                        final_capital REAL NOT NULL,
                        total_return REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL,
                        profit_factor REAL,
                        total_trades INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        avg_trade_duration REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # ========================================
                # STRATEGY-SPECIFIC RESULTS
                # ========================================
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        total_signals INTEGER NOT NULL,
                        executed_signals INTEGER NOT NULL,
                        rejected_signals INTEGER NOT NULL,
                        total_trades INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        losing_trades INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        total_pnl REAL NOT NULL,
                        avg_pnl REAL NOT NULL,
                        max_profit REAL NOT NULL,
                        max_loss REAL NOT NULL,
                        profit_factor REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        avg_trade_duration REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES backtest_sessions(session_id)
                    )
                """)
                
                # ========================================
                # BACKTEST SIGNALS (ALL GENERATED)
                # ========================================
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        strategy_name TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        reasoning TEXT,
                        stop_loss REAL,
                        target1 REAL,
                        target2 REAL,
                        target3 REAL,
                        position_multiplier REAL DEFAULT 1.0,
                        status TEXT DEFAULT 'GENERATED', -- GENERATED, EXECUTED, REJECTED
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES backtest_sessions(session_id)
                    )
                """)
                
                # ========================================
                # REJECTED SIGNALS (DETAILED REASONS)
                # ========================================
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_rejected_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        signal_id INTEGER NOT NULL,
                        timestamp DATETIME NOT NULL,
                        strategy_name TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        rejection_reason TEXT NOT NULL,
                        rejection_category TEXT NOT NULL, -- LOW_CONFIDENCE, EXPOSURE_LIMIT, RISK_LIMIT, etc.
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES backtest_sessions(session_id),
                        FOREIGN KEY (signal_id) REFERENCES backtest_signals(id)
                    )
                """)
                
                # ========================================
                # BACKTEST TRADES (EXECUTED SIGNALS)
                # ========================================
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        signal_id INTEGER NOT NULL,
                        strategy_name TEXT NOT NULL,
                        entry_timestamp DATETIME NOT NULL,
                        exit_timestamp DATETIME,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        quantity INTEGER NOT NULL,
                        pnl REAL,
                        returns REAL,
                        duration_minutes REAL,
                        exit_reason TEXT, -- TARGET_HIT, STOP_LOSS, TIME_EXIT, etc.
                        status TEXT DEFAULT 'OPEN', -- OPEN, CLOSED
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES backtest_sessions(session_id),
                        FOREIGN KEY (signal_id) REFERENCES backtest_signals(id)
                    )
                """)
                
                # ========================================
                # EQUITY CURVE (DAILY CAPITAL TRACKING)
                # ========================================
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_equity_curve (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        capital REAL NOT NULL,
                        open_trades INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        cumulative_return REAL DEFAULT 0,
                        drawdown REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES backtest_sessions(session_id)
                    )
                """)
                
                # ========================================
                # STRATEGY PERFORMANCE BREAKDOWN
                # ========================================
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance_breakdown (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        date DATE NOT NULL,
                        signals_generated INTEGER DEFAULT 0,
                        signals_executed INTEGER DEFAULT 0,
                        signals_rejected INTEGER DEFAULT 0,
                        trades_closed INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES backtest_sessions(session_id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_signals_session ON backtest_signals(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_signals_strategy ON backtest_signals(strategy_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_signals_timestamp ON backtest_signals(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_trades_session ON backtest_trades(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_trades_strategy ON backtest_trades(strategy_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_rejected_session ON backtest_rejected_signals(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_results_session ON strategy_results(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_curve_session ON backtest_equity_curve(session_id)")
                
                conn.commit()
                logger.info("✅ Backtest database schema created successfully")
                
        except Exception as e:
            logger.error(f"❌ Error initializing backtest database: {e}")
            raise
    
    def create_backtest_session(self, session_id: str, session_name: str, start_date: datetime, 
                               end_date: datetime, symbol: str, timeframe: str, initial_capital: float) -> bool:
        """Create a new backtest session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO backtest_sessions 
                    (session_id, session_name, start_date, end_date, symbol, timeframe, initial_capital, 
                     final_capital, total_return, max_drawdown, sharpe_ratio, profit_factor, 
                     total_trades, winning_trades, win_rate, avg_trade_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, session_name, start_date, end_date, symbol, timeframe, initial_capital,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0))
                conn.commit()
                logger.info(f"✅ Created backtest session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Error creating backtest session: {e}")
            return False
    
    def save_backtest_signal(self, session_id: str, timestamp: datetime, strategy_name: str, 
                           signal_type: str, price: float, confidence: float, reasoning: str = "",
                           stop_loss: float = None, target1: float = None, target2: float = None, 
                           target3: float = None, position_multiplier: float = 1.0) -> int:
        """Save a backtest signal and return the signal ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert timestamp to string if it's a datetime object
                if isinstance(timestamp, datetime):
                    timestamp_str = timestamp.isoformat()
                else:
                    timestamp_str = str(timestamp)
                
                # Ensure all numeric values are properly converted
                price_val = float(price) if price is not None else 0.0
                confidence_val = float(confidence) if confidence is not None else 0.0
                stop_loss_val = float(stop_loss) if stop_loss is not None else None
                target1_val = float(target1) if target1 is not None else None
                target2_val = float(target2) if target2 is not None else None
                target3_val = float(target3) if target3 is not None else None
                position_multiplier_val = float(position_multiplier) if position_multiplier is not None else 1.0
                
                cursor.execute("""
                    INSERT INTO backtest_signals 
                    (session_id, timestamp, strategy_name, signal_type, price, confidence, reasoning,
                     stop_loss, target1, target2, target3, position_multiplier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, timestamp_str, strategy_name, signal_type, price_val, confidence_val, reasoning,
                      stop_loss_val, target1_val, target2_val, target3_val, position_multiplier_val))
                signal_id = cursor.lastrowid
                conn.commit()
                return signal_id
        except Exception as e:
            logger.error(f"❌ Error saving backtest signal: {e}")
            return None
    
    def save_rejected_signal(self, session_id: str, signal_id: int, timestamp: datetime, 
                           strategy_name: str, signal_type: str, price: float, confidence: float,
                           rejection_reason: str, rejection_category: str) -> bool:
        """Save a rejected signal with detailed reason."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO backtest_rejected_signals 
                    (session_id, signal_id, timestamp, strategy_name, signal_type, price, confidence,
                     rejection_reason, rejection_category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, signal_id, timestamp, strategy_name, signal_type, price, confidence,
                      rejection_reason, rejection_category))
                
                # Update signal status to REJECTED
                cursor.execute("""
                    UPDATE backtest_signals SET status = 'REJECTED' WHERE id = ?
                """, (signal_id,))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Error saving rejected signal: {e}")
            return False
    
    def save_backtest_trade(self, session_id: str, signal_id: int, strategy_name: str, 
                          entry_timestamp: datetime, entry_price: float, quantity: int) -> int:
        """Save a backtest trade and return the trade ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert timestamp to string if it's a datetime object
                if isinstance(entry_timestamp, datetime):
                    entry_timestamp_str = entry_timestamp.isoformat()
                else:
                    entry_timestamp_str = str(entry_timestamp)
                
                # Ensure all numeric values are properly converted
                entry_price_val = float(entry_price) if entry_price is not None else 0.0
                quantity_val = int(quantity) if quantity is not None else 1
                signal_id_val = int(signal_id) if signal_id is not None else 0
                
                cursor.execute("""
                    INSERT INTO backtest_trades 
                    (session_id, signal_id, strategy_name, entry_timestamp, entry_price, quantity)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, signal_id_val, strategy_name, entry_timestamp_str, entry_price_val, quantity_val))
                trade_id = cursor.lastrowid
                
                # Update signal status to EXECUTED
                cursor.execute("""
                    UPDATE backtest_signals SET status = 'EXECUTED' WHERE id = ?
                """, (signal_id_val,))
                
                conn.commit()
                return trade_id
        except Exception as e:
            logger.error(f"❌ Error saving backtest trade: {e}")
            return None
    
    def close_backtest_trade(self, trade_id: int, exit_timestamp: datetime, exit_price: float,
                           pnl: float, exit_reason: str) -> bool:
        """Close a backtest trade."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert timestamp to string if it's a datetime object
                if isinstance(exit_timestamp, datetime):
                    exit_timestamp_str = exit_timestamp.isoformat()
                else:
                    exit_timestamp_str = str(exit_timestamp)
                
                # Ensure all numeric values are properly converted
                exit_price_val = float(exit_price) if exit_price is not None else 0.0
                pnl_val = float(pnl) if pnl is not None else 0.0
                trade_id_val = int(trade_id) if trade_id is not None else 0
                
                # Get trade details for duration calculation
                cursor.execute("SELECT entry_timestamp FROM backtest_trades WHERE id = ?", (trade_id_val,))
                result = cursor.fetchone()
                if not result:
                    return False
                
                entry_timestamp = datetime.fromisoformat(result[0])
                duration_minutes = (exit_timestamp - entry_timestamp).total_seconds() / 60
                returns = (pnl_val / (exit_price_val * 1)) * 100  # Assuming quantity = 1 for simplicity
                
                cursor.execute("""
                    UPDATE backtest_trades 
                    SET exit_timestamp = ?, exit_price = ?, pnl = ?, returns = ?, 
                        duration_minutes = ?, exit_reason = ?, status = 'CLOSED'
                    WHERE id = ?
                """, (exit_timestamp_str, exit_price_val, pnl_val, returns, duration_minutes, exit_reason, trade_id_val))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Error closing backtest trade: {e}")
            return False
    
    def save_equity_point(self, session_id: str, timestamp: datetime, capital: float, 
                         open_trades: int, daily_pnl: float, cumulative_return: float, drawdown: float) -> bool:
        """Save an equity curve point."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO backtest_equity_curve 
                    (session_id, timestamp, capital, open_trades, daily_pnl, cumulative_return, drawdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, timestamp, capital, open_trades, daily_pnl, cumulative_return, drawdown))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Error saving equity point: {e}")
            return False
    
    def finalize_backtest_session(self, session_id: str, final_capital: float, total_return: float,
                                max_drawdown: float, sharpe_ratio: float, profit_factor: float,
                                total_trades: int, winning_trades: int, win_rate: float,
                                avg_trade_duration: float) -> bool:
        """Finalize a backtest session with final metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE backtest_sessions 
                    SET final_capital = ?, total_return = ?, max_drawdown = ?, sharpe_ratio = ?,
                        profit_factor = ?, total_trades = ?, winning_trades = ?, win_rate = ?, avg_trade_duration = ?
                    WHERE session_id = ?
                """, (final_capital, total_return, max_drawdown, sharpe_ratio, profit_factor,
                      total_trades, winning_trades, win_rate, avg_trade_duration, session_id))
                conn.commit()
                logger.info(f"✅ Finalized backtest session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Error finalizing backtest session: {e}")
            return False
    
    def save_strategy_results(self, session_id: str, strategy_name: str, total_signals: int,
                            executed_signals: int, rejected_signals: int, total_trades: int,
                            winning_trades: int, losing_trades: int, win_rate: float, total_pnl: float,
                            avg_pnl: float, max_profit: float, max_loss: float, profit_factor: float,
                            sharpe_ratio: float, max_drawdown: float, avg_trade_duration: float) -> bool:
        """Save strategy-specific results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_results 
                    (session_id, strategy_name, total_signals, executed_signals, rejected_signals,
                     total_trades, winning_trades, losing_trades, win_rate, total_pnl, avg_pnl,
                     max_profit, max_loss, profit_factor, sharpe_ratio, max_drawdown, avg_trade_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, strategy_name, total_signals, executed_signals, rejected_signals,
                      total_trades, winning_trades, losing_trades, win_rate, total_pnl, avg_pnl,
                      max_profit, max_loss, profit_factor, sharpe_ratio, max_drawdown, avg_trade_duration))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Error saving strategy results: {e}")
            return False
    
    def get_backtest_summary(self, session_id: str) -> Dict:
        """Get comprehensive backtest summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get session details
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM backtest_sessions WHERE session_id = ?", (session_id,))
                session = cursor.fetchone()
                
                if not session:
                    return None
                
                # Get strategy results
                cursor.execute("SELECT * FROM strategy_results WHERE session_id = ?", (session_id,))
                strategies = cursor.fetchall()
                
                # Get signal counts
                cursor.execute("""
                    SELECT COUNT(*) as total, 
                           SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed,
                           SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected
                    FROM backtest_signals WHERE session_id = ?
                """, (session_id,))
                signal_counts = cursor.fetchone()
                
                return {
                    'session': session,
                    'strategies': strategies,
                    'signal_counts': signal_counts
                }
        except Exception as e:
            logger.error(f"❌ Error getting backtest summary: {e}")
            return None
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all backtest sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, session_name, start_date, end_date, symbol, timeframe,
                           initial_capital, final_capital, total_return, total_trades, win_rate
                    FROM backtest_sessions 
                    ORDER BY created_at DESC
                """)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"❌ Error getting sessions: {e}")
            return [] 