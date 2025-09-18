#!/usr/bin/env python3
"""
Enhanced Database with Timezone-Aware Timestamps
Proper timezone handling for all database operations
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from zoneinfo import ZoneInfo
import pandas as pd

logger = logging.getLogger(__name__)

class EnhancedTimezoneDatabase:
    """Enhanced database with timezone-aware timestamps"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        self.timezone = ZoneInfo('Asia/Kolkata')
        self.init_database()
    
    def init_database(self):
        """Initialize database with timezone-aware schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Market-specific tables
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS indian_market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,  -- ISO format with timezone
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS crypto_market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,  -- ISO format with timezone
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        timezone TEXT DEFAULT 'UTC',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Signal tables with timezone
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS entry_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        price REAL NOT NULL,
                        timestamp TEXT NOT NULL,  -- ISO format with timezone
                        timeframe TEXT NOT NULL,
                        strength TEXT,
                        indicator_values TEXT,  -- JSON
                        market_condition TEXT,
                        volatility REAL,
                        position_size REAL,
                        stop_loss_price REAL,
                        take_profit_price REAL,
                        confirmed BOOLEAN DEFAULT FALSE,
                        executed BOOLEAN DEFAULT FALSE,
                        execution_reason TEXT,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS exit_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT UNIQUE NOT NULL,
                        trade_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        exit_type TEXT NOT NULL,
                        exit_price REAL NOT NULL,
                        timestamp TEXT NOT NULL,  -- ISO format with timezone
                        reason TEXT NOT NULL,
                        pnl REAL,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS rejected_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        rejection_reason TEXT NOT NULL,
                        timestamp TEXT NOT NULL,  -- ISO format with timezone
                        confidence REAL,
                        price REAL,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Trade tables
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS open_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        timestamp TEXT NOT NULL,  -- ISO format with timezone
                        stop_loss REAL,
                        take_profit REAL,
                        current_price REAL,
                        unrealized_pnl REAL,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS closed_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        entry_timestamp TEXT NOT NULL,  -- ISO format with timezone
                        exit_timestamp TEXT NOT NULL,   -- ISO format with timezone
                        pnl REAL NOT NULL,
                        exit_reason TEXT NOT NULL,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Daily summary table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS daily_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,  -- YYYY-MM-DD format
                        market TEXT NOT NULL,
                        total_signals INTEGER DEFAULT 0,
                        entry_signals INTEGER DEFAULT 0,
                        exit_signals INTEGER DEFAULT 0,
                        rejected_signals INTEGER DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        realized_pnl REAL DEFAULT 0.0,
                        unrealized_pnl REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        sharpe_ratio REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.0,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, market)
                    )
                ''')
                
                # Market conditions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_conditions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,  -- ISO format with timezone
                        regime TEXT NOT NULL,
                        volatility REAL NOT NULL,
                        trend_strength REAL NOT NULL,
                        volume_ratio REAL NOT NULL,
                        momentum REAL NOT NULL,
                        confidence REAL NOT NULL,
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_entry_signals_timestamp ON entry_signals(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_entry_signals_symbol ON entry_signals(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_exit_signals_timestamp ON exit_signals(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_exit_signals_trade_id ON exit_signals(trade_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_open_trades_symbol ON open_trades(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_closed_trades_date ON closed_trades(exit_timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_market_conditions_timestamp ON market_conditions(timestamp)')
                
                conn.commit()
                logger.info("✅ Enhanced timezone database initialized")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def _ensure_timezone_aware(self, timestamp: Union[str, datetime]) -> str:
        """Ensure timestamp is timezone-aware and in ISO format"""
        try:
            if isinstance(timestamp, str):
                # Try to parse as datetime
                try:
                    dt = datetime.fromisoformat(timestamp)
                except ValueError:
                    # Try parsing with different formats
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    dt = dt.replace(tzinfo=self.timezone)
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                raise ValueError(f"Invalid timestamp type: {type(timestamp)}")
            
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=self.timezone)
            
            # Convert to ISO format
            return dt.isoformat()
            
        except Exception as e:
            logger.error(f"❌ Timezone conversion failed: {e}")
            # Fallback to current time
            return datetime.now(self.timezone).isoformat()
    
    def save_market_data(self, symbol: str, data: Dict[str, Any], market: str = "indian"):
        """Save market data with timezone-aware timestamp"""
        try:
            table_name = f"{market}_market_data"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f'''
                    INSERT INTO {table_name} 
                    (symbol, timestamp, open, high, low, close, volume, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    self._ensure_timezone_aware(data['timestamp']),
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close'],
                    data['volume'],
                    self.timezone.key
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save market data: {e}")
    
    def save_entry_signal(self, signal_data: Dict[str, Any]):
        """Save entry signal with timezone-aware timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO entry_signals 
                    (signal_id, symbol, strategy, signal_type, confidence, price, timestamp,
                     timeframe, strength, indicator_values, market_condition, volatility,
                     position_size, stop_loss_price, take_profit_price, confirmed, executed,
                     execution_reason, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['symbol'],
                    signal_data['strategy'],
                    signal_data['signal_type'],
                    signal_data['confidence'],
                    signal_data['price'],
                    self._ensure_timezone_aware(signal_data['timestamp']),
                    signal_data.get('timeframe', '1h'),
                    signal_data.get('strength', 'medium'),
                    json.dumps(signal_data.get('indicator_values', {})),
                    signal_data.get('market_condition', 'unknown'),
                    signal_data.get('volatility', 0.0),
                    signal_data.get('position_size', 0.0),
                    signal_data.get('stop_loss_price', 0.0),
                    signal_data.get('take_profit_price', 0.0),
                    signal_data.get('confirmed', False),
                    signal_data.get('executed', False),
                    signal_data.get('execution_reason', ''),
                    self.timezone.key
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save entry signal: {e}")
    
    def save_exit_signal(self, signal_data: Dict[str, Any]):
        """Save exit signal with timezone-aware timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO exit_signals 
                    (signal_id, trade_id, symbol, strategy, exit_type, exit_price, timestamp,
                     reason, pnl, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['trade_id'],
                    signal_data['symbol'],
                    signal_data['strategy'],
                    signal_data['exit_type'],
                    signal_data['exit_price'],
                    self._ensure_timezone_aware(signal_data['timestamp']),
                    signal_data['reason'],
                    signal_data.get('pnl', 0.0),
                    self.timezone.key
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save exit signal: {e}")
    
    def save_rejected_signal(self, signal_data: Dict[str, Any]):
        """Save rejected signal with timezone-aware timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO rejected_signals 
                    (signal_id, symbol, strategy, signal_type, rejection_reason, timestamp,
                     confidence, price, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['symbol'],
                    signal_data['strategy'],
                    signal_data['signal_type'],
                    signal_data['rejection_reason'],
                    self._ensure_timezone_aware(signal_data['timestamp']),
                    signal_data.get('confidence', 0.0),
                    signal_data.get('price', 0.0),
                    self.timezone.key
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save rejected signal: {e}")
    
    def save_trade(self, trade_data: Dict[str, Any]):
        """Save trade with timezone-aware timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO open_trades 
                    (trade_id, symbol, strategy, signal_type, entry_price, quantity, timestamp,
                     stop_loss, take_profit, current_price, unrealized_pnl, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['trade_id'],
                    trade_data['symbol'],
                    trade_data['strategy'],
                    trade_data['signal_type'],
                    trade_data['entry_price'],
                    trade_data['quantity'],
                    self._ensure_timezone_aware(trade_data['timestamp']),
                    trade_data.get('stop_loss', 0.0),
                    trade_data.get('take_profit', 0.0),
                    trade_data.get('current_price', 0.0),
                    trade_data.get('unrealized_pnl', 0.0),
                    self.timezone.key
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save trade: {e}")
    
    def update_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]):
        """Update trade with exit information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get trade data
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM open_trades WHERE trade_id = ?', (trade_id,))
                trade = cursor.fetchone()
                
                if trade:
                    # Calculate P&L
                    entry_price = trade[4]  # entry_price
                    quantity = trade[5]     # quantity
                    exit_price = exit_data['exit_price']
                    
                    if trade[3] == 'BUY':  # signal_type
                        pnl = (exit_price - entry_price) * quantity
                    else:  # SELL
                        pnl = (entry_price - exit_price) * quantity
                    
                    # Move to closed_trades
                    conn.execute('''
                        INSERT INTO closed_trades 
                        (trade_id, symbol, strategy, signal_type, entry_price, exit_price, quantity,
                         entry_timestamp, exit_timestamp, pnl, exit_reason, timezone)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_id,
                        trade[1],  # symbol
                        trade[2],  # strategy
                        trade[3],  # signal_type
                        entry_price,
                        exit_price,
                        quantity,
                        trade[6],  # entry timestamp
                        self._ensure_timezone_aware(exit_data['timestamp']),
                        pnl,
                        exit_data['reason'],
                        self.timezone.key
                    ))
                    
                    # Remove from open_trades
                    conn.execute('DELETE FROM open_trades WHERE trade_id = ?', (trade_id,))
                    
                    conn.commit()
                    logger.info(f"✅ Trade {trade_id} closed with P&L: {pnl:.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to update trade exit: {e}")
    
    def save_market_condition(self, condition_data: Dict[str, Any]):
        """Save market condition with timezone-aware timestamp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO market_conditions 
                    (symbol, timestamp, regime, volatility, trend_strength, volume_ratio,
                     momentum, confidence, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    condition_data['symbol'],
                    self._ensure_timezone_aware(condition_data['timestamp']),
                    condition_data['regime'],
                    condition_data['volatility'],
                    condition_data['trend_strength'],
                    condition_data['volume_ratio'],
                    condition_data['momentum'],
                    condition_data['confidence'],
                    self.timezone.key
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save market condition: {e}")
    
    def update_daily_summary(self, date: str, market: str, summary_data: Dict[str, Any]):
        """Update daily summary with timezone-aware date"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO daily_summary 
                    (date, market, total_signals, entry_signals, exit_signals, rejected_signals,
                     total_trades, winning_trades, losing_trades, total_pnl, realized_pnl,
                     unrealized_pnl, max_drawdown, sharpe_ratio, win_rate, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date,
                    market,
                    summary_data.get('total_signals', 0),
                    summary_data.get('entry_signals', 0),
                    summary_data.get('exit_signals', 0),
                    summary_data.get('rejected_signals', 0),
                    summary_data.get('total_trades', 0),
                    summary_data.get('winning_trades', 0),
                    summary_data.get('losing_trades', 0),
                    summary_data.get('total_pnl', 0.0),
                    summary_data.get('realized_pnl', 0.0),
                    summary_data.get('unrealized_pnl', 0.0),
                    summary_data.get('max_drawdown', 0.0),
                    summary_data.get('sharpe_ratio', 0.0),
                    summary_data.get('win_rate', 0.0),
                    self.timezone.key
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to update daily summary: {e}")
    
    def get_daily_summary(self, date: str, market: str) -> Optional[Dict[str, Any]]:
        """Get daily summary for specific date and market"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM daily_summary WHERE date = ? AND market = ?
                ''', (date, market))
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get daily summary: {e}")
            return None
    
    def get_market_statistics(self, market: str, days: int = 30) -> Dict[str, Any]:
        """Get market statistics for specified period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get date range
                end_date = datetime.now(self.timezone).date()
                start_date = end_date - timedelta(days=days)
                
                # Get daily summaries
                cursor.execute('''
                    SELECT * FROM daily_summary 
                    WHERE market = ? AND date >= ? AND date <= ?
                    ORDER BY date DESC
                ''', (market, start_date.isoformat(), end_date.isoformat()))
                
                summaries = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                if not summaries:
                    return {}
                
                # Calculate statistics
                total_pnl = sum(row[columns.index('total_pnl')] for row in summaries)
                total_trades = sum(row[columns.index('total_trades')] for row in summaries)
                winning_trades = sum(row[columns.index('winning_trades')] for row in summaries)
                total_signals = sum(row[columns.index('total_signals')] for row in summaries)
                
                return {
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_signals': total_signals,
                    'avg_daily_pnl': total_pnl / len(summaries) if summaries else 0,
                    'period_days': len(summaries)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get market statistics: {e}")
            return {}
    
    def get_entry_signals(self, symbol: str = None, strategy: str = None, 
                         days: int = 7) -> List[Dict[str, Any]]:
        """Get entry signals with timezone filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = 'SELECT * FROM entry_signals WHERE 1=1'
                params = []
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                if strategy:
                    query += ' AND strategy = ?'
                    params.append(strategy)
                
                # Add date filter
                end_date = datetime.now(self.timezone)
                start_date = end_date - timedelta(days=days)
                query += ' AND timestamp >= ? AND timestamp <= ?'
                params.extend([start_date.isoformat(), end_date.isoformat()])
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get entry signals: {e}")
            return []
    
    def get_exit_signals(self, trade_id: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Get exit signals with timezone filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = 'SELECT * FROM exit_signals WHERE 1=1'
                params = []
                
                if trade_id:
                    query += ' AND trade_id = ?'
                    params.append(trade_id)
                
                # Add date filter
                end_date = datetime.now(self.timezone)
                start_date = end_date - timedelta(days=days)
                query += ' AND timestamp >= ? AND timestamp <= ?'
                params.extend([start_date.isoformat(), end_date.isoformat()])
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get exit signals: {e}")
            return []
    
    def get_rejected_signals(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Get rejected signals with timezone filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = 'SELECT * FROM rejected_signals WHERE 1=1'
                params = []
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                # Add date filter
                end_date = datetime.now(self.timezone)
                start_date = end_date - timedelta(days=days)
                query += ' AND timestamp >= ? AND timestamp <= ?'
                params.extend([start_date.isoformat(), end_date.isoformat()])
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get rejected signals: {e}")
            return []

# Global enhanced database instance
enhanced_timezone_db = EnhancedTimezoneDatabase()

# Convenience functions
def save_entry_signal(signal_data: Dict[str, Any]):
    """Save entry signal with timezone awareness"""
    enhanced_timezone_db.save_entry_signal(signal_data)

def save_exit_signal(signal_data: Dict[str, Any]):
    """Save exit signal with timezone awareness"""
    enhanced_timezone_db.save_exit_signal(signal_data)

def save_rejected_signal(signal_data: Dict[str, Any]):
    """Save rejected signal with timezone awareness"""
    enhanced_timezone_db.save_rejected_signal(signal_data)

def save_trade(trade_data: Dict[str, Any]):
    """Save trade with timezone awareness"""
    enhanced_timezone_db.save_trade(trade_data)

def update_trade_exit(trade_id: str, exit_data: Dict[str, Any]):
    """Update trade with exit information"""
    enhanced_timezone_db.update_trade_exit(trade_id, exit_data)

def get_daily_summary(date: str, market: str) -> Optional[Dict[str, Any]]:
    """Get daily summary for specific date and market"""
    return enhanced_timezone_db.get_daily_summary(date, market)

def get_market_statistics(market: str, days: int = 30) -> Dict[str, Any]:
    """Get market statistics for specified period"""
    return enhanced_timezone_db.get_market_statistics(market, days)
