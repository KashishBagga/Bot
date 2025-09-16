#!/usr/bin/env python3
"""
Enhanced Trading Database with Proper Market Separation
=====================================================
Separate tables for Indian/Crypto markets, symbols, strategies, and comprehensive tracking
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

class EnhancedTradingDatabase:
    """Enhanced database with proper market separation and comprehensive tracking"""
    
    def __init__(self, db_path: str = "data/enhanced_trading.db"):
        self.db_path = db_path
        self.tz = ZoneInfo("Asia/Kolkata")
        self.init_database()
    
    def init_database(self):
        """Initialize database with comprehensive table structure"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create market-specific signal tables
                for market in ['indian', 'crypto']:
                    # Entry signals table
                    cursor.execute(f'''
                        CREATE TABLE IF NOT EXISTS {market}_entry_signals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            signal_id TEXT UNIQUE NOT NULL,
                            symbol TEXT NOT NULL,
                            strategy TEXT NOT NULL,
                            signal_type TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            price REAL NOT NULL,
                            timestamp TEXT NOT NULL,
                            timeframe TEXT NOT NULL,
                            strength TEXT,
                            indicator_values JSON,
                            market_condition TEXT,
                            volatility REAL,
                            position_size REAL,
                            stop_loss_price REAL,
                            take_profit_price REAL,
                            confirmed BOOLEAN DEFAULT FALSE,
                            executed BOOLEAN DEFAULT FALSE,
                            execution_reason TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Exit signals table
                    cursor.execute(f'''
                        CREATE TABLE IF NOT EXISTS {market}_exit_signals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            exit_signal_id TEXT UNIQUE NOT NULL,
                            trade_id TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            strategy TEXT NOT NULL,
                            exit_type TEXT NOT NULL,
                            exit_price REAL NOT NULL,
                            timestamp TEXT NOT NULL,
                            exit_reason TEXT NOT NULL,
                            pnl REAL,
                            duration_minutes INTEGER,
                            indicator_values JSON,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Rejected signals table
                    cursor.execute(f'''
                        CREATE TABLE IF NOT EXISTS {market}_rejected_signals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            signal_id TEXT UNIQUE NOT NULL,
                            symbol TEXT NOT NULL,
                            strategy TEXT NOT NULL,
                            signal_type TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            price REAL NOT NULL,
                            timestamp TEXT NOT NULL,
                            rejection_reason TEXT NOT NULL,
                            indicator_values JSON,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Symbol-specific tables
                    for symbol in ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX', 'NSE:FINNIFTY-INDEX']:
                        table_name = symbol.replace(':', '_').replace('-', '_').lower()
                        cursor.execute(f'''
                            CREATE TABLE IF NOT EXISTS {market}_{table_name}_trades (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                trade_id TEXT UNIQUE NOT NULL,
                                strategy TEXT NOT NULL,
                                signal_type TEXT NOT NULL,
                                entry_price REAL NOT NULL,
                                exit_price REAL,
                                quantity REAL NOT NULL,
                                entry_time TEXT NOT NULL,
                                exit_time TEXT,
                                stop_loss_price REAL,
                                take_profit_price REAL,
                                exit_reason TEXT,
                                pnl REAL,
                                commission REAL DEFAULT 0.0,
                                duration_minutes INTEGER,
                                status TEXT DEFAULT 'OPEN',
                                indicator_values JSON,
                                created_at TEXT DEFAULT CURRENT_TIMESTAMP
                            )
                        ''')
                
                # Strategy-specific performance tables
                strategies = ['simple_ema', 'ema_crossover_enhanced', 'supertrend_macd_rsi_ema', 'supertrend_ema']
                for strategy in strategies:
                    cursor.execute(f'''
                        CREATE TABLE IF NOT EXISTS {strategy}_performance (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            date TEXT NOT NULL,
                            market TEXT NOT NULL,
                            total_signals INTEGER DEFAULT 0,
                            executed_signals INTEGER DEFAULT 0,
                            rejected_signals INTEGER DEFAULT 0,
                            total_trades INTEGER DEFAULT 0,
                            winning_trades INTEGER DEFAULT 0,
                            losing_trades INTEGER DEFAULT 0,
                            total_pnl REAL DEFAULT 0.0,
                            win_rate REAL DEFAULT 0.0,
                            avg_win REAL DEFAULT 0.0,
                            avg_loss REAL DEFAULT 0.0,
                            max_drawdown REAL DEFAULT 0.0,
                            sharpe_ratio REAL DEFAULT 0.0,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                
                # Daily summary table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        market TEXT NOT NULL,
                        total_signals INTEGER DEFAULT 0,
                        executed_signals INTEGER DEFAULT 0,
                        rejected_signals INTEGER DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        open_trades INTEGER DEFAULT 0,
                        closed_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        realized_pnl REAL DEFAULT 0.0,
                        unrealized_pnl REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.0,
                        avg_trade_duration REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        volatility REAL DEFAULT 0.0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, market)
                    )
                ''')
                
                # Market conditions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_conditions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        market TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        volatility REAL NOT NULL,
                        trend_strength REAL NOT NULL,
                        volume_profile TEXT,
                        support_levels JSON,
                        resistance_levels JSON,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                for market in ['indian', 'crypto']:
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{market}_entry_signals_symbol ON {market}_entry_signals(symbol)')
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{market}_entry_signals_timestamp ON {market}_entry_signals(timestamp)')
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{market}_entry_signals_strategy ON {market}_entry_signals(strategy)')
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{market}_exit_signals_trade_id ON {market}_exit_signals(trade_id)')
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{market}_rejected_signals_symbol ON {market}_rejected_signals(symbol)')
                
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_summary_date_market ON daily_summary(date, market)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_conditions_date_symbol ON market_conditions(date, symbol)')
                
                conn.commit()
                logger.info("✅ Enhanced database initialized with comprehensive table structure")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced database: {e}")
            raise
    
    def save_entry_signal(self, market: str, signal_id: str, symbol: str, strategy: str, 
                         signal_type: str, confidence: float, price: float, timestamp: str,
                         timeframe: str, strength: str = None, indicator_values: Dict = None,
                         market_condition: str = None, volatility: float = None,
                         position_size: float = None, stop_loss_price: float = None,
                         take_profit_price: float = None) -> bool:
        """Save entry signal to market-specific table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'''
                    INSERT INTO {market}_entry_signals 
                    (signal_id, symbol, strategy, signal_type, confidence, price, timestamp,
                     timeframe, strength, indicator_values, market_condition, volatility,
                     position_size, stop_loss_price, take_profit_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (signal_id, symbol, strategy, signal_type, confidence, price, timestamp,
                      timeframe, strength, json.dumps(indicator_values) if indicator_values else None,
                      market_condition, volatility, position_size, stop_loss_price, take_profit_price))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save entry signal: {e}")
            return False
    
    def save_exit_signal(self, market: str, exit_signal_id: str, trade_id: str, symbol: str,
                        strategy: str, exit_type: str, exit_price: float, timestamp: str,
                        exit_reason: str, pnl: float = None, duration_minutes: int = None,
                        indicator_values: Dict = None) -> bool:
        """Save exit signal to market-specific table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'''
                    INSERT INTO {market}_exit_signals 
                    (exit_signal_id, trade_id, symbol, strategy, exit_type, exit_price,
                     timestamp, exit_reason, pnl, duration_minutes, indicator_values)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (exit_signal_id, trade_id, symbol, strategy, exit_type, exit_price,
                      timestamp, exit_reason, pnl, duration_minutes,
                      json.dumps(indicator_values) if indicator_values else None))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save exit signal: {e}")
            return False
    
    def save_rejected_signal(self, market: str, signal_id: str, symbol: str, strategy: str,
                           signal_type: str, confidence: float, price: float, timestamp: str,
                           rejection_reason: str, indicator_values: Dict = None) -> bool:
        """Save rejected signal to market-specific table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'''
                    INSERT INTO {market}_rejected_signals 
                    (signal_id, symbol, strategy, signal_type, confidence, price, timestamp,
                     rejection_reason, indicator_values)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (signal_id, symbol, strategy, signal_type, confidence, price, timestamp,
                      rejection_reason, json.dumps(indicator_values) if indicator_values else None))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save rejected signal: {e}")
            return False
    
    def save_trade(self, market: str, symbol: str, trade_id: str, strategy: str, signal_type: str,
                  entry_price: float, quantity: float, entry_time: str, stop_loss_price: float = None,
                  take_profit_price: float = None, indicator_values: Dict = None) -> bool:
        """Save trade to symbol-specific table"""
        try:
            table_name = symbol.replace(':', '_').replace('-', '_').lower()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'''
                    INSERT INTO {market}_{table_name}_trades 
                    (trade_id, strategy, signal_type, entry_price, quantity, entry_time,
                     stop_loss_price, take_profit_price, indicator_values)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (trade_id, strategy, signal_type, entry_price, quantity, entry_time,
                      stop_loss_price, take_profit_price,
                      json.dumps(indicator_values) if indicator_values else None))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save trade: {e}")
            return False
    
    def update_trade_exit(self, market: str, symbol: str, trade_id: str, exit_price: float,
                         exit_time: str, exit_reason: str, pnl: float, duration_minutes: int) -> bool:
        """Update trade with exit information"""
        try:
            table_name = symbol.replace(':', '_').replace('-', '_').lower()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'''
                    UPDATE {market}_{table_name}_trades 
                    SET exit_price = ?, exit_time = ?, exit_reason = ?, pnl = ?, 
                        duration_minutes = ?, status = 'CLOSED'
                    WHERE trade_id = ?
                ''', (exit_price, exit_time, exit_reason, pnl, duration_minutes, trade_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to update trade exit: {e}")
            return False
    
    def get_daily_summary(self, date: str, market: str) -> Optional[Dict]:
        """Get daily summary for a specific date and market"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM daily_summary 
                    WHERE date = ? AND market = ?
                ''', (date, market))
                
                result = cursor.fetchone()
                if result:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, result))
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get daily summary: {e}")
            return None
    
    def update_daily_summary(self, date: str, market: str, summary_data: Dict) -> bool:
        """Update daily summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if summary exists
                existing = self.get_daily_summary(date, market)
                
                if existing:
                    # Update existing
                    cursor.execute('''
                        UPDATE daily_summary 
                        SET total_signals = ?, executed_signals = ?, rejected_signals = ?,
                            total_trades = ?, open_trades = ?, closed_trades = ?,
                            total_pnl = ?, realized_pnl = ?, unrealized_pnl = ?,
                            win_rate = ?, avg_trade_duration = ?, max_drawdown = ?,
                            volatility = ?
                        WHERE date = ? AND market = ?
                    ''', (summary_data.get('total_signals', 0),
                          summary_data.get('executed_signals', 0),
                          summary_data.get('rejected_signals', 0),
                          summary_data.get('total_trades', 0),
                          summary_data.get('open_trades', 0),
                          summary_data.get('closed_trades', 0),
                          summary_data.get('total_pnl', 0.0),
                          summary_data.get('realized_pnl', 0.0),
                          summary_data.get('unrealized_pnl', 0.0),
                          summary_data.get('win_rate', 0.0),
                          summary_data.get('avg_trade_duration', 0.0),
                          summary_data.get('max_drawdown', 0.0),
                          summary_data.get('volatility', 0.0),
                          date, market))
                else:
                    # Insert new
                    cursor.execute('''
                        INSERT INTO daily_summary 
                        (date, market, total_signals, executed_signals, rejected_signals,
                         total_trades, open_trades, closed_trades, total_pnl, realized_pnl,
                         unrealized_pnl, win_rate, avg_trade_duration, max_drawdown, volatility)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (date, market,
                          summary_data.get('total_signals', 0),
                          summary_data.get('executed_signals', 0),
                          summary_data.get('rejected_signals', 0),
                          summary_data.get('total_trades', 0),
                          summary_data.get('open_trades', 0),
                          summary_data.get('closed_trades', 0),
                          summary_data.get('total_pnl', 0.0),
                          summary_data.get('realized_pnl', 0.0),
                          summary_data.get('unrealized_pnl', 0.0),
                          summary_data.get('win_rate', 0.0),
                          summary_data.get('avg_trade_duration', 0.0),
                          summary_data.get('max_drawdown', 0.0),
                          summary_data.get('volatility', 0.0)))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to update daily summary: {e}")
            return False
    
    def get_market_statistics(self, market: str) -> Dict:
        """Get comprehensive market statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get entry signals count
                cursor.execute(f'SELECT COUNT(*) FROM {market}_entry_signals')
                total_signals = cursor.fetchone()[0]
                
                cursor.execute(f'SELECT COUNT(*) FROM {market}_entry_signals WHERE executed = 1')
                executed_signals = cursor.fetchone()[0]
                
                cursor.execute(f'SELECT COUNT(*) FROM {market}_rejected_signals')
                rejected_signals = cursor.fetchone()[0]
                
                # Get trades count
                cursor.execute(f'SELECT COUNT(*) FROM {market}_nse_nifty50_index_trades WHERE status = "OPEN"')
                open_trades = cursor.fetchone()[0]
                
                cursor.execute(f'SELECT COUNT(*) FROM {market}_nse_nifty50_index_trades WHERE status = "CLOSED"')
                closed_trades = cursor.fetchone()[0]
                
                # Get P&L
                cursor.execute(f'SELECT SUM(pnl) FROM {market}_nse_nifty50_index_trades WHERE status = "CLOSED"')
                total_pnl = cursor.fetchone()[0] or 0.0
                
                return {
                    'total_signals': total_signals,
                    'executed_signals': executed_signals,
                    'rejected_signals': rejected_signals,
                    'open_trades': open_trades,
                    'closed_trades': closed_trades,
                    'total_pnl': total_pnl,
                    'execution_rate': (executed_signals / total_signals * 100) if total_signals > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get market statistics: {e}")
            return {}
