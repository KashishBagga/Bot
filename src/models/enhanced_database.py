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
                
                # ── Trade Intelligence Warehouse (V2: Profitability Discovery) ──
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE NOT NULL,
                        date TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        strategy_version TEXT DEFAULT 'v1.0',
                        feature_version TEXT DEFAULT 'v1.0',
                        market_regime TEXT,
                        daily_bias TEXT,
                        hourly_bias TEXT,
                        trend_strength REAL,
                        rvol REAL,
                        atr REAL,
                        distance_from_supply REAL,
                        distance_from_demand REAL,
                        nearest_supply_strength REAL,
                        nearest_demand_strength REAL,
                        distance_to_liquidity_pool REAL,
                        session_type TEXT,
                        day_type TEXT,
                        liquidity_score REAL,
                        fft_score REAL,
                        entry_time TEXT,
                        exit_time TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        mfe REAL DEFAULT 0.0,
                        mae REAL DEFAULT 0.0,
                        pnl REAL,
                        win_loss TEXT,
                        indicator_snapshot JSON,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # ── Signal Snapshot Table (Priority 1) ────────────────
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_signals_snapshot (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        market_regime TEXT,
                        signal_strength REAL,
                        accepted INTEGER,  -- 1 for accepted, 0 for rejected
                        rejected_reason TEXT,
                        executed INTEGER,  -- 1 for executed, 0 for not
                        result TEXT,       -- Potential result if tracked
                        context_snapshot JSON,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # ── Exit Analytics Table (Priority 2) ─────────────────
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS exit_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE NOT NULL,
                        exit_reason TEXT NOT NULL, -- TP, SL, TSL, MANUAL, RISK
                        target_hit REAL,
                        sl_hit REAL,
                        trailing_sl REAL,
                        manual_exit REAL,
                        risk_manager_exit REAL,
                        time_in_trade INTEGER, -- Seconds
                        mfe REAL,
                        mae REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # ── Option Warehouse (Priority 6) ──────────────────────
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS option_chain_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        underlying TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        atm_strike REAL NOT NULL,
                        chain_data JSON NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_features_trade_id ON trade_features(trade_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_option_snapshots_underlying_time ON option_chain_snapshots(underlying, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_snapshots_timestamp ON trade_signals_snapshot(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_exit_analysis_trade_id ON exit_analysis(trade_id)')
                
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

    def save_market_condition(self, market: str, symbol: str, condition: str, 
                             volatility: float, trend_strength: float, volume_profile: str,
                             support_levels: List[float] = None, resistance_levels: List[float] = None) -> bool:
        """Save market condition"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO market_conditions 
                    (date, market, symbol, condition, volatility, trend_strength, 
                     volume_profile, support_levels, resistance_levels)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (datetime.now().strftime("%Y-%m-%d"), market, symbol, condition, 
                      volatility, trend_strength, volume_profile,
                      json.dumps(support_levels) if support_levels else None,
                      json.dumps(resistance_levels) if resistance_levels else None))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save market condition: {e}")
            return False
    
    def get_entry_signals(self, market: str, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get entry signals for market and optionally symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute(f'''
                        SELECT * FROM {market}_entry_signals 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute(f'''
                        SELECT * FROM {market}_entry_signals 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in results]
                
        except Exception as e:
            logger.error(f"❌ Failed to get entry signals: {e}")
            return []
    
    def get_exit_signals(self, market: str, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get exit signals for market and optionally symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute(f'''
                        SELECT * FROM {market}_exit_signals 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute(f'''
                        SELECT * FROM {market}_exit_signals 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in results]
                
        except Exception as e:
            logger.error(f"❌ Failed to get exit signals: {e}")
            return []
    
    def get_rejected_signals(self, market: str, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get rejected signals for market and optionally symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute(f'''
                        SELECT * FROM {market}_rejected_signals 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute(f'''
                        SELECT * FROM {market}_rejected_signals 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in results]
                
        except Exception as e:
            logger.error(f"❌ Failed to get rejected signals: {e}")
            return []
    
    def update_daily_summary_kwargs(self, market: str, date: str, **kwargs) -> bool:
        """Update daily summary with keyword arguments"""
        try:
            summary_data = {
                'total_signals': kwargs.get('total_signals', 0),
                'executed_signals': kwargs.get('executed_signals', 0),
                'rejected_signals': kwargs.get('rejected_signals', 0),
                'total_trades': kwargs.get('total_trades', 0),
                'open_trades': kwargs.get('open_trades', 0),
                'closed_trades': kwargs.get('closed_trades', 0),
                'total_pnl': kwargs.get('total_pnl', 0.0),
                'realized_pnl': kwargs.get('realized_pnl', 0.0),
                'unrealized_pnl': kwargs.get('unrealized_pnl', 0.0),
                'win_rate': kwargs.get('win_rate', 0.0),
                'avg_trade_duration': kwargs.get('avg_trade_duration', 0.0),
                'max_drawdown': kwargs.get('max_drawdown', 0.0),
                'volatility': kwargs.get('volatility', 0.0)
            }
            return self.update_daily_summary(date, market, summary_data)
        except Exception as e:
            logger.error(f"❌ Failed to update daily summary: {e}")
            return False
    def save_trade_features(self, features: Dict[str, Any]) -> bool:
        """Save exhaustive trade features to intelligence warehouse"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if entry exists to perform insert or update
                cursor.execute("SELECT id FROM trade_features WHERE trade_id = ?", (features.get('trade_id'),))
                row = cursor.fetchone()
                
                columns = [
                    'trade_id', 'date', 'strategy_name', 'strategy_version', 'feature_version',
                    'market_regime', 'daily_bias', 'hourly_bias', 'trend_strength', 'rvol',
                    'atr', 'distance_from_supply', 'distance_from_demand', 'nearest_supply_strength',
                    'nearest_demand_strength', 'distance_to_liquidity_pool', 'session_type',
                    'day_type', 'liquidity_score', 'fft_score', 'entry_time', 'exit_time',
                    'entry_price', 'exit_price', 'mfe', 'mae', 'pnl', 'win_loss', 'indicator_snapshot'
                ]
                
                if row:
                    # Update
                    update_cols = [f"{col} = ?" for col in columns if col in features]
                    vals = [features[col] if col != 'indicator_snapshot' else json.dumps(features[col]) 
                            for col in columns if col in features]
                    vals.append(features['trade_id'])
                    
                    query = f"UPDATE trade_features SET {', '.join(update_cols)} WHERE trade_id = ?"
                    cursor.execute(query, vals)
                else:
                    # Insert
                    present_cols = [col for col in columns if col in features]
                    placeholders = ", ".join(["?" for _ in present_cols])
                    vals = [features[col] if col != 'indicator_snapshot' else json.dumps(features[col])
                            for col in present_cols]
                    
                    query = f"INSERT INTO trade_features ({', '.join(present_cols)}) VALUES ({placeholders})"
                    cursor.execute(query, vals)
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save trade features: {e}")
            return False

    def save_option_snapshot(self, underlying: str, atm_strike: float, chain_data: Dict) -> bool:
        """Save raw option chain snapshot"""
        try:
            timestamp = datetime.now(self.tz).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO option_chain_snapshots (underlying, timestamp, atm_strike, chain_data)
                    VALUES (?, ?, ?, ?)
                ''', (underlying, timestamp, atm_strike, json.dumps(chain_data)))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save option snapshot: {e}")
            return False

    def get_trade_features(self, trade_id: str) -> Optional[Dict]:
        """Retrieve features for a specific trade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trade_features WHERE trade_id = ?", (trade_id,))
                result = cursor.fetchone()
                if result:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, result))
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get trade features: {e}")
            return None
    def save_signal_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Save a snapshot of every signal generated"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                columns = [
                    'signal_id', 'timestamp', 'strategy', 'symbol', 'market_regime',
                    'signal_strength', 'accepted', 'rejected_reason', 'executed',
                    'result', 'context_snapshot'
                ]
                present_cols = [col for col in columns if col in snapshot]
                placeholders = ", ".join(["?" for _ in present_cols])
                vals = [snapshot[col] if col != 'context_snapshot' else json.dumps(snapshot[col])
                        for col in present_cols]
                
                query = f"INSERT OR REPLACE INTO trade_signals_snapshot ({', '.join(present_cols)}) VALUES ({placeholders})"
                cursor.execute(query, vals)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save signal snapshot: {e}")
            return False

    def save_exit_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Save detailed exit analytics for a trade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                columns = [
                    'trade_id', 'exit_reason', 'target_hit', 'sl_hit', 'trailing_sl',
                    'manual_exit', 'risk_manager_exit', 'time_in_trade', 'mfe', 'mae'
                ]
                present_cols = [col for col in columns if col in analysis]
                placeholders = ", ".join(["?" for _ in present_cols])
                vals = [analysis[col] for col in present_cols]
                
                query = f"INSERT OR REPLACE INTO exit_analysis ({', '.join(present_cols)}) VALUES ({placeholders})"
                cursor.execute(query, vals)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save exit analysis: {e}")
            return False

    def get_data_completeness_score(self) -> Dict[str, float]:
        """Calculate population rates for intelligence features"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trade_features")
                total = cursor.fetchone()[0]
                if total == 0: return {}
                
                columns = [
                    'market_regime', 'daily_bias', 'rvol', 'atr', 
                    'distance_from_supply', 'liquidity_score', 'fft_score',
                    'mfe', 'mae', 'session_type', 'day_type'
                ]
                
                scores = {}
                for col in columns:
                    cursor.execute(f"SELECT COUNT(*) FROM trade_features WHERE {col} IS NOT NULL")
                    populated = cursor.fetchone()[0]
                    scores[col] = (populated / total) * 100
                    
                return scores
        except Exception as e:
            logger.error(f"❌ Failed to calculate completeness scores: {e}")
            return {}
