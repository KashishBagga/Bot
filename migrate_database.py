#!/usr/bin/env python3
"""
Database Migration Script
Adds missing columns to existing database schema.
"""

import sqlite3
import logging

logger = logging.getLogger('database_migration')

def migrate_database():
    """Migrate existing database to enhanced schema."""
    try:
        conn = sqlite3.connect('unified_trading.db')
        cursor = conn.cursor()
        
        # Check if data_quality_score column exists
        cursor.execute("PRAGMA table_info(raw_options_chain)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add missing columns to raw_options_chain
        if 'data_quality_score' not in columns:
            logger.info("Adding data_quality_score column to raw_options_chain")
            cursor.execute("ALTER TABLE raw_options_chain ADD COLUMN data_quality_score REAL DEFAULT 0.0")
        
        if 'is_market_open' not in columns:
            logger.info("Adding is_market_open column to raw_options_chain")
            cursor.execute("ALTER TABLE raw_options_chain ADD COLUMN is_market_open BOOLEAN DEFAULT FALSE")
        
        # Create new tables if they don't exist
        tables_to_create = [
            ('options_data', '''
                CREATE TABLE IF NOT EXISTS options_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_chain_id INTEGER,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price INTEGER NOT NULL,
                    expiry_date TEXT NOT NULL,
                    ltp REAL DEFAULT 0.0,
                    bid REAL DEFAULT 0.0,
                    ask REAL DEFAULT 0.0,
                    volume INTEGER DEFAULT 0,
                    oi INTEGER DEFAULT 0,
                    oi_change INTEGER DEFAULT 0,
                    oi_change_pct REAL DEFAULT 0.0,
                    prev_oi INTEGER DEFAULT 0,
                    implied_volatility REAL DEFAULT 0.0,
                    delta REAL DEFAULT 0.0,
                    gamma REAL DEFAULT 0.0,
                    theta REAL DEFAULT 0.0,
                    vega REAL DEFAULT 0.0,
                    bid_ask_spread REAL DEFAULT 0.0,
                    spread_pct REAL DEFAULT 0.0,
                    is_atm BOOLEAN DEFAULT FALSE,
                    is_itm BOOLEAN DEFAULT FALSE,
                    is_otm BOOLEAN DEFAULT FALSE,
                    underlying_price REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (raw_chain_id) REFERENCES raw_options_chain(id)
                )
            '''),
            ('market_summary', '''
                CREATE TABLE IF NOT EXISTS market_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    underlying_price REAL NOT NULL,
                    call_oi INTEGER DEFAULT 0,
                    put_oi INTEGER DEFAULT 0,
                    total_oi INTEGER DEFAULT 0,
                    pcr REAL DEFAULT 0.0,
                    indiavix REAL DEFAULT 0.0,
                    total_volume INTEGER DEFAULT 0,
                    total_options INTEGER DEFAULT 0,
                    total_strikes INTEGER DEFAULT 0,
                    atm_strike INTEGER DEFAULT 0,
                    atm_call_ltp REAL DEFAULT 0.0,
                    atm_put_ltp REAL DEFAULT 0.0,
                    atm_call_iv REAL DEFAULT 0.0,
                    atm_put_iv REAL DEFAULT 0.0,
                    is_market_open BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL
                )
            '''),
            ('alerts', '''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    symbol TEXT,
                    message TEXT NOT NULL,
                    details TEXT,
                    is_acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    created_at TEXT NOT NULL
                )
            '''),
            ('data_quality_log', '''
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.0,
                    issues TEXT,
                    warnings TEXT,
                    errors TEXT,
                    data_freshness_minutes INTEGER DEFAULT 0,
                    api_latency_ms INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            '''),
            ('ohlc_candles', '''
                CREATE TABLE IF NOT EXISTS ohlc_candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            '''),
            ('greeks_analysis', '''
                CREATE TABLE IF NOT EXISTS greeks_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    delta REAL DEFAULT 0.0,
                    gamma REAL DEFAULT 0.0,
                    theta REAL DEFAULT 0.0,
                    vega REAL DEFAULT 0.0,
                    rho REAL DEFAULT 0.0,
                    implied_volatility REAL DEFAULT 0.0,
                    historical_volatility REAL DEFAULT 0.0,
                    iv_percentile REAL DEFAULT 0.0,
                    iv_rank REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
            '''),
            ('volatility_surface', '''
                CREATE TABLE IF NOT EXISTS volatility_surface (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strike_price INTEGER NOT NULL,
                    expiry_date TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    implied_volatility REAL DEFAULT 0.0,
                    moneyness REAL DEFAULT 0.0,
                    days_to_expiry INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            '''),
            ('strategy_signals', '''
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    signal_strength REAL DEFAULT 0.0,
                    entry_price REAL DEFAULT 0.0,
                    stop_loss REAL DEFAULT 0.0,
                    take_profit REAL DEFAULT 0.0,
                    option_symbol TEXT,
                    strike_price INTEGER,
                    option_type TEXT,
                    implied_volatility REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    market_conditions TEXT,
                    created_at TEXT NOT NULL
                )
            '''),
            ('performance_metrics', '''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    sharpe_ratio REAL DEFAULT 0.0,
                    avg_trade_pnl REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
            ''')
        ]
        
        for table_name, create_sql in tables_to_create:
            logger.info(f"Creating table: {table_name}")
            cursor.execute(create_sql)
        
        # Create indexes for performance
        indexes = [
            ("idx_raw_symbol_timestamp", "raw_options_chain(symbol, timestamp)"),
            ("idx_raw_timestamp", "raw_options_chain(timestamp)"),
            ("idx_raw_quality", "raw_options_chain(data_quality_score)"),
            ("idx_options_symbol_timestamp", "options_data(symbol, timestamp)"),
            ("idx_options_strike_type", "options_data(strike_price, option_type)"),
            ("idx_options_iv", "options_data(implied_volatility)"),
            ("idx_options_volume", "options_data(volume)"),
            ("idx_market_symbol_timestamp", "market_summary(symbol, timestamp)"),
            ("idx_market_pcr", "market_summary(pcr)"),
            ("idx_market_vix", "market_summary(indiavix)"),
            ("idx_ohlc_symbol_timestamp", "ohlc_candles(symbol, timestamp)"),
            ("idx_ohlc_timeframe", "ohlc_candles(timeframe)"),
            ("idx_alerts_timestamp", "alerts(timestamp)"),
            ("idx_alerts_type", "alerts(alert_type)"),
            ("idx_alerts_severity", "alerts(severity)"),
            ("idx_alerts_acknowledged", "alerts(is_acknowledged)"),
            ("idx_quality_symbol_timestamp", "data_quality_log(symbol, timestamp)"),
            ("idx_quality_score", "data_quality_log(quality_score)"),
            ("idx_greeks_symbol_timestamp", "greeks_analysis(symbol, timestamp)"),
            ("idx_greeks_iv", "greeks_analysis(implied_volatility)"),
            ("idx_vol_surface_symbol_timestamp", "volatility_surface(symbol, timestamp)"),
            ("idx_vol_surface_strike_expiry", "volatility_surface(strike_price, expiry_date)"),
            ("idx_signals_symbol_timestamp", "strategy_signals(symbol, timestamp)"),
            ("idx_signals_strategy", "strategy_signals(strategy_name)"),
            ("idx_signals_type", "strategy_signals(signal_type)"),
            ("idx_performance_symbol_strategy", "performance_metrics(symbol, strategy_name)"),
            ("idx_performance_timestamp", "performance_metrics(timestamp)")
        ]
        
        for index_name, index_sql in indexes:
            try:
                logger.info(f"Creating index: {index_name}")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {index_sql}")
            except Exception as e:
                logger.warning(f"Index {index_name} already exists or failed: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database migration failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_database() 