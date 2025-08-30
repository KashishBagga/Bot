#!/usr/bin/env python3
"""
Updated Unified Trading Database
===============================

Enhanced database with raw options chain storage capabilities.
"""

import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime
from src.config.settings import setup_logging

logger = setup_logging('unified_database')

class UnifiedTradingDatabase:
    """Unified database for all trading data including raw options chain."""
    
    def __init__(self, db_path: str = "unified_trading.db"):
        """Initialize the unified database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the unified database with all required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence_score REAL,
                    price REAL,
                    volume REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create rejected signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rejected_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence_score REAL,
                    rejection_reason TEXT NOT NULL,
                    price REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create open option positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS open_option_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    expiry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    confidence_score REAL,
                    entry_timestamp TEXT NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create closed option positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS closed_option_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    expiry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    confidence_score REAL,
                    entry_timestamp TEXT NOT NULL,
                    exit_timestamp TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create equity curve table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_exposure REAL NOT NULL,
                    max_drawdown REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create raw options chain table for storing complete API responses
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_options_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    raw_data TEXT NOT NULL,  -- JSON string of complete API response
                    call_oi INTEGER,
                    put_oi INTEGER,
                    indiavix REAL,
                    total_options INTEGER,
                    total_strikes INTEGER,
                    api_response_code INTEGER,
                    api_message TEXT,
                    api_status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rejected_signals_timestamp ON rejected_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rejected_signals_symbol ON rejected_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_open_positions_timestamp ON open_option_positions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_open_positions_symbol ON open_option_positions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_positions_timestamp ON closed_option_positions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_positions_symbol ON closed_option_positions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_equity_curve_timestamp ON equity_curve(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_options_chain_timestamp ON raw_options_chain(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_options_chain_symbol ON raw_options_chain(symbol)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Updated unified database schema created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating database schema: {e}")
            raise

    def save_raw_options_chain(self, raw_data: Dict):
        """Save raw option chain data to database for historical analysis.
        
        Args:
            raw_data: Raw response from Fyers Option Chain API
        """
        try:
            import json
            
            # Extract metadata
            metadata = raw_data.get('_metadata', {})
            symbol = metadata.get('symbol', '')
            timestamp = metadata.get('timestamp', '')
            api_response_code = metadata.get('api_response_code', 0)
            api_message = metadata.get('api_message', '')
            api_status = metadata.get('api_status', '')
            
            # Extract key metrics from the new data structure
            data = raw_data.get('data', {})
            options_chain = data.get('optionsChain', [])
            call_oi = data.get('callOi', 0)
            put_oi = data.get('putOi', 0)
            indiavix = data.get('indiavixData', {}).get('ltp', 0)
            
            # Count real strikes
            real_strikes = set()
            for option in options_chain:
                if option.get('option_type') in ['CE', 'PE']:
                    strike = option.get('strike_price', -1)
                    if strike > 0:
                        real_strikes.add(strike)
            
            total_strikes = len(real_strikes)
            total_options = len(options_chain)
            
            # Convert raw data to JSON string
            raw_data_json = json.dumps(raw_data, default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO raw_options_chain 
                    (timestamp, symbol, raw_data, call_oi, put_oi, indiavix, 
                     total_options, total_strikes, api_response_code, api_message, api_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    symbol,
                    raw_data_json,
                    call_oi,
                    put_oi,
                    indiavix,
                    total_options,
                    total_strikes,
                    api_response_code,
                    api_message,
                    api_status
                ))
                conn.commit()
                
            logger.info(f"✅ Saved raw option chain data for {symbol}: {total_options} options, {total_strikes} strikes")
            
        except Exception as e:
            logger.error(f"❌ Error saving raw option chain data: {e}")

    def get_raw_options_chain_summary(self) -> Dict:
        """Get summary of accumulated raw option chain data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM raw_options_chain")
                total_records = cursor.fetchone()[0]
                
                # Get unique symbols
                cursor.execute("SELECT DISTINCT symbol FROM raw_options_chain")
                symbols = [row[0] for row in cursor.fetchall()]
                
                # Get date range
                cursor.execute("""
                    SELECT MIN(timestamp), MAX(timestamp) 
                    FROM raw_options_chain
                """)
                date_range = cursor.fetchone()
                
                # Get records per symbol
                symbol_counts = {}
                for symbol in symbols:
                    cursor.execute("SELECT COUNT(*) FROM raw_options_chain WHERE symbol = ?", (symbol,))
                    symbol_counts[symbol] = cursor.fetchone()[0]
                
                # Get latest data for each symbol
                latest_data = {}
                for symbol in symbols:
                    cursor.execute("""
                        SELECT timestamp, call_oi, put_oi, indiavix, total_options, total_strikes
                        FROM raw_options_chain 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (symbol,))
                    latest = cursor.fetchone()
                    if latest:
                        latest_data[symbol] = {
                            'timestamp': latest[0],
                            'call_oi': latest[1],
                            'put_oi': latest[2],
                            'indiavix': latest[3],
                            'total_options': latest[4],
                            'total_strikes': latest[5]
                        }
                
                return {
                    'total_records': total_records,
                    'symbols': symbols,
                    'date_range': {
                        'start': date_range[0] if date_range[0] else None,
                        'end': date_range[1] if date_range[1] else None
                    },
                    'records_per_symbol': symbol_counts,
                    'latest_data': latest_data
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting raw options chain summary: {e}")
            return {}

    def get_raw_options_chain_data(self, symbol: str, start_date: str = None, end_date: str = None, limit: int = 1000) -> List[Dict]:
        """Get raw option chain data from database.
        
        Args:
            symbol: Trading symbol
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of raw option chain data records
        """
        try:
            import json
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM raw_options_chain WHERE symbol = ?"
                params = [symbol]
                
                if start_date:
                    query += " AND DATE(timestamp) >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND DATE(timestamp) <= ?"
                    params.append(end_date)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                columns = [description[0] for description in cursor.description]
                data = []
                for row in rows:
                    record = dict(zip(columns, row))
                    # Parse JSON raw_data
                    if record.get('raw_data'):
                        try:
                            record['raw_data'] = json.loads(record['raw_data'])
                        except:
                            pass
                    data.append(record)
                
                logger.info(f"✅ Retrieved {len(data)} raw option chain records for {symbol}")
                return data
                
        except Exception as e:
            logger.error(f"❌ Error retrieving raw options chain data: {e}")
            return []

# Legacy compatibility
class UnifiedDatabase(UnifiedTradingDatabase):
    """Legacy class for backward compatibility."""
    pass 