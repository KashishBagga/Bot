#!/usr/bin/env python3
"""
Updated Unified Trading Database
===============================

Enhanced database with raw options chain storage capabilities.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

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
                    status TEXT DEFAULT 'OPEN',
                    exit_price REAL DEFAULT NULL,
                    exit_time TEXT DEFAULT NULL,
                    pnl REAL DEFAULT NULL,
                    exit_reason TEXT DEFAULT NULL,
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
            
            # Create options_data table for storing individual option contracts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS options_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    expiry_date TEXT NOT NULL,
                    lot_size INTEGER NOT NULL,
                    underlying_price REAL,
                    bid_price REAL,
                    ask_price REAL,
                    last_traded_price REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility REAL,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    bid_ask_spread REAL,
                    data_quality_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create raw_options_chain table for storing raw Fyers data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_options_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    raw_data JSON NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create live trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_trading_signals (
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
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rejected_signals_timestamp ON rejected_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rejected_signals_symbol ON rejected_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_open_option_positions_timestamp ON open_option_positions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_open_option_positions_symbol ON open_option_positions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_option_positions_timestamp ON closed_option_positions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_closed_option_positions_symbol ON closed_option_positions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_equity_curve_timestamp ON equity_curve(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_data_timestamp ON options_data(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_data_symbol ON options_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_data_strike ON options_data(strike_price)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_data_expiry ON options_data(expiry_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_options_chain_timestamp ON raw_options_chain(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_options_chain_symbol ON raw_options_chain(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trading_signals_timestamp ON live_trading_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trading_signals_symbol ON live_trading_signals(symbol)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Updated unified database schema created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating database schema: {e}")
            raise

    def save_options_data(self, options_data: Dict):
        """Save processed options data directly to options_data table."""
        try:
            symbol = options_data.get('symbol', 'UNKNOWN')
            timestamp = options_data.get('timestamp', datetime.now().isoformat())
            
            # Get the options chain from the Fyers API response structure
            data = options_data.get('data', {})
            options_chain = data.get('optionsChain', [])
            underlying_price = data.get('underlyingPrice', 0)
            
            # Process individual options
            processed_count = 0
            total_options = len(options_chain)
            
            if total_options == 0:
                logger.warning(f"⚠️ No options in chain for {symbol}")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for option in options_chain:
                try:
                    # Extract option details from Fyers structure
                    option_symbol = option.get('symbol', '')
                    option_type = option.get('option_type', '')
                    strike_price = option.get('strike_price', 0)
                    
                    # Skip if essential data is missing
                    if not option_symbol or not option_type or strike_price <= 0:
                        continue
                    
                    # Extract other fields with defaults
                    ltp = option.get('ltp', 0)
                    bid = option.get('bid', 0)
                    ask = option.get('ask', 0)
                    volume = option.get('volume', 0)
                    oi = option.get('oi', 0)
                    
                    # Calculate derived fields
                    bid_ask_spread = ask - bid if ask > 0 and bid > 0 else 0
                    
                    # Calculate data quality score for individual option
                    option_quality = 0
                    if bid > 0 and ask > 0:
                        option_quality += 30  # Valid bid/ask
                    if ltp > 0:
                        option_quality += 20  # Valid LTP
                    if volume > 0:
                        option_quality += 20  # Valid volume
                    if oi > 0:
                        option_quality += 20  # Valid OI
                    if bid_ask_spread > 0:
                        option_quality += 10  # Valid spread
                    
                    # Insert individual option data
                    cursor.execute("""
                        INSERT OR REPLACE INTO options_data 
                        (timestamp, symbol, option_symbol, option_type, strike_price,
                         expiry_date, lot_size, underlying_price, bid_price, ask_price, 
                         last_traded_price, volume, open_interest, implied_volatility, 
                         delta, gamma, theta, vega, bid_ask_spread, data_quality_score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, symbol, option_symbol, option_type, strike_price,
                        timestamp,  # Using timestamp as expiry_date for now
                        50,  # Default lot size for Nifty
                        underlying_price, bid, ask, ltp, volume, oi,
                        0, 0, 0, 0, 0,  # Greeks placeholder
                        bid_ask_spread, option_quality, datetime.now().isoformat()
                    ))
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error processing option {option.get('strikePrice', 'UNKNOWN')}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Options data saved for {symbol}: {processed_count}/{total_options} valid options")
            return processed_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error saving options data: {e}")
            raise

    def update_option_position_status(self, trade_id: str, status: str, exit_price: float = None, 
                                    exit_time: str = None, pnl: float = None, exit_reason: str = None):
        """Update option position status when trade is closed."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First, check if trade_id column exists, if not add it
            cursor.execute("PRAGMA table_info(open_option_positions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'trade_id' not in columns:
                cursor.execute("ALTER TABLE open_option_positions ADD COLUMN trade_id TEXT")
                conn.commit()
            
            if status == 'CLOSED':
                cursor.execute("""
                    UPDATE open_option_positions 
                    SET status = ?, exit_price = ?, exit_time = ?, pnl = ?, exit_reason = ?, trade_id = ?
                    WHERE id = (SELECT id FROM open_option_positions WHERE trade_id = ? LIMIT 1)
                """, (status, exit_price, exit_time, pnl, exit_reason, trade_id, trade_id))
            else:
                cursor.execute("""
                    UPDATE open_option_positions 
                    SET status = ?, trade_id = ?
                    WHERE id = (SELECT id FROM open_option_positions WHERE trade_id = ? LIMIT 1)
                """, (status, trade_id, trade_id))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Updated option position status for {trade_id} to {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating option position status: {e}")
            return False

    def get_latest_options_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest options data for a symbol from the options_data table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM options_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (symbol, limit))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            options_data = []
            for row in rows:
                option_dict = dict(zip(columns, row))
                options_data.append(option_dict)
            
            conn.close()
            return options_data
            
        except Exception as e:
            logger.error(f"❌ Error getting latest options data for {symbol}: {e}")
            return []

    def _calculate_data_quality_score(self, data: Dict) -> float:
        """Calculate data quality score (0-100)."""
        try:
            score = 100.0
            options_chain = data.get('optionsChain', [])
            
            # Check for basic structure
            if not options_chain:
                score -= 30
            
            # Check for essential fields
            if not data.get('callOi'):
                score -= 20
            if not data.get('putOi'):
                score -= 20
            
            # Check for underlying price
            if not data.get('underlying_price'):
                score -= 15
            
            # Check for valid options
            valid_options = 0
            for option in options_chain:
                if option.get('strike_price', 0) > 0:
                    valid_options += 1
            
            if valid_options < 10:
                score -= 15
            
            return max(0, score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating data quality score: {e}")
            return 0.0
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:15 AM - 3:30 PM IST, Mon-Fri)."""
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            
            # Check if it's a weekday (Monday = 0, Sunday = 6)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Check if it's within market hours (9:15 AM - 3:30 PM IST)
            market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_start <= now <= market_end
            
        except Exception as e:
            logger.error(f"❌ Error checking market hours: {e}")
            return True  # Default to True if we can't determine

    def save_trading_signal(self, signal_data: Dict) -> bool:
        """Save trading signal to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trading_signals 
                (timestamp, symbol, strategy, signal_type, confidence_score, price, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('timestamp'),
                signal_data.get('symbol'),
                signal_data.get('strategy'),
                signal_data.get('signal_type'),
                signal_data.get('confidence', 0),
                signal_data.get('price', 0),
                signal_data.get('created_at')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Trading signal saved: {signal_data.get('strategy')} {signal_data.get('signal_type')} for {signal_data.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving trading signal: {e}")
            return False

    def save_rejected_signal(self, rejected_data: Dict) -> bool:
        """Save rejected signal to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO rejected_signals 
                (timestamp, symbol, strategy, signal_type, confidence_score, rejection_reason, price, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rejected_data.get('timestamp'),
                rejected_data.get('symbol'),
                rejected_data.get('strategy'),
                rejected_data.get('signal_type'),
                rejected_data.get('confidence', 0),
                rejected_data.get('rejection_reason', ''),
                rejected_data.get('price', 0),
                rejected_data.get('created_at')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Rejected signal saved: {rejected_data.get('strategy')} {rejected_data.get('signal_type')} for {rejected_data.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving rejected signal: {e}")
            return False

    def get_trading_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trading signals from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM trading_signals 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM trading_signals 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            signals = []
            for row in rows:
                signal_dict = dict(zip(columns, row))
                signals.append(signal_dict)
            
            conn.close()
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting trading signals: {e}")
            return []

    def get_rejected_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get rejected signals from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM rejected_signals 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM rejected_signals 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            signals = []
            for row in rows:
                signal_dict = dict(zip(columns, row))
                signals.append(signal_dict)
            
            conn.close()
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting rejected signals: {e}")
            return []

    def save_live_trading_signal(self, signal_data: Dict) -> bool:
        """Save live trading signal to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO live_trading_signals 
                (timestamp, symbol, strategy, signal_type, confidence_score, price, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('timestamp'),
                signal_data.get('symbol'),
                signal_data.get('strategy'),
                signal_data.get('signal_type'),
                signal_data.get('confidence', 0),
                signal_data.get('price', 0),
                signal_data.get('volume', 0),
                signal_data.get('created_at')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Live trading signal saved: {signal_data.get('strategy')} {signal_data.get('signal_type')} for {signal_data.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving live trading signal: {e}")
            return False

    def save_capital_rejection_log(self, rejection_data: Dict) -> bool:
        """Save capital rejection log to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO capital_rejection_logs (
                    timestamp, symbol, strategy, signal_type, 
                    required_capital, available_capital, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rejection_data.get('timestamp'),
                rejection_data.get('symbol'),
                rejection_data.get('strategy'),
                rejection_data.get('signal_type'),
                rejection_data.get('required_capital'),
                rejection_data.get('available_capital'),
                rejection_data.get('reason')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Capital rejection log saved: {rejection_data.get('strategy')} {rejection_data.get('signal_type')} for {rejection_data.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving capital rejection log: {e}")
            return False

    def save_raw_options_chain(self, raw_option_chain: Dict) -> bool:
        """Save raw options chain data to database."""
        try:
            # Extract basic info
            symbol = raw_option_chain.get('symbol', 'UNKNOWN')
            timestamp = raw_option_chain.get('timestamp', datetime.now().isoformat())
            
            # Get the options chain from the Fyers API response structure
            data = raw_option_chain.get('data', {})
            options_chain = data.get('optionsChain', [])
            underlying_price = data.get('underlyingPrice', 0)
            
            # Process individual options
            processed_count = 0
            total_options = len(options_chain)
            
            if total_options == 0:
                logger.warning(f"⚠️ No options in chain for {symbol}")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for option in options_chain:
                try:
                    # Extract option details from Fyers structure
                    option_symbol = option.get('symbol', '')
                    option_type = option.get('option_type', '')
                    strike_price = option.get('strike_price', 0)
                    
                    # Skip if essential data is missing
                    if not option_symbol or not option_type or strike_price <= 0:
                        continue
                    
                    # Extract other fields with defaults
                    ltp = option.get('ltp', 0)
                    bid = option.get('bid', 0)
                    ask = option.get('ask', 0)
                    volume = option.get('volume', 0)
                    oi = option.get('oi', 0)
                    
                    # Calculate derived fields
                    bid_ask_spread = ask - bid if ask > 0 and bid > 0 else 0
                    
                    # Calculate data quality score for individual option
                    option_quality = 0
                    if bid > 0 and ask > 0:
                        option_quality += 30  # Valid bid/ask
                    if ltp > 0:
                        option_quality += 20  # Valid LTP
                    if volume > 0:
                        option_quality += 20  # Valid volume
                    if oi > 0:
                        option_quality += 20  # Valid OI
                    if bid_ask_spread > 0:
                        option_quality += 10  # Valid spread
                    
                    # Insert individual option data
                    cursor.execute("""
                        INSERT OR REPLACE INTO options_data 
                        (timestamp, symbol, option_symbol, option_type, strike_price,
                         expiry_date, lot_size, underlying_price, bid_price, ask_price, 
                         last_traded_price, volume, open_interest, implied_volatility, 
                         delta, gamma, theta, vega, bid_ask_spread, data_quality_score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, symbol, option_symbol, option_type, strike_price,
                        timestamp,  # Using timestamp as expiry_date for now
                        50,  # Default lot size for Nifty
                        underlying_price, bid, ask, ltp, volume, oi,
                        0, 0, 0, 0, 0,  # Greeks placeholder
                        bid_ask_spread, option_quality, datetime.now().isoformat()
                    ))
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error processing option {option.get('strikePrice', 'UNKNOWN')}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Raw options chain saved for {symbol}: {processed_count}/{total_options} valid options")
            return processed_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error saving raw options chain: {e}")
            return False

    def get_live_trading_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get live trading signals from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM live_trading_signals 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM live_trading_signals 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            signals = []
            for row in rows:
                signal = dict(zip(columns, row))
                signals.append(signal)
            
            conn.close()
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting live trading signals: {e}")
            return []

# Legacy compatibility
class UnifiedDatabase(UnifiedTradingDatabase):
    """Legacy class for backward compatibility."""
    pass 
    def save_open_option_position(self, trade):
        """Save open option position to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create option_positions table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS option_positions (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    contract_symbol TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    lot_size INTEGER NOT NULL,
                    strike REAL NOT NULL,
                    expiry TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    commission REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    reasoning TEXT DEFAULT '',
                    stop_loss REAL,
                    target1 REAL,
                    target2 REAL,
                    target3 REAL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    exit_reason TEXT,
                    entry_value REAL,
                    entry_commission REAL,
                    entry_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("""
                INSERT OR REPLACE INTO option_positions 
                (trade_id, timestamp, contract_symbol, underlying, strategy, signal_type,
                 entry_price, quantity, lot_size, strike, expiry, option_type, status,
                 commission, confidence, reasoning, stop_loss, target1, target2, target3,
                 exit_price, exit_time, pnl, exit_reason, entry_value, entry_commission, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.id, 
                trade.timestamp.isoformat(), 
                trade.contract_symbol,
                trade.underlying, 
                trade.strategy, 
                trade.signal_type,
                trade.entry_price, 
                trade.quantity, 
                trade.lot_size,
                trade.strike, 
                trade.expiry.isoformat() if trade.expiry else None, 
                trade.option_type,
                trade.status, 
                trade.commission, 
                trade.confidence,
                trade.reasoning, 
                trade.stop_loss, 
                trade.target1, 
                trade.target2, 
                trade.target3,
                trade.exit_price, 
                trade.exit_time.isoformat() if trade.exit_time else None, 
                trade.pnl, 
                trade.exit_reason, 
                trade.entry_value, 
                trade.entry_commission,
                trade.entry_time.isoformat() if trade.entry_time else None
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"✅ Saved open option position: {trade.id[:8]}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving open option position: {e}")
            return False

    def fetch_open_option_positions(self):
        """Fetch all open option positions from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM option_positions 
                WHERE status = 'OPEN'
                ORDER BY timestamp DESC
            """)
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            positions = []
            for row in rows:
                position = dict(zip(columns, row))
                positions.append(position)
            
            conn.close()
            return positions
            
        except Exception as e:
            logger.error(f"❌ Error fetching open option positions: {e}")
            return []

        """Save open option position to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS option_positions (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    contract_symbol TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    lot_size INTEGER NOT NULL,
                    strike REAL NOT NULL,
                    expiry TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    commission REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    reasoning TEXT DEFAULT '',
                    entry_value REAL,
                    entry_commission REAL,
                    entry_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("""
                INSERT OR REPLACE INTO option_positions 
                (trade_id, timestamp, contract_symbol, underlying, strategy, signal_type,
                 entry_price, quantity, lot_size, strike, expiry, option_type, status,
                 commission, confidence, reasoning, entry_value, entry_commission, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.id, trade.timestamp.isoformat(), trade.contract_symbol,
                trade.underlying, trade.strategy, trade.signal_type,
                trade.entry_price, trade.quantity, trade.lot_size,
                trade.strike, trade.expiry.isoformat() if trade.expiry else None, 
                trade.option_type, trade.status, trade.commission, 
                trade.confidence, trade.reasoning, trade.entry_value, 
                trade.entry_commission, trade.entry_time.isoformat() if trade.entry_time else None
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving open option position: {e}")
            return False

class UnifiedDatabase(UnifiedTradingDatabase):
    """Legacy class for backward compatibility."""
    pass 
    def save_open_option_position(self, trade):
        """Save open option position to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create option_positions table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS option_positions (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    contract_symbol TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    lot_size INTEGER NOT NULL,
                    strike REAL NOT NULL,
                    expiry TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    commission REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    reasoning TEXT DEFAULT '',
                    stop_loss REAL,
                    target1 REAL,
                    target2 REAL,
                    target3 REAL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    exit_reason TEXT,
                    entry_value REAL,
                    entry_commission REAL,
                    entry_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("""
                INSERT OR REPLACE INTO option_positions 
                (trade_id, timestamp, contract_symbol, underlying, strategy, signal_type,
                 entry_price, quantity, lot_size, strike, expiry, option_type, status,
                 commission, confidence, reasoning, stop_loss, target1, target2, target3,
                 exit_price, exit_time, pnl, exit_reason, entry_value, entry_commission, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.id, 
                trade.timestamp.isoformat(), 
                trade.contract_symbol,
                trade.underlying, 
                trade.strategy, 
                trade.signal_type,
                trade.entry_price, 
                trade.quantity, 
                trade.lot_size,
                trade.strike, 
                trade.expiry.isoformat() if trade.expiry else None, 
                trade.option_type,
                trade.status, 
                trade.commission, 
                trade.confidence,
                trade.reasoning, 
                trade.stop_loss, 
                trade.target1, 
                trade.target2, 
                trade.target3,
                trade.exit_price, 
                trade.exit_time.isoformat() if trade.exit_time else None, 
                trade.pnl, 
                trade.exit_reason, 
                trade.entry_value, 
                trade.entry_commission,
                trade.entry_time.isoformat() if trade.entry_time else None
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"✅ Saved open option position: {trade.id[:8]}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving open option position: {e}")
            return False

    def fetch_open_option_positions(self):
        """Fetch all open option positions from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM option_positions 
                WHERE status = 'OPEN'
                ORDER BY timestamp DESC
            """)
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            positions = []
            for row in rows:
                position = dict(zip(columns, row))
                positions.append(position)
            
            conn.close()
            return positions
            
        except Exception as e:
            logger.error(f"❌ Error fetching open option positions: {e}")
            return []

        """Save open option position to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS option_positions (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    contract_symbol TEXT NOT NULL,
                    underlying TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    lot_size INTEGER NOT NULL,
                    strike REAL NOT NULL,
                    expiry TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    commission REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    reasoning TEXT DEFAULT '',
                    entry_value REAL,
                    entry_commission REAL,
                    entry_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute("""
                INSERT OR REPLACE INTO option_positions 
                (trade_id, timestamp, contract_symbol, underlying, strategy, signal_type,
                 entry_price, quantity, lot_size, strike, expiry, option_type, status,
                 commission, confidence, reasoning, entry_value, entry_commission, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.id, trade.timestamp.isoformat(), trade.contract_symbol,
                trade.underlying, trade.strategy, trade.signal_type,
                trade.entry_price, trade.quantity, trade.lot_size,
                trade.strike, trade.expiry.isoformat() if trade.expiry else None, 
                trade.option_type, trade.status, trade.commission, 
                trade.confidence, trade.reasoning, trade.entry_value, 
                trade.entry_commission, trade.entry_time.isoformat() if trade.entry_time else None
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving open option position: {e}")
            return False

    def save_open_trade(self, trade_id: str, symbol: str, strategy: str, signal: str, 
                       entry_price: float, quantity: float, entry_time: datetime,
                       stop_loss_price: float, take_profit_price: float) -> bool:
        """Save an open trade to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO open_trades 
                (trade_id, symbol, strategy, signal, entry_price, quantity, entry_time, 
                 stop_loss_price, take_profit_price, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (trade_id, symbol, strategy, signal, entry_price, quantity, 
                  entry_time.isoformat(), stop_loss_price, take_profit_price, 'OPEN'))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving open trade: {e}")
            return False
    
    def close_trade(self, trade_id: str, exit_price: float, exit_time: datetime, 
                   exit_reason: str, pnl: float) -> bool:
        """Close a trade and move to closed trades table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the open trade
            cursor.execute('SELECT * FROM open_trades WHERE trade_id = ?', (trade_id,))
            open_trade = cursor.fetchone()
            
            if not open_trade:
                logger.warning(f"Open trade {trade_id} not found")
                return False
            
            # Insert into closed trades
            cursor.execute('''
                INSERT INTO closed_trades 
                (trade_id, symbol, strategy, signal, entry_price, exit_price, quantity, 
                 entry_time, exit_time, exit_reason, pnl, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (trade_id, open_trade[1], open_trade[2], open_trade[3], open_trade[4], 
                  exit_price, open_trade[5], open_trade[6], exit_time.isoformat(), 
                  exit_reason, pnl, 'CLOSED'))
            
            # Remove from open trades
            cursor.execute('DELETE FROM open_trades WHERE trade_id = ?', (trade_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return False

            # Create open trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS open_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
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
            
            # Create closed trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
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
                    status TEXT DEFAULT 'CLOSED',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

