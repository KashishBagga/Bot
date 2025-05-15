"""
Database module for the trading bot application.
Handles all database operations, table setup, and data logging.
"""
import sqlite3
from datetime import datetime
import importlib
import os
from src.config.settings import DATABASE_PATH, TIMEZONE
import logging

class Database:
    """Database manager for the trading bot application."""
    
    def __init__(self, db_path=DATABASE_PATH):
        """Initialize database connection with provided path."""
        self.db_path = db_path
    
    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query, params=None, commit=True):
        """Execute a SQL query with optional parameters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if commit:
                    conn.commit()
                
                return cursor
            except Exception as e:
                logging.error(f"Database error: {e}")
                conn.rollback()
                raise
    
    def setup_signals_table(self):
        """Create the signals table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_time TEXT,
                index_name TEXT,
                signal TEXT,
                strike_price INTEGER,
                stop_loss INTEGER,
                target INTEGER,
                target2 INTEGER,
                target3 INTEGER,
                price REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                ema_20 REAL,
                atr REAL,
                outcome TEXT,
                rsi_reason TEXT,
                macd_reason TEXT,
                price_reason TEXT,
                confidence TEXT,
                trade_type TEXT,
                option_chain_confirmation TEXT,
                pnl REAL,
                targets_hit INTEGER,
                stoploss_count INTEGER,
                failure_reason TEXT,
                UNIQUE(index_name, signal_time)
            )
        """
        self.execute_query(query)
    
    def setup_backtesting_table(self):
        """Create the backtesting table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS backtesting (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_time TEXT,
                index_name TEXT,
                signal TEXT,
                strike_price INTEGER,
                stop_loss INTEGER,
                target INTEGER,
                target2 INTEGER,
                target3 INTEGER,
                price REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                ema_20 REAL,
                atr REAL,
                outcome TEXT,
                rsi_reason TEXT,
                macd_reason TEXT,
                price_reason TEXT,
                confidence TEXT,
                trade_type TEXT,
                option_chain_confirmation TEXT,
                pnl REAL,
                targets_hit INTEGER,
                stoploss_count INTEGER,
                failure_reason TEXT
            )
        """
        self.execute_query(query)
    
    def log_trade(self, index_name, signal_data):
        """Log a trade signal to the database."""
        self.setup_signals_table()
        
        # Calculate targets based on ATR
        atr = signal_data.get('atr', 0)
        stop_loss = int(round(atr))
        target = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))

        # Ensure signal_time is present or add it
        if 'signal_time' not in signal_data:
            signal_data['signal_time'] = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        
        # Set strike price as rounded to nearest 50
        strike_price = int(round(signal_data.get('price', 0) / 50) * 50)

        query = """
            INSERT OR IGNORE INTO signals (
                signal_time, index_name, signal, strike_price, stop_loss, target, target2, target3,
                price, rsi, macd, macd_signal, ema_20, atr, outcome,
                rsi_reason, macd_reason, price_reason, confidence, trade_type,
                option_chain_confirmation, pnl, targets_hit, stoploss_count, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            signal_data.get('signal_time'),
            index_name,
            signal_data.get('signal'),
            strike_price,
            stop_loss,
            target,
            target2,
            target3,
            float(signal_data.get('price', 0)),
            float(signal_data.get('rsi', 0)),
            float(signal_data.get('macd', 0)),
            float(signal_data.get('macd_signal', 0)),
            float(signal_data.get('ema_20', 0)),
            float(signal_data.get('atr', 0)),
            "Pending",
            signal_data.get('rsi_reason', ''),
            signal_data.get('macd_reason', ''),
            signal_data.get('price_reason', ''),
            signal_data.get('confidence', 'Low'),
            signal_data.get('trade_type', 'Intraday'),
            signal_data.get('option_chain_confirmation', 'No'),
            0.0,
            0,
            0,
            ""
        )
        
        try:
            self.execute_query(query, params)
            print(f"✅ Trade logged in SQLite: {signal_data.get('signal')} at {signal_data.get('price')}")
            return True
        except Exception as e:
            print(f"❌ Failed to insert signal: {e}")
            return False
    
    def log_backtesting_trade(self, index_name, signal_data):
        """Log a backtesting trade to the database."""
        self.setup_backtesting_table()
        
        # Calculate targets based on ATR
        atr = signal_data.get('atr', 0)
        stop_loss = int(round(atr))
        target = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))

        # Ensure signal_time is present
        signal_time = signal_data.get('signal_time')

        query = """
            INSERT INTO backtesting (
                signal_time, index_name, signal, strike_price, stop_loss, target, target2, target3,
                price, rsi, macd, macd_signal, ema_20, atr, outcome,
                rsi_reason, macd_reason, price_reason, confidence, trade_type,
                option_chain_confirmation, pnl, targets_hit, stoploss_count, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            signal_time,
            index_name,
            signal_data.get('signal'),
            int(round(signal_data.get('price', 0) / 50) * 50),
            stop_loss,
            target,
            target2,
            target3,
            float(signal_data.get('price', 0)),
            float(signal_data.get('rsi', 0)),
            float(signal_data.get('macd', 0)),
            float(signal_data.get('macd_signal', 0)),
            float(signal_data.get('ema_20', 0)),
            float(signal_data.get('atr', 0)),
            signal_data.get('outcome', 'Pending'),
            signal_data.get('rsi_reason', ''),
            signal_data.get('macd_reason', ''),
            signal_data.get('price_reason', ''),
            signal_data.get('confidence', 'Low'),
            signal_data.get('trade_type', 'Intraday'),
            signal_data.get('option_chain_confirmation', 'No'),
            signal_data.get('pnl', 0.0),
            signal_data.get('targets_hit', 0),
            signal_data.get('stoploss_count', 0),
            signal_data.get('failure_reason', '')
        )
        
        self.execute_query(query, params)
        print(f"✅ Backtesting trade logged in SQLite: {signal_data.get('signal')} at {signal_data.get('price')}")
    
    def log_strategy_trade(self, strategy_name, signal_data):
        """Log a trade for a specific strategy."""
        # Dynamically import the appropriate schema module
        try:
            schema_module = importlib.import_module(f"src.models.schema.{strategy_name}")
            setup_func = getattr(schema_module, f"setup_{strategy_name}_table")
            fields_list = getattr(schema_module, f"{strategy_name}_fields")
        except (ImportError, AttributeError) as e:
            print(f"Error loading schema for {strategy_name}: {e}")
            print(f"Creating generic schema for {strategy_name}")
            # Fall back to a generic setup if specific schema not found
            from src.models.schema.generic import setup_generic_table, generic_fields
            setup_func = setup_generic_table
            fields_list = generic_fields
        
        # Call the setup function to ensure the table exists
        setup_func(self)
        
        # Generate the query and parameters dynamically from the fields list
        placeholders = ", ".join(["?"] * len(fields_list))
        columns = ", ".join(fields_list)
        
        query = f"""
            INSERT INTO {strategy_name} ({columns})
            VALUES ({placeholders})
        """
        
        # Extract values from signal_data based on fields_list
        params = [signal_data.get(field, None) for field in fields_list]
        
        self.execute_query(query, params)
        print(f"✅ Strategy trade logged in SQLite: {strategy_name} - {signal_data.get('signal')} at {signal_data.get('price')}")

# Create a database instance for direct imports
db = Database() 